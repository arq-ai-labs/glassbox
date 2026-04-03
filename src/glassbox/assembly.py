"""Context assembler — builds a context window with token budget enforcement.

This is the piece that populates the rejection ledger. When you control context
assembly (RAG pipelines, agent scaffolding, etc.), use ContextAssembler to
track what was included, what was excluded, and why.

Usage:
    from glassbox import ContextAssembler
    from glassbox.format.sections import RetrievalSection

    assembler = ContextAssembler(budget=8000)

    # System prompt always fits
    assembler.add(SystemPromptSection(
        section_id="sys", source="system",
        token_count=500, content="You are a helpful assistant.",
    ))

    # RAG results — some may be rejected if budget is exceeded
    for doc in retrieved_docs:
        assembler.add(RetrievalSection(
            section_id=doc.id, source="rag:pinecone",
            token_count=estimate_tokens(doc.text), content=doc.text,
            query=query, score=doc.score,
        ))

    sections = assembler.sections          # what got in
    budget = assembler.build_budget()       # includes rejected candidates
"""

from __future__ import annotations

from typing import Optional

from .format.sections import ContextSection
from .format.token_budget import (
    RejectedCandidate,
    RejectionReason,
    SectionTokenAllocation,
    TokenBudget,
)


class ContextAssembler:
    """Builds a context window with token budget enforcement and rejection tracking.

    Sections are added in order. If a section would exceed the budget, it's
    recorded as a rejected candidate with the reason why.
    """

    def __init__(self, budget: int) -> None:
        """Initialize with a total token budget.

        Args:
            budget: Maximum tokens available for context assembly.
        """
        if budget < 0:
            raise ValueError(f"Budget must be non-negative, got {budget}")
        self.budget = budget
        self.sections: list[ContextSection] = []
        self.rejected: list[RejectedCandidate] = []
        self._used = 0

    @property
    def used(self) -> int:
        """Tokens consumed so far."""
        return self._used

    @property
    def remaining(self) -> int:
        """Tokens remaining in budget."""
        return max(0, self.budget - self._used)

    def add(
        self,
        section: ContextSection,
        *,
        reason: RejectionReason = "token_limit_exceeded",
        reason_detail: Optional[str] = None,
    ) -> bool:
        """Add a section if it fits within the budget.

        Args:
            section: The context section to add.
            reason: Rejection reason if the section doesn't fit (default: token_limit_exceeded).
            reason_detail: Optional human-readable explanation.

        Returns:
            True if the section was added, False if rejected.
        """
        if self._used + section.token_count <= self.budget:
            self.sections.append(section)
            self._used += section.token_count
            return True

        # Doesn't fit — record rejection
        self.rejected.append(RejectedCandidate(
            section_type=section.type,
            source=section.source,
            token_count=section.token_count,
            reason=reason,
            reason_detail=reason_detail or (
                f"Would need {section.token_count} tokens but only "
                f"{self.remaining} remaining of {self.budget} budget"
            ),
        ))
        return False

    def reject(
        self,
        section: ContextSection,
        reason: RejectionReason,
        reason_detail: Optional[str] = None,
    ) -> None:
        """Explicitly reject a section without attempting to add it.

        Use this when you know a section should be excluded for reasons other
        than token budget (e.g., policy, staleness, relevance).
        """
        self.rejected.append(RejectedCandidate(
            section_type=section.type,
            source=section.source,
            token_count=section.token_count,
            reason=reason,
            reason_detail=reason_detail,
        ))

    def build_budget(self) -> TokenBudget:
        """Build a complete TokenBudget with allocations and rejections."""
        return TokenBudget(
            total_budget=self.budget,
            total_used=self._used,
            by_section=[
                SectionTokenAllocation(
                    section_id=s.section_id,
                    section_type=s.type,
                    token_count=s.token_count,
                )
                for s in self.sections
            ],
            rejected=list(self.rejected),
        )

    def reset(self) -> None:
        """Clear all sections and rejections."""
        self.sections.clear()
        self.rejected.clear()
        self._used = 0
