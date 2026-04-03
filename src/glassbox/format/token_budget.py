"""Token budget — the allocation ledger and rejection record for a ContextPack."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


RejectionReason = Literal[
    "token_limit_exceeded",
    "relevance_below_threshold",
    "staleness",
    "redundant",
    "policy_excluded",
    "priority_displaced",
    "custom",
]
"""Why a candidate section was excluded from the ContextPack."""


class RejectedCandidate(BaseModel):
    """A candidate that was considered but not included."""

    section_type: str
    """What kind of section this would have been."""

    source: Optional[str] = None
    """Origin label for the rejected content."""

    token_count: int = Field(ge=0)
    """How many tokens this candidate would have consumed."""

    reason: RejectionReason
    """Why it was excluded."""

    reason_detail: Optional[str] = None
    """Human-readable explanation of the rejection."""


class SectionTokenAllocation(BaseModel):
    section_id: str
    section_type: str
    token_count: int = Field(ge=0)


class TokenBudget(BaseModel):
    """Token allocation and rejection ledger for a ContextPack."""

    total_budget: int = Field(ge=0)
    """Maximum tokens available for context assembly."""

    total_used: int = Field(ge=0)
    """Tokens actually consumed by included sections."""

    by_section: list[SectionTokenAllocation]
    """Per-section allocation breakdown."""

    rejected: list[RejectedCandidate] = Field(default_factory=list)
    """Candidates that were considered but excluded."""
