"""Tests for ContextAssembler — token budget enforcement and rejection tracking."""

import pytest

from glassbox.assembly import ContextAssembler
from glassbox.format.sections import (
    RetrievalSection,
    SystemPromptSection,
    UserMessageSection,
)


def _section(section_id, token_count, section_type="user_message"):
    if section_type == "system_prompt":
        return SystemPromptSection(
            section_id=section_id, token_count=token_count, content="x" * token_count,
        )
    if section_type == "retrieval":
        return RetrievalSection(
            section_id=section_id, token_count=token_count, content="x" * token_count,
        )
    return UserMessageSection(
        section_id=section_id, token_count=token_count, content="x" * token_count,
    )


class TestContextAssembler:
    def test_add_within_budget(self):
        asm = ContextAssembler(budget=100)
        assert asm.add(_section("s1", 50)) is True
        assert asm.add(_section("s2", 30)) is True
        assert asm.used == 80
        assert asm.remaining == 20
        assert len(asm.sections) == 2
        assert len(asm.rejected) == 0

    def test_add_exceeds_budget(self):
        asm = ContextAssembler(budget=100)
        asm.add(_section("s1", 80))
        assert asm.add(_section("s2", 30)) is False
        assert asm.used == 80
        assert len(asm.sections) == 1
        assert len(asm.rejected) == 1
        assert asm.rejected[0].reason == "token_limit_exceeded"
        assert asm.rejected[0].token_count == 30

    def test_exact_budget(self):
        asm = ContextAssembler(budget=100)
        assert asm.add(_section("s1", 100)) is True
        assert asm.remaining == 0
        assert asm.add(_section("s2", 1)) is False

    def test_zero_budget(self):
        asm = ContextAssembler(budget=0)
        assert asm.add(_section("s1", 1)) is False
        assert len(asm.rejected) == 1

    def test_negative_budget_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ContextAssembler(budget=-1)

    def test_explicit_reject(self):
        asm = ContextAssembler(budget=1000)
        section = _section("stale1", 50, "retrieval")
        asm.reject(section, reason="staleness", reason_detail="Data is 7 days old")
        assert len(asm.rejected) == 1
        assert asm.rejected[0].reason == "staleness"
        assert asm.rejected[0].reason_detail == "Data is 7 days old"
        assert asm.used == 0  # explicit reject doesn't consume budget

    def test_mixed_add_and_reject(self):
        asm = ContextAssembler(budget=100)
        asm.add(_section("sys", 20, "system_prompt"))
        asm.reject(_section("policy_doc", 40, "retrieval"), reason="policy_excluded")
        asm.add(_section("user", 30))
        asm.add(_section("rag_big", 60, "retrieval"))  # won't fit

        assert len(asm.sections) == 2
        assert len(asm.rejected) == 2
        assert asm.rejected[0].reason == "policy_excluded"
        assert asm.rejected[1].reason == "token_limit_exceeded"

    def test_build_budget(self):
        asm = ContextAssembler(budget=200)
        asm.add(_section("s1", 80, "system_prompt"))
        asm.add(_section("u1", 50))
        asm.add(_section("r1", 100, "retrieval"))  # rejected

        budget = asm.build_budget()
        assert budget.total_budget == 200
        assert budget.total_used == 130
        assert len(budget.by_section) == 2
        assert len(budget.rejected) == 1
        assert budget.by_section[0].section_id == "s1"
        assert budget.by_section[1].section_id == "u1"

    def test_build_budget_empty(self):
        asm = ContextAssembler(budget=100)
        budget = asm.build_budget()
        assert budget.total_budget == 100
        assert budget.total_used == 0
        assert len(budget.by_section) == 0
        assert len(budget.rejected) == 0

    def test_reset(self):
        asm = ContextAssembler(budget=100)
        asm.add(_section("s1", 50))
        asm.add(_section("s2", 60))  # rejected
        assert asm.used == 50
        assert len(asm.rejected) == 1

        asm.reset()
        assert asm.used == 0
        assert asm.remaining == 100
        assert len(asm.sections) == 0
        assert len(asm.rejected) == 0

    def test_rejection_detail_auto_generated(self):
        asm = ContextAssembler(budget=50)
        asm.add(_section("s1", 40))
        asm.add(_section("s2", 20))  # rejected

        assert "20 tokens" in asm.rejected[0].reason_detail
        assert "10 remaining" in asm.rejected[0].reason_detail
        assert "50 budget" in asm.rejected[0].reason_detail

    def test_all_rejection_reasons(self):
        """All 7 rejection reasons should be accepted."""
        asm = ContextAssembler(budget=1000)
        reasons = [
            "token_limit_exceeded",
            "relevance_below_threshold",
            "staleness",
            "redundant",
            "policy_excluded",
            "priority_displaced",
            "custom",
        ]
        for reason in reasons:
            asm.reject(_section(f"s_{reason}", 10), reason=reason)
        assert len(asm.rejected) == 7
