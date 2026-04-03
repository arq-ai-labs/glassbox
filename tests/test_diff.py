"""Tests for diff engine — section-level comparison between ContextPacks."""

import pytest

from glassbox.diff import diff_context_packs
from glassbox.format.context_pack import ContextPack, ContextPackMetrics
from glassbox.format.model_info import ModelInfo
from glassbox.format.output import OutputRecord
from glassbox.format.sections import SystemPromptSection, UserMessageSection
from glassbox.format.token_budget import SectionTokenAllocation, TokenBudget
from glassbox.format.version import FORMAT_VERSION


def _make_pack(step_id, step_index, sections):
    total = sum(s.token_count for s in sections)
    return ContextPack(
        format_version=FORMAT_VERSION,
        run_id="run_001",
        step_id=step_id,
        step_index=step_index,
        started_at="2026-03-29T10:00:00.000Z",
        completed_at="2026-03-29T10:00:01.000Z",
        sections=sections,
        token_budget=TokenBudget(
            total_budget=total, total_used=total,
            by_section=[
                SectionTokenAllocation(section_id=s.section_id, section_type=s.type, token_count=s.token_count)
                for s in sections
            ],
        ),
        model=ModelInfo(provider="test", model="test-model"),
        output=OutputRecord(type="text", text="output"),
        metrics=ContextPackMetrics(latency_ms=100, input_tokens=total, output_tokens=5),
    )


class TestDiffContextPacks:
    def test_identical_packs(self):
        sections = [
            SystemPromptSection(section_id="s1", token_count=10, content="Be helpful."),
            UserMessageSection(section_id="u1", token_count=5, content="Hello"),
        ]
        pack_a = _make_pack("step_0", 0, sections)
        pack_b = _make_pack("step_1", 1, sections)

        diff = diff_context_packs(pack_a, pack_b)
        assert diff.sections_added == 0
        assert diff.sections_removed == 0
        assert diff.sections_modified == 0
        assert diff.token_delta == 0
        assert all(c.change_type == "unchanged" for c in diff.changes)

    def test_added_section(self):
        sections_a = [
            SystemPromptSection(section_id="s1", token_count=10, content="System"),
        ]
        sections_b = [
            SystemPromptSection(section_id="s1", token_count=10, content="System"),
            UserMessageSection(section_id="u1", token_count=20, content="New message"),
        ]
        diff = diff_context_packs(
            _make_pack("step_0", 0, sections_a),
            _make_pack("step_1", 1, sections_b),
        )
        assert diff.sections_added == 1
        assert diff.sections_removed == 0
        assert diff.token_delta == 20

    def test_removed_section(self):
        sections_a = [
            SystemPromptSection(section_id="s1", token_count=10, content="System"),
            UserMessageSection(section_id="u1", token_count=20, content="Old message"),
        ]
        sections_b = [
            SystemPromptSection(section_id="s1", token_count=10, content="System"),
        ]
        diff = diff_context_packs(
            _make_pack("step_0", 0, sections_a),
            _make_pack("step_1", 1, sections_b),
        )
        assert diff.sections_added == 0
        assert diff.sections_removed == 1
        assert diff.token_delta == -20

    def test_modified_section(self):
        sections_a = [
            UserMessageSection(section_id="u1", token_count=10, content="Original"),
        ]
        sections_b = [
            UserMessageSection(section_id="u1", token_count=25, content="Modified with more content"),
        ]
        diff = diff_context_packs(
            _make_pack("step_0", 0, sections_a),
            _make_pack("step_1", 1, sections_b),
        )
        assert diff.sections_modified == 1
        assert diff.token_delta == 15

        modified = [c for c in diff.changes if c.change_type == "modified"]
        assert len(modified) == 1
        assert modified[0].from_tokens == 10
        assert modified[0].to_tokens == 25

    def test_empty_packs(self):
        diff = diff_context_packs(
            _make_pack("step_0", 0, []),
            _make_pack("step_1", 1, []),
        )
        assert diff.sections_added == 0
        assert diff.sections_removed == 0
        assert diff.sections_modified == 0
        assert diff.token_delta == 0
        assert len(diff.changes) == 0

    def test_complex_diff(self):
        """Mix of adds, removes, modifies, and unchanged."""
        sections_a = [
            SystemPromptSection(section_id="s1", token_count=10, content="System"),
            UserMessageSection(section_id="u1", token_count=20, content="User msg 1"),
            UserMessageSection(section_id="u2", token_count=15, content="User msg 2"),
        ]
        sections_b = [
            SystemPromptSection(section_id="s1", token_count=10, content="System"),  # unchanged
            UserMessageSection(section_id="u1", token_count=30, content="User msg 1 EDITED"),  # modified
            UserMessageSection(section_id="u3", token_count=25, content="New user msg"),  # added
            # u2 removed
        ]
        diff = diff_context_packs(
            _make_pack("step_0", 0, sections_a),
            _make_pack("step_1", 1, sections_b),
        )
        assert diff.sections_added == 1
        assert diff.sections_removed == 1
        assert diff.sections_modified == 1
        # Token delta: was 10+20+15=45, now 10+30+25=65, delta = +20
        assert diff.token_delta == 20

    def test_step_ids_preserved(self):
        diff = diff_context_packs(
            _make_pack("step_0", 0, []),
            _make_pack("step_1", 1, []),
        )
        assert diff.from_step_id == "step_0"
        assert diff.to_step_id == "step_1"
        assert diff.from_step_index == 0
        assert diff.to_step_index == 1
