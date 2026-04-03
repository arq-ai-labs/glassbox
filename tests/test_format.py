"""Tests for Pydantic format models — serialization, deserialization, discriminated union."""

import json
import pytest

from glassbox.format.sections import (
    AssistantMessageSection,
    CustomSection,
    InstructionSection,
    MemorySection,
    RetrievalSection,
    SystemPromptSection,
    ToolResultSection,
    UserMessageSection,
)
from glassbox.format.token_budget import (
    RejectedCandidate,
    SectionTokenAllocation,
    TokenBudget,
)
from glassbox.format.model_info import ModelInfo
from glassbox.format.output import OutputRecord, ToolCall
from glassbox.format.context_pack import ContextPack, ContextPackMetrics, MultiAgentLink
from glassbox.format.run_metadata import RunMetadata
from glassbox.format.version import FORMAT_VERSION


# ---------------------------------------------------------------------------
# Section types
# ---------------------------------------------------------------------------


class TestSectionTypes:
    def test_system_prompt(self):
        s = SystemPromptSection(
            section_id="s1", source="system",
            token_count=10, content="Be helpful.",
        )
        assert s.type == "system_prompt"
        assert s.token_count == 10

    def test_user_message(self):
        s = UserMessageSection(
            section_id="u1", source="user",
            token_count=5, content="Hello",
        )
        assert s.type == "user_message"

    def test_assistant_message(self):
        s = AssistantMessageSection(
            section_id="a1", source="assistant",
            token_count=8, content="Hi there!",
        )
        assert s.type == "assistant_message"

    def test_tool_result(self):
        s = ToolResultSection(
            section_id="t1", source="tool:search",
            token_count=20, content="result data",
            tool_name="search", tool_call_id="tc_123",
        )
        assert s.type == "tool_result"
        assert s.tool_name == "search"

    def test_retrieval(self):
        s = RetrievalSection(
            section_id="r1", source="rag:pinecone",
            token_count=100, content="retrieved doc",
            query="test query", score=0.95, document_id="doc_1",
        )
        assert s.type == "retrieval"
        assert s.score == 0.95

    def test_memory(self):
        s = MemorySection(
            section_id="m1", token_count=30, content="past context",
            memory_type="conversation",
        )
        assert s.type == "memory"

    def test_instruction(self):
        s = InstructionSection(
            section_id="i1", token_count=15, content="Follow these rules",
        )
        assert s.type == "instruction"

    def test_custom(self):
        s = CustomSection(
            section_id="c1", token_count=10, content="custom data",
            custom_type="my_app_context",
        )
        assert s.type == "custom"
        assert s.custom_type == "my_app_context"


# ---------------------------------------------------------------------------
# Section serialization round-trip
# ---------------------------------------------------------------------------


class TestSectionRoundTrip:
    def test_all_types_round_trip(self):
        sections = [
            SystemPromptSection(section_id="s1", token_count=10, content="sys"),
            UserMessageSection(section_id="u1", token_count=5, content="hi"),
            AssistantMessageSection(section_id="a1", token_count=8, content="hello"),
            ToolResultSection(section_id="t1", token_count=20, content="result", tool_name="search"),
            RetrievalSection(section_id="r1", token_count=100, content="doc", score=0.5),
            MemorySection(section_id="m1", token_count=30, content="memory", memory_type="long_term"),
            InstructionSection(section_id="i1", token_count=15, content="instructions"),
            CustomSection(section_id="c1", token_count=10, content="custom", custom_type="test"),
        ]
        for section in sections:
            data = section.model_dump(mode="json")
            json_str = json.dumps(data)
            restored = type(section).model_validate_json(json_str)
            assert restored.section_id == section.section_id
            assert restored.type == section.type
            assert restored.content == section.content

    def test_metadata_field(self):
        s = SystemPromptSection(
            section_id="s1", token_count=10, content="test",
            metadata={"version": "2.0", "author": "glassbox"},
        )
        data = s.model_dump(mode="json")
        assert data["metadata"]["version"] == "2.0"


# ---------------------------------------------------------------------------
# TokenBudget
# ---------------------------------------------------------------------------


class TestTokenBudget:
    def test_with_rejections(self):
        budget = TokenBudget(
            total_budget=1000,
            total_used=800,
            by_section=[
                SectionTokenAllocation(section_id="s1", section_type="system_prompt", token_count=300),
                SectionTokenAllocation(section_id="u1", section_type="user_message", token_count=500),
            ],
            rejected=[
                RejectedCandidate(
                    section_type="retrieval", token_count=300,
                    reason="token_limit_exceeded",
                    reason_detail="Would exceed budget by 100 tokens",
                ),
            ],
        )
        assert budget.total_budget == 1000
        assert len(budget.rejected) == 1
        assert budget.rejected[0].reason == "token_limit_exceeded"

    def test_empty_rejections(self):
        budget = TokenBudget(
            total_budget=500, total_used=200,
            by_section=[], rejected=[],
        )
        assert len(budget.rejected) == 0


# ---------------------------------------------------------------------------
# ContextPack full round-trip
# ---------------------------------------------------------------------------


class TestContextPack:
    def _make_pack(self, **overrides):
        defaults = dict(
            format_version=FORMAT_VERSION,
            run_id="run_001",
            agent_name="test-agent",
            step_id="step_001",
            step_index=0,
            started_at="2026-03-29T10:00:00.000Z",
            completed_at="2026-03-29T10:00:01.500Z",
            sections=[
                SystemPromptSection(section_id="s1", token_count=10, content="Be helpful."),
                UserMessageSection(section_id="u1", token_count=5, content="Hello"),
            ],
            token_budget=TokenBudget(
                total_budget=100, total_used=15,
                by_section=[
                    SectionTokenAllocation(section_id="s1", section_type="system_prompt", token_count=10),
                    SectionTokenAllocation(section_id="u1", section_type="user_message", token_count=5),
                ],
            ),
            model=ModelInfo(provider="openai", model="gpt-4o-mini"),
            output=OutputRecord(type="text", text="Hi there!"),
            metrics=ContextPackMetrics(latency_ms=1500, input_tokens=15, output_tokens=8),
        )
        defaults.update(overrides)
        return ContextPack(**defaults)

    def test_round_trip_json(self):
        pack = self._make_pack()
        json_str = json.dumps(pack.model_dump(mode="json"))
        restored = ContextPack.model_validate_json(json_str)
        assert restored.run_id == "run_001"
        assert restored.step_index == 0
        assert len(restored.sections) == 2
        assert restored.sections[0].type == "system_prompt"
        assert restored.metrics.latency_ms == 1500

    def test_multi_agent_link(self):
        pack = self._make_pack(
            multi_agent=MultiAgentLink(
                parent_run_id="parent_run",
                delegation_scope="subtask",
            ),
        )
        assert pack.multi_agent is not None
        assert pack.multi_agent.parent_run_id == "parent_run"

    def test_extensions(self):
        pack = self._make_pack(extensions={"custom_key": "custom_value"})
        data = pack.model_dump(mode="json")
        assert data["extensions"]["custom_key"] == "custom_value"

    def test_tool_call_output(self):
        pack = self._make_pack(
            output=OutputRecord(
                type="tool_calls",
                tool_calls=[
                    ToolCall(tool_call_id="tc1", tool_name="search", arguments={"q": "test"}),
                ],
            ),
        )
        assert pack.output.type == "tool_calls"
        assert len(pack.output.tool_calls) == 1


# ---------------------------------------------------------------------------
# RunMetadata
# ---------------------------------------------------------------------------


class TestRunMetadata:
    def test_basic(self):
        meta = RunMetadata(
            run_id="run_001",
            started_at="2026-03-29T10:00:00.000Z",
            step_count=5,
            total_input_tokens=1000,
            total_output_tokens=500,
            total_latency_ms=3000,
            status="completed",
            models_used=["gpt-4o-mini", "claude-3.5-sonnet"],
        )
        assert meta.step_count == 5
        assert meta.status == "completed"

    def test_content_hash(self):
        meta = RunMetadata(
            run_id="run_001",
            started_at="2026-03-29T10:00:00.000Z",
            step_count=1,
            total_input_tokens=100,
            total_output_tokens=50,
            total_latency_ms=500,
            status="completed",
            models_used=["gpt-4o"],
            content_hash="abc123def456",
        )
        assert meta.content_hash == "abc123def456"
        data = meta.model_dump(mode="json")
        assert data["content_hash"] == "abc123def456"
