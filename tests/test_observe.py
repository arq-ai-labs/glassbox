"""Integration tests for observe() — LangGraph callback handler + wrapper.

Uses mock LangGraph graphs and LangChain message types to verify
ContextPack capture without real LLM calls or langgraph installed.
"""

import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
from uuid import uuid4

from glassbox.integrations.langgraph import (
    GlassboxCallbackHandler,
    observe,
    _ObservedGraph,
)
from glassbox.format.context_pack import ContextPack, MultiAgentLink


# ---------------------------------------------------------------------------
# Fake LangChain message types
# ---------------------------------------------------------------------------

class FakeSystemMessage:
    type = "system"
    content = "You are a helpful assistant."
    def _get_type(self):
        return "system"

class FakeHumanMessage:
    type = "human"
    content = "What's the weather in NYC?"
    def _get_type(self):
        return "human"

class FakeAIMessage:
    type = "ai"
    content = "The weather is sunny!"
    additional_kwargs = {}
    tool_calls = []
    def _get_type(self):
        return "ai"

class FakeToolMessage:
    type = "tool"
    content = '{"temp": 72, "condition": "sunny"}'
    name = "get_weather"
    tool_call_id = "call_xyz"
    def _get_type(self):
        return "tool"


def _fake_llm_result(text="Hello!", tool_calls=None, finish_reason="stop"):
    """Build a mock LLMResult like LangChain returns."""
    ak = {}
    if tool_calls:
        ak["tool_calls"] = tool_calls

    msg = SimpleNamespace(
        content=text,
        additional_kwargs=ak,
        tool_calls=tool_calls or [],
    )
    gen = SimpleNamespace(
        text=text,
        message=msg,
        generation_info={"finish_reason": finish_reason},
    )
    return SimpleNamespace(
        generations=[[gen]],
        llm_output={"token_usage": {"prompt_tokens": 100, "completion_tokens": 30}},
    )


# ---------------------------------------------------------------------------
# Tests: GlassboxCallbackHandler — the core callback
# ---------------------------------------------------------------------------

class TestGlassboxCallbackHandler:
    def test_basic_chat_cycle(self):
        """on_chat_model_start + on_llm_end → emits a ContextPack."""
        packs = []
        handler = GlassboxCallbackHandler(
            agent_name="test-agent",
            storage_dir=None,
            on_step=lambda p: packs.append(p),
        )
        # Disable storage writes
        handler._storage = MagicMock()

        run_id = uuid4()
        parent_id = uuid4()

        handler.on_chat_model_start(
            serialized={"id": ["langchain", "chat_models", "ChatOpenAI"]},
            messages=[[FakeSystemMessage(), FakeHumanMessage()]],
            run_id=run_id,
            parent_run_id=parent_id,
        )

        handler.on_llm_end(
            response=_fake_llm_result("The weather is sunny!"),
            run_id=run_id,
        )

        assert len(packs) == 1
        pack = packs[0]
        assert isinstance(pack, ContextPack)
        assert pack.agent_name == "test-agent"

    def test_captures_sections_from_messages(self):
        packs = []
        handler = GlassboxCallbackHandler(on_step=lambda p: packs.append(p))
        handler._storage = MagicMock()

        run_id = uuid4()
        handler.on_chat_model_start(
            serialized={"id": ["ChatOpenAI"]},
            messages=[[FakeSystemMessage(), FakeHumanMessage()]],
            run_id=run_id,
        )
        handler.on_llm_end(response=_fake_llm_result(), run_id=run_id)

        sections = packs[0].sections
        types = [s.type for s in sections]
        assert "system_prompt" in types
        assert "user_message" in types

    def test_captures_tool_result_sections(self):
        packs = []
        handler = GlassboxCallbackHandler(on_step=lambda p: packs.append(p))
        handler._storage = MagicMock()

        run_id = uuid4()
        handler.on_chat_model_start(
            serialized={"id": ["ChatOpenAI"]},
            messages=[[FakeHumanMessage(), FakeToolMessage()]],
            run_id=run_id,
        )
        handler.on_llm_end(response=_fake_llm_result(), run_id=run_id)

        types = [s.type for s in packs[0].sections]
        assert "tool_result" in types

    def test_captures_token_usage(self):
        packs = []
        handler = GlassboxCallbackHandler(on_step=lambda p: packs.append(p))
        handler._storage = MagicMock()

        run_id = uuid4()
        handler.on_chat_model_start(
            serialized={"id": ["ChatOpenAI"]},
            messages=[[FakeHumanMessage()]],
            run_id=run_id,
        )
        handler.on_llm_end(response=_fake_llm_result(), run_id=run_id)

        met = packs[0].metrics
        assert met.input_tokens == 100
        assert met.output_tokens == 30

    def test_captures_output(self):
        packs = []
        handler = GlassboxCallbackHandler(on_step=lambda p: packs.append(p))
        handler._storage = MagicMock()

        run_id = uuid4()
        handler.on_chat_model_start(
            serialized={"id": ["ChatOpenAI"]},
            messages=[[FakeHumanMessage()]],
            run_id=run_id,
        )
        handler.on_llm_end(response=_fake_llm_result("Got it!"), run_id=run_id)

        out = packs[0].output
        assert out.type == "text"
        assert out.text == "Got it!"
        assert out.stop_reason == "stop"

    def test_multi_step_increments(self):
        packs = []
        handler = GlassboxCallbackHandler(on_step=lambda p: packs.append(p))
        handler._storage = MagicMock()

        for i in range(3):
            rid = uuid4()
            handler.on_chat_model_start(
                serialized={"id": ["ChatOpenAI"]},
                messages=[[FakeHumanMessage()]],
                run_id=rid,
            )
            handler.on_llm_end(response=_fake_llm_result(), run_id=rid)

        assert len(packs) == 3
        assert packs[0].step_index == 0
        assert packs[1].step_index == 1
        assert packs[2].step_index == 2

    def test_on_llm_error(self):
        packs = []
        handler = GlassboxCallbackHandler(on_step=lambda p: packs.append(p))
        handler._storage = MagicMock()

        run_id = uuid4()
        handler.on_chat_model_start(
            serialized={"id": ["ChatOpenAI"]},
            messages=[[FakeHumanMessage()]],
            run_id=run_id,
        )
        handler.on_llm_error(
            error=RuntimeError("Rate limit exceeded"),
            run_id=run_id,
        )

        assert len(packs) == 1
        assert packs[0].output.type == "error"
        assert "Rate limit" in packs[0].output.error

    def test_on_llm_start_fallback(self):
        """Completion-style (non-chat) on_llm_start path."""
        packs = []
        handler = GlassboxCallbackHandler(on_step=lambda p: packs.append(p))
        handler._storage = MagicMock()

        run_id = uuid4()
        handler.on_llm_start(
            serialized={"id": ["OpenAI"]},
            prompts=["Tell me a joke"],
            run_id=run_id,
        )
        handler.on_llm_end(response=_fake_llm_result("Why did the..."), run_id=run_id)

        assert len(packs) == 1
        assert packs[0].sections[0].type == "user_message"


# ---------------------------------------------------------------------------
# Tests: Multi-agent delegation link
# ---------------------------------------------------------------------------

class TestMultiAgentLink:
    def test_parent_run_creates_link(self):
        packs = []
        handler = GlassboxCallbackHandler(on_step=lambda p: packs.append(p))
        handler._storage = MagicMock()

        run_id = uuid4()
        parent_id = uuid4()
        handler.on_chat_model_start(
            serialized={"id": ["ChatOpenAI"]},
            messages=[[FakeHumanMessage()]],
            run_id=run_id,
            parent_run_id=parent_id,
        )
        handler.on_llm_end(response=_fake_llm_result(), run_id=run_id)

        link = packs[0].multi_agent
        assert link is not None
        assert link.parent_run_id == str(parent_id)

    def test_no_parent_no_link(self):
        packs = []
        handler = GlassboxCallbackHandler(on_step=lambda p: packs.append(p))
        handler._storage = MagicMock()

        run_id = uuid4()
        handler.on_chat_model_start(
            serialized={"id": ["ChatOpenAI"]},
            messages=[[FakeHumanMessage()]],
            run_id=run_id,
            parent_run_id=None,
        )
        handler.on_llm_end(response=_fake_llm_result(), run_id=run_id)

        assert packs[0].multi_agent is None

    def test_delegation_scope_from_tags(self):
        packs = []
        handler = GlassboxCallbackHandler(on_step=lambda p: packs.append(p))
        handler._storage = MagicMock()

        run_id = uuid4()
        parent_id = uuid4()
        handler.on_chat_model_start(
            serialized={"id": ["ChatOpenAI"]},
            messages=[[FakeHumanMessage()]],
            run_id=run_id,
            parent_run_id=parent_id,
            tags=["scope:resolve_shipping_complaint", "agent:support"],
        )
        handler.on_llm_end(response=_fake_llm_result(), run_id=run_id)

        link = packs[0].multi_agent
        assert link.delegation_scope == "resolve_shipping_complaint"

    def test_delegation_scope_from_metadata(self):
        packs = []
        handler = GlassboxCallbackHandler(on_step=lambda p: packs.append(p))
        handler._storage = MagicMock()

        run_id = uuid4()
        parent_id = uuid4()
        handler.on_chat_model_start(
            serialized={"id": ["ChatOpenAI"]},
            messages=[[FakeHumanMessage()]],
            run_id=run_id,
            parent_run_id=parent_id,
            metadata={"delegation_scope": "billing_inquiry"},
        )
        handler.on_llm_end(response=_fake_llm_result(), run_id=run_id)

        link = packs[0].multi_agent
        assert link.delegation_scope == "billing_inquiry"

    def test_parent_step_id_from_metadata(self):
        packs = []
        handler = GlassboxCallbackHandler(on_step=lambda p: packs.append(p))
        handler._storage = MagicMock()

        run_id = uuid4()
        parent_id = uuid4()
        handler.on_chat_model_start(
            serialized={"id": ["ChatOpenAI"]},
            messages=[[FakeHumanMessage()]],
            run_id=run_id,
            parent_run_id=parent_id,
            metadata={"parent_step_id": "step_007"},
        )
        handler.on_llm_end(response=_fake_llm_result(), run_id=run_id)

        link = packs[0].multi_agent
        assert link.parent_step_id == "step_007"

    def test_inherited_sections_from_metadata(self):
        packs = []
        handler = GlassboxCallbackHandler(on_step=lambda p: packs.append(p))
        handler._storage = MagicMock()

        run_id = uuid4()
        parent_id = uuid4()
        handler.on_chat_model_start(
            serialized={"id": ["ChatOpenAI"]},
            messages=[[FakeHumanMessage()]],
            run_id=run_id,
            parent_run_id=parent_id,
            metadata={"inherited_sections": ["sec_abc", "sec_def"]},
        )
        handler.on_llm_end(response=_fake_llm_result(), run_id=run_id)

        link = packs[0].multi_agent
        assert link.inherited_sections == ["sec_abc", "sec_def"]

    def test_full_multi_agent_link(self):
        """All four fields populated together."""
        packs = []
        handler = GlassboxCallbackHandler(on_step=lambda p: packs.append(p))
        handler._storage = MagicMock()

        run_id = uuid4()
        parent_id = uuid4()
        handler.on_chat_model_start(
            serialized={"id": ["ChatOpenAI"]},
            messages=[[FakeHumanMessage()]],
            run_id=run_id,
            parent_run_id=parent_id,
            tags=["scope:resolve_complaint"],
            metadata={
                "parent_step_id": "step_003",
                "inherited_sections": ["sec_001"],
            },
        )
        handler.on_llm_end(response=_fake_llm_result(), run_id=run_id)

        link = packs[0].multi_agent
        assert link.parent_run_id == str(parent_id)
        assert link.parent_step_id == "step_003"
        assert link.delegation_scope == "resolve_complaint"
        assert link.inherited_sections == ["sec_001"]


# ---------------------------------------------------------------------------
# Tests: observe() wrapper
# ---------------------------------------------------------------------------

class TestObserve:
    def test_observe_wraps_compiled_graph(self):
        graph = MagicMock()
        graph.invoke = MagicMock(return_value={"output": "done"})
        # Has invoke → already compiled
        observed = observe(graph, storage_dir=None)
        assert isinstance(observed, _ObservedGraph)

    def test_observe_compiles_state_graph(self):
        graph = MagicMock()
        # Has compile but no invoke → StateGraph
        del graph.invoke
        compiled = MagicMock()
        compiled.invoke = MagicMock(return_value={"output": "done"})
        graph.compile = MagicMock(return_value=compiled)

        observed = observe(graph, storage_dir=None)
        graph.compile.assert_called_once()
        assert isinstance(observed, _ObservedGraph)

    def test_observed_graph_proxies_invoke(self):
        graph = MagicMock()
        graph.invoke = MagicMock(return_value={"result": "ok"})

        observed = observe(graph, storage_dir=None)
        result = observed.invoke({"input": "test"})
        assert result == {"result": "ok"}

    def test_observed_graph_injects_callbacks(self):
        graph = MagicMock()
        graph.invoke = MagicMock(return_value={})

        observed = observe(graph, storage_dir=None)
        observed.invoke({"input": "test"})

        # Check that invoke was called with a config containing callbacks
        call_args = graph.invoke.call_args
        config = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("config")
        assert "callbacks" in config
        assert any(isinstance(cb, GlassboxCallbackHandler) for cb in config["callbacks"])

    def test_observed_graph_proxies_attributes(self):
        graph = MagicMock()
        graph.invoke = MagicMock()
        graph.get_graph = MagicMock(return_value="graph_data")

        observed = observe(graph, storage_dir=None)
        assert observed.get_graph() == "graph_data"
