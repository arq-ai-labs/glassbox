"""Integration tests for wrap() — OpenAI and Anthropic client wrapping.

Uses mock clients to verify ContextPack capture without real API calls.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from glassbox import wrap
from glassbox.core.emitter import GlassboxEmitter
from glassbox.format.context_pack import ContextPack


# ---------------------------------------------------------------------------
# Fake OpenAI client
# ---------------------------------------------------------------------------

class _FakeOpenAIClient:
    """Minimal fake that looks like openai.OpenAI() but NOT like Anthropic."""
    pass

def _make_openai_client():
    """Build a mock that looks like openai.OpenAI()."""
    client = _FakeOpenAIClient()

    # Build the chat.completions.create chain
    usage = SimpleNamespace(prompt_tokens=50, completion_tokens=20)
    message = SimpleNamespace(
        content="Hello! How can I help?",
        tool_calls=None,
    )
    choice = SimpleNamespace(message=message, finish_reason="stop")
    response = SimpleNamespace(choices=[choice], usage=usage)

    completions = SimpleNamespace(create=MagicMock(return_value=response))
    client.chat = SimpleNamespace(completions=completions)
    return client, response


def _make_openai_tool_response():
    """Response with tool calls."""
    tc = SimpleNamespace(
        id="call_abc123",
        function=SimpleNamespace(
            name="get_weather",
            arguments='{"location": "NYC"}',
        ),
    )
    message = SimpleNamespace(content=None, tool_calls=[tc])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    usage = SimpleNamespace(prompt_tokens=80, completion_tokens=30)
    return SimpleNamespace(choices=[choice], usage=usage)


# ---------------------------------------------------------------------------
# Fake Anthropic client
# ---------------------------------------------------------------------------

def _make_anthropic_client():
    """Build a mock that looks like anthropic.Anthropic()."""
    client = MagicMock()

    text_block = SimpleNamespace(type="text", text="I can help with that!")
    usage = SimpleNamespace(
        input_tokens=40, output_tokens=15,
        cache_read_input_tokens=None, cache_creation_input_tokens=None,
    )
    response = SimpleNamespace(
        content=[text_block],
        usage=usage,
        stop_reason="end_turn",
    )

    client.messages.create = MagicMock(return_value=response)
    return client, response


# ---------------------------------------------------------------------------
# Tests: wrap() detection
# ---------------------------------------------------------------------------

class TestWrapDetection:
    def test_rejects_unknown_client(self):
        with pytest.raises(ValueError, match="Unrecognized client type"):
            wrap("not_a_client")

    def test_detects_openai_client(self):
        client, _ = _make_openai_client()
        wrapped = wrap(client, storage=False)
        gb = getattr(wrapped, "__glassbox", None)
        assert gb is not None
        assert gb["run"] is not None

    def test_detects_anthropic_client(self):
        client, _ = _make_anthropic_client()
        wrapped = wrap(client, storage=False)
        assert getattr(wrapped, "__glassbox", None) is not None

    def test_returns_same_client(self):
        client, _ = _make_openai_client()
        wrapped = wrap(client, storage=False)
        assert wrapped is client


# ---------------------------------------------------------------------------
# Tests: OpenAI wrap() — ContextPack capture
# ---------------------------------------------------------------------------

class TestWrapOpenAI:
    def test_emits_step_complete(self):
        client, _ = _make_openai_client()
        packs = []

        def capture(pack):
            packs.append(pack)

        wrapped = wrap(client, storage=False, on_step=capture)
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi there"},
            ],
        )

        assert len(packs) == 1
        pack = packs[0]
        assert isinstance(pack, ContextPack)

    def test_captures_sections(self):
        client, _ = _make_openai_client()
        packs = []

        wrapped = wrap(client, storage=False, on_step=lambda p: packs.append(p))
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a poet."},
                {"role": "user", "content": "Write a haiku"},
            ],
        )

        pack = packs[0]
        assert len(pack.sections) == 2
        assert pack.sections[0].type == "system_prompt"
        assert pack.sections[1].type == "user_message"

    def test_captures_model_info(self):
        client, _ = _make_openai_client()
        packs = []

        wrapped = wrap(client, storage=False, on_step=lambda p: packs.append(p))
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
            max_tokens=100,
        )

        m = packs[0].model
        assert m.provider == "openai"
        assert m.model == "gpt-4o"
        assert m.temperature == 0.7
        assert m.max_tokens == 100

    def test_captures_output(self):
        client, _ = _make_openai_client()
        packs = []

        wrapped = wrap(client, storage=False, on_step=lambda p: packs.append(p))
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

        out = packs[0].output
        assert out.type == "text"
        assert out.text == "Hello! How can I help?"
        assert out.stop_reason == "stop"

    def test_captures_token_counts(self):
        client, _ = _make_openai_client()
        packs = []

        wrapped = wrap(client, storage=False, on_step=lambda p: packs.append(p))
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

        met = packs[0].metrics
        assert met.input_tokens == 50
        assert met.output_tokens == 20
        assert met.latency_ms >= 0  # mock calls are near-instant

    def test_captures_tool_calls(self):
        client, _ = _make_openai_client()
        client.chat.completions.create = MagicMock(return_value=_make_openai_tool_response())
        packs = []

        wrapped = wrap(client, storage=False, on_step=lambda p: packs.append(p))
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What's the weather?"}],
        )

        out = packs[0].output
        assert out.type == "tool_calls"
        assert len(out.tool_calls) == 1
        assert out.tool_calls[0].tool_name == "get_weather"
        assert out.tool_calls[0].arguments == {"location": "NYC"}

    def test_multi_step_increments_index(self):
        client, _ = _make_openai_client()
        packs = []

        wrapped = wrap(client, storage=False, on_step=lambda p: packs.append(p))
        for _ in range(3):
            wrapped.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert len(packs) == 3
        assert packs[0].step_index == 0
        assert packs[1].step_index == 1
        assert packs[2].step_index == 2
        # Same run
        assert packs[0].run_id == packs[1].run_id == packs[2].run_id

    def test_custom_agent_name(self):
        client, _ = _make_openai_client()
        packs = []

        wrapped = wrap(client, storage=False, agent_name="my-bot", on_step=lambda p: packs.append(p))
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert packs[0].agent_name == "my-bot"

    def test_token_budget_populated(self):
        client, _ = _make_openai_client()
        packs = []

        wrapped = wrap(client, storage=False, on_step=lambda p: packs.append(p))
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=500,
        )

        tb = packs[0].token_budget
        assert tb.total_used > 0
        assert tb.total_budget > tb.total_used  # includes max_tokens headroom
        assert len(tb.by_section) == 1

    def test_error_captured(self):
        client, _ = _make_openai_client()
        client.chat.completions.create = MagicMock(side_effect=RuntimeError("API down"))
        packs = []

        wrapped = wrap(client, storage=False, on_step=lambda p: packs.append(p))
        with pytest.raises(RuntimeError):
            wrapped.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert len(packs) == 1
        assert packs[0].output.type == "error"
        assert "API down" in packs[0].output.error


# ---------------------------------------------------------------------------
# Tests: Anthropic wrap() — ContextPack capture
# ---------------------------------------------------------------------------

class TestWrapAnthropic:
    def test_emits_step_complete(self):
        client, _ = _make_anthropic_client()
        packs = []

        wrapped = wrap(client, storage=False, on_step=lambda p: packs.append(p))
        wrapped.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert len(packs) == 1
        assert isinstance(packs[0], ContextPack)

    def test_captures_anthropic_model_info(self):
        client, _ = _make_anthropic_client()
        packs = []

        wrapped = wrap(client, storage=False, on_step=lambda p: packs.append(p))
        wrapped.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.3,
        )

        m = packs[0].model
        assert m.provider == "anthropic"
        assert m.model == "claude-sonnet-4-20250514"
        assert m.max_tokens == 1024

    def test_captures_system_prompt(self):
        client, _ = _make_anthropic_client()
        packs = []

        wrapped = wrap(client, storage=False, on_step=lambda p: packs.append(p))
        wrapped.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Hi"}],
        )

        types = [s.type for s in packs[0].sections]
        assert "system_prompt" in types
        assert "user_message" in types

    def test_captures_output_text(self):
        client, _ = _make_anthropic_client()
        packs = []

        wrapped = wrap(client, storage=False, on_step=lambda p: packs.append(p))
        wrapped.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert packs[0].output.text == "I can help with that!"
        assert packs[0].output.stop_reason == "end_turn"

    def test_captures_anthropic_tokens(self):
        client, _ = _make_anthropic_client()
        packs = []

        wrapped = wrap(client, storage=False, on_step=lambda p: packs.append(p))
        wrapped.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": "Hi"}],
        )

        met = packs[0].metrics
        assert met.input_tokens == 40
        assert met.output_tokens == 15

    def test_anthropic_tool_use(self):
        client, _ = _make_anthropic_client()
        tool_block = SimpleNamespace(
            type="tool_use", id="toolu_01", name="calculator",
            input={"expression": "2+2"},
        )
        client.messages.create.return_value = SimpleNamespace(
            content=[tool_block],
            usage=SimpleNamespace(
                input_tokens=60, output_tokens=25,
                cache_read_input_tokens=None, cache_creation_input_tokens=None,
            ),
            stop_reason="tool_use",
        )
        packs = []

        wrapped = wrap(client, storage=False, on_step=lambda p: packs.append(p))
        wrapped.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

        out = packs[0].output
        assert out.type == "tool_calls"
        assert out.tool_calls[0].tool_name == "calculator"
