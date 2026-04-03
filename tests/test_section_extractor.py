"""Tests for section extraction — OpenAI and Anthropic message format conversion."""

import pytest

from glassbox.wrappers.section_extractor import (
    estimate_tokens,
    extract_anthropic_sections,
    extract_openai_sections,
)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_text(self):
        result = estimate_tokens("Hello, world!")
        assert result > 0
        assert isinstance(result, int)

    def test_longer_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        result = estimate_tokens(text)
        # Should be roughly len/4 (with tiktoken) or len/4*1.1 (heuristic)
        assert result > 50
        assert result < len(text)  # More than 1 char per token minimum

    def test_code_text(self):
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""
        result = estimate_tokens(code)
        assert result > 0

    def test_consistency(self):
        """Same input should always give same output."""
        text = "Test consistency"
        assert estimate_tokens(text) == estimate_tokens(text)


# ---------------------------------------------------------------------------
# OpenAI message extraction
# ---------------------------------------------------------------------------


class TestExtractOpenAISections:
    def test_system_message(self):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        sections = extract_openai_sections(messages)
        assert len(sections) == 1
        assert sections[0].type == "system_prompt"
        assert "helpful assistant" in sections[0].content

    def test_user_message(self):
        messages = [{"role": "user", "content": "What is 2+2?"}]
        sections = extract_openai_sections(messages)
        assert len(sections) == 1
        assert sections[0].type == "user_message"

    def test_assistant_message(self):
        messages = [{"role": "assistant", "content": "The answer is 4."}]
        sections = extract_openai_sections(messages)
        assert len(sections) == 1
        assert sections[0].type == "assistant_message"

    def test_tool_message(self):
        messages = [{"role": "tool", "content": "search results here", "name": "web_search", "tool_call_id": "tc1"}]
        sections = extract_openai_sections(messages)
        assert len(sections) == 1
        assert sections[0].type == "tool_result"
        assert sections[0].tool_name == "web_search"
        assert sections[0].tool_call_id == "tc1"

    def test_developer_role(self):
        """Developer role should be treated as system prompt."""
        messages = [{"role": "developer", "content": "System instructions here."}]
        sections = extract_openai_sections(messages)
        assert len(sections) == 1
        assert sections[0].type == "system_prompt"

    def test_multi_message_conversation(self):
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What's the weather?"},
        ]
        sections = extract_openai_sections(messages)
        assert len(sections) == 4
        assert sections[0].type == "system_prompt"
        assert sections[1].type == "user_message"
        assert sections[2].type == "assistant_message"
        assert sections[3].type == "user_message"

    def test_content_blocks(self):
        """Content can be a list of blocks (vision, etc.)."""
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
        ]}]
        sections = extract_openai_sections(messages)
        assert len(sections) == 1
        assert "Describe this image" in sections[0].content

    def test_empty_content_skipped(self):
        messages = [{"role": "user", "content": ""}]
        sections = extract_openai_sections(messages)
        assert len(sections) == 0

    def test_unique_section_ids(self):
        messages = [
            {"role": "user", "content": "msg1"},
            {"role": "user", "content": "msg2"},
        ]
        sections = extract_openai_sections(messages)
        ids = [s.section_id for s in sections]
        assert len(ids) == len(set(ids))  # all unique


# ---------------------------------------------------------------------------
# Anthropic message extraction
# ---------------------------------------------------------------------------


class TestExtractAnthropicSections:
    def test_system_string(self):
        sections = extract_anthropic_sections(
            system="You are a helpful assistant.",
            messages=[],
        )
        assert len(sections) == 1
        assert sections[0].type == "system_prompt"

    def test_system_blocks(self):
        sections = extract_anthropic_sections(
            system=[
                {"type": "text", "text": "First instruction"},
                {"type": "text", "text": "Second instruction"},
            ],
            messages=[],
        )
        assert len(sections) == 1
        assert "First instruction" in sections[0].content
        assert "Second instruction" in sections[0].content

    def test_no_system(self):
        sections = extract_anthropic_sections(
            system=None,
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert len(sections) == 1
        assert sections[0].type == "user_message"

    def test_user_and_assistant(self):
        sections = extract_anthropic_sections(
            system=None,
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
        )
        assert len(sections) == 2
        assert sections[0].type == "user_message"
        assert sections[1].type == "assistant_message"

    def test_content_blocks_with_tool_result(self):
        sections = extract_anthropic_sections(
            system=None,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_123", "content": "search result data"},
                    {"type": "text", "text": "Please analyze the above."},
                ],
            }],
        )
        # Should have both a tool_result and a user_message
        types = [s.type for s in sections]
        assert "user_message" in types
        assert "tool_result" in types

    def test_token_counts_positive(self):
        sections = extract_anthropic_sections(
            system="System prompt",
            messages=[{"role": "user", "content": "User message"}],
        )
        for s in sections:
            assert s.token_count > 0
