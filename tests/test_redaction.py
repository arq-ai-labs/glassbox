"""Tests for content redaction — privacy controls for enterprise deployments."""

import pytest

from glassbox.redaction import (
    RedactionPolicy,
    apply_redaction,
    redact_pack_dict,
    set_redaction_policy,
    get_redaction_policy,
)


@pytest.fixture(autouse=True)
def reset_policy():
    """Reset redaction policy to NONE after each test."""
    yield
    set_redaction_policy(RedactionPolicy.NONE)


class TestRedactionPolicy:
    def test_default_is_none(self):
        set_redaction_policy(RedactionPolicy.NONE)
        assert get_redaction_policy() == RedactionPolicy.NONE

    def test_none_passes_through(self):
        set_redaction_policy(RedactionPolicy.NONE)
        section = {"section_id": "s1", "type": "user_message", "content": "secret data", "token_count": 5}
        result = apply_redaction(section)
        assert result["content"] == "secret data"

    def test_hash_replaces_content(self):
        set_redaction_policy(RedactionPolicy.HASH)
        section = {"section_id": "s1", "type": "user_message", "content": "secret data", "token_count": 5}
        result = apply_redaction(section)
        assert result["content"].startswith("sha256:")
        assert len(result["content"]) == 71  # "sha256:" + 64 hex chars
        assert result["token_count"] == 5  # preserved

    def test_hash_is_deterministic(self):
        set_redaction_policy(RedactionPolicy.HASH)
        s1 = apply_redaction({"content": "same text"})
        s2 = apply_redaction({"content": "same text"})
        assert s1["content"] == s2["content"]

    def test_truncate(self):
        set_redaction_policy(RedactionPolicy.TRUNCATE)
        long_content = "x" * 500
        result = apply_redaction({"content": long_content, "token_count": 100})
        assert len(result["content"]) == 103  # 100 + "..."
        assert result["content"].endswith("...")

    def test_truncate_short_content(self):
        set_redaction_policy(RedactionPolicy.TRUNCATE)
        result = apply_redaction({"content": "short", "token_count": 2})
        assert result["content"] == "short"  # no truncation needed

    def test_drop_content(self):
        set_redaction_policy(RedactionPolicy.DROP_CONTENT)
        result = apply_redaction({
            "section_id": "s1", "type": "system_prompt",
            "content": "secret instructions", "token_count": 50,
        })
        assert result["content"] == ""
        assert result["section_id"] == "s1"
        assert result["token_count"] == 50

    def test_custom_function(self):
        def redactor(section):
            if section.get("type") == "user_message":
                section = dict(section)
                section["content"] = "[PII REDACTED]"
            return section

        set_redaction_policy(redactor)
        user = apply_redaction({"type": "user_message", "content": "my SSN is 123-45-6789"})
        system = apply_redaction({"type": "system_prompt", "content": "Be helpful"})
        assert user["content"] == "[PII REDACTED]"
        assert system["content"] == "Be helpful"

    def test_custom_function_drop(self):
        """Custom function returning None drops the section entirely."""
        def drop_tools(section):
            if section.get("type") == "tool_result":
                return None
            return section

        set_redaction_policy(drop_tools)
        result = apply_redaction({"type": "tool_result", "content": "data"})
        assert result is None

    def test_custom_function_error_fallback(self):
        """If custom function raises, original section is preserved."""
        def buggy(section):
            raise RuntimeError("oops")

        set_redaction_policy(buggy)
        result = apply_redaction({"content": "data"})
        assert result["content"] == "data"  # preserved despite error


class TestRedactPackDict:
    def test_none_is_fast_path(self):
        set_redaction_policy(RedactionPolicy.NONE)
        pack = {"sections": [{"content": "data"}]}
        result = redact_pack_dict(pack)
        assert result is pack  # same object, not copied

    def test_hash_all_sections(self):
        set_redaction_policy(RedactionPolicy.HASH)
        pack = {
            "sections": [
                {"section_id": "s1", "type": "system_prompt", "content": "secret A"},
                {"section_id": "s2", "type": "user_message", "content": "secret B"},
            ],
            "run_id": "run_001",
        }
        result = redact_pack_dict(pack)
        assert result["run_id"] == "run_001"
        for s in result["sections"]:
            assert s["content"].startswith("sha256:")

    def test_drop_filters_sections(self):
        def drop_memory(section):
            if section.get("type") == "memory":
                return None
            return section

        set_redaction_policy(drop_memory)
        pack = {
            "sections": [
                {"type": "system_prompt", "content": "keep"},
                {"type": "memory", "content": "drop"},
                {"type": "user_message", "content": "keep"},
            ]
        }
        result = redact_pack_dict(pack)
        assert len(result["sections"]) == 2
        types = [s["type"] for s in result["sections"]]
        assert "memory" not in types
