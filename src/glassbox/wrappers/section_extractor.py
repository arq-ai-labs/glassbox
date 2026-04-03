"""Section extractor — converts provider-specific message formats to ContextPack sections."""

from __future__ import annotations

import logging
import math
from typing import Any, Union
from uuid import uuid4

from ..format.sections import (
    AssistantMessageSection,
    ContextSection,
    SystemPromptSection,
    ToolResultSection,
    UserMessageSection,
)

logger = logging.getLogger("glassbox")


# ---------------------------------------------------------------------------
# Token estimation — uses tiktoken when available, heuristic fallback
# ---------------------------------------------------------------------------

_tiktoken_enc: Union[object, bool, None] = None
_warned_heuristic = False


def _get_tiktoken():
    """Lazy-load tiktoken encoder. Returns encoder or None."""
    global _tiktoken_enc
    if _tiktoken_enc is None:
        try:
            import tiktoken
            _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        except (ImportError, Exception):
            _tiktoken_enc = False  # sentinel: don't retry on failure
    return _tiktoken_enc if _tiktoken_enc else None


def estimate_tokens(text: str) -> int:
    """Estimate token count. Uses tiktoken (accurate) when installed, else heuristic.

    Install tiktoken for accurate counts: pip install glassbox-ctx[accurate]
    """
    global _warned_heuristic
    if not text:
        return 0
    enc = _get_tiktoken()
    if enc:
        return len(enc.encode(text))
    # Fallback: ~4 chars per token + 10% safety buffer
    if not _warned_heuristic:
        _warned_heuristic = True
        logger.warning(
            "tiktoken not installed — token counts are estimates (~20%% error). "
            "Install for accurate counts: pip install glassbox-ctx[accurate]"
        )
    return math.ceil(len(text) / 4 * 1.1)


def _make_id() -> str:
    return uuid4().hex


# ---------------------------------------------------------------------------
# OpenAI message format
# ---------------------------------------------------------------------------


def extract_openai_sections(messages: list[dict[str, Any]]) -> list[ContextSection]:
    """Convert OpenAI chat messages to ContextPack sections."""
    sections: list[ContextSection] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Content blocks (vision, etc.) — extract text parts
            content = "\n".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        if not content:
            continue

        if role == "system" or role == "developer":
            sections.append(SystemPromptSection(
                section_id=_make_id(), source="system",
                token_count=estimate_tokens(content), content=content,
            ))
        elif role == "user":
            sections.append(UserMessageSection(
                section_id=_make_id(), source="user",
                token_count=estimate_tokens(content), content=content,
            ))
        elif role == "assistant":
            sections.append(AssistantMessageSection(
                section_id=_make_id(), source="assistant",
                token_count=estimate_tokens(content), content=content,
            ))
        elif role == "tool":
            sections.append(ToolResultSection(
                section_id=_make_id(), source=f"tool:{msg.get('name', 'unknown')}",
                token_count=estimate_tokens(content), content=content,
                tool_name=msg.get("name", "unknown"),
                tool_call_id=msg.get("tool_call_id"),
            ))

    return sections


# ---------------------------------------------------------------------------
# Anthropic message format
# ---------------------------------------------------------------------------


def extract_anthropic_sections(
    system: Any,
    messages: list[dict[str, Any]],
) -> list[ContextSection]:
    """Convert Anthropic messages (with separate system param) to sections."""
    sections: list[ContextSection] = []

    # System prompt — can be string or list of text blocks
    if system:
        if isinstance(system, str):
            text = system
        elif isinstance(system, list):
            text = "\n".join(
                b.get("text", "") for b in system
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            text = str(system)
        if text:
            sections.append(SystemPromptSection(
                section_id=_make_id(), source="system",
                token_count=estimate_tokens(text), content=text,
            ))

    # Messages
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Simple string content
        if isinstance(content, str):
            if not content:
                continue
            if role == "user":
                sections.append(UserMessageSection(
                    section_id=_make_id(), source="user",
                    token_count=estimate_tokens(content), content=content,
                ))
            elif role == "assistant":
                sections.append(AssistantMessageSection(
                    section_id=_make_id(), source="assistant",
                    token_count=estimate_tokens(content), content=content,
                ))
            continue

        # Content blocks (list)
        if isinstance(content, list):
            # Extract text blocks
            text_parts = [
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            text = "\n".join(t for t in text_parts if t)
            if text:
                if role == "user":
                    sections.append(UserMessageSection(
                        section_id=_make_id(), source="user",
                        token_count=estimate_tokens(text), content=text,
                    ))
                elif role == "assistant":
                    sections.append(AssistantMessageSection(
                        section_id=_make_id(), source="assistant",
                        token_count=estimate_tokens(text), content=text,
                    ))

            # Extract tool_result blocks
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        result_content = "\n".join(
                            b.get("text", "") for b in result_content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    if result_content:
                        sections.append(ToolResultSection(
                            section_id=_make_id(),
                            source=f"tool:{block.get('tool_use_id', 'unknown')}",
                            token_count=estimate_tokens(str(result_content)),
                            content=str(result_content),
                            tool_name=block.get("tool_use_id", "unknown"),
                            tool_call_id=block.get("tool_use_id"),
                        ))

    return sections