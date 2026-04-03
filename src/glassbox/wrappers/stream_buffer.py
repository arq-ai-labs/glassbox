"""Stream buffer — accumulates streaming LLM responses without blocking the stream."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

logger = logging.getLogger("glassbox")


@dataclass
class StreamBufferResult:
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None


# ---------------------------------------------------------------------------
# OpenAI stream tapping
# ---------------------------------------------------------------------------


def tap_openai_stream_sync(stream: Iterator[Any]) -> tuple[Iterator[Any], StreamBufferResult]:
    """Wrap a sync OpenAI stream. Yields chunks unchanged while buffering."""
    buf = StreamBufferResult()
    partial_tools: dict[int, dict[str, str]] = {}

    def tapped() -> Iterator[Any]:
        try:
            for chunk in stream:
                _process_openai_chunk(chunk, buf, partial_tools)
                yield chunk
        finally:
            _finalize_openai_tools(buf, partial_tools)

    return tapped(), buf


async def tap_openai_stream_async(stream: AsyncIterator[Any]) -> tuple[AsyncIterator[Any], StreamBufferResult]:
    """Wrap an async OpenAI stream."""
    buf = StreamBufferResult()
    partial_tools: dict[int, dict[str, str]] = {}

    async def tapped() -> AsyncIterator[Any]:
        try:
            async for chunk in stream:
                _process_openai_chunk(chunk, buf, partial_tools)
                yield chunk
        finally:
            _finalize_openai_tools(buf, partial_tools)

    return tapped(), buf


def _process_openai_chunk(chunk: Any, buf: StreamBufferResult, partial_tools: dict) -> None:
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return
    choice = choices[0]
    delta = getattr(choice, "delta", None)

    if delta:
        # Text content
        content = getattr(delta, "content", None)
        if content:
            buf.text += content

        # Tool calls — accumulate by index
        tool_calls = getattr(delta, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                idx = getattr(tc, "index", 0)
                if idx not in partial_tools:
                    partial_tools[idx] = {"id": "", "name": "", "args": ""}
                partial = partial_tools[idx]
                tc_id = getattr(tc, "id", None)
                if tc_id:
                    partial["id"] = tc_id
                fn = getattr(tc, "function", None)
                if fn:
                    name = getattr(fn, "name", None)
                    if name:
                        partial["name"] = name
                    args = getattr(fn, "arguments", None)
                    if args:
                        partial["args"] += args

    # Finish reason
    finish = getattr(choice, "finish_reason", None)
    if finish:
        buf.stop_reason = finish

    # Usage (typically on last chunk)
    usage = getattr(chunk, "usage", None)
    if usage:
        buf.input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        buf.output_tokens = getattr(usage, "completion_tokens", 0) or 0


def _finalize_openai_tools(buf: StreamBufferResult, partial_tools: dict) -> None:
    for partial in partial_tools.values():
        args: dict[str, Any] = {}
        try:
            args = json.loads(partial["args"]) if partial["args"] else {}
        except json.JSONDecodeError:
            pass
        buf.tool_calls.append({
            "tool_call_id": partial["id"],
            "tool_name": partial["name"],
            "arguments": args,
        })


# ---------------------------------------------------------------------------
# Anthropic stream tapping
# ---------------------------------------------------------------------------


def tap_anthropic_stream(stream: Any) -> tuple[Any, StreamBufferResult]:
    """Tap an Anthropic MessageStream. Returns the original stream + buffer.

    Works by hooking into the stream's event callbacks if available,
    or falling back to post-completion extraction.
    """
    buf = StreamBufferResult()

    if hasattr(stream, "on"):
        # Event-based stream (Anthropic SDK MessageStream)
        def on_text(text: str) -> None:
            buf.text += text

        def on_message(msg: Any) -> None:
            usage = getattr(msg, "usage", None)
            if usage:
                buf.input_tokens = getattr(usage, "input_tokens", 0) or 0
                buf.output_tokens = getattr(usage, "output_tokens", 0) or 0
                buf.cache_read_tokens = getattr(usage, "cache_read_input_tokens", None)
                buf.cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", None)
            buf.stop_reason = getattr(msg, "stop_reason", None)

            for block in getattr(msg, "content", []):
                if getattr(block, "type", None) == "tool_use":
                    buf.tool_calls.append({
                        "tool_call_id": getattr(block, "id", ""),
                        "tool_name": getattr(block, "name", ""),
                        "arguments": getattr(block, "input", {}),
                    })

        try:
            stream.on("text", on_text)
            stream.on("message", on_message)
        except Exception:
            logger.debug("Could not attach stream listeners", exc_info=True)

    return stream, buf
