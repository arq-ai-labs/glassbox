"""OpenAI SDK wrapper — intercepts chat.completions.create() to capture ContextPacks."""

from __future__ import annotations

import inspect
import json
import logging
import time
from typing import Any, Optional
from uuid import uuid4

from ..core.context import get_current_run
from ..core.emitter import emitter
from ..core.file_storage import FileStorage
from ..core.run import GlassboxRun, RunOptions
from ..format.context_pack import ContextPack, ContextPackMetrics
from ..format.model_info import ModelInfo
from ..format.output import OutputRecord, ToolCall
from ..format.sections import ContextSection
from ..format.token_budget import SectionTokenAllocation, TokenBudget
from ..format.version import FORMAT_VERSION
from ..pricing import estimate_cost
from .section_extractor import extract_openai_sections
from .stream_buffer import (
    StreamBufferResult,
    tap_openai_stream_async,
    tap_openai_stream_sync,
)

logger = logging.getLogger("glassbox")


class WrapOptions:
    def __init__(
        self,
        agent_name: Optional[str] = None,
        app_name: Optional[str] = None,
        storage_dir: Optional[str] = None,
        storage: Any = None,  # StorageAdapter | False
        on_step: Optional[Any] = None,
    ):
        self.agent_name = agent_name
        self.app_name = app_name
        self.storage_dir = storage_dir
        self.storage = storage
        self.on_step = on_step


def wrap_openai(client: Any, options: Optional[WrapOptions] = None) -> Any:
    """Wrap an OpenAI client to auto-capture ContextPacks.

    Monkey-patches client.chat.completions.create() to intercept calls.
    Returns the same client with a __glassbox attribute attached.
    """
    opts = options or WrapOptions()
    run = get_current_run() or GlassboxRun(RunOptions(
        agent_name=opts.agent_name,
        app_name=opts.app_name,
    ))

    storage: Any = None
    if opts.storage is not False:
        storage = opts.storage or FileStorage(opts.storage_dir)

    original_create = client.chat.completions.create

    def _build_pack(
        params: dict,
        sections: list[ContextSection],
        output: OutputRecord,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
    ) -> ContextPack:
        step = run.next_step()
        total_section_tokens = sum(s.token_count for s in sections)
        now_ms = time.time() * 1000

        return ContextPack(
            format_version=FORMAT_VERSION,
            run_id=run.run_id,
            agent_name=run.agent_name,
            app_name=run.app_name,
            step_id=step.step_id,
            step_index=step.step_index,
            started_at=_iso(now_ms - latency_ms),
            completed_at=_iso(now_ms),
            sections=sections,
            token_budget=TokenBudget(
                total_budget=(
                    total_section_tokens + (params.get("max_tokens") or params.get("max_completion_tokens") or 0)
                ),
                total_used=total_section_tokens,
                by_section=[
                    SectionTokenAllocation(
                        section_id=s.section_id, section_type=s.type,
                        token_count=s.token_count,
                    )
                    for s in sections
                ],
                rejected=[],
            ),
            model=ModelInfo(
                provider="openai",
                model=params.get("model", "unknown"),
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens") or params.get("max_completion_tokens"),
                top_p=params.get("top_p"),
                stop_sequences=(
                    [params["stop"]] if isinstance(params.get("stop"), str)
                    else params.get("stop")
                ),
            ),
            output=output,
            metrics=ContextPackMetrics(
                latency_ms=round(latency_ms),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_estimate_usd=estimate_cost(
                    params.get("model", "unknown"), input_tokens, output_tokens
                ),
            ),
        )

    def _emit_pack(pack: ContextPack) -> None:
        emitter.emit("step:complete", pack)
        if opts.on_step:
            try:
                opts.on_step(pack)
            except Exception:
                logger.debug("on_step callback error", exc_info=True)
        if storage:
            try:
                storage.write_step_background(pack)
            except Exception:
                logger.debug("Storage write error", exc_info=True)

    def _extract_output(response: Any) -> tuple[OutputRecord, int, int]:
        msg = response.choices[0].message
        text = getattr(msg, "content", None) or ""
        tool_calls_raw = getattr(msg, "tool_calls", None) or []

        tc_list = []
        for tc in tool_calls_raw:
            args: dict = {}
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except (json.JSONDecodeError, AttributeError):
                pass
            tc_list.append(ToolCall(
                tool_call_id=tc.id or "",
                tool_name=getattr(tc.function, "name", "unknown") or "unknown",
                arguments=args,
            ))

        has_text = bool(text)
        has_tools = bool(tc_list)
        output_type = "mixed" if (has_text and has_tools) else ("tool_calls" if has_tools else "text")

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0 if usage else 0

        return OutputRecord(
            type=output_type,
            text=text or None,
            tool_calls=tc_list or None,
            stop_reason=response.choices[0].finish_reason,
        ), input_tokens, output_tokens

    if inspect.iscoroutinefunction(original_create):
        async def async_create(*args: Any, **kwargs: Any) -> Any:
            params = kwargs if kwargs else (args[0] if args else {})
            messages = params.get("messages", [])
            sections = extract_openai_sections(messages)
            is_streaming = params.get("stream", False)

            start = time.perf_counter()
            if is_streaming:
                response = await original_create(*args, **kwargs)
                tapped, buf = await tap_openai_stream_async(response)

                async def emit_after() -> None:
                    # Will be called after stream is consumed
                    latency = (time.perf_counter() - start) * 1000
                    out = _buf_to_output(buf)
                    pack = _build_pack(params, sections, out, latency, buf.input_tokens, buf.output_tokens)
                    _emit_pack(pack)

                # Wrap to emit after full consumption
                return _AsyncStreamWrapper(tapped, emit_after)
            else:
                error = None
                try:
                    response = await original_create(*args, **kwargs)
                except Exception as e:
                    error = e
                    raise
                finally:
                    latency = (time.perf_counter() - start) * 1000
                    if error:
                        out = OutputRecord(type="error", error=str(error))
                        pack = _build_pack(params, sections, out, latency, 0, 0)
                    else:
                        out, inp_t, out_t = _extract_output(response)
                        pack = _build_pack(params, sections, out, latency, inp_t, out_t)
                    _emit_pack(pack)
                return response

        client.chat.completions.create = async_create
    else:
        def sync_create(*args: Any, **kwargs: Any) -> Any:
            params = kwargs if kwargs else (args[0] if args else {})
            messages = params.get("messages", [])
            sections = extract_openai_sections(messages)
            is_streaming = params.get("stream", False)

            start = time.perf_counter()
            if is_streaming:
                response = original_create(*args, **kwargs)
                tapped, buf = tap_openai_stream_sync(response)

                def emit_after() -> None:
                    latency = (time.perf_counter() - start) * 1000
                    out = _buf_to_output(buf)
                    pack = _build_pack(params, sections, out, latency, buf.input_tokens, buf.output_tokens)
                    _emit_pack(pack)

                return _SyncStreamWrapper(tapped, emit_after)
            else:
                error = None
                try:
                    response = original_create(*args, **kwargs)
                except Exception as e:
                    error = e
                    raise
                finally:
                    latency = (time.perf_counter() - start) * 1000
                    if error:
                        out = OutputRecord(type="error", error=str(error))
                        pack = _build_pack(params, sections, out, latency, 0, 0)
                    else:
                        out, inp_t, out_t = _extract_output(response)
                        pack = _build_pack(params, sections, out, latency, inp_t, out_t)
                    _emit_pack(pack)
                return response

        client.chat.completions.create = sync_create

    # Attach glassbox metadata
    client.__glassbox = {"run": run, "options": opts}
    return client


def _buf_to_output(buf: StreamBufferResult) -> OutputRecord:
    has_text = bool(buf.text)
    has_tools = bool(buf.tool_calls)
    tc_list = [
        ToolCall(
            tool_call_id=tc["tool_call_id"],
            tool_name=tc["tool_name"],
            arguments=tc["arguments"],
        )
        for tc in buf.tool_calls
    ]
    return OutputRecord(
        type="mixed" if (has_text and has_tools) else ("tool_calls" if has_tools else "text"),
        text=buf.text or None,
        tool_calls=tc_list or None,
        stop_reason=buf.stop_reason,
    )


def _iso(ms_epoch: float) -> str:
    """Convert epoch millis to ISO 8601 with Z suffix."""
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ms_epoch / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


class _SyncStreamWrapper:
    """Wraps a sync stream to emit ContextPack after full consumption."""

    def __init__(self, stream: Any, on_complete: Any) -> None:
        self._stream = stream
        self._on_complete = on_complete

    def __iter__(self) -> Any:
        try:
            yield from self._stream
        finally:
            try:
                self._on_complete()
            except Exception:
                pass


class _AsyncStreamWrapper:
    """Wraps an async stream to emit ContextPack after full consumption."""

    def __init__(self, stream: Any, on_complete: Any) -> None:
        self._stream = stream
        self._on_complete = on_complete

    async def __aiter__(self) -> Any:
        try:
            async for chunk in self._stream:
                yield chunk
        finally:
            try:
                await self._on_complete()
            except Exception:
                pass
