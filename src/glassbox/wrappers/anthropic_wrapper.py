"""Anthropic SDK wrapper — intercepts messages.create() to capture ContextPacks."""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any, Optional

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
from .openai_wrapper import WrapOptions
from .section_extractor import extract_anthropic_sections
from .stream_buffer import tap_anthropic_stream

logger = logging.getLogger("glassbox")


def wrap_anthropic(client: Any, options: Optional[WrapOptions] = None) -> Any:
    """Wrap an Anthropic client to auto-capture ContextPacks."""
    opts = options or WrapOptions()
    run = get_current_run() or GlassboxRun(RunOptions(
        agent_name=opts.agent_name,
        app_name=opts.app_name,
    ))

    storage: Any = None
    if opts.storage is not False:
        storage = opts.storage or FileStorage(opts.storage_dir)

    original_create = client.messages.create

    def _build_pack(
        params: dict,
        sections: list[ContextSection],
        output: OutputRecord,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        cache_read: Optional[int] = None,
        cache_creation: Optional[int] = None,
    ) -> ContextPack:
        step = run.next_step()
        total = sum(s.token_count for s in sections)
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
                total_budget=total + (params.get("max_tokens") or 0),
                total_used=total,
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
                provider="anthropic",
                model=params.get("model", "unknown"),
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens"),
                top_p=params.get("top_p"),
                stop_sequences=params.get("stop_sequences"),
            ),
            output=output,
            metrics=ContextPackMetrics(
                latency_ms=round(latency_ms),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_creation,
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

    def _extract_output(response: Any) -> tuple[OutputRecord, int, int, Optional[int], Optional[int]]:
        content_blocks = getattr(response, "content", [])

        text_parts = []
        tc_list = []
        for block in content_blocks:
            btype = getattr(block, "type", "")
            if btype == "text":
                text_parts.append(getattr(block, "text", ""))
            elif btype == "tool_use":
                tc_list.append(ToolCall(
                    tool_call_id=getattr(block, "id", ""),
                    tool_name=getattr(block, "name", "unknown"),
                    arguments=getattr(block, "input", {}) or {},
                ))

        text = "".join(text_parts)
        has_text = bool(text)
        has_tools = bool(tc_list)

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) or 0 if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0 if usage else 0
        cache_read = getattr(usage, "cache_read_input_tokens", None) if usage else None
        cache_creation = getattr(usage, "cache_creation_input_tokens", None) if usage else None

        return OutputRecord(
            type="mixed" if (has_text and has_tools) else ("tool_calls" if has_tools else "text"),
            text=text or None,
            tool_calls=tc_list or None,
            stop_reason=getattr(response, "stop_reason", None),
        ), input_tokens, output_tokens, cache_read, cache_creation

    if inspect.iscoroutinefunction(original_create):
        async def async_create(*args: Any, **kwargs: Any) -> Any:
            params = kwargs if kwargs else (args[0] if args else {})
            system = params.get("system")
            messages = params.get("messages", [])
            sections = extract_anthropic_sections(system, messages)

            start = time.perf_counter()
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
                    out, inp_t, out_t, cr, cc = _extract_output(response)
                    pack = _build_pack(params, sections, out, latency, inp_t, out_t, cr, cc)
                _emit_pack(pack)
            return response

        client.messages.create = async_create
    else:
        def sync_create(*args: Any, **kwargs: Any) -> Any:
            params = kwargs if kwargs else (args[0] if args else {})
            system = params.get("system")
            messages = params.get("messages", [])
            sections = extract_anthropic_sections(system, messages)

            start = time.perf_counter()
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
                    out, inp_t, out_t, cr, cc = _extract_output(response)
                    pack = _build_pack(params, sections, out, latency, inp_t, out_t, cr, cc)
                _emit_pack(pack)
            return response

        client.messages.create = sync_create

    client.__glassbox = {"run": run, "options": opts}
    return client


def _iso(ms_epoch: float) -> str:
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ms_epoch / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"
