"""LangGraph / LangChain integration — callback handler + observe() wrapper.

Usage:
    from glassbox import observe
    graph = observe(build_my_agent())
    result = graph.invoke({"input": "..."})
    # ContextPacks are now in .glassbox/ and visible at localhost:4100
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from ..core.emitter import emitter
from ..core.file_storage import FileStorage
from ..core.run import GlassboxRun, RunOptions
from ..format.context_pack import ContextPack, ContextPackMetrics, MultiAgentLink
from ..format.model_info import ModelInfo
from ..format.output import OutputRecord, ToolCall
from ..format.sections import (
    AssistantMessageSection,
    ContextSection,
    SystemPromptSection,
    ToolResultSection,
    UserMessageSection,
)
from ..format.token_budget import SectionTokenAllocation, TokenBudget
from ..format.version import FORMAT_VERSION
from ..wrappers.section_extractor import estimate_tokens

logger = logging.getLogger("glassbox")


@dataclass
class _PendingStep:
    run_id: str
    parent_run_id: Optional[str]
    start_time: float
    sections: list[ContextSection]
    model: str
    extra_params: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _iso(ms_epoch: float) -> str:
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ms_epoch / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


class GlassboxCallbackHandler:
    """LangChain/LangGraph callback handler that captures ContextPacks.

    Intercepts on_chat_model_start/on_llm_end to build a ContextPack
    for every LLM call within a LangGraph execution.
    """

    name = "GlassboxCallbackHandler"

    def __init__(
        self,
        agent_name: Optional[str] = None,
        app_name: Optional[str] = None,
        storage_dir: Optional[str] = None,
        on_step: Optional[Any] = None,
    ) -> None:
        self._run = GlassboxRun(RunOptions(agent_name=agent_name, app_name=app_name))
        self._storage = FileStorage(storage_dir)
        self._on_step = on_step
        self._pending: dict[str, _PendingStep] = {}
        self._lock = threading.Lock()

    @property
    def run(self) -> GlassboxRun:
        return self._run

    # ------------------------------------------------------------------
    # Chat model callbacks (message-based — the common path for LangGraph)
    # ------------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        sections: list[ContextSection] = []

        for batch in messages:
            for msg in batch:
                text = _extract_message_text(msg)
                if not text:
                    continue

                msg_type = _get_message_type(msg)
                section_id = uuid4().hex

                if msg_type == "system":
                    sections.append(SystemPromptSection(
                        section_id=section_id, source="system",
                        token_count=estimate_tokens(text), content=text,
                    ))
                elif msg_type == "human":
                    sections.append(UserMessageSection(
                        section_id=section_id, source="user",
                        token_count=estimate_tokens(text), content=text,
                    ))
                elif msg_type == "ai":
                    sections.append(AssistantMessageSection(
                        section_id=section_id, source="assistant",
                        token_count=estimate_tokens(text), content=text,
                    ))
                elif msg_type == "tool":
                    tool_name = getattr(msg, "name", None) or "unknown"
                    tool_call_id = getattr(msg, "tool_call_id", None)
                    if not tool_call_id:
                        ak = getattr(msg, "additional_kwargs", {})
                        tool_call_id = ak.get("tool_call_id")
                    sections.append(ToolResultSection(
                        section_id=section_id,
                        source=f"tool:{tool_name}",
                        token_count=estimate_tokens(text), content=text,
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                    ))
                else:
                    sections.append(UserMessageSection(
                        section_id=section_id, source=msg_type,
                        token_count=estimate_tokens(text), content=text,
                    ))

        model = _extract_model_name(serialized, kwargs)

        with self._lock:
            self._pending[str(run_id)] = _PendingStep(
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                start_time=time.perf_counter(),
                sections=sections,
                model=model,
                extra_params=kwargs.get("invocation_params", {}),
                tags=list(tags or []),
                metadata=dict(metadata or {}),
            )

    # ------------------------------------------------------------------
    # Completion-style fallback
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        sections = [
            UserMessageSection(
                section_id=uuid4().hex, source=f"prompt_{i}",
                token_count=estimate_tokens(p), content=p,
            )
            for i, p in enumerate(prompts) if p
        ]

        model = _extract_model_name(serialized, kwargs)

        with self._lock:
            self._pending[str(run_id)] = _PendingStep(
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                start_time=time.perf_counter(),
                sections=sections,
                model=model,
                extra_params=kwargs.get("invocation_params", {}),
                tags=list(tags or []),
                metadata=dict(metadata or {}),
            )

    # ------------------------------------------------------------------
    # End / Error
    # ------------------------------------------------------------------

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            pending = self._pending.pop(str(run_id), None)
        if not pending:
            return

        latency_ms = (time.perf_counter() - pending.start_time) * 1000
        step = self._run.next_step()

        # Extract output
        all_text = ""
        tc_list: list[ToolCall] = []

        generations = getattr(response, "generations", [])
        for gen_batch in generations:
            for gen in gen_batch:
                all_text += getattr(gen, "text", "") or ""

                # Check for tool calls on the message
                msg = getattr(gen, "message", None)
                if msg:
                    ak = getattr(msg, "additional_kwargs", {})
                    for tc in ak.get("tool_calls", []):
                        fn = tc.get("function", {})
                        args_raw = fn.get("arguments", "{}")
                        args: dict = {}
                        try:
                            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                        except json.JSONDecodeError:
                            pass
                        tc_list.append(ToolCall(
                            tool_call_id=tc.get("id", uuid4().hex),
                            tool_name=fn.get("name") or tc.get("name", "unknown"),
                            arguments=args,
                        ))

                    # Also check tool_calls attribute directly
                    for tc in getattr(msg, "tool_calls", []):
                        if isinstance(tc, dict):
                            tc_list.append(ToolCall(
                                tool_call_id=tc.get("id", uuid4().hex),
                                tool_name=tc.get("name", "unknown"),
                                arguments=tc.get("args", {}),
                            ))

        has_text = bool(all_text)
        has_tools = bool(tc_list)
        output = OutputRecord(
            type="mixed" if (has_text and has_tools) else ("tool_calls" if has_tools else "text"),
            text=all_text or None,
            tool_calls=tc_list or None,
            stop_reason=_extract_stop_reason(response),
        )

        # Token usage
        llm_output = getattr(response, "llm_output", {}) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        input_tokens = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("promptTokens")
            or 0
        )
        output_tokens = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("completionTokens")
            or 0
        )

        total_section_tokens = sum(s.token_count for s in pending.sections)
        now_ms = time.time() * 1000

        # Wire up multi-agent delegation link when a parent run exists
        multi_agent = None
        if pending.parent_run_id:
            # Derive delegation_scope from tags (e.g. "scope:resolve_complaint")
            # or metadata (e.g. {"delegation_scope": "..."})
            scope = pending.metadata.get("delegation_scope")
            if not scope:
                for tag in pending.tags:
                    if tag.startswith("scope:"):
                        scope = tag[6:]
                        break
            # Derive parent_step_id from metadata if the orchestrator provides it
            parent_step = pending.metadata.get("parent_step_id")
            # inherited_sections: IDs of sections carried over from parent context
            inherited = pending.metadata.get("inherited_sections")
            multi_agent = MultiAgentLink(
                parent_run_id=pending.parent_run_id,
                parent_step_id=parent_step,
                delegation_scope=scope,
                inherited_sections=inherited,
            )

        pack = ContextPack(
            format_version=FORMAT_VERSION,
            run_id=self._run.run_id,
            agent_name=self._run.agent_name,
            app_name=self._run.app_name,
            step_id=step.step_id,
            step_index=step.step_index,
            started_at=_iso(now_ms - latency_ms),
            completed_at=_iso(now_ms),
            sections=pending.sections,
            token_budget=TokenBudget(
                total_budget=total_section_tokens,
                total_used=total_section_tokens,
                by_section=[
                    SectionTokenAllocation(
                        section_id=s.section_id, section_type=s.type,
                        token_count=s.token_count,
                    )
                    for s in pending.sections
                ],
                rejected=[],
            ),
            model=ModelInfo(
                provider="langchain",
                model=pending.model,
                temperature=pending.extra_params.get("temperature"),
                max_tokens=pending.extra_params.get("max_tokens")
                or pending.extra_params.get("maxTokens"),
            ),
            output=output,
            metrics=ContextPackMetrics(
                latency_ms=round(latency_ms),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=usage.get("cache_read_input_tokens"),
            ),
            multi_agent=multi_agent,
        )

        emitter.emit("step:complete", pack)
        if self._on_step:
            try:
                self._on_step(pack)
            except Exception:
                pass
        try:
            self._storage.write_step_background(pack)
        except Exception:
            pass

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            pending = self._pending.pop(str(run_id), None)
        if not pending:
            return

        latency_ms = (time.perf_counter() - pending.start_time) * 1000
        step = self._run.next_step()
        total = sum(s.token_count for s in pending.sections)
        now_ms = time.time() * 1000

        pack = ContextPack(
            format_version=FORMAT_VERSION,
            run_id=self._run.run_id,
            agent_name=self._run.agent_name,
            app_name=self._run.app_name,
            step_id=step.step_id,
            step_index=step.step_index,
            started_at=_iso(now_ms - latency_ms),
            completed_at=_iso(now_ms),
            sections=pending.sections,
            token_budget=TokenBudget(
                total_budget=total, total_used=total,
                by_section=[
                    SectionTokenAllocation(
                        section_id=s.section_id, section_type=s.type,
                        token_count=s.token_count,
                    )
                    for s in pending.sections
                ],
                rejected=[],
            ),
            model=ModelInfo(provider="langchain", model=pending.model),
            output=OutputRecord(type="error", error=str(error)),
            metrics=ContextPackMetrics(
                latency_ms=round(latency_ms), input_tokens=0, output_tokens=0,
            ),
        )

        emitter.emit("step:error", {
            "run_id": self._run.run_id,
            "step_id": step.step_id,
            "error": error,
        })
        emitter.emit("step:complete", pack)
        if self._on_step:
            try:
                self._on_step(pack)
            except Exception:
                pass
        try:
            self._storage.write_step_background(pack)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# observe() — the primary LangGraph integration point
# ---------------------------------------------------------------------------


def observe(graph: Any, **kwargs: Any) -> Any:
    """Wrap a LangGraph graph to auto-capture ContextPacks.

    Args:
        graph: A StateGraph (will be compiled) or CompiledGraph.
        **kwargs: Passed to GlassboxCallbackHandler (agent_name, app_name, etc.)

    Returns:
        An ObservedGraph that injects the callback on every invoke/stream call.
    """
    handler = GlassboxCallbackHandler(**kwargs)

    # If it's a StateGraph (has .compile), compile it
    if hasattr(graph, "compile") and not hasattr(graph, "invoke"):
        graph = graph.compile()

    return _ObservedGraph(graph, handler)


class _ObservedGraph:
    """Wraps a compiled LangGraph to inject Glassbox callbacks."""

    def __init__(self, graph: Any, handler: GlassboxCallbackHandler) -> None:
        self._graph = graph
        self._handler = handler

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        config = _inject_callback(config, self._handler)
        return self._graph.invoke(input, config, **kwargs)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        config = _inject_callback(config, self._handler)
        return await self._graph.ainvoke(input, config, **kwargs)

    def stream(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        config = _inject_callback(config, self._handler)
        return self._graph.stream(input, config, **kwargs)

    async def astream(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        config = _inject_callback(config, self._handler)
        return self._graph.astream(input, config, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph, name)


def _inject_callback(config: Any, handler: GlassboxCallbackHandler) -> dict:
    """Ensure the Glassbox callback handler is in the config."""
    if config is None:
        config = {}
    if isinstance(config, dict):
        callbacks = list(config.get("callbacks", []))
        callbacks.append(handler)
        config["callbacks"] = callbacks
    else:
        # RunnableConfig object
        callbacks = list(getattr(config, "callbacks", None) or [])
        callbacks.append(handler)
        config = dict(config) if hasattr(config, "__iter__") else {"callbacks": callbacks}
        config["callbacks"] = callbacks
    return config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_message_text(msg: Any) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) and b.get("type") == "text"
            else str(b) if not isinstance(b, dict) else ""
            for b in content
        )
    return str(content) if content else ""


def _get_message_type(msg: Any) -> str:
    if hasattr(msg, "_get_type"):
        return msg._get_type()
    if hasattr(msg, "type"):
        return msg.type
    return "unknown"


def _extract_model_name(serialized: dict, kwargs: dict) -> str:
    ip = kwargs.get("invocation_params", {})
    for key in ("model", "model_name", "modelName"):
        if key in ip:
            return str(ip[key])

    # From serialized
    sid = serialized.get("id", [])
    if sid:
        return str(sid[-1]) if isinstance(sid, list) else str(sid)

    # Fallback
    return serialized.get("name", "unknown")


def _extract_stop_reason(response: Any) -> Optional[str]:
    """Extract stop/finish reason from a LangChain LLMResult."""
    generations = getattr(response, "generations", [])
    for gen_batch in generations:
        for gen in gen_batch:
            info = getattr(gen, "generation_info", {}) or {}
            if "finish_reason" in info:
                return str(info["finish_reason"])
            if "stop_reason" in info:
                return str(info["stop_reason"])
    return None