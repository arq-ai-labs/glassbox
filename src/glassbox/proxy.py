"""Glassbox Proxy — unified real-time intercept for Anthropic, OpenAI, and Ollama.

Single process: starts the proxy + viewer together. Handles both API formats
on the same port. Detects the format from the request path automatically.

Usage:
    from glassbox import proxy
    proxy()                                      # Anthropic (default)
    proxy(provider="openai")                     # OpenAI
    proxy(provider="ollama")                     # Ollama (local)
    proxy(provider="openai", target="https://...") # Custom endpoint

CLI:
    glassbox proxy                               # Anthropic
    glassbox proxy --provider openai             # OpenAI
    glassbox proxy --provider ollama             # Ollama
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger("glassbox.proxy")

# Defaults per provider
_PROVIDER_DEFAULTS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
    "ollama": "http://localhost:11434",
}

_ENV_HINT = {
    "anthropic": "ANTHROPIC_BASE_URL",
    "openai": "OPENAI_BASE_URL",
    "ollama": "OPENAI_BASE_URL",
}


def proxy(
    provider: str = "anthropic",
    target: Optional[str] = None,
    proxy_port: int = 4050,
    viewer_port: int = 4100,
    storage_dir: Optional[str] = None,
    working_dir: Optional[str] = None,
) -> None:
    """Start the Glassbox proxy + viewer in a single process.

    Args:
        provider: "anthropic", "openai", or "ollama"
        target: Upstream API URL. Defaults per provider.
        proxy_port: Port for the proxy (default 4050).
        viewer_port: Port for the viewer UI (default 4100).
        storage_dir: Path to .glassbox/ directory.
        working_dir: Project directory to scan for source inventory.
            Defaults to current working directory.
    """
    try:
        import uvicorn
        import httpx  # noqa: F401
        from starlette.applications import Starlette  # noqa: F401
    except ImportError:
        print("Proxy requires server extras. Install with: pip install glassbox[server]")
        print("Also: pip install httpx")
        return

    provider = provider.lower()
    if provider not in _PROVIDER_DEFAULTS:
        print(f"Unknown provider: {provider}. Use: anthropic, openai, or ollama")
        return

    target_url = target or _PROVIDER_DEFAULTS[provider]
    storage_path = storage_dir or str(Path.home() / ".glassbox")
    env_var = _ENV_HINT[provider]

    if provider == "ollama":
        proxy_url = f"http://localhost:{proxy_port}/v1"
    elif provider == "openai":
        proxy_url = f"http://localhost:{proxy_port}/v1"
    else:
        proxy_url = f"http://localhost:{proxy_port}"

    # Banner
    print()
    print("  ┌──────────────────────────────────────────────────┐")
    print(f"  │  GLASSBOX PROXY — {provider.upper():<31} │")
    print("  ├──────────────────────────────────────────────────┤")
    print(f"  │  Proxy:   http://localhost:{proxy_port:<23} │")
    print(f"  │  Viewer:  http://localhost:{viewer_port:<23} │")
    print(f"  │  Target:  {target_url[:40]:<40} │")
    print("  ├──────────────────────────────────────────────────┤")
    print(f"  │  Set in your environment:                        │")
    print(f"  │  {env_var}={proxy_url:<27} │")
    print("  │                                                  │")
    print("  │  Then use your client normally.                   │")
    print("  │  Every API call appears in the viewer.            │")
    print("  └──────────────────────────────────────────────────┘")
    print()

    # Start viewer in background
    import socket
    def port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) == 0

    if port_in_use(viewer_port):
        print(f"  Viewer already running on port {viewer_port}\n")
    else:
        from .viewer.server import create_app as create_viewer_app
        viewer_app = create_viewer_app(storage_path)

        def run_viewer():
            uvicorn.run(viewer_app, host="0.0.0.0", port=viewer_port, log_level="warning")

        threading.Thread(target=run_viewer, daemon=True).start()
        print(f"  Viewer started on port {viewer_port}\n")

    # Discover sources
    work_dir = working_dir or os.getcwd()
    from .discovery import discover_sources
    inventory = discover_sources(working_dir=work_dir)
    print(f"  Source inventory: {inventory.total_sources} sources discovered ({inventory.total_tokens_available or 0:,} tokens)")
    print(f"  Working dir: {work_dir}\n")

    # Build and start proxy
    app = _create_proxy_app(provider, target_url, storage_path, inventory)
    uvicorn.run(app, host="0.0.0.0", port=proxy_port, log_level="warning")


# ---------------------------------------------------------------------------
# Proxy app builder
# ---------------------------------------------------------------------------


def _create_proxy_app(provider: str, target_url: str, storage_dir: str, inventory: Any = None) -> Any:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response, StreamingResponse
    from starlette.routing import Route

    from .core.emitter import emitter
    from .core.file_storage import FileStorage
    from .core.run import GlassboxRun, RunOptions
    from .format.context_pack import ContextPack, ContextPackMetrics
    from .format.model_info import ModelInfo
    from .format.output import OutputRecord, ToolCall
    from .format.run_metadata import RunMetadata
    from .format.sections import (
        AssistantMessageSection,
        SystemPromptSection,
        ToolResultSection,
        UserMessageSection,
    )
    from .format.token_budget import SectionTokenAllocation, TokenBudget
    from .format.version import FORMAT_VERSION
    from .pricing import estimate_cost
    from .wrappers.section_extractor import estimate_tokens

    storage = FileStorage(storage_dir)
    run = GlassboxRun(RunOptions(agent_name=provider, app_name="glassbox-proxy"))

    # Headers that must be stripped from upstream responses.
    # httpx auto-decompresses, so forwarding content-encoding causes
    # the client to attempt double decompression.
    _HOP_HEADERS = {"content-encoding", "content-length", "transfer-encoding"}

    def clean_resp_headers(raw_headers) -> dict:
        return {k: v for k, v in dict(raw_headers).items()
                if k.lower() not in _HOP_HEADERS}

    # Shared state
    state = {
        "call_count": 0,
        "total_input": 0,
        "total_output": 0,
        "total_latency": 0.0,
        "models_seen": set(),
    }
    lock = threading.Lock()

    def iso_now() -> str:
        from datetime import datetime, timezone
        dt = datetime.now(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"

    def accumulate(pack: ContextPack) -> None:
        with lock:
            state["total_input"] += pack.metrics.input_tokens
            state["total_output"] += pack.metrics.output_tokens
            state["total_latency"] += pack.metrics.latency_ms
            state["models_seen"].add(pack.model.model)

    def update_index() -> None:
        try:
            storage.complete_run(RunMetadata(
                run_id=run.run_id,
                agent_name=provider,
                app_name="glassbox-proxy",
                started_at=run.started_at,
                completed_at=iso_now(),
                step_count=run.step_count,
                total_input_tokens=state["total_input"],
                total_output_tokens=state["total_output"],
                total_latency_ms=round(state["total_latency"]),
                status="running",
                models_used=sorted(state["models_seen"]),
            ))
        except Exception:
            pass

    def emit_pack(pack: ContextPack, call_num: int, model: str, latency: float, streaming: bool) -> None:
        accumulate(pack)
        emitter.emit("step:complete", pack)
        storage.write_step(pack)
        update_index()
        tag = " | streaming" if streaming else ""
        logger.info(
            f"  [{call_num}] {model} | "
            f"{pack.metrics.input_tokens} in / {pack.metrics.output_tokens} out | "
            f"{latency:.0f}ms{tag}"
        )

    # ---------------------------------------------------------------
    # Section extractors
    # ---------------------------------------------------------------

    def extract_anthropic_sections(body: dict) -> list:
        sections = []
        system = body.get("system")
        if system:
            text = system if isinstance(system, str) else "\n".join(
                b.get("text", "") for b in system
                if isinstance(b, dict) and b.get("type") == "text"
            ) if isinstance(system, list) else str(system)
            if text:
                sections.append(SystemPromptSection(
                    section_id=uuid4().hex, source="system",
                    token_count=estimate_tokens(text),
                    content=text[:5000] + ("..." if len(text) > 5000 else ""),
                ))
        for msg in body.get("messages", []):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_result":
                            result = block.get("content", "")
                            if isinstance(result, list):
                                result = "\n".join(
                                    b.get("text", "") for b in result
                                    if isinstance(b, dict) and b.get("type") == "text"
                                )
                            if not isinstance(result, str):
                                result = str(result)
                            sections.append(ToolResultSection(
                                section_id=uuid4().hex,
                                source=f"tool:{block.get('tool_use_id', 'unknown')}",
                                token_count=estimate_tokens(result),
                                content=result[:3000],
                                tool_name=block.get("tool_use_id", "unknown"),
                                tool_call_id=block.get("tool_use_id"),
                            ))
                        elif block.get("type") == "tool_use":
                            inp = json.dumps(block.get("input", {}))
                            sections.append(ToolResultSection(
                                section_id=uuid4().hex,
                                source=f"tool_call:{block.get('name', 'unknown')}",
                                token_count=estimate_tokens(inp),
                                content=inp[:3000],
                                tool_name=block.get("name", "unknown"),
                                tool_call_id=block.get("id"),
                            ))
                content = "\n".join(t for t in text_parts if t)
            if not content or not isinstance(content, str):
                continue
            tokens = estimate_tokens(content)
            display = content[:5000] + ("..." if len(content) > 5000 else "")
            if role == "user":
                sections.append(UserMessageSection(
                    section_id=uuid4().hex, source="user",
                    token_count=tokens, content=display,
                ))
            elif role == "assistant":
                sections.append(AssistantMessageSection(
                    section_id=uuid4().hex, source="assistant",
                    token_count=tokens, content=display,
                ))
        return sections

    def extract_openai_sections(body: dict) -> list:
        sections = []
        for msg in body.get("messages", []):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            if not content or not isinstance(content, str):
                if role == "tool":
                    content = msg.get("content", "") or ""
                else:
                    continue
            if not content:
                continue
            tokens = estimate_tokens(content)
            display = content[:5000] + ("..." if len(content) > 5000 else "")
            if role in ("system", "developer"):
                sections.append(SystemPromptSection(
                    section_id=uuid4().hex, source="system",
                    token_count=tokens, content=display,
                ))
            elif role == "user":
                sections.append(UserMessageSection(
                    section_id=uuid4().hex, source="user",
                    token_count=tokens, content=display,
                ))
            elif role == "assistant":
                sections.append(AssistantMessageSection(
                    section_id=uuid4().hex, source="assistant",
                    token_count=tokens, content=display,
                ))
            elif role == "tool":
                sections.append(ToolResultSection(
                    section_id=uuid4().hex,
                    source=f"tool:{msg.get('tool_call_id', 'unknown')}",
                    token_count=tokens, content=display,
                    tool_name=msg.get("name", "unknown"),
                    tool_call_id=msg.get("tool_call_id"),
                ))
        return sections

    # ---------------------------------------------------------------
    # Pack builders
    # ---------------------------------------------------------------

    def build_anthropic_pack(body, sections, output, latency_ms, inp, out, cr, cc):
        step = run.next_step()
        st = sum(s.token_count for s in sections)
        effective = max(inp, st)
        if cr and cr > 0:
            effective = max(inp, st)
        model_name = body.get("model", "unknown")
        inv = copy.deepcopy(inventory) if inventory else None
        return ContextPack(
            format_version=FORMAT_VERSION, run_id=run.run_id,
            agent_name=provider, app_name="glassbox-proxy",
            step_id=step.step_id, step_index=step.step_index,
            step_label=f"messages.create ({model_name})",
            started_at=iso_now(), completed_at=iso_now(),
            sections=sections,
            token_budget=TokenBudget(
                total_budget=st + (body.get("max_tokens") or 4096),
                total_used=st,
                by_section=[SectionTokenAllocation(section_id=s.section_id, section_type=s.type, token_count=s.token_count) for s in sections],
                rejected=[],
            ),
            model=ModelInfo(provider="anthropic", model=model_name,
                temperature=body.get("temperature"), max_tokens=body.get("max_tokens"),
                top_p=body.get("top_p"), stop_sequences=body.get("stop_sequences")),
            output=output,
            metrics=ContextPackMetrics(
                latency_ms=round(latency_ms), input_tokens=effective, output_tokens=out,
                cache_read_tokens=cr, cache_creation_tokens=cc,
                cost_estimate_usd=estimate_cost(model_name, effective, out)),
            context_inventory=inv,
        )

    def build_openai_pack(body, sections, output, latency_ms, inp, out):
        step = run.next_step()
        st = sum(s.token_count for s in sections)
        effective = max(inp, st)
        model_name = body.get("model", "unknown")
        inv = copy.deepcopy(inventory) if inventory else None
        return ContextPack(
            format_version=FORMAT_VERSION, run_id=run.run_id,
            agent_name=provider, app_name="glassbox-proxy",
            step_id=step.step_id, step_index=step.step_index,
            step_label=f"chat.completions.create ({model_name})",
            started_at=iso_now(), completed_at=iso_now(),
            sections=sections,
            token_budget=TokenBudget(
                total_budget=st + (body.get("max_tokens") or body.get("max_completion_tokens") or 4096),
                total_used=st,
                by_section=[SectionTokenAllocation(section_id=s.section_id, section_type=s.type, token_count=s.token_count) for s in sections],
                rejected=[],
            ),
            model=ModelInfo(provider=provider, model=model_name,
                temperature=body.get("temperature"),
                max_tokens=body.get("max_tokens") or body.get("max_completion_tokens"),
                top_p=body.get("top_p"),
                stop_sequences=[body["stop"]] if isinstance(body.get("stop"), str) else body.get("stop")),
            output=output,
            metrics=ContextPackMetrics(
                latency_ms=round(latency_ms), input_tokens=effective, output_tokens=out,
                cost_estimate_usd=estimate_cost(model_name, effective, out)),
            context_inventory=inv,
        )

    # ---------------------------------------------------------------
    # Anthropic handler: /v1/messages
    # ---------------------------------------------------------------

    async def handle_anthropic_messages(request: Request) -> Response:
        import httpx

        raw_body = await request.body()
        body = json.loads(raw_body)
        is_streaming = body.get("stream", False)
        sections = extract_anthropic_sections(body)

        headers = {}
        for key in ["x-api-key", "anthropic-version", "anthropic-beta", "content-type", "authorization"]:
            val = request.headers.get(key)
            if val:
                headers[key] = val
        headers["content-type"] = "application/json"

        url = f"{target_url}/v1/messages"
        start = time.perf_counter()
        state["call_count"] += 1
        call_num = state["call_count"]

        if is_streaming:
            async def stream():
                text = ""
                tools = []
                usage = {}
                stop = None

                async with httpx.AsyncClient(timeout=300.0) as c:
                    async with c.stream("POST", url, content=raw_body, headers=headers) as resp:
                        async for line in resp.aiter_lines():
                            yield line + "\n"
                            if line.startswith("data: "):
                                ds = line[6:]
                                if ds.strip() == "[DONE]":
                                    continue
                                try:
                                    ev = json.loads(ds)
                                    et = ev.get("type", "")
                                    if et == "content_block_delta":
                                        d = ev.get("delta", {})
                                        if d.get("type") == "text_delta":
                                            text += d.get("text", "")
                                    elif et == "content_block_start":
                                        cb = ev.get("content_block", {})
                                        if cb.get("type") == "tool_use":
                                            tools.append({"id": cb.get("id", ""), "name": cb.get("name", ""), "input": {}})
                                    elif et == "message_delta":
                                        d = ev.get("delta", {})
                                        stop = d.get("stop_reason") or stop
                                        u = ev.get("usage", {})
                                        if u: usage.update(u)
                                    elif et == "message_start":
                                        u = ev.get("message", {}).get("usage", {})
                                        if u: usage.update(u)
                                except json.JSONDecodeError:
                                    pass

                latency = (time.perf_counter() - start) * 1000
                tc = [ToolCall(tool_call_id=t["id"], tool_name=t["name"], arguments=t.get("input", {})) for t in tools]
                ht, htc = bool(text), bool(tc)
                out = OutputRecord(
                    type="mixed" if (ht and htc) else ("tool_calls" if htc else "text"),
                    text=text[:5000] or None, tool_calls=tc or None, stop_reason=stop)
                pack = build_anthropic_pack(body, sections, out, latency,
                    usage.get("input_tokens", 0), usage.get("output_tokens", 0),
                    usage.get("cache_read_input_tokens"), usage.get("cache_creation_input_tokens"))
                emit_pack(pack, call_num, body.get("model", "?"), latency, True)

            return StreamingResponse(stream(), media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
        else:
            async with httpx.AsyncClient(timeout=300.0) as c:
                resp = await c.post(url, content=raw_body, headers=headers)
            latency = (time.perf_counter() - start) * 1000
            rb = resp.json()
            blocks = rb.get("content", [])
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            tc = [ToolCall(tool_call_id=b.get("id", ""), tool_name=b.get("name", "unknown"), arguments=b.get("input", {}))
                  for b in blocks if b.get("type") == "tool_use"]
            ht, htc = bool(text), bool(tc)
            usage = rb.get("usage", {})
            out = OutputRecord(
                type="mixed" if (ht and htc) else ("tool_calls" if htc else "text"),
                text=text[:5000] or None, tool_calls=tc or None, stop_reason=rb.get("stop_reason"))
            pack = build_anthropic_pack(body, sections, out, latency,
                usage.get("input_tokens", 0), usage.get("output_tokens", 0),
                usage.get("cache_read_input_tokens"), usage.get("cache_creation_input_tokens"))
            emit_pack(pack, call_num, body.get("model", "?"), latency, False)
            return Response(content=resp.content, status_code=resp.status_code, headers=clean_resp_headers(resp.headers))

    # ---------------------------------------------------------------
    # OpenAI handler: /v1/chat/completions
    # ---------------------------------------------------------------

    async def handle_openai_completions(request: Request) -> Response:
        import httpx

        raw_body = await request.body()
        body = json.loads(raw_body)
        is_streaming = body.get("stream", False)
        sections = extract_openai_sections(body)

        headers = {}
        for key in ["authorization", "content-type", "api-key", "x-api-key"]:
            val = request.headers.get(key)
            if val:
                headers[key] = val
        headers["content-type"] = "application/json"

        url = f"{target_url}/v1/chat/completions"
        start = time.perf_counter()
        state["call_count"] += 1
        call_num = state["call_count"]

        if is_streaming:
            async def stream():
                text = ""
                tools: dict[int, dict] = {}
                usage = {}
                stop = None

                async with httpx.AsyncClient(timeout=300.0) as c:
                    async with c.stream("POST", url, content=raw_body, headers=headers) as resp:
                        async for line in resp.aiter_lines():
                            yield line + "\n"
                            if line.startswith("data: "):
                                ds = line[6:]
                                if ds.strip() == "[DONE]":
                                    continue
                                try:
                                    chunk = json.loads(ds)
                                    ch = chunk.get("choices", [])
                                    if ch:
                                        delta = ch[0].get("delta", {})
                                        if delta.get("content"):
                                            text += delta["content"]
                                        for tc in delta.get("tool_calls", []):
                                            idx = tc.get("index", 0)
                                            if idx not in tools:
                                                tools[idx] = {"id": tc.get("id", ""), "name": tc.get("function", {}).get("name", ""), "arguments": ""}
                                            fn = tc.get("function", {})
                                            if fn.get("name"):
                                                tools[idx]["name"] = fn["name"]
                                            if "arguments" in fn:
                                                tools[idx]["arguments"] += fn["arguments"]
                                        fr = ch[0].get("finish_reason")
                                        if fr:
                                            stop = fr
                                    u = chunk.get("usage")
                                    if u:
                                        usage.update(u)
                                except json.JSONDecodeError:
                                    pass

                latency = (time.perf_counter() - start) * 1000
                tc_list = []
                for idx in sorted(tools.keys()):
                    t = tools[idx]
                    args = {}
                    try:
                        args = json.loads(t["arguments"]) if t["arguments"] else {}
                    except json.JSONDecodeError:
                        pass
                    tc_list.append(ToolCall(tool_call_id=t["id"] or uuid4().hex, tool_name=t["name"] or "unknown", arguments=args))
                ht, htc = bool(text), bool(tc_list)
                out = OutputRecord(
                    type="mixed" if (ht and htc) else ("tool_calls" if htc else "text"),
                    text=text[:5000] or None, tool_calls=tc_list or None, stop_reason=stop)
                pack = build_openai_pack(body, sections, out, latency,
                    usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
                emit_pack(pack, call_num, body.get("model", "?"), latency, True)

            return StreamingResponse(stream(), media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
        else:
            async with httpx.AsyncClient(timeout=300.0) as c:
                resp = await c.post(url, content=raw_body, headers=headers)
            latency = (time.perf_counter() - start) * 1000
            rb = resp.json()
            ch = rb.get("choices", [])
            if ch:
                msg = ch[0].get("message", {})
                text = msg.get("content") or ""
                tcs_raw = msg.get("tool_calls") or []
                tc_list = []
                for tc in tcs_raw:
                    fn = tc.get("function", {})
                    args = {}
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        pass
                    tc_list.append(ToolCall(tool_call_id=tc.get("id", ""), tool_name=fn.get("name", "unknown"), arguments=args))
                ht, htc = bool(text), bool(tc_list)
                usage = rb.get("usage", {})
                out = OutputRecord(
                    type="mixed" if (ht and htc) else ("tool_calls" if htc else "text"),
                    text=text[:5000] or None, tool_calls=tc_list or None,
                    stop_reason=ch[0].get("finish_reason"))
            else:
                usage = rb.get("usage", {})
                out = OutputRecord(type="text", text=None)
            pack = build_openai_pack(body, sections, out, latency,
                usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            emit_pack(pack, call_num, body.get("model", "?"), latency, False)
            return Response(content=resp.content, status_code=resp.status_code, headers=clean_resp_headers(resp.headers))

    # ---------------------------------------------------------------
    # Catchall + models
    # ---------------------------------------------------------------

    async def handle_models(request: Request) -> Response:
        import httpx
        headers = {}
        for key in ["authorization", "api-key", "x-api-key"]:
            val = request.headers.get(key)
            if val:
                headers[key] = val
        async with httpx.AsyncClient(timeout=30.0) as c:
            resp = await c.get(f"{target_url}/v1/models", headers=headers)
        return Response(content=resp.content, status_code=resp.status_code, headers=clean_resp_headers(resp.headers))

    async def handle_catchall(request: Request) -> Response:
        import httpx
        raw_body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "transfer-encoding")}
        path = request.url.path
        async with httpx.AsyncClient(timeout=120.0) as c:
            resp = await c.request(request.method, f"{target_url}{path}",
                content=raw_body, headers=headers, params=dict(request.query_params))
        return Response(content=resp.content, status_code=resp.status_code, headers=clean_resp_headers(resp.headers))

    # ---------------------------------------------------------------
    # Routes — both Anthropic and OpenAI endpoints on the same app
    # ---------------------------------------------------------------

    routes = [
        Route("/v1/messages", handle_anthropic_messages, methods=["POST"]),
        Route("/v1/chat/completions", handle_openai_completions, methods=["POST"]),
        Route("/v1/models", handle_models, methods=["GET"]),
        Route("/{path:path}", handle_catchall, methods=["GET", "POST", "PUT", "DELETE", "PATCH"]),
    ]

    return Starlette(routes=routes)
