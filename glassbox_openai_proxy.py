"""Glassbox OpenAI-Compatible Proxy — intercepts OpenAI and Ollama API calls.

Works with any OpenAI-compatible API: OpenAI, Ollama, Azure OpenAI, etc.
Captures every request/response as a ContextPack and forwards transparently.

Usage (OpenAI):
    1. python glassbox_openai_proxy.py
    2. Set OPENAI_BASE_URL=http://localhost:4060/v1
    3. Use any OpenAI client normally

Usage (Ollama):
    1. OPENAI_REAL_URL=http://localhost:11434 python glassbox_openai_proxy.py
    2. Set OPENAI_BASE_URL=http://localhost:4060/v1
    3. Use any OpenAI client normally (pointed at Ollama models)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

sys.path.insert(0, str(Path(__file__).parent / "src"))

from glassbox.format.context_pack import ContextPack, ContextPackMetrics
from glassbox.format.model_info import ModelInfo
from glassbox.format.output import OutputRecord, ToolCall
from glassbox.format.run_metadata import RunMetadata
from glassbox.format.sections import (
    AssistantMessageSection,
    SystemPromptSection,
    ToolResultSection,
    UserMessageSection,
)
from glassbox.format.token_budget import SectionTokenAllocation, TokenBudget
from glassbox.format.version import FORMAT_VERSION
from glassbox.core.emitter import emitter
from glassbox.core.file_storage import FileStorage
from glassbox.core.run import GlassboxRun, RunOptions
from glassbox.wrappers.section_extractor import estimate_tokens
from glassbox.pricing import estimate_cost

logging.basicConfig(level=logging.INFO, format="  %(message)s")
logger = logging.getLogger("glassbox.openai_proxy")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROXY_PORT = int(os.environ.get("GLASSBOX_OPENAI_PROXY_PORT", "4060"))
VIEWER_PORT = int(os.environ.get("GLASSBOX_VIEWER_PORT", "4100"))
OPENAI_API_URL = os.environ.get("OPENAI_REAL_URL", "https://api.openai.com")
STORAGE_DIR = os.environ.get("GLASSBOX_DIR", str(Path.home() / ".glassbox"))

# Detect provider name from target URL
_provider = "openai"
if "localhost" in OPENAI_API_URL or "127.0.0.1" in OPENAI_API_URL:
    _provider = "ollama"
if "azure" in OPENAI_API_URL.lower():
    _provider = "azure-openai"

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

storage = FileStorage(STORAGE_DIR)
run = GlassboxRun(RunOptions(agent_name=_provider, app_name="glassbox-proxy"))
call_count = 0

_metrics_lock = threading.Lock()
_total_input_tokens = 0
_total_output_tokens = 0
_total_latency_ms = 0.0
_models_seen: set[str] = set()


def _accumulate_metrics(pack: ContextPack) -> None:
    global _total_input_tokens, _total_output_tokens, _total_latency_ms
    with _metrics_lock:
        _total_input_tokens += pack.metrics.input_tokens
        _total_output_tokens += pack.metrics.output_tokens
        _total_latency_ms += pack.metrics.latency_ms
        _models_seen.add(pack.model.model)


def _update_run_index():
    try:
        storage.complete_run(RunMetadata(
            run_id=run.run_id,
            agent_name=_provider,
            app_name="glassbox-proxy",
            started_at=run.started_at,
            completed_at=iso_now(),
            step_count=run.step_count,
            total_input_tokens=_total_input_tokens,
            total_output_tokens=_total_output_tokens,
            total_latency_ms=round(_total_latency_ms),
            status="running",
            models_used=sorted(_models_seen),
        ))
    except Exception:
        pass


def iso_now() -> str:
    from datetime import datetime, timezone
    dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


# ---------------------------------------------------------------------------
# Section extraction from OpenAI request
# ---------------------------------------------------------------------------


def extract_sections(body: dict) -> list:
    sections = []
    for msg in body.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Handle content blocks (vision, etc.)
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            content = "\n".join(text_parts)

        if not content or not isinstance(content, str):
            # Tool call messages might not have text content
            if role == "tool":
                content = msg.get("content", "") or ""
            else:
                continue

        if not content:
            continue

        tokens = estimate_tokens(content)
        display = content[:5000] + ("..." if len(content) > 5000 else "")

        if role == "system" or role == "developer":
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
            tool_call_id = msg.get("tool_call_id")
            sections.append(ToolResultSection(
                section_id=uuid4().hex,
                source=f"tool:{tool_call_id or 'unknown'}",
                token_count=tokens, content=display,
                tool_name=msg.get("name", "unknown"),
                tool_call_id=tool_call_id,
            ))

    return sections


# ---------------------------------------------------------------------------
# Output extraction from OpenAI response
# ---------------------------------------------------------------------------


def extract_output(resp_body: dict) -> tuple[OutputRecord, int, int]:
    choices = resp_body.get("choices", [])
    if not choices:
        return OutputRecord(type="text", text=None), 0, 0

    msg = choices[0].get("message", {})
    text = msg.get("content") or ""
    tool_calls_raw = msg.get("tool_calls") or []

    tc_list = []
    for tc in tool_calls_raw:
        fn = tc.get("function", {})
        args_raw = fn.get("arguments", "{}")
        args = {}
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except json.JSONDecodeError:
            pass
        tc_list.append(ToolCall(
            tool_call_id=tc.get("id", uuid4().hex),
            tool_name=fn.get("name", "unknown"),
            arguments=args,
        ))

    has_text = bool(text)
    has_tools = bool(tc_list)

    usage = resp_body.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    return OutputRecord(
        type="mixed" if (has_text and has_tools) else ("tool_calls" if has_tools else "text"),
        text=text[:5000] or None,
        tool_calls=tc_list or None,
        stop_reason=choices[0].get("finish_reason"),
    ), input_tokens, output_tokens


# ---------------------------------------------------------------------------
# Build ContextPack
# ---------------------------------------------------------------------------


def build_pack(
    body: dict,
    sections: list,
    output: OutputRecord,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
) -> ContextPack:
    step = run.next_step()
    section_total = sum(s.token_count for s in sections)
    now = iso_now()
    effective_input = max(input_tokens, section_total)
    model_name = body.get("model", "unknown")

    return ContextPack(
        format_version=FORMAT_VERSION,
        run_id=run.run_id,
        agent_name=_provider,
        app_name="glassbox-proxy",
        step_id=step.step_id,
        step_index=step.step_index,
        step_label=f"chat.completions.create ({model_name})",
        started_at=iso_now(),
        completed_at=now,
        sections=sections,
        token_budget=TokenBudget(
            total_budget=section_total + (body.get("max_tokens") or body.get("max_completion_tokens") or 4096),
            total_used=section_total,
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
            provider=_provider,
            model=model_name,
            temperature=body.get("temperature"),
            max_tokens=body.get("max_tokens") or body.get("max_completion_tokens"),
            top_p=body.get("top_p"),
            stop_sequences=(
                [body["stop"]] if isinstance(body.get("stop"), str)
                else body.get("stop")
            ),
        ),
        output=output,
        metrics=ContextPackMetrics(
            latency_ms=round(latency_ms),
            input_tokens=effective_input,
            output_tokens=output_tokens,
            cost_estimate_usd=estimate_cost(model_name, effective_input, output_tokens),
        ),
    )


# ---------------------------------------------------------------------------
# Proxy handler
# ---------------------------------------------------------------------------


async def proxy_chat_completions(request: Request) -> Response:
    """Forward /v1/chat/completions and capture a ContextPack."""
    global call_count

    raw_body = await request.body()
    body = json.loads(raw_body)
    is_streaming = body.get("stream", False)

    sections = extract_sections(body)

    # Forward headers
    headers = {}
    for key in ["authorization", "content-type", "api-key", "x-api-key"]:
        val = request.headers.get(key)
        if val:
            headers[key] = val
    headers["content-type"] = "application/json"

    target_url = f"{OPENAI_API_URL}/v1/chat/completions"
    start = time.perf_counter()
    call_count += 1
    call_num = call_count

    if is_streaming:
        async def stream_and_capture():
            accumulated_text = ""
            accumulated_tools: dict[int, dict] = {}  # index -> {id, name, arguments}
            usage = {}
            stop_reason = None

            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST", target_url, content=raw_body, headers=headers,
                ) as resp:
                    async for line in resp.aiter_lines():
                        yield line + "\n"

                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                continue
                            try:
                                chunk = json.loads(data_str)
                                choices = chunk.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    # Text content
                                    if "content" in delta and delta["content"]:
                                        accumulated_text += delta["content"]
                                    # Tool calls
                                    for tc in delta.get("tool_calls", []):
                                        idx = tc.get("index", 0)
                                        if idx not in accumulated_tools:
                                            accumulated_tools[idx] = {
                                                "id": tc.get("id", ""),
                                                "name": tc.get("function", {}).get("name", ""),
                                                "arguments": "",
                                            }
                                        fn = tc.get("function", {})
                                        if "name" in fn and fn["name"]:
                                            accumulated_tools[idx]["name"] = fn["name"]
                                        if "arguments" in fn:
                                            accumulated_tools[idx]["arguments"] += fn["arguments"]
                                    # Finish reason
                                    fr = choices[0].get("finish_reason")
                                    if fr:
                                        stop_reason = fr

                                # Usage (some providers send it in the last chunk)
                                u = chunk.get("usage")
                                if u:
                                    usage.update(u)

                            except json.JSONDecodeError:
                                pass

            latency = (time.perf_counter() - start) * 1000

            tc_list = []
            for idx in sorted(accumulated_tools.keys()):
                t = accumulated_tools[idx]
                args = {}
                try:
                    args = json.loads(t["arguments"]) if t["arguments"] else {}
                except json.JSONDecodeError:
                    pass
                tc_list.append(ToolCall(
                    tool_call_id=t["id"] or uuid4().hex,
                    tool_name=t["name"] or "unknown",
                    arguments=args,
                ))

            has_text = bool(accumulated_text)
            has_tools = bool(tc_list)

            output = OutputRecord(
                type="mixed" if (has_text and has_tools) else ("tool_calls" if has_tools else "text"),
                text=accumulated_text[:5000] if accumulated_text else None,
                tool_calls=tc_list or None,
                stop_reason=stop_reason,
            )

            pack = build_pack(
                body, sections, output, latency,
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            )

            _accumulate_metrics(pack)
            emitter.emit("step:complete", pack)
            storage.write_step(pack)
            _update_run_index()

            logger.info(
                f"  [{call_num}] {body.get('model', '?')} | "
                f"{pack.metrics.input_tokens} in / {pack.metrics.output_tokens} out | "
                f"{latency:.0f}ms | streaming"
            )

        return StreamingResponse(
            stream_and_capture(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(target_url, content=raw_body, headers=headers)

        latency = (time.perf_counter() - start) * 1000
        resp_body = resp.json()

        output, inp_t, out_t = extract_output(resp_body)
        pack = build_pack(body, sections, output, latency, inp_t, out_t)

        _accumulate_metrics(pack)
        emitter.emit("step:complete", pack)
        storage.write_step(pack)
        _update_run_index()

        logger.info(
            f"  [{call_num}] {body.get('model', '?')} | "
            f"{inp_t} in / {out_t} out | {latency:.0f}ms"
        )

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )


async def proxy_models(request: Request) -> Response:
    """Forward /v1/models so clients can discover available models."""
    headers = {}
    for key in ["authorization", "api-key", "x-api-key"]:
        val = request.headers.get(key)
        if val:
            headers[key] = val

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{OPENAI_API_URL}/v1/models", headers=headers)

    return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))


async def proxy_catchall(request: Request) -> Response:
    """Forward any other API call transparently."""
    raw_body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "transfer-encoding")}
    path = request.url.path
    target_url = f"{OPENAI_API_URL}{path}"

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.request(
            request.method, target_url,
            content=raw_body, headers=headers,
            params=dict(request.query_params),
        )

    return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def create_proxy_app() -> Starlette:
    routes = [
        Route("/v1/chat/completions", proxy_chat_completions, methods=["POST"]),
        Route("/v1/models", proxy_models, methods=["GET"]),
        Route("/{path:path}", proxy_catchall, methods=["GET", "POST", "PUT", "DELETE", "PATCH"]),
    ]
    return Starlette(routes=routes)


def main():
    import uvicorn

    target_label = OPENAI_API_URL
    if _provider == "ollama":
        target_label = f"{OPENAI_API_URL} (Ollama)"

    print()
    print("  ┌──────────────────────────────────────────────────┐")
    print("  │       GLASSBOX OPENAI-COMPATIBLE PROXY           │")
    print("  ├──────────────────────────────────────────────────┤")
    print(f"  │  Proxy:    http://localhost:{PROXY_PORT}               │")
    print(f"  │  Viewer:   http://localhost:{VIEWER_PORT}               │")
    print(f"  │  Target:   {target_label[:38]:<38} │")
    print("  ├──────────────────────────────────────────────────┤")
    print("  │  Set this in your environment:                   │")
    print(f"  │  OPENAI_BASE_URL=http://localhost:{PROXY_PORT}/v1      │")
    print("  │                                                  │")
    print("  │  Then use any OpenAI client normally.            │")
    print("  │  Every API call will appear in the viewer.       │")
    print("  └──────────────────────────────────────────────────┘")
    print()

    import socket
    def port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) == 0

    if port_in_use(VIEWER_PORT):
        print(f"  Viewer already running on port {VIEWER_PORT} — skipping\n")
    else:
        from glassbox.viewer.server import create_app as create_viewer_app
        viewer_app = create_viewer_app(STORAGE_DIR)

        def run_viewer():
            uvicorn.run(viewer_app, host="0.0.0.0", port=VIEWER_PORT, log_level="warning")

        viewer_thread = threading.Thread(target=run_viewer, daemon=True)
        viewer_thread.start()
        print(f"  Viewer started on port {VIEWER_PORT}\n")

    proxy_app = create_proxy_app()
    uvicorn.run(proxy_app, host="0.0.0.0", port=PROXY_PORT, log_level="warning")


if __name__ == "__main__":
    main()
