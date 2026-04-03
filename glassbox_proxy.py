"""Glassbox Proxy — intercepts real Anthropic API calls in real-time.

Sits between Claude Code (or any Anthropic client) and the Anthropic API.
Captures every request/response as a ContextPack. Forwards everything
transparently — your app works exactly the same, but now you can see
what the model saw.

Usage:
    1. Start this proxy:
       python glassbox_proxy.py

    2. Configure Claude Code to use it:
       Set ANTHROPIC_BASE_URL=http://localhost:4050 in your environment

    3. Open the viewer:
       http://localhost:4100

    4. Use Claude Code normally — every call appears in the viewer in real-time.
"""

from __future__ import annotations

import asyncio
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
from starlette.routing import Route, Mount

# Add the src directory so we can import glassbox
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

logging.basicConfig(level=logging.INFO, format="  %(message)s")
logger = logging.getLogger("glassbox.proxy")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROXY_PORT = int(os.environ.get("GLASSBOX_PROXY_PORT", "4050"))
VIEWER_PORT = int(os.environ.get("GLASSBOX_VIEWER_PORT", "4100"))
ANTHROPIC_API_URL = os.environ.get("ANTHROPIC_REAL_URL", "https://api.anthropic.com")
STORAGE_DIR = os.environ.get("GLASSBOX_DIR", str(Path.home() / ".glassbox"))

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

storage = FileStorage(STORAGE_DIR)
# One run per proxy session — all calls grouped together
run = GlassboxRun(RunOptions(agent_name="claude-code", app_name="glassbox-proxy"))
call_count = 0

# ---------------------------------------------------------------------------
# Run-level metric accumulators (updated after every step, thread-safe)
# ---------------------------------------------------------------------------
_metrics_lock = threading.Lock()
_total_input_tokens = 0
_total_output_tokens = 0
_total_latency_ms = 0.0
_models_seen: set[str] = set()


def _accumulate_metrics(pack: ContextPack) -> None:
    """Add a step's metrics to the run-level accumulators (thread-safe)."""
    global _total_input_tokens, _total_output_tokens, _total_latency_ms
    with _metrics_lock:
        _total_input_tokens += pack.metrics.input_tokens
        _total_output_tokens += pack.metrics.output_tokens
        _total_latency_ms += pack.metrics.latency_ms
        _models_seen.add(pack.model.model)


def _update_run_index():
    """Update the run index after every step so the viewer can see it immediately."""
    try:
        storage.complete_run(RunMetadata(
            run_id=run.run_id,
            agent_name="claude-code",
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


# Use the SDK's token estimator and cost calculator
from glassbox.wrappers.section_extractor import estimate_tokens
from glassbox.pricing import estimate_cost


def iso_now() -> str:
    from datetime import datetime, timezone
    dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


# ---------------------------------------------------------------------------
# Section extraction from Anthropic request
# ---------------------------------------------------------------------------


def extract_sections(body: dict) -> list:
    sections = []

    # System prompt
    system = body.get("system")
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
            # Truncate display but keep full token count
            sections.append(SystemPromptSection(
                section_id=uuid4().hex, source="system",
                token_count=estimate_tokens(text),
                content=text[:5000] + ("..." if len(text) > 5000 else ""),
            ))

    # Messages
    for msg in body.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, str):
            if not content:
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
            continue

        if isinstance(content, list):
            # Text blocks
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
                            content=result[:3000] + ("..." if len(result) > 3000 else ""),
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

            text = "\n".join(t for t in text_parts if t)
            if text:
                tokens = estimate_tokens(text)
                display = text[:5000] + ("..." if len(text) > 5000 else "")
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


# ---------------------------------------------------------------------------
# Output extraction from Anthropic response
# ---------------------------------------------------------------------------


def extract_output(response_body: dict) -> tuple[OutputRecord, int, int, Optional[int], Optional[int]]:
    content_blocks = response_body.get("content", [])
    text_parts = []
    tc_list = []

    for block in content_blocks:
        btype = block.get("type", "")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tc_list.append(ToolCall(
                tool_call_id=block.get("id", ""),
                tool_name=block.get("name", "unknown"),
                arguments=block.get("input", {}),
            ))

    text = "".join(text_parts)
    has_text = bool(text)
    has_tools = bool(tc_list)

    usage = response_body.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens")
    cache_creation = usage.get("cache_creation_input_tokens")

    return OutputRecord(
        type="mixed" if (has_text and has_tools) else ("tool_calls" if has_tools else "text"),
        text=text[:5000] + ("..." if len(text) > 5000 else "") if text else None,
        tool_calls=tc_list or None,
        stop_reason=response_body.get("stop_reason"),
    ), input_tokens, output_tokens, cache_read, cache_creation


# ---------------------------------------------------------------------------
# Build ContextPack from request + response
# ---------------------------------------------------------------------------


def build_pack(
    body: dict,
    sections: list,
    output: OutputRecord,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    cache_read: Optional[int],
    cache_creation: Optional[int],
) -> ContextPack:
    step = run.next_step()
    section_total = sum(s.token_count for s in sections)
    now = iso_now()

    # Use API-reported input_tokens when available and plausible.
    # Streaming responses often report incomplete usage — fall back to section estimates.
    effective_input = input_tokens if input_tokens >= section_total else section_total

    # For cache-enabled requests, the real input can be much lower than section total
    # because cached tokens aren't counted in input_tokens. Account for that.
    if cache_read and cache_read > 0:
        effective_input = max(input_tokens, section_total)

    return ContextPack(
        format_version=FORMAT_VERSION,
        run_id=run.run_id,
        agent_name="claude-code",
        app_name="glassbox-proxy",
        step_id=step.step_id,
        step_index=step.step_index,
        step_label=f"messages.create ({body.get('model', 'unknown')})",
        started_at=iso_now(),
        completed_at=now,
        sections=sections,
        token_budget=TokenBudget(
            total_budget=section_total + (body.get("max_tokens") or 4096),
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
            provider="anthropic",
            model=body.get("model", "unknown"),
            temperature=body.get("temperature"),
            max_tokens=body.get("max_tokens"),
            top_p=body.get("top_p"),
            stop_sequences=body.get("stop_sequences"),
        ),
        output=output,
        metrics=ContextPackMetrics(
            latency_ms=round(latency_ms),
            input_tokens=effective_input,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
            cost_estimate_usd=estimate_cost(
                body.get("model", "unknown"), effective_input, output_tokens
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Proxy handler
# ---------------------------------------------------------------------------


async def proxy_messages(request: Request) -> Response:
    """Forward /v1/messages to the real Anthropic API and capture a ContextPack."""
    global call_count

    raw_body = await request.body()
    body = json.loads(raw_body)
    is_streaming = body.get("stream", False)

    # Extract sections from request
    sections = extract_sections(body)

    # Forward headers (pass through API key, etc.)
    headers = {}
    for key in ["x-api-key", "anthropic-version", "anthropic-beta", "content-type", "authorization"]:
        val = request.headers.get(key)
        if val:
            headers[key] = val
    headers["content-type"] = "application/json"

    target_url = f"{ANTHROPIC_API_URL}/v1/messages"
    start = time.perf_counter()
    call_count += 1
    call_num = call_count

    if is_streaming:
        # Streaming: forward SSE events while buffering the response
        async def stream_and_capture():
            accumulated_text = ""
            accumulated_tools = []
            usage = {}
            stop_reason = None

            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST", target_url, content=raw_body, headers=headers,
                ) as resp:
                    async for line in resp.aiter_lines():
                        yield line + "\n"

                        # Parse SSE data events
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                continue
                            try:
                                event = json.loads(data_str)
                                etype = event.get("type", "")

                                if etype == "content_block_delta":
                                    delta = event.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        accumulated_text += delta.get("text", "")
                                    elif delta.get("type") == "input_json_delta":
                                        # Tool input streaming — accumulate
                                        pass

                                elif etype == "content_block_start":
                                    cb = event.get("content_block", {})
                                    if cb.get("type") == "tool_use":
                                        accumulated_tools.append({
                                            "id": cb.get("id", ""),
                                            "name": cb.get("name", ""),
                                            "input": {},
                                        })

                                elif etype == "message_delta":
                                    delta = event.get("delta", {})
                                    stop_reason = delta.get("stop_reason")
                                    u = event.get("usage", {})
                                    if u:
                                        usage.update(u)

                                elif etype == "message_start":
                                    msg = event.get("message", {})
                                    u = msg.get("usage", {})
                                    if u:
                                        usage.update(u)

                            except json.JSONDecodeError:
                                pass

            # Stream complete — build ContextPack
            latency = (time.perf_counter() - start) * 1000

            tc_list = [
                ToolCall(tool_call_id=t["id"], tool_name=t["name"], arguments=t.get("input", {}))
                for t in accumulated_tools
            ]
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
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("cache_read_input_tokens"),
                usage.get("cache_creation_input_tokens"),
            )

            _accumulate_metrics(pack)
            emitter.emit("step:complete", pack)
            storage.write_step(pack)
            _update_run_index()

            actual_in = pack.metrics.input_tokens
            actual_out = pack.metrics.output_tokens
            logger.info(
                f"  [{call_num}] {body.get('model', '?')} | "
                f"{actual_in} in / {actual_out} out | "
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
        # Non-streaming: simple forward
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(target_url, content=raw_body, headers=headers)

        latency = (time.perf_counter() - start) * 1000
        resp_body = resp.json()

        output, inp_t, out_t, cr, cc = extract_output(resp_body)
        pack = build_pack(body, sections, output, latency, inp_t, out_t, cr, cc)

        _accumulate_metrics(pack)
        emitter.emit("step:complete", pack)
        storage.write_step_background(pack)
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


async def proxy_catchall(request: Request) -> Response:
    """Forward any other Anthropic API call transparently."""
    raw_body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "transfer-encoding")}

    path = request.url.path
    target_url = f"{ANTHROPIC_API_URL}{path}"

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.request(
            request.method, target_url,
            content=raw_body, headers=headers,
            params=dict(request.query_params),
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )


# ---------------------------------------------------------------------------
# Combined app: proxy + viewer on one server (or two ports)
# ---------------------------------------------------------------------------


def create_proxy_app() -> Starlette:
    routes = [
        Route("/v1/messages", proxy_messages, methods=["POST"]),
        Route("/{path:path}", proxy_catchall, methods=["GET", "POST", "PUT", "DELETE", "PATCH"]),
    ]
    return Starlette(routes=routes)


def main():
    import uvicorn

    print()
    print("  ┌──────────────────────────────────────────────────┐")
    print("  │         GLASSBOX REAL-TIME PROXY                 │")
    print("  ├──────────────────────────────────────────────────┤")
    print(f"  │  Proxy:   http://localhost:{PROXY_PORT}                │")
    print(f"  │  Viewer:  http://localhost:{VIEWER_PORT}                │")
    print("  ├──────────────────────────────────────────────────┤")
    print("  │  Set this in your environment:                   │")
    print(f"  │  ANTHROPIC_BASE_URL=http://localhost:{PROXY_PORT}       │")
    print("  │                                                  │")
    print("  │  Then use Claude Code normally.                  │")
    print("  │  Every API call will appear in the viewer.       │")
    print("  └──────────────────────────────────────────────────┘")
    print()

    # Start viewer on separate port in background (skip if already running)
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

    # Start proxy on main thread
    proxy_app = create_proxy_app()
    uvicorn.run(proxy_app, host="0.0.0.0", port=PROXY_PORT, log_level="warning")


if __name__ == "__main__":
    main()
