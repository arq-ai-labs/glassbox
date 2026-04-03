"""Viewer server — Starlette ASGI app serving the inspector UI + JSON API.

Matches the TypeScript Express server API routes exactly for cross-compat.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("glassbox")

def _load_html() -> str:
    html_path = Path(__file__).parent / "viewer.html"
    return html_path.read_text(encoding="utf-8")


def create_app(directory: Optional[str] = None) -> Any:
    """Create the Starlette ASGI app."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Route
    from sse_starlette.sse import EventSourceResponse

    from ..core.emitter import emitter
    from ..core.file_storage import FileStorage
    from ..diff import diff_context_packs

    storage = FileStorage(directory)

    async def index(request: Request) -> HTMLResponse:
        return HTMLResponse(_load_html())

    async def list_runs(request: Request) -> JSONResponse:
        limit = int(request.query_params.get("limit", "50"))
        offset = int(request.query_params.get("offset", "0"))
        runs = storage.list_runs(limit, offset)
        return JSONResponse({"runs": [r.model_dump(mode="json") for r in runs]})

    async def get_run(request: Request) -> JSONResponse:
        run = storage.get_run(request.path_params["run_id"])
        if not run:
            return JSONResponse({"error": "Run not found"}, status_code=404)
        return JSONResponse(run.model_dump(mode="json"))

    async def get_steps(request: Request) -> JSONResponse:
        steps = storage.get_steps(request.path_params["run_id"])
        return JSONResponse({"steps": [s.model_dump(mode="json") for s in steps]})

    async def get_step(request: Request) -> JSONResponse:
        step = storage.get_step(
            request.path_params["run_id"],
            request.path_params["step_id"],
        )
        if not step:
            return JSONResponse({"error": "Step not found"}, status_code=404)
        return JSONResponse(step.model_dump(mode="json"))

    async def get_diff(request: Request) -> JSONResponse:
        run_id = request.path_params["run_id"]
        from_id = request.query_params.get("from")
        to_id = request.query_params.get("to")
        if not from_id or not to_id:
            return JSONResponse({"error": "from and to query params required"}, status_code=400)
        from_pack = storage.get_step(run_id, from_id)
        to_pack = storage.get_step(run_id, to_id)
        if not from_pack or not to_pack:
            return JSONResponse({"error": "One or both steps not found"}, status_code=404)
        diff = diff_context_packs(from_pack, to_pack)
        return JSONResponse(diff.model_dump(mode="json"))

    async def live(request: Request) -> EventSourceResponse:
        import asyncio

        queue: asyncio.Queue[str] = asyncio.Queue()

        def on_step(pack: Any) -> None:
            data = json.dumps({
                "type": "step:complete",
                "run_id": pack.run_id,
                "step_id": pack.step_id,
                "step_index": pack.step_index,
                "agent_name": pack.agent_name,
                "model": pack.model.model,
                "input_tokens": pack.metrics.input_tokens,
                "output_tokens": pack.metrics.output_tokens,
            })
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                pass

        def on_run(meta: Any) -> None:
            data = json.dumps({
                "type": "run:complete",
                "run_id": meta.get("run_id") if isinstance(meta, dict) else getattr(meta, "run_id", ""),
            })
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                pass

        unsub_step = emitter.on("step:complete", on_step)
        unsub_run = emitter.on("run:complete", on_run)

        async def event_generator() -> Any:
            try:
                while True:
                    data = await queue.get()
                    yield {"data": data}
            except asyncio.CancelledError:
                pass
            finally:
                unsub_step()
                unsub_run()

        return EventSourceResponse(event_generator())

    routes = [
        Route("/", index),
        Route("/api/runs", list_runs),
        Route("/api/runs/{run_id}", get_run),
        Route("/api/runs/{run_id}/steps", get_steps),
        Route("/api/runs/{run_id}/steps/{step_id}", get_step),
        Route("/api/runs/{run_id}/diff", get_diff),
        Route("/api/live", live),
    ]

    return Starlette(routes=routes)


def serve(port: int = 4100, directory: Optional[str] = None) -> None:
    """Start the Glassbox viewer server.

    Args:
        port: Port to listen on (default 4100).
        directory: Path to .glassbox/ storage directory.
    """
    import uvicorn

    app = create_app(directory)
    print(f"\n  Glassbox Inspector running at http://localhost:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
