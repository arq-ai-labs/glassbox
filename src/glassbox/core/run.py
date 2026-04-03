"""GlassboxRun — lifecycle management for a single agent execution."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from .emitter import emitter


def _now_iso() -> str:
    """ISO 8601 timestamp matching JavaScript's toISOString() output."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
        f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"


@dataclass
class StepHandle:
    step_id: str
    step_index: int


@dataclass
class RunOptions:
    agent_name: Optional[str] = None
    app_name: Optional[str] = None
    tags: Optional[dict[str, str]] = None


class GlassboxRun:
    """Tracks a single agent execution run."""

    def __init__(self, options: Optional[RunOptions] = None) -> None:
        opts = options or RunOptions()
        self.run_id: str = uuid4().hex
        self.agent_name: Optional[str] = opts.agent_name
        self.app_name: Optional[str] = opts.app_name
        self.tags: Optional[dict[str, str]] = opts.tags
        self.started_at: str = _now_iso()
        self._step_counter: int = 0
        self._completed: bool = False
        self._lock = threading.Lock()

        emitter.emit("run:start", {
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "app_name": self.app_name,
        })

    def next_step(self) -> StepHandle:
        """Allocate the next step in this run."""
        with self._lock:
            if self._completed:
                raise RuntimeError(f"Run {self.run_id} is already completed")
            handle = StepHandle(
                step_id=uuid4().hex,
                step_index=self._step_counter,
            )
            self._step_counter += 1

        emitter.emit("step:start", {
            "run_id": self.run_id,
            "step_id": handle.step_id,
            "step_index": handle.step_index,
        })
        return handle

    @property
    def step_count(self) -> int:
        return self._step_counter

    @property
    def completed(self) -> bool:
        return self._completed

    def complete(self) -> None:
        self._completed = True
