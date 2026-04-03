"""Run metadata — aggregate information about a complete run."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


RunStatus = Literal["running", "completed", "errored"]


class RunMetadata(BaseModel):
    run_id: str
    agent_name: Optional[str] = None
    app_name: Optional[str] = None
    started_at: str
    completed_at: Optional[str] = None
    step_count: int = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_latency_ms: float = Field(ge=0)
    total_cost_estimate_usd: Optional[float] = Field(default=None, ge=0)
    status: RunStatus
    models_used: list[str]
    tags: Optional[dict[str, str]] = None
    content_hash: Optional[str] = None
