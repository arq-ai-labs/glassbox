"""ContextPack — the core Glassbox primitive.

A structured, versioned record of exactly what context an LLM saw during a call,
why those pieces were selected, what was excluded, and what the model produced.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .context_source import ContextInventory
from .model_info import ModelInfo
from .output import OutputRecord
from .sections import ContextSection
from .token_budget import TokenBudget
from .version import FORMAT_VERSION


class ContextPackMetrics(BaseModel):
    """Performance and cost data for a single step."""

    latency_ms: float = Field(ge=0)
    """Wall-clock time from request to response in milliseconds."""

    input_tokens: int = Field(ge=0)
    """Tokens consumed by the input/context."""

    output_tokens: int = Field(ge=0)
    """Tokens generated in the output."""

    cache_read_tokens: Optional[int] = Field(default=None, ge=0)
    cache_creation_tokens: Optional[int] = Field(default=None, ge=0)
    cost_estimate_usd: Optional[float] = Field(default=None, ge=0)


class MultiAgentLink(BaseModel):
    """Delegation metadata for multi-agent architectures."""

    parent_run_id: Optional[str] = None
    parent_step_id: Optional[str] = None
    delegation_scope: Optional[str] = None
    inherited_sections: Optional[list[str]] = None


class ContextPack(BaseModel):
    """The core Glassbox observable — one per LLM call."""

    # --- Envelope ---
    format_version: str = FORMAT_VERSION

    # --- Run-level identity ---
    run_id: str
    agent_name: Optional[str] = None
    app_name: Optional[str] = None

    # --- Step-level identity ---
    step_id: str
    step_index: int = Field(ge=0)
    step_label: Optional[str] = None

    # --- Timestamps ---
    started_at: str
    """ISO 8601 timestamp when the step began."""

    completed_at: str
    """ISO 8601 timestamp when the step completed."""

    # --- Context Assembly ---
    sections: list[ContextSection]
    token_budget: TokenBudget

    # --- Model ---
    model: ModelInfo

    # --- Output ---
    output: OutputRecord

    # --- Metrics ---
    metrics: ContextPackMetrics

    # --- Pre-assembly inventory (optional) ---
    context_inventory: Optional[ContextInventory] = None
    """The full upstream inventory — what was available to the assembly
    process, not just what made it into sections. Enables source-level
    drift detection and assembly coverage analysis.
    """

    # --- Multi-agent (optional) ---
    multi_agent: Optional[MultiAgentLink] = None

    # --- Extension point ---
    extensions: Optional[dict[str, Any]] = None
