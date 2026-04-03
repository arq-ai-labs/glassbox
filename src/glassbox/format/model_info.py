"""Model info — which model was used and with what parameters."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class ModelInfo(BaseModel):
    provider: str
    """Provider name: 'anthropic', 'openai', 'azure', 'ollama', etc."""

    model: str
    """Full model identifier, e.g. 'claude-sonnet-4-20250514', 'gpt-4o'."""

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[list[str]] = None
    additional_params: Optional[dict[str, Any]] = None
