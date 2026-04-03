"""Output record — what the model produced."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel


class ToolCall(BaseModel):
    tool_call_id: str
    """Provider-assigned tool call identifier."""

    tool_name: str
    """Name of the tool being invoked."""

    arguments: dict[str, Any]
    """Arguments passed to the tool."""


class OutputRecord(BaseModel):
    type: Literal["text", "tool_calls", "mixed", "error"]
    """Discriminator for the output shape."""

    text: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    structured: Optional[Any] = None
    stop_reason: Optional[str] = None
    error: Optional[str] = None
