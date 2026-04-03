"""Context diff types — step-to-step comparison."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SectionChange(BaseModel):
    section_id: str
    change_type: Literal["added", "removed", "modified", "unchanged"]
    section_type: str
    from_tokens: int = Field(ge=0)
    to_tokens: int = Field(ge=0)
    summary: Optional[str] = None


class ContextPackDiff(BaseModel):
    from_step_id: str
    to_step_id: str
    from_step_index: int = Field(ge=0)
    to_step_index: int = Field(ge=0)
    changes: list[SectionChange]
    sections_added: int = Field(ge=0)
    sections_removed: int = Field(ge=0)
    sections_modified: int = Field(ge=0)
    token_delta: int
