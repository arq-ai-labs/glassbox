"""Context section types — the discriminated union at the heart of ContextPack."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class SectionBase(BaseModel):
    """Fields shared by every section type."""

    section_id: str
    """Stable identifier for this section within a ContextPack."""

    type: str
    """Discriminator — determines the section's semantic role."""

    source: Optional[str] = None
    """Human-readable origin label. E.g. 'system', 'tool:web_search', 'rag:pinecone'."""

    token_count: int = Field(ge=0)
    """Estimated token count for this section's content."""

    content: str
    """The actual text content that was (or would be) sent to the model."""

    metadata: Optional[dict[str, Any]] = None
    """Arbitrary key-value pairs for vendor or application-specific data."""


class SystemPromptSection(SectionBase):
    type: Literal["system_prompt"] = "system_prompt"


class UserMessageSection(SectionBase):
    type: Literal["user_message"] = "user_message"


class AssistantMessageSection(SectionBase):
    type: Literal["assistant_message"] = "assistant_message"


class ToolResultSection(SectionBase):
    type: Literal["tool_result"] = "tool_result"

    tool_name: str
    """Name of the tool that produced this result."""

    tool_call_id: Optional[str] = None
    """Provider-assigned tool call ID, if available."""


class RetrievalSection(SectionBase):
    type: Literal["retrieval"] = "retrieval"

    query: Optional[str] = None
    """The query used to retrieve this content."""

    source_collection: Optional[str] = None
    """Collection or index the content was retrieved from."""

    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    """Retrieval relevance score (0-1)."""

    document_id: Optional[str] = None
    """Identifier of the source document."""


class MemorySection(SectionBase):
    type: Literal["memory"] = "memory"

    memory_type: Optional[Literal["conversation", "long_term", "working", "episodic"]] = None
    """Classification of memory type."""


class InstructionSection(SectionBase):
    type: Literal["instruction"] = "instruction"


class CustomSection(SectionBase):
    """Extension point — allows consumers to define their own section types."""

    type: Literal["custom"] = "custom"

    custom_type: str
    """Application-defined sub-type name."""


ContextSection = Annotated[
    Union[
        SystemPromptSection,
        UserMessageSection,
        AssistantMessageSection,
        ToolResultSection,
        RetrievalSection,
        MemorySection,
        InstructionSection,
        CustomSection,
    ],
    Field(discriminator="type"),
]
"""Discriminated union of all section types."""
