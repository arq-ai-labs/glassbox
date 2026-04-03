"""Context Sources — the pre-assembly inventory layer.

This module models what was AVAILABLE to the context assembly process,
not just what made it into the wire call. It captures the full upstream
pipeline: files on disk, skill definitions, environment rules, memory
stores, and retrieval corpora — the raw material before it gets
compressed into messages[].

The ContextSource is to assembly what the RejectedCandidate is to the
token budget: it tells you what existed in the world, whether it was
considered, and what happened to it.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Source status — the lifecycle of a context source
# ---------------------------------------------------------------------------

SourceStatus = Literal[
    "included",       # Made it into the final context (maps to a section)
    "considered",     # Was evaluated but not selected (e.g. low relevance)
    "available",      # Existed but was never evaluated (e.g. file in dir)
    "excluded",       # Explicitly blocked by policy
    "stale",          # Detected but too old / outdated
    "unknown",        # Status could not be determined
]
"""Lifecycle state of a context source relative to the assembly process."""


SourceType = Literal[
    "file",           # File on disk (code, config, data)
    "skill",          # Skill definition (SKILL.md, .cursorrules, etc.)
    "agent_config",   # Agent configuration (CLAUDE.md, .clinerules, etc.)
    "memory_store",   # Memory backend (conversation, long-term, vector DB)
    "retrieval_corpus",  # RAG corpus or collection available for search
    "tool_definition",   # Tool/function schema available to the model
    "environment",    # Environment variable, runtime config
    "conversation_history",  # Prior turns available in the session
    "api_schema",     # API specification (OpenAPI, GraphQL, etc.)
    "custom",         # Extension point
]
"""Classification of the upstream context source."""


# ---------------------------------------------------------------------------
# ContextSource — one entry per upstream source
# ---------------------------------------------------------------------------

class ContextSource(BaseModel):
    """A single upstream context source — something that existed and could
    have contributed to the LLM's context window.

    This is NOT the content itself (that's in the sections). This is the
    metadata about what was available, whether it was considered, and
    what happened to it during assembly.
    """

    source_id: str
    """Stable identifier for this source (e.g. file path hash, skill name)."""

    type: SourceType
    """What kind of source this is."""

    name: str
    """Human-readable name (e.g. filename, skill name, memory store name)."""

    path: Optional[str] = None
    """Filesystem path, URL, or logical address of the source.
    For files: the absolute or relative path.
    For skills: path to the SKILL.md or definition file.
    For memory: the store name or connection string (redacted).
    For retrieval: the collection/index name.
    """

    status: SourceStatus = "unknown"
    """What happened to this source during assembly."""

    section_id: Optional[str] = None
    """If status is 'included', the section_id in the ContextPack it mapped to.
    This links the source inventory to the wire-level sections.
    """

    token_count: Optional[int] = Field(default=None, ge=0)
    """Estimated tokens if this source were included in full.
    For files: token count of the file content.
    For skills: token count of the skill definition.
    None if not estimated.
    """

    token_count_included: Optional[int] = Field(default=None, ge=0)
    """Tokens actually included (may be less than token_count if truncated)."""

    content_hash: Optional[str] = None
    """Hash of the source content (sha256 prefix). Used for change detection
    without storing the actual content. Enables drift detection at the
    source level — did the SKILL.md change between steps?
    """

    last_modified: Optional[str] = None
    """ISO 8601 timestamp of when this source was last modified.
    For files: filesystem mtime.
    For memory: last write time.
    """

    exclusion_reason: Optional[str] = None
    """If status is 'considered' or 'excluded', why it wasn't included.
    Free-text explanation (e.g. 'relevance score 0.32 below threshold 0.5',
    'file too large (48k tokens)', 'blocked by .glassboxignore').
    """

    priority: Optional[int] = None
    """Assembly priority (lower = higher priority). Used to understand
    displacement decisions when the budget is tight.
    """

    tags: Optional[dict[str, str]] = None
    """Arbitrary key-value metadata. Examples:
    - language: 'python'
    - framework: 'langchain'
    - skill_version: '2.1'
    - memory_type: 'episodic'
    """


# ---------------------------------------------------------------------------
# ContextInventory — the full pre-assembly snapshot
# ---------------------------------------------------------------------------

class ContextInventory(BaseModel):
    """The complete inventory of what was available to the context assembly
    process for a single LLM call.

    This is the 'before' picture. The sections in the ContextPack are the
    'after' picture. Together they answer: what existed, what was selected,
    what was left behind, and why.
    """

    sources: list[ContextSource] = Field(default_factory=list)
    """All sources that existed at assembly time."""

    working_directory: Optional[str] = None
    """The working directory of the agent at assembly time.
    For file-based agents, this is the root of the project.
    """

    discovery_method: Optional[str] = None
    """How sources were discovered. Examples:
    - 'filesystem_scan' — walked the directory tree
    - 'explicit' — sources were manually registered by the framework
    - 'hybrid' — combination of scan + explicit
    - 'proxy_inferred' — inferred from the API payload (limited)
    """

    discovered_at: Optional[str] = None
    """ISO 8601 timestamp of when the inventory was captured."""

    # --- Aggregate stats ---
    total_sources: int = 0
    """Total number of sources discovered."""

    sources_included: int = 0
    """Number of sources that made it into sections."""

    sources_considered: int = 0
    """Number of sources that were evaluated but not included."""

    sources_available: int = 0
    """Number of sources that existed but were never evaluated."""

    sources_excluded: int = 0
    """Number of sources explicitly blocked by policy."""

    total_tokens_available: Optional[int] = None
    """Sum of token_count across all sources (the full 'universe' of content).
    Compared with token_budget.total_used, this shows compression ratio.
    """

    def compute_stats(self) -> None:
        """Recompute aggregate stats from the sources list."""
        self.total_sources = len(self.sources)
        self.sources_included = sum(1 for s in self.sources if s.status == "included")
        self.sources_considered = sum(1 for s in self.sources if s.status == "considered")
        self.sources_available = sum(1 for s in self.sources if s.status == "available")
        self.sources_excluded = sum(1 for s in self.sources if s.status == "excluded")
        counts = [s.token_count for s in self.sources if s.token_count is not None]
        self.total_tokens_available = sum(counts) if counts else None


# ---------------------------------------------------------------------------
# SourceChange — for inventory-level drift detection
# ---------------------------------------------------------------------------

class SourceChange(BaseModel):
    """A change to a single context source between two steps."""

    source_id: str
    name: str
    type: SourceType

    change_type: Literal["added", "removed", "modified", "status_changed", "unchanged"]
    """What changed about this source.
    - added: source appeared (new file, new skill)
    - removed: source disappeared (file deleted)
    - modified: content changed (content_hash differs)
    - status_changed: same content but different assembly decision
    - unchanged: identical
    """

    from_status: Optional[SourceStatus] = None
    to_status: Optional[SourceStatus] = None

    from_content_hash: Optional[str] = None
    to_content_hash: Optional[str] = None

    from_token_count: Optional[int] = None
    to_token_count: Optional[int] = None

    summary: Optional[str] = None
    """Human-readable description of the change."""


class InventoryDiff(BaseModel):
    """Diff between two ContextInventory snapshots — source-level drift."""

    from_step_id: str
    to_step_id: str
    from_step_index: int
    to_step_index: int

    changes: list[SourceChange] = Field(default_factory=list)

    sources_added: int = 0
    sources_removed: int = 0
    sources_modified: int = 0
    sources_status_changed: int = 0
    sources_unchanged: int = 0

    # --- The key insight metric ---
    content_changed_but_status_same: int = 0
    """Sources where the content hash changed but assembly decision didn't.
    High values here mean the agent is ignoring upstream changes — a
    leading indicator of stale context.
    """

    status_changed_but_content_same: int = 0
    """Sources where assembly decision changed but content didn't.
    High values here mean the assembly logic is non-deterministic or
    priority-sensitive — the same content gets different treatment.
    """
