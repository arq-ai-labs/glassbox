"""Glassbox — AI context observability. See what your LLM saw.

Quick start:
    from glassbox import wrap
    from openai import OpenAI

    client = wrap(OpenAI())
    # All calls now emit ContextPacks to .glassbox/

    # View them:
    from glassbox import serve
    serve()  # http://localhost:4100

LangGraph:
    from glassbox import observe
    graph = observe(build_my_agent())
    graph.invoke({"input": "..."})
"""

from __future__ import annotations

# Always available — no optional deps needed
from .format import (
    FORMAT_VERSION,
    ContextPack,
    ContextPackDiff,
    ContextPackMetrics,
    ContextSection,
    ContextInventory,
    ContextSource,
    InventoryDiff,
    SourceChange,
    SourceStatus,
    SourceType,
    ModelInfo,
    MultiAgentLink,
    OutputRecord,
    RejectedCandidate,
    RejectionReason,
    RunMetadata,
    RunStatus,
    SectionBase,
    SectionChange,
    SectionTokenAllocation,
    SystemPromptSection,
    UserMessageSection,
    AssistantMessageSection,
    ToolResultSection,
    RetrievalSection,
    MemorySection,
    InstructionSection,
    CustomSection,
    TokenBudget,
    ToolCall,
)
from .core import (
    GlassboxRun,
    FileStorage,
    RunOptions,
    StepHandle,
    StorageAdapter,
    emitter,
    get_current_run,
    with_run,
)
from .diff import diff_context_packs
from .inventory_diff import diff_inventories
from .discovery import discover_sources, link_sources_to_sections
from .assembly import ContextAssembler
from .redaction import RedactionPolicy, set_redaction_policy, get_redaction_policy
from .pricing import estimate_cost


# Lazy imports to avoid pulling in optional dependencies at import time
def wrap(client, **kwargs):
    """Wrap an OpenAI or Anthropic client to auto-capture ContextPacks."""
    from .wrappers import wrap as _wrap
    return _wrap(client, **kwargs)


def observe(graph, **kwargs):
    """Wrap a LangGraph graph to auto-capture ContextPacks."""
    from .integrations.langgraph import observe as _observe
    return _observe(graph, **kwargs)


def serve(port: int = 4100, directory=None):
    """Start the Glassbox viewer UI at http://localhost:{port}."""
    from .viewer.server import serve as _serve
    _serve(port=port, directory=directory)


def proxy(provider: str = "anthropic", target=None, proxy_port: int = 4050, viewer_port: int = 4100, storage_dir=None, working_dir=None):
    """Start the Glassbox proxy + viewer in a single process.

    Point your LLM client at http://localhost:{proxy_port} and open
    http://localhost:{viewer_port} to see every call in real time.
    """
    from .proxy import proxy as _proxy
    _proxy(provider=provider, target=target, proxy_port=proxy_port, viewer_port=viewer_port, storage_dir=storage_dir, working_dir=working_dir)


__version__ = FORMAT_VERSION

__all__ = [
    # Primary API
    "wrap",
    "observe",
    "serve",
    "proxy",
    # Format types
    "FORMAT_VERSION",
    "ContextPack",
    "ContextPackMetrics",
    "ContextPackDiff",
    "ContextSection",
    "SectionBase",
    "SectionChange",
    "SystemPromptSection",
    "UserMessageSection",
    "AssistantMessageSection",
    "ToolResultSection",
    "RetrievalSection",
    "MemorySection",
    "InstructionSection",
    "CustomSection",
    "ModelInfo",
    "MultiAgentLink",
    "OutputRecord",
    "RejectedCandidate",
    "RejectionReason",
    "RunMetadata",
    "RunStatus",
    "TokenBudget",
    "SectionTokenAllocation",
    "ToolCall",
    # Core
    "GlassboxRun",
    "FileStorage",
    "RunOptions",
    "StepHandle",
    "StorageAdapter",
    "emitter",
    "get_current_run",
    "with_run",
    "diff_context_packs",
    "diff_inventories",
    "discover_sources",
    "link_sources_to_sections",
    "ContextInventory",
    "ContextSource",
    "InventoryDiff",
    "SourceChange",
    "SourceStatus",
    "SourceType",
    "ContextAssembler",
    # Redaction
    "RedactionPolicy",
    "set_redaction_policy",
    "get_redaction_policy",
    # Pricing
    "estimate_cost",
]