"""Context Source Discovery — scans the upstream environment to build a ContextInventory.

This module discovers what was AVAILABLE to the context assembly process:
files on disk, skill definitions, agent configs, and tool schemas. It builds
the pre-assembly inventory that ContextPack.context_inventory captures.

Discovery is designed to be:
- Fast: hashes but doesn't tokenize by default (tokenization is optional)
- Non-invasive: read-only filesystem access, no content stored in inventory
- Configurable: .glassboxignore for excluding paths
- Extensible: register custom source discoverers

Usage:
    from glassbox.discovery import discover_sources

    inventory = discover_sources(
        working_dir="/path/to/project",
        include_files=True,
        include_skills=True,
    )
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

from .format.context_source import ContextInventory, ContextSource

logger = logging.getLogger("glassbox.discovery")


# ---------------------------------------------------------------------------
# Known skill / config file patterns
# ---------------------------------------------------------------------------

SKILL_PATTERNS: dict[str, str] = {
    # Claude ecosystem
    "SKILL.md": "skill",
    "CLAUDE.md": "agent_config",
    ".claude/settings.json": "agent_config",
    ".claude/commands/**": "skill",
    # Cursor ecosystem
    ".cursorrules": "agent_config",
    ".cursor/rules/**": "agent_config",
    # Windsurf
    ".windsurfrules": "agent_config",
    # Aider
    ".aider.conf.yml": "agent_config",
    ".aiderignore": "agent_config",
    # Cline
    ".clinerules": "agent_config",
    ".cline/rules/**": "agent_config",
    # Copilot
    ".github/copilot-instructions.md": "agent_config",
    # MCP (Model Context Protocol)
    ".mcp.json": "tool_definition",
    "mcp.json": "tool_definition",
    ".mcp/config.json": "tool_definition",
    "claude_desktop_config.json": "tool_definition",
    # Generic
    "system_prompt.txt": "agent_config",
    "system_prompt.md": "agent_config",
    ".env": "environment",
    ".env.local": "environment",
}

# File extensions to scan (code, config, docs)
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".rb",
    ".cpp", ".c", ".h", ".cs", ".swift", ".kt", ".scala", ".php",
    ".sql", ".sh", ".bash", ".zsh", ".ps1", ".bat",
}
CONFIG_EXTENSIONS = {
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".xml", ".env", ".properties",
}
DOC_EXTENSIONS = {
    ".md", ".txt", ".rst", ".adoc", ".org",
}

# Directories to always skip
SKIP_DIRS = {
    "__pycache__", ".git", ".svn", ".hg", "node_modules", ".venv",
    "venv", "env", ".env", ".tox", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "dist", "build", ".eggs", "*.egg-info",
    ".glassbox", ".next", ".nuxt", "target", "bin", "obj",
}

# Default max file size to hash (skip large binaries)
MAX_FILE_SIZE = 512 * 1024  # 512KB


# ---------------------------------------------------------------------------
# .glassboxignore support
# ---------------------------------------------------------------------------

def _load_ignore_patterns(working_dir: str) -> list[str]:
    """Load patterns from .glassboxignore file."""
    ignore_file = Path(working_dir) / ".glassboxignore"
    if not ignore_file.exists():
        return []
    try:
        lines = ignore_file.read_text(encoding="utf-8").splitlines()
        return [
            line.strip() for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]
    except Exception:
        return []


def _matches_ignore(path: str, patterns: list[str]) -> bool:
    """Check if a path matches any ignore pattern (simple glob-style)."""
    for pattern in patterns:
        # Simple matching: exact name, prefix, or fnmatch-style
        name = os.path.basename(path)
        if pattern == name:
            return True
        if path.endswith(pattern):
            return True
        if pattern.endswith("/") and pattern[:-1] in path:
            return True
        # Simple wildcard: *.ext
        if pattern.startswith("*") and name.endswith(pattern[1:]):
            return True
    return False


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------

def _hash_file(path: Path) -> Optional[str]:
    """Compute sha256 prefix of file content. Returns None for large/binary files."""
    try:
        size = path.stat().st_size
        if size > MAX_FILE_SIZE:
            return None
        content = path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]
    except Exception:
        return None


def _hash_content(content: str) -> str:
    """Hash a string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Token estimation (lightweight, no tiktoken dependency)
# ---------------------------------------------------------------------------

def _estimate_file_tokens(path: Path) -> Optional[int]:
    """Quick token estimate based on file size (avg 4 chars per token)."""
    try:
        size = path.stat().st_size
        if size > MAX_FILE_SIZE:
            return None
        return max(1, size // 4)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Source discovery functions
# ---------------------------------------------------------------------------

def _discover_skills(working_dir: Path, ignore_patterns: list[str]) -> list[ContextSource]:
    """Find skill definitions and agent config files."""
    sources: list[ContextSource] = []

    for pattern, source_type in SKILL_PATTERNS.items():
        if "**" in pattern:
            # Glob pattern
            base, glob_part = pattern.split("**", 1)
            search_dir = working_dir / base.rstrip("/")
            if search_dir.exists():
                for f in search_dir.rglob("*"):
                    if f.is_file() and not _matches_ignore(str(f), ignore_patterns):
                        rel = str(f.relative_to(working_dir))
                        sources.append(ContextSource(
                            source_id=_hash_content(rel),
                            type=source_type,
                            name=f.name,
                            path=rel,
                            status="available",
                            token_count=_estimate_file_tokens(f),
                            content_hash=_hash_file(f),
                            last_modified=_file_mtime(f),
                        ))
        else:
            # Exact path
            target = working_dir / pattern
            if target.exists() and target.is_file():
                rel = pattern
                if not _matches_ignore(rel, ignore_patterns):
                    sources.append(ContextSource(
                        source_id=_hash_content(rel),
                        type=source_type,
                        name=target.name,
                        path=rel,
                        status="available",
                        token_count=_estimate_file_tokens(target),
                        content_hash=_hash_file(target),
                        last_modified=_file_mtime(target),
                    ))

    return sources


def _discover_files(
    working_dir: Path,
    ignore_patterns: list[str],
    extensions: set[str] | None = None,
    max_depth: int = 4,
    max_files: int = 500,
) -> list[ContextSource]:
    """Scan the directory tree for relevant source files."""
    if extensions is None:
        extensions = CODE_EXTENSIONS | CONFIG_EXTENSIONS | DOC_EXTENSIONS

    sources: list[ContextSource] = []
    count = 0

    for root, dirs, files in os.walk(working_dir):
        # Prune skipped directories
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS
            and not d.startswith(".")
            and not _matches_ignore(d, ignore_patterns)
        ]

        # Check depth
        depth = str(root).replace(str(working_dir), "").count(os.sep)
        if depth >= max_depth:
            dirs.clear()
            continue

        for fname in files:
            if count >= max_files:
                break

            fpath = Path(root) / fname
            ext = fpath.suffix.lower()

            if ext not in extensions:
                continue

            rel = str(fpath.relative_to(working_dir)).replace("\\", "/")
            if _matches_ignore(rel, ignore_patterns):
                continue

            sources.append(ContextSource(
                source_id=_hash_content(rel),
                type="file",
                name=fname,
                path=rel,
                status="available",
                token_count=_estimate_file_tokens(fpath),
                content_hash=_hash_file(fpath),
                last_modified=_file_mtime(fpath),
                tags={"extension": ext} if ext else None,
            ))
            count += 1

        if count >= max_files:
            break

    return sources


def _discover_tool_definitions(working_dir: Path) -> list[ContextSource]:
    """Look for tool/function definition files (OpenAPI specs, etc.)."""
    sources: list[ContextSource] = []

    # Common locations for API/tool definitions
    patterns = [
        "openapi.yaml", "openapi.yml", "openapi.json",
        "swagger.yaml", "swagger.yml", "swagger.json",
        "tools.json", "functions.json",
        "schema.graphql", "schema.gql",
    ]

    for pattern in patterns:
        target = working_dir / pattern
        if target.exists() and target.is_file():
            sources.append(ContextSource(
                source_id=_hash_content(pattern),
                type="api_schema" if "api" in pattern or "swagger" in pattern or "graphql" in pattern.lower() else "tool_definition",
                name=target.name,
                path=pattern,
                status="available",
                token_count=_estimate_file_tokens(target),
                content_hash=_hash_file(target),
                last_modified=_file_mtime(target),
            ))

    return sources


def _file_mtime(path: Path) -> Optional[str]:
    """Get file modification time as ISO 8601."""
    try:
        mtime = path.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main discovery function
# ---------------------------------------------------------------------------

def discover_sources(
    working_dir: str | Path | None = None,
    include_files: bool = True,
    include_skills: bool = True,
    include_tools: bool = True,
    file_extensions: set[str] | None = None,
    max_depth: int = 4,
    max_files: int = 500,
    extra_sources: list[ContextSource] | None = None,
) -> ContextInventory:
    """Discover all context sources in the working environment.

    Args:
        working_dir: Project root to scan. Defaults to cwd.
        include_files: Scan for source code files.
        include_skills: Look for SKILL.md, CLAUDE.md, .cursorrules, etc.
        include_tools: Look for tool/API definition files.
        file_extensions: Custom set of extensions to include.
        max_depth: Maximum directory depth to scan.
        max_files: Maximum number of files to inventory.
        extra_sources: Additional sources to include (e.g. from memory stores).

    Returns:
        A ContextInventory with all discovered sources.
    """
    if working_dir is None:
        working_dir = Path.cwd()
    else:
        working_dir = Path(working_dir)

    ignore_patterns = _load_ignore_patterns(str(working_dir))
    all_sources: list[ContextSource] = []
    method_parts: list[str] = []

    if include_skills:
        all_sources.extend(_discover_skills(working_dir, ignore_patterns))
        method_parts.append("skills")

    if include_files:
        all_sources.extend(_discover_files(
            working_dir, ignore_patterns,
            extensions=file_extensions,
            max_depth=max_depth,
            max_files=max_files,
        ))
        method_parts.append("filesystem_scan")

    if include_tools:
        all_sources.extend(_discover_tool_definitions(working_dir))
        method_parts.append("tool_definitions")

    if extra_sources:
        all_sources.extend(extra_sources)
        method_parts.append("explicit")

    # Deduplicate by source_id
    seen: set[str] = set()
    deduped: list[ContextSource] = []
    for s in all_sources:
        if s.source_id not in seen:
            seen.add(s.source_id)
            deduped.append(s)

    inventory = ContextInventory(
        sources=deduped,
        working_directory=str(working_dir),
        discovery_method=" + ".join(method_parts) if method_parts else "none",
        discovered_at=datetime.now(timezone.utc).isoformat(),
    )
    inventory.compute_stats()

    return inventory


# ---------------------------------------------------------------------------
# Helper: mark sources as included based on section mapping
# ---------------------------------------------------------------------------

def link_sources_to_sections(
    inventory: ContextInventory,
    section_source_map: dict[str, str],
) -> ContextInventory:
    """Update source statuses based on which sections they mapped to.

    Args:
        inventory: The discovered inventory.
        section_source_map: Maps source path/name -> section_id in the ContextPack.

    Returns:
        The updated inventory (mutated in place).
    """
    source_by_path = {s.path: s for s in inventory.sources if s.path}
    source_by_name = {s.name: s for s in inventory.sources}

    for source_key, section_id in section_source_map.items():
        source = source_by_path.get(source_key) or source_by_name.get(source_key)
        if source:
            source.status = "included"
            source.section_id = section_id

    inventory.compute_stats()
    return inventory
