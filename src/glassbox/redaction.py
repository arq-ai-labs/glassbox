"""Content redaction — hooks for masking, hashing, or dropping sensitive content.

Enterprise deployments need control over what gets stored. This module provides
configurable redaction that runs before any ContextPack is written to storage.

Usage:
    from glassbox import set_redaction_policy, RedactionPolicy

    # Hash all content (keeps structure, removes raw text)
    set_redaction_policy(RedactionPolicy.HASH)

    # Or use a custom function
    def my_redactor(section):
        if "PII" in section.metadata:
            return section.model_copy(update={"content": "[REDACTED]"})
        return section

    set_redaction_policy(my_redactor)
"""

from __future__ import annotations

import hashlib
import logging
from enum import Enum
from typing import Any, Callable, Optional, Union

logger = logging.getLogger("glassbox")

# Type for a redaction function: takes a section dict, returns modified dict or None (to drop)
RedactionFn = Callable[[dict[str, Any]], Optional[dict[str, Any]]]


class RedactionPolicy(Enum):
    """Built-in redaction strategies."""

    NONE = "none"
    """Store full content (default). No redaction."""

    HASH = "hash"
    """Replace content with SHA-256 hash. Preserves token counts and structure."""

    TRUNCATE = "truncate"
    """Truncate content to first 100 chars + '...' suffix."""

    DROP_CONTENT = "drop_content"
    """Remove content entirely, keep all metadata (type, token_count, source, etc.)."""


# Global redaction policy
_redaction_policy: Union[RedactionPolicy, RedactionFn] = RedactionPolicy.NONE


def set_redaction_policy(policy: Union[RedactionPolicy, RedactionFn]) -> None:
    """Set the global redaction policy applied to all sections before storage.

    Args:
        policy: A RedactionPolicy enum value, or a callable that takes a section
                dict and returns a modified dict (or None to drop the section entirely).
    """
    global _redaction_policy
    _redaction_policy = policy
    logger.info("Glassbox redaction policy set to: %s",
                policy.value if isinstance(policy, RedactionPolicy) else "custom")


def get_redaction_policy() -> Union[RedactionPolicy, RedactionFn]:
    """Get the current redaction policy."""
    return _redaction_policy


def _hash_content(text: str) -> str:
    """SHA-256 hash of content for privacy-preserving storage."""
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def apply_redaction(section_dict: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Apply the current redaction policy to a section dict.

    Returns the (possibly modified) section dict, or None if the section should be dropped.
    """
    policy = _redaction_policy

    if isinstance(policy, RedactionPolicy):
        if policy == RedactionPolicy.NONE:
            return section_dict

        if policy == RedactionPolicy.HASH:
            content = section_dict.get("content", "")
            section_dict = dict(section_dict)
            section_dict["content"] = _hash_content(content) if content else ""
            return section_dict

        if policy == RedactionPolicy.TRUNCATE:
            content = section_dict.get("content", "")
            section_dict = dict(section_dict)
            if len(content) > 100:
                section_dict["content"] = content[:100] + "..."
            return section_dict

        if policy == RedactionPolicy.DROP_CONTENT:
            section_dict = dict(section_dict)
            section_dict["content"] = ""
            return section_dict

    elif callable(policy):
        try:
            return policy(section_dict)
        except Exception:
            logger.warning("Custom redaction function raised an error, storing unredacted",
                           exc_info=True)
            return section_dict

    return section_dict


def redact_pack_dict(pack_dict: dict[str, Any]) -> dict[str, Any]:
    """Apply redaction to all sections in a ContextPack dict (pre-serialization).

    This is called by FileStorage before writing to disk.
    """
    if _redaction_policy == RedactionPolicy.NONE:
        return pack_dict  # fast path

    sections = pack_dict.get("sections", [])
    redacted_sections = []
    for section in sections:
        result = apply_redaction(section)
        if result is not None:
            redacted_sections.append(result)

    pack_dict = dict(pack_dict)
    pack_dict["sections"] = redacted_sections
    return pack_dict
