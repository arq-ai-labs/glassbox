"""Inventory-level drift detection — compares pre-assembly context sources.

While diff.py compares the WIRE-LEVEL sections (what was sent to the LLM),
this module compares the SOURCE-LEVEL inventory (what was available to the
assembly process). Together they answer:

    Wire diff:      "The system prompt grew by 200 tokens"
    Inventory diff: "Because SKILL.md was updated and a new file appeared"

This is the causal layer — it explains WHY the wire-level context changed.
"""

from __future__ import annotations

from .format.context_source import (
    ContextInventory,
    InventoryDiff,
    SourceChange,
)


def diff_inventories(
    from_pack,  # ContextPack
    to_pack,    # ContextPack
) -> InventoryDiff | None:
    """Compare context inventories between two steps.

    Returns None if either step has no context_inventory.
    """
    from_inv = from_pack.context_inventory
    to_inv = to_pack.context_inventory

    if from_inv is None or to_inv is None:
        return None

    from_map = {s.source_id: s for s in from_inv.sources}
    to_map = {s.source_id: s for s in to_inv.sources}

    all_ids = set(from_map.keys()) | set(to_map.keys())
    changes: list[SourceChange] = []

    counts = {
        "added": 0,
        "removed": 0,
        "modified": 0,
        "status_changed": 0,
        "unchanged": 0,
    }
    content_changed_status_same = 0
    status_changed_content_same = 0

    for sid in sorted(all_ids):
        f = from_map.get(sid)
        t = to_map.get(sid)

        if f is None and t is not None:
            # New source appeared
            changes.append(SourceChange(
                source_id=sid,
                name=t.name,
                type=t.type,
                change_type="added",
                to_status=t.status,
                to_content_hash=t.content_hash,
                to_token_count=t.token_count,
                summary=f"New {t.type} '{t.name}' appeared",
            ))
            counts["added"] += 1

        elif f is not None and t is None:
            # Source disappeared
            changes.append(SourceChange(
                source_id=sid,
                name=f.name,
                type=f.type,
                change_type="removed",
                from_status=f.status,
                from_content_hash=f.content_hash,
                from_token_count=f.token_count,
                summary=f"{f.type} '{f.name}' no longer available",
            ))
            counts["removed"] += 1

        else:
            # Both exist — check for changes
            assert f is not None and t is not None

            hash_changed = (
                f.content_hash is not None
                and t.content_hash is not None
                and f.content_hash != t.content_hash
            )
            status_changed = f.status != t.status

            if hash_changed and status_changed:
                change_type = "modified"
                summary = (
                    f"{t.type} '{t.name}' content and status changed "
                    f"({f.status} -> {t.status})"
                )
                counts["modified"] += 1
            elif hash_changed:
                change_type = "modified"
                summary = f"{t.type} '{t.name}' content changed (status still {t.status})"
                counts["modified"] += 1
                content_changed_status_same += 1
            elif status_changed:
                change_type = "status_changed"
                summary = f"{t.type} '{t.name}' status: {f.status} -> {t.status}"
                counts["status_changed"] += 1
                status_changed_content_same += 1
            else:
                change_type = "unchanged"
                summary = None
                counts["unchanged"] += 1

            if change_type != "unchanged":
                changes.append(SourceChange(
                    source_id=sid,
                    name=t.name,
                    type=t.type,
                    change_type=change_type,
                    from_status=f.status,
                    to_status=t.status,
                    from_content_hash=f.content_hash,
                    to_content_hash=t.content_hash,
                    from_token_count=f.token_count,
                    to_token_count=t.token_count,
                    summary=summary,
                ))

    return InventoryDiff(
        from_step_id=from_pack.step_id,
        to_step_id=to_pack.step_id,
        from_step_index=from_pack.step_index,
        to_step_index=to_pack.step_index,
        changes=changes,
        sources_added=counts["added"],
        sources_removed=counts["removed"],
        sources_modified=counts["modified"],
        sources_status_changed=counts["status_changed"],
        sources_unchanged=counts["unchanged"],
        content_changed_but_status_same=content_changed_status_same,
        status_changed_but_content_same=status_changed_content_same,
    )
