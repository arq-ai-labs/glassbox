"""Diff engine — section-level comparison between two ContextPacks."""

from __future__ import annotations

from .format.context_pack import ContextPack
from .format.diff import ContextPackDiff, SectionChange


def diff_context_packs(from_pack: ContextPack, to_pack: ContextPack) -> ContextPackDiff:
    """Compute a section-level diff between two ContextPacks."""
    from_map = {s.section_id: s for s in from_pack.sections}
    to_map = {s.section_id: s for s in to_pack.sections}
    all_ids = set(from_map.keys()) | set(to_map.keys())

    changes: list[SectionChange] = []
    added = removed = modified = 0

    for sid in all_ids:
        from_sec = from_map.get(sid)
        to_sec = to_map.get(sid)

        if from_sec and not to_sec:
            changes.append(SectionChange(
                section_id=sid, change_type="removed",
                section_type=from_sec.type,
                from_tokens=from_sec.token_count, to_tokens=0,
            ))
            removed += 1
        elif not from_sec and to_sec:
            changes.append(SectionChange(
                section_id=sid, change_type="added",
                section_type=to_sec.type,
                from_tokens=0, to_tokens=to_sec.token_count,
            ))
            added += 1
        elif from_sec and to_sec:
            content_changed = from_sec.content != to_sec.content
            changes.append(SectionChange(
                section_id=sid,
                change_type="modified" if content_changed else "unchanged",
                section_type=to_sec.type,
                from_tokens=from_sec.token_count,
                to_tokens=to_sec.token_count,
                summary=(
                    f"Content changed ({from_sec.token_count} -> {to_sec.token_count} tokens)"
                    if content_changed else None
                ),
            ))
            if content_changed:
                modified += 1

    from_total = sum(s.token_count for s in from_pack.sections)
    to_total = sum(s.token_count for s in to_pack.sections)

    return ContextPackDiff(
        from_step_id=from_pack.step_id,
        to_step_id=to_pack.step_id,
        from_step_index=from_pack.step_index,
        to_step_index=to_pack.step_index,
        changes=changes,
        sections_added=added,
        sections_removed=removed,
        sections_modified=modified,
        token_delta=to_total - from_total,
    )
