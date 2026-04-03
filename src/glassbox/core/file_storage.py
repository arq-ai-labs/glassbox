"""File-based storage — writes ContextPacks to .glassbox/ directory.

File layout (identical to TypeScript version for cross-language compat):
  .glassbox/
    index.json
    runs/<run_id>/run.json
    runs/<run_id>/steps/0000_<step_id>.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional

from ..format.context_pack import ContextPack
from ..format.run_metadata import RunMetadata
from ..redaction import redact_pack_dict
from .emitter import emitter

logger = logging.getLogger("glassbox")


def _dump(model: Any, apply_redaction: bool = False) -> str:
    """Serialize a Pydantic model to JSON string, optionally applying redaction."""
    if hasattr(model, "model_dump"):
        data = model.model_dump(mode="json")
        if apply_redaction:
            data = redact_pack_dict(data)
        return json.dumps(data, indent=2)
    return json.dumps(model, indent=2, default=str)


class FileStorage:
    """Writes ContextPacks and run metadata to .glassbox/ on disk."""

    def __init__(self, directory: Optional[str] = None) -> None:
        self.root = Path(directory) if directory else Path.cwd() / ".glassbox"
        self._runs_dir = self.root / "runs"
        self._index_path = self.root / "index.json"
        self._index_lock = threading.Lock()

    def write_step(self, pack: ContextPack) -> None:
        try:
            steps_dir = self._runs_dir / pack.run_id / "steps"
            steps_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{pack.step_index:04d}_{pack.step_id}.json"
            (steps_dir / filename).write_text(
                _dump(pack, apply_redaction=True), encoding="utf-8"
            )
        except Exception as e:
            logger.debug("write_step error: %s", e, exc_info=True)
            emitter.emit("storage:error", {"error": e, "context": f"write_step({pack.run_id}/{pack.step_id})"})

    def write_step_background(self, pack: ContextPack) -> None:
        """Fire-and-forget write in a daemon thread."""
        t = threading.Thread(target=self.write_step, args=(pack,), daemon=True)
        t.start()

    def compute_content_hash(self, run_id: str) -> Optional[str]:
        """Compute SHA-256 hash of all step files for tamper-evident provenance."""
        steps_dir = self._runs_dir / run_id / "steps"
        if not steps_dir.exists():
            return None
        try:
            hasher = hashlib.sha256()
            for f in sorted(steps_dir.iterdir()):
                if f.suffix == ".json":
                    hasher.update(f.read_bytes())
            return hasher.hexdigest()
        except Exception:
            return None

    def complete_run(self, run_meta: RunMetadata) -> None:
        try:
            run_dir = self._runs_dir / run_meta.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            # Compute content hash if not already set
            if not run_meta.content_hash:
                run_meta.content_hash = self.compute_content_hash(run_meta.run_id)
            (run_dir / "run.json").write_text(_dump(run_meta), encoding="utf-8")
            self._append_to_index(run_meta)
        except Exception as e:
            emitter.emit("storage:error", {"error": e, "context": f"complete_run({run_meta.run_id})"})

    def list_runs(self, limit: int = 50, offset: int = 0) -> list[RunMetadata]:
        index = self._read_index()
        runs = sorted(index, key=lambda r: r.get("started_at", ""), reverse=True)
        return [RunMetadata.model_validate(r) for r in runs[offset:offset + limit]]

    def get_run(self, run_id: str) -> Optional[RunMetadata]:
        run_path = self._runs_dir / run_id / "run.json"
        if not run_path.exists():
            return None
        try:
            raw = json.loads(run_path.read_text(encoding="utf-8"))
            return RunMetadata.model_validate(raw)
        except Exception:
            return None

    def get_steps(self, run_id: str) -> list[ContextPack]:
        steps_dir = self._runs_dir / run_id / "steps"
        if not steps_dir.exists():
            return []
        try:
            files = sorted(f for f in steps_dir.iterdir() if f.suffix == ".json")
            packs = []
            for f in files:
                raw = json.loads(f.read_text(encoding="utf-8"))
                packs.append(ContextPack.model_validate(raw))
            return packs
        except Exception:
            return []

    def get_step(self, run_id: str, step_id: str) -> Optional[ContextPack]:
        steps_dir = self._runs_dir / run_id / "steps"
        if not steps_dir.exists():
            return None
        try:
            for f in steps_dir.iterdir():
                if step_id in f.name:
                    raw = json.loads(f.read_text(encoding="utf-8"))
                    return ContextPack.model_validate(raw)
        except Exception:
            pass
        return None

    def _read_index(self) -> list[dict]:
        if not self._index_path.exists():
            return []
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            return data.get("runs", [])
        except Exception:
            return []

    def _append_to_index(self, run_meta: RunMetadata) -> None:
        """Append or update a run entry in index.json (thread-safe)."""
        with self._index_lock:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                existing = self._read_index()
                entry = run_meta.model_dump(mode="json")
                # Replace if run_id already exists, else append
                existing = [r for r in existing if r.get("run_id") != run_meta.run_id]
                existing.append(entry)
                self._index_path.write_text(
                    json.dumps({"runs": existing}, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                emitter.emit("storage:error", {"error": e, "context": "append_to_index"})