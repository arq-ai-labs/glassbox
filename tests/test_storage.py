"""Tests for FileStorage — directory structure, read/write, concurrent safety."""

import json
import threading
import pytest

from glassbox.core.file_storage import FileStorage
from glassbox.format.context_pack import ContextPack, ContextPackMetrics
from glassbox.format.model_info import ModelInfo
from glassbox.format.output import OutputRecord
from glassbox.format.run_metadata import RunMetadata
from glassbox.format.sections import SystemPromptSection, UserMessageSection
from glassbox.format.token_budget import SectionTokenAllocation, TokenBudget
from glassbox.format.version import FORMAT_VERSION


def _make_pack(run_id="run_001", step_id="step_001", step_index=0):
    return ContextPack(
        format_version=FORMAT_VERSION,
        run_id=run_id,
        step_id=step_id,
        step_index=step_index,
        started_at="2026-03-29T10:00:00.000Z",
        completed_at="2026-03-29T10:00:01.000Z",
        sections=[
            SystemPromptSection(section_id="s1", token_count=10, content="Be helpful."),
            UserMessageSection(section_id="u1", token_count=5, content="Hello"),
        ],
        token_budget=TokenBudget(
            total_budget=100, total_used=15,
            by_section=[
                SectionTokenAllocation(section_id="s1", section_type="system_prompt", token_count=10),
                SectionTokenAllocation(section_id="u1", section_type="user_message", token_count=5),
            ],
        ),
        model=ModelInfo(provider="test", model="test-model"),
        output=OutputRecord(type="text", text="Hi!"),
        metrics=ContextPackMetrics(latency_ms=100, input_tokens=15, output_tokens=3),
    )


def _make_run_meta(run_id="run_001", **overrides):
    defaults = dict(
        run_id=run_id,
        started_at="2026-03-29T10:00:00.000Z",
        completed_at="2026-03-29T10:00:05.000Z",
        step_count=2,
        total_input_tokens=100,
        total_output_tokens=50,
        total_latency_ms=500,
        status="completed",
        models_used=["test-model"],
    )
    defaults.update(overrides)
    return RunMetadata(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFileStorage:
    def test_write_step_creates_directory(self, tmp_path):
        storage = FileStorage(str(tmp_path / ".glassbox"))
        pack = _make_pack()
        storage.write_step(pack)

        steps_dir = tmp_path / ".glassbox" / "runs" / "run_001" / "steps"
        assert steps_dir.exists()
        files = list(steps_dir.iterdir())
        assert len(files) == 1
        assert files[0].name == "0000_step_001.json"

    def test_written_json_is_valid(self, tmp_path):
        storage = FileStorage(str(tmp_path / ".glassbox"))
        pack = _make_pack()
        storage.write_step(pack)

        step_file = tmp_path / ".glassbox" / "runs" / "run_001" / "steps" / "0000_step_001.json"
        data = json.loads(step_file.read_text())
        assert data["run_id"] == "run_001"
        assert data["step_id"] == "step_001"
        assert len(data["sections"]) == 2

    def test_complete_run_writes_run_json(self, tmp_path):
        storage = FileStorage(str(tmp_path / ".glassbox"))
        meta = _make_run_meta()
        storage.complete_run(meta)

        run_file = tmp_path / ".glassbox" / "runs" / "run_001" / "run.json"
        assert run_file.exists()
        data = json.loads(run_file.read_text())
        assert data["run_id"] == "run_001"
        assert data["status"] == "completed"

    def test_complete_run_creates_index(self, tmp_path):
        storage = FileStorage(str(tmp_path / ".glassbox"))
        meta = _make_run_meta()
        storage.complete_run(meta)

        index_file = tmp_path / ".glassbox" / "index.json"
        assert index_file.exists()
        data = json.loads(index_file.read_text())
        assert len(data["runs"]) == 1
        assert data["runs"][0]["run_id"] == "run_001"

    def test_index_is_idempotent(self, tmp_path):
        storage = FileStorage(str(tmp_path / ".glassbox"))
        meta = _make_run_meta()
        storage.complete_run(meta)
        storage.complete_run(meta)  # write again

        data = json.loads((tmp_path / ".glassbox" / "index.json").read_text())
        assert len(data["runs"]) == 1  # should not duplicate

    def test_list_runs(self, tmp_path):
        storage = FileStorage(str(tmp_path / ".glassbox"))
        storage.complete_run(_make_run_meta("run_001"))
        storage.complete_run(_make_run_meta("run_002"))

        runs = storage.list_runs()
        assert len(runs) == 2
        run_ids = {r.run_id for r in runs}
        assert "run_001" in run_ids
        assert "run_002" in run_ids

    def test_list_runs_empty(self, tmp_path):
        storage = FileStorage(str(tmp_path / ".glassbox"))
        runs = storage.list_runs()
        assert runs == []

    def test_get_run(self, tmp_path):
        storage = FileStorage(str(tmp_path / ".glassbox"))
        storage.complete_run(_make_run_meta("run_001"))

        run = storage.get_run("run_001")
        assert run is not None
        assert run.run_id == "run_001"

    def test_get_run_nonexistent(self, tmp_path):
        storage = FileStorage(str(tmp_path / ".glassbox"))
        assert storage.get_run("nonexistent") is None

    def test_get_steps(self, tmp_path):
        storage = FileStorage(str(tmp_path / ".glassbox"))
        storage.write_step(_make_pack("run_001", "step_001", 0))
        storage.write_step(_make_pack("run_001", "step_002", 1))

        steps = storage.get_steps("run_001")
        assert len(steps) == 2
        assert steps[0].step_index == 0
        assert steps[1].step_index == 1

    def test_get_step(self, tmp_path):
        storage = FileStorage(str(tmp_path / ".glassbox"))
        storage.write_step(_make_pack("run_001", "step_001", 0))

        step = storage.get_step("run_001", "step_001")
        assert step is not None
        assert step.step_id == "step_001"

    def test_get_steps_empty_run(self, tmp_path):
        storage = FileStorage(str(tmp_path / ".glassbox"))
        assert storage.get_steps("nonexistent") == []

    def test_content_hash(self, tmp_path):
        storage = FileStorage(str(tmp_path / ".glassbox"))
        storage.write_step(_make_pack("run_001", "step_001", 0))
        storage.write_step(_make_pack("run_001", "step_002", 1))

        hash_val = storage.compute_content_hash("run_001")
        assert hash_val is not None
        assert len(hash_val) == 64  # SHA-256 hex

        # Same content → same hash
        hash_val2 = storage.compute_content_hash("run_001")
        assert hash_val == hash_val2

    def test_content_hash_auto_populated(self, tmp_path):
        storage = FileStorage(str(tmp_path / ".glassbox"))
        storage.write_step(_make_pack("run_001", "step_001", 0))

        meta = _make_run_meta("run_001")
        assert meta.content_hash is None
        storage.complete_run(meta)

        # Should be auto-populated
        assert meta.content_hash is not None
        assert len(meta.content_hash) == 64

    def test_concurrent_writes(self, tmp_path):
        """Multiple threads writing to the same run shouldn't corrupt the index."""
        storage = FileStorage(str(tmp_path / ".glassbox"))
        errors = []

        def write_run(run_id):
            try:
                storage.complete_run(_make_run_meta(run_id))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=write_run, args=(f"run_{i:03d}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        runs = storage.list_runs(limit=100)
        assert len(runs) == 10
