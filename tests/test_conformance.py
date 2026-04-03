"""Conformance tests — validate ContextPack fixtures against the schema.

These tests define the ContextPack format contract. Any implementation (Python,
TypeScript, Go, Rust, etc.) that passes these fixtures is conformant.

Run: pytest tests/test_conformance.py -v
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from glassbox.format.context_pack import ContextPack

FIXTURES_DIR = Path(__file__).parent.parent / "spec" / "fixtures"


def _load_fixture(name: str) -> dict:
    path = FIXTURES_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Valid fixtures — MUST parse without error
# ---------------------------------------------------------------------------


class TestValidFixtures:
    """All valid_*.json fixtures must parse into a ContextPack."""

    def test_minimal(self):
        data = _load_fixture("valid_minimal.json")
        pack = ContextPack.model_validate(data)
        assert pack.run_id == "run_001"
        assert pack.step_index == 0
        assert len(pack.sections) == 1
        assert pack.sections[0].type == "user_message"

    def test_full(self):
        data = _load_fixture("valid_full.json")
        pack = ContextPack.model_validate(data)
        assert pack.agent_name == "support-agent"
        assert len(pack.sections) == 6

        # All 6 section types present
        types = {s.type for s in pack.sections}
        assert types == {"system_prompt", "user_message", "retrieval", "tool_result", "memory", "instruction"}

        # Rejection ledger
        assert len(pack.token_budget.rejected) == 3
        reasons = {r.reason for r in pack.token_budget.rejected}
        assert "token_limit_exceeded" in reasons
        assert "relevance_below_threshold" in reasons
        assert "staleness" in reasons

        # Multi-agent
        assert pack.multi_agent is not None
        assert pack.multi_agent.parent_run_id == "run_orchestrator_001"
        assert pack.multi_agent.delegation_scope == "resolve_shipping_complaint"

        # Extensions
        assert pack.extensions is not None
        assert pack.extensions["ticket_id"] == "TICKET-456"

        # Cost
        assert pack.metrics.cost_estimate_usd is not None
        assert pack.metrics.cost_estimate_usd > 0

    def test_tool_calls_only(self):
        data = _load_fixture("valid_tool_calls_only.json")
        pack = ContextPack.model_validate(data)
        assert pack.output.type == "tool_calls"
        assert pack.output.tool_calls is not None
        assert len(pack.output.tool_calls) == 1
        assert pack.output.tool_calls[0].tool_name == "get_weather"

    def test_error_output(self):
        data = _load_fixture("valid_error_output.json")
        pack = ContextPack.model_validate(data)
        assert pack.output.type == "error"
        assert pack.output.error is not None
        assert "RateLimitError" in pack.output.error

    def test_custom_section(self):
        data = _load_fixture("valid_custom_section.json")
        pack = ContextPack.model_validate(data)
        custom = [s for s in pack.sections if s.type == "custom"]
        assert len(custom) == 1
        assert custom[0].custom_type == "guardrail_result"
        assert custom[0].metadata is not None

    def test_all_valid_round_trip(self):
        """Every valid fixture must survive serialize -> deserialize."""
        for f in sorted(FIXTURES_DIR.glob("valid_*.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            pack = ContextPack.model_validate(data)
            # Round-trip
            json_str = json.dumps(pack.model_dump(mode="json"))
            pack2 = ContextPack.model_validate_json(json_str)
            assert pack2.run_id == pack.run_id
            assert pack2.step_id == pack.step_id
            assert len(pack2.sections) == len(pack.sections)


# ---------------------------------------------------------------------------
# Invalid fixtures — MUST raise ValidationError
# ---------------------------------------------------------------------------


class TestInvalidFixtures:
    """All invalid_*.json fixtures must raise ValidationError."""

    def test_missing_required(self):
        data = _load_fixture("invalid_missing_required.json")
        with pytest.raises(ValidationError):
            ContextPack.model_validate(data)

    def test_negative_tokens(self):
        data = _load_fixture("invalid_negative_tokens.json")
        with pytest.raises(ValidationError):
            ContextPack.model_validate(data)

    def test_bad_section_type(self):
        data = _load_fixture("invalid_bad_section_type.json")
        with pytest.raises(ValidationError):
            ContextPack.model_validate(data)

    def test_bad_rejection_reason(self):
        data = _load_fixture("invalid_bad_rejection_reason.json")
        with pytest.raises(ValidationError):
            ContextPack.model_validate(data)

    def test_all_invalid_reject(self):
        """Every invalid fixture must fail validation."""
        for f in sorted(FIXTURES_DIR.glob("invalid_*.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            with pytest.raises(ValidationError, match=".*"):
                ContextPack.model_validate(data)
