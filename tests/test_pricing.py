"""Tests for model pricing — cost estimation accuracy."""

import pytest

from glassbox.pricing import estimate_cost, MODEL_PRICES


class TestEstimateCost:
    def test_known_model(self):
        cost = estimate_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        assert cost is not None
        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        expected = (1000 / 1e6) * 0.15 + (500 / 1e6) * 0.60
        assert abs(cost - expected) < 1e-8

    def test_claude_sonnet(self):
        cost = estimate_cost("claude-sonnet-4-20250514", input_tokens=10000, output_tokens=2000)
        assert cost is not None
        # $3/1M in, $15/1M out
        expected = (10000 / 1e6) * 3.0 + (2000 / 1e6) * 15.0
        assert abs(cost - expected) < 1e-8

    def test_unknown_model(self):
        cost = estimate_cost("totally-unknown-model-v99", input_tokens=100, output_tokens=50)
        assert cost is None

    def test_zero_tokens(self):
        cost = estimate_cost("gpt-4o", input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_large_usage(self):
        cost = estimate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=500_000)
        assert cost is not None
        # $2.50/1M in, $10/1M out = $2.50 + $5.00 = $7.50
        assert abs(cost - 7.50) < 0.01

    def test_fuzzy_match_with_date_suffix(self):
        """Models with date suffixes should match base model."""
        cost = estimate_cost("gpt-4o-2024-08-06", input_tokens=1000, output_tokens=500)
        assert cost is not None

    def test_pricing_table_not_empty(self):
        assert len(MODEL_PRICES) > 10

    def test_all_prices_positive(self):
        for model, (inp, out) in MODEL_PRICES.items():
            assert inp >= 0, f"{model} has negative input price"
            assert out >= 0, f"{model} has negative output price"
