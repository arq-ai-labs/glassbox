"""Model pricing — estimates cost per LLM call.

Prices in USD per 1M tokens. Updated March 2026.
Sources: https://openai.com/pricing, https://docs.anthropic.com/en/docs/about-claude/models
"""

from __future__ import annotations

from typing import Optional


# (input_per_1M, output_per_1M) in USD
MODEL_PRICES: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-sonnet-20240620": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-11-20": (2.50, 10.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4-turbo-2024-04-09": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o1-preview": (15.00, 60.00),
    "o3-mini": (1.10, 4.40),
    # Google
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
}

# Fuzzy matching: strip date suffixes for fallback lookups
_FUZZY_CACHE: dict[str, Optional[tuple[float, float]]] = {}


def _fuzzy_lookup(model: str) -> Optional[tuple[float, float]]:
    """Try exact match, then progressively strip suffixes."""
    if model in _FUZZY_CACHE:
        return _FUZZY_CACHE[model]

    # Exact match
    if model in MODEL_PRICES:
        _FUZZY_CACHE[model] = MODEL_PRICES[model]
        return MODEL_PRICES[model]

    # Strip date suffix: "claude-3-5-sonnet-20241022" -> "claude-3-5-sonnet"
    parts = model.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 8:
        base = parts[0]
        if base in MODEL_PRICES:
            _FUZZY_CACHE[model] = MODEL_PRICES[base]
            return MODEL_PRICES[base]

    # Prefix match: find longest matching key
    best = None
    best_len = 0
    for key, price in MODEL_PRICES.items():
        if model.startswith(key) and len(key) > best_len:
            best = price
            best_len = len(key)
    if not best:
        for key, price in MODEL_PRICES.items():
            if key.startswith(model) and len(key) > best_len:
                best = price
                best_len = len(key)

    _FUZZY_CACHE[model] = best
    return best


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> Optional[float]:
    """Estimate cost in USD for a single LLM call.

    Returns None if the model is not in the pricing table.
    """
    prices = _fuzzy_lookup(model)
    if not prices:
        return None
    input_cost = (input_tokens / 1_000_000) * prices[0]
    output_cost = (output_tokens / 1_000_000) * prices[1]
    return round(input_cost + output_cost, 6)
