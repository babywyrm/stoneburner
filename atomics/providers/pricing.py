"""Centralized, cache-aware model pricing.

All rates are USD per 1M tokens as ``(input_rate, output_rate)``. Each provider
keeps its own lookup table (model identifiers differ across APIs) but shares a
single cost function so cache accounting and rounding stay consistent.
"""

from __future__ import annotations

import re

Price = tuple[float, float]
_DATE_SNAPSHOT_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")

# Anthropic prompt-caching multipliers, relative to the base input rate: cache
# writes bill at 1.25x and cache reads at 0.10x. Cached tokens are reported
# separately from input_tokens, so they are additive in the cost calculation.
# Providers without prompt caching simply pass 0 cached tokens.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10

CLAUDE_PRICING: dict[str, Price] = {
    # Current (validated against live API 2026-06-22)
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-fable-5": (10.0, 50.0),
    # Deprecated (kept for historical cost lookups on stored results)
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-opus-4-20250514": (15.0, 75.0),
}
CLAUDE_DEFAULT: Price = (3.0, 15.0)

OPENAI_PRICING: dict[str, Price] = {
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-2024-11-20": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5": (15.0, 60.0),
    "gpt-5-turbo": (5.00, 20.0),
    "gpt-5.3": (10.0, 40.0),
    "gpt-5.5": (15.0, 60.0),
    "gpt-5.6": (5.0, 30.0),
    "o3": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o3-pro": (20.0, 80.0),
    "o4-mini": (1.10, 4.40),
    "codex-mini-latest": (1.50, 6.00),
}
OPENAI_DEFAULT: Price = (2.50, 10.0)

BEDROCK_PRICING: dict[str, Price] = {
    "us.anthropic.claude-sonnet-4-6": (3.0, 15.0),
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": (1.0, 5.0),
    "us.anthropic.claude-opus-4-6-v1": (5.0, 25.0),
    "us.anthropic.claude-sonnet-4-20250514-v1:0": (3.0, 15.0),
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": (3.0, 15.0),
    "anthropic.claude-sonnet-4-20250514-v1:0": (3.0, 15.0),
    "anthropic.claude-3-5-sonnet-20241022-v2:0": (3.0, 15.0),
    "anthropic.claude-3-5-haiku-20241022-v1:0": (0.80, 4.0),
}
BEDROCK_DEFAULT: Price = (3.0, 15.0)


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    table: dict[str, Price],
    default: Price,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Cost in USD for a single response.

    Cached tokens are billed against the input rate using the Anthropic
    multipliers; providers without prompt caching pass 0 for both and the cache
    terms vanish.
    """
    price = table.get(model)
    if price is None:
        base_model = _DATE_SNAPSHOT_SUFFIX.sub("", model)
        price = table.get(base_model, default)
    inp_price, out_price = price
    cost = (
        input_tokens * inp_price
        + cache_write_tokens * inp_price * CACHE_WRITE_MULTIPLIER
        + cache_read_tokens * inp_price * CACHE_READ_MULTIPLIER
        + output_tokens * out_price
    )
    return cost / 1_000_000
