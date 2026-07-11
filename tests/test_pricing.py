"""Tests for the centralized, cache-aware pricing module."""

from __future__ import annotations

from atomics.providers import pricing


def test_estimate_cost_basic_no_cache():
    cost = pricing.estimate_cost(
        "claude-sonnet-4-6",
        1000,
        500,
        table=pricing.CLAUDE_PRICING,
        default=pricing.CLAUDE_DEFAULT,
    )
    assert abs(cost - (1000 * 3.0 + 500 * 15.0) / 1_000_000) < 1e-9


def test_estimate_cost_cache_aware():
    cost = pricing.estimate_cost(
        "claude-sonnet-4-6",
        1000,
        500,
        table=pricing.CLAUDE_PRICING,
        default=pricing.CLAUDE_DEFAULT,
        cache_read_tokens=2000,
        cache_write_tokens=400,
    )
    expected = (
        1000 * 3.0
        + 400 * 3.0 * pricing.CACHE_WRITE_MULTIPLIER
        + 2000 * 3.0 * pricing.CACHE_READ_MULTIPLIER
        + 500 * 15.0
    ) / 1_000_000
    assert abs(cost - expected) < 1e-9


def test_estimate_cost_unknown_model_uses_default():
    cost = pricing.estimate_cost(
        "totally-unknown",
        1000,
        1000,
        table=pricing.OPENAI_PRICING,
        default=pricing.OPENAI_DEFAULT,
    )
    inp, out = pricing.OPENAI_DEFAULT
    assert abs(cost - (1000 * inp + 1000 * out) / 1_000_000) < 1e-9


def test_estimate_cost_resolves_openai_date_snapshot_to_base_model():
    cost = pricing.estimate_cost(
        "gpt-5.6-2026-06-01",
        1_000_000,
        1_000_000,
        table=pricing.OPENAI_PRICING,
        default=pricing.OPENAI_DEFAULT,
    )

    assert cost == 35.0


def test_estimate_cost_exact_snapshot_entry_wins():
    cost = pricing.estimate_cost(
        "gpt-4o-2024-11-20",
        1_000_000,
        1_000_000,
        table=pricing.OPENAI_PRICING,
        default=(99.0, 99.0),
    )

    assert cost == 12.5


def test_estimate_cost_does_not_prefix_match_unknown_snapshot():
    cost = pricing.estimate_cost(
        "gpt-5.6-experimental-2026-06-01",
        1_000_000,
        1_000_000,
        table=pricing.OPENAI_PRICING,
        default=pricing.OPENAI_DEFAULT,
    )

    assert cost == 12.5


def test_cache_multipliers_match_anthropic_docs():
    assert pricing.CACHE_WRITE_MULTIPLIER == 1.25
    assert pricing.CACHE_READ_MULTIPLIER == 0.10


def test_latest_frontier_model_pricing():
    assert pricing.OPENAI_PRICING["gpt-5.6"] == (5.0, 30.0)
    assert pricing.CLAUDE_PRICING["claude-fable-5"] == (10.0, 50.0)


def test_provider_wrappers_delegate_to_central():
    """Each provider's _estimate_cost must match the central calculation."""
    from atomics.providers.bedrock import _estimate_cost as bedrock_cost
    from atomics.providers.claude import _estimate_cost as claude_cost
    from atomics.providers.openai import _estimate_cost as openai_cost

    assert claude_cost("claude-sonnet-4-6", 1000, 500, 2000, 400) == pricing.estimate_cost(
        "claude-sonnet-4-6",
        1000,
        500,
        table=pricing.CLAUDE_PRICING,
        default=pricing.CLAUDE_DEFAULT,
        cache_read_tokens=2000,
        cache_write_tokens=400,
    )
    assert openai_cost("gpt-4o", 1000, 500) == pricing.estimate_cost(
        "gpt-4o", 1000, 500, table=pricing.OPENAI_PRICING, default=pricing.OPENAI_DEFAULT
    )
    assert bedrock_cost("us.anthropic.claude-sonnet-4-6", 1000, 500) == pricing.estimate_cost(
        "us.anthropic.claude-sonnet-4-6",
        1000,
        500,
        table=pricing.BEDROCK_PRICING,
        default=pricing.BEDROCK_DEFAULT,
    )
