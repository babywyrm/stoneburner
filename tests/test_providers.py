"""Tests for provider interface conformance."""

import pytest

from atomics.providers.base import BaseProvider


def test_claude_provider_implements_interface():
    from atomics.providers.claude import ClaudeProvider

    provider = ClaudeProvider(api_key="fake-key")
    assert isinstance(provider, BaseProvider)
    assert provider.name == "claude"


def test_bedrock_provider_implements_interface():
    """Bedrock requires boto3 — test import gating if not available."""
    try:
        from atomics.providers.bedrock import BedrockProvider

        provider = BedrockProvider.__new__(BedrockProvider)
        assert isinstance(provider, BaseProvider)
        assert provider.name == "bedrock"
    except ImportError:
        pytest.skip("boto3 not installed, skipping bedrock interface test")


def test_claude_cost_estimation():
    from atomics.providers.claude import _estimate_cost

    cost = _estimate_cost("claude-sonnet-4-20250514", 1000, 500)
    expected = (1000 * 3.0 + 500 * 15.0) / 1_000_000
    assert abs(cost - expected) < 0.000001


def test_bedrock_cost_estimation():
    from atomics.providers.bedrock import _estimate_cost

    cost = _estimate_cost("anthropic.claude-3-5-sonnet-20241022-v2:0", 1000, 500)
    expected = (1000 * 3.0 + 500 * 15.0) / 1_000_000
    assert abs(cost - expected) < 0.000001
