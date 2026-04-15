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

    cost = _estimate_cost("us.anthropic.claude-sonnet-4-6", 1000, 500)
    expected = (1000 * 3.0 + 500 * 15.0) / 1_000_000
    assert abs(cost - expected) < 0.000001


def test_openai_provider_implements_interface():
    from atomics.providers.openai import OpenAIProvider

    class FakeChat:
        completions = None

    client = type("FakeClient", (), {"chat": FakeChat()})()
    provider = OpenAIProvider(api_key="fake-key", client=client)
    assert isinstance(provider, BaseProvider)
    assert provider.name == "openai"


def test_openai_cost_estimation():
    from atomics.providers.openai import _estimate_cost

    cost = _estimate_cost("gpt-4o", 1000, 500)
    expected = (1000 * 2.50 + 500 * 10.0) / 1_000_000
    assert abs(cost - expected) < 0.000001
