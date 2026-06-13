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


def test_claude_cost_estimation_cache_aware():
    from atomics.providers.claude import _estimate_cost

    # input rate 3.0/M, output 15.0/M; cache write 1.25x, cache read 0.10x.
    cost = _estimate_cost(
        "claude-sonnet-4-20250514",
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=2000,
        cache_write_tokens=400,
    )
    expected = (
        1000 * 3.0
        + 400 * 3.0 * 1.25
        + 2000 * 3.0 * 0.10
        + 500 * 15.0
    ) / 1_000_000
    assert abs(cost - expected) < 0.000001


def test_claude_cost_estimation_cache_defaults_to_zero():
    """Cache args are optional — omitting them matches the no-cache cost."""
    from atomics.providers.claude import _estimate_cost

    assert _estimate_cost("claude-sonnet-4-20250514", 1000, 500) == _estimate_cost(
        "claude-sonnet-4-20250514", 1000, 500, 0, 0
    )


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


# ── providers/__init__ lazy __getattr__ ──────────────────────────────────────

def test_providers_init_ollama_lazy():
    import atomics.providers as p
    OllamaProvider = p.OllamaProvider
    assert OllamaProvider.__name__ == "OllamaProvider"


def test_providers_init_openai_lazy():
    import atomics.providers as p
    OpenAIProvider = p.OpenAIProvider
    assert OpenAIProvider.__name__ == "OpenAIProvider"


def test_providers_init_brain_gateway_lazy():
    import atomics.providers as p
    BrainGatewayProvider = p.BrainGatewayProvider
    assert BrainGatewayProvider.__name__ == "BrainGatewayProvider"


def test_providers_init_bedrock_lazy():
    import atomics.providers as p
    BedrockProvider = p.BedrockProvider
    assert BedrockProvider.__name__ == "BedrockProvider"


def test_providers_init_unknown_attr():
    import atomics.providers as p
    import pytest
    with pytest.raises(AttributeError, match="no attribute"):
        _ = p.NonExistentProvider
