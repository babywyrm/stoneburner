"""Shared --effort / --reasoning-mode mapping and native request payloads."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from atomics.providers.effort import (
    EffortError,
    claude_request,
    normalize_effort,
    normalize_reasoning_mode,
    openai_chat_effort,
    openai_reasoning,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("high", "high"),
        ("HIGH", "high"),
        ("xl", "xhigh"),
        ("xhigh", "xhigh"),
        ("ultra", "max"),
        ("minimal", "minimal"),
        ("none", "none"),
    ],
)
def test_normalize_effort_aliases(raw: str | None, expected: str | None) -> None:
    assert normalize_effort(raw) == expected


@pytest.mark.unit
def test_normalize_effort_rejects_unknown() -> None:
    with pytest.raises(EffortError, match="unknown effort"):
        normalize_effort("ludicrous")


@pytest.mark.unit
def test_normalize_reasoning_mode() -> None:
    assert normalize_reasoning_mode(None) is None
    assert normalize_reasoning_mode("PRO") == "pro"
    assert normalize_reasoning_mode("standard") == "standard"
    with pytest.raises(EffortError, match="unknown reasoning mode"):
        normalize_reasoning_mode("turbo")


@pytest.mark.unit
def test_openai_reasoning_payload() -> None:
    assert openai_reasoning(None, None) is None
    assert openai_reasoning("high", None) == {"effort": "high"}
    assert openai_reasoning("xl", "pro") == {"effort": "xhigh", "mode": "pro"}
    assert openai_reasoning(None, "pro") == {"mode": "pro"}
    assert openai_reasoning("none", None, thinking=False) == {"effort": "none"}


@pytest.mark.unit
def test_openai_chat_effort_is_scalar() -> None:
    assert openai_chat_effort(None) is None
    assert openai_chat_effort("ultra") == "max"
    assert openai_chat_effort("none") == "none"


@pytest.mark.unit
def test_claude_adaptive_plus_effort_on_opus_4_6() -> None:
    thinking, extra = claude_request(
        model="claude-opus-4-6",
        thinking=None,
        thinking_budget=None,
        effort="high",
    )
    assert thinking == {"type": "adaptive"}
    assert extra["output_config"] == {"effort": "high"}


@pytest.mark.unit
def test_claude_maps_xhigh_to_max_on_4_6() -> None:
    _thinking, extra = claude_request(
        model="claude-sonnet-4-6",
        thinking=True,
        thinking_budget=8000,
        effort="xl",
    )
    assert extra["output_config"] == {"effort": "max"}


@pytest.mark.unit
def test_claude_keeps_budget_tokens_when_effort_omitted() -> None:
    thinking, extra = claude_request(
        model="claude-sonnet-4-6",
        thinking=True,
        thinking_budget=20000,
        effort=None,
    )
    assert thinking == {"type": "enabled", "budget_tokens": 20000}
    assert "output_config" not in extra


@pytest.mark.unit
def test_claude_no_thinking_still_sends_effort() -> None:
    thinking, extra = claude_request(
        model="claude-opus-4-6",
        thinking=False,
        thinking_budget=None,
        effort="low",
    )
    assert thinking is None
    assert extra["output_config"] == {"effort": "low"}


@pytest.mark.asyncio
@pytest.mark.unit
async def test_openai_completions_sends_reasoning_effort() -> None:
    from atomics.providers.openai import OpenAIProvider
    from tests.test_providers_openai import FakeOpenAIClient

    client = FakeOpenAIClient()
    provider = OpenAIProvider(api_key="fake", client=client)
    resp = await provider.generate("ping", model="gpt-5.6-sol", effort="high")
    call = client.chat.completions.create_calls[0]
    assert call["reasoning_effort"] == "high"
    assert resp.effort == "high"
    assert resp.reasoning_request == {"effort": "high"}


@pytest.mark.asyncio
@pytest.mark.unit
async def test_openai_pro_mode_uses_responses_api() -> None:
    from atomics.providers.openai import OpenAIProvider
    from tests.test_providers_openai import FakeOAuthClient

    client = FakeOAuthClient()
    provider = OpenAIProvider(api_key="fake", client=client)
    resp = await provider.generate(
        "ping",
        model="gpt-5.6-sol",
        effort="max",
        reasoning_mode="pro",
    )
    assert client.responses.create_calls
    assert not client.chat.completions.create_calls
    call = client.responses.create_calls[0]
    assert call["reasoning"] == {"effort": "max", "mode": "pro"}
    assert resp.reasoning_mode == "pro"
    assert resp.reasoning_request == {"effort": "max", "mode": "pro"}


@pytest.mark.asyncio
@pytest.mark.unit
async def test_claude_generate_sends_adaptive_and_effort() -> None:
    from atomics.providers.claude import ClaudeProvider

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "4"
    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 5
    usage.thinking_tokens = 0
    mock_response = MagicMock()
    mock_response.content = [text_block]
    mock_response.usage = usage
    mock_response.model_dump.return_value = {}
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    provider = ClaudeProvider(api_key="test", client=mock_client)
    resp = await provider.generate(
        "What is 2+2?",
        model="claude-opus-4-6",
        effort="high",
    )
    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "high"}
    assert resp.effort == "high"
    assert resp.reasoning_request == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "command",
    [
        "eval",
        "provider-test",
        "run",
        "sweep",
        "adversarial",
        "redblue",
        "refusal",
        "toolcall",
        "codereview",
    ],
)
def test_cli_wired_commands_expose_effort(command: str) -> None:
    from click.testing import CliRunner

    from atomics.cli import cli

    result = CliRunner().invoke(cli, [command, "--help"])
    assert result.exit_code == 0, result.output
    assert "--effort" in result.output
    assert "--reasoning-mode" in result.output


@pytest.mark.unit
def test_cli_eval_rejects_unknown_effort() -> None:
    from click.testing import CliRunner

    from atomics.cli import cli

    result = CliRunner().invoke(cli, ["eval", "--effort", "ludicrous", "--no-save"])
    assert result.exit_code != 0
    assert "unknown effort" in result.output.lower()
