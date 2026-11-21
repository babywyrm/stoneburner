"""Tests for the resilient judge scoring fallback chain.

Verifies that score_response handles thinking-mode models gracefully:
1. Direct response (thinking=False) — normal path
2. Empty response → retry with thinking enabled
3. Still empty → parse from thinking_text field
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from atomics.eval.judge import score_response
from atomics.providers.base import ProviderResponse


def _make_provider_response(text: str = "", thinking_text: str = "", **kwargs) -> ProviderResponse:
    defaults = {
        "text": text,
        "input_tokens": 10,
        "output_tokens": 50,
        "total_tokens": 60,
        "model": "test-model",
        "latency_ms": 100.0,
        "estimated_cost_usd": 0.0,
        "tokens_per_second": 50.0,
        "thinking_tokens": 0,
        "thinking_text": thinking_text,
    }
    defaults.update(kwargs)
    return ProviderResponse(**defaults)


@pytest.mark.asyncio
async def test_judge_direct_response_parses():
    """Normal path: thinking=False produces a parseable score on first try."""
    provider = AsyncMock()
    provider.generate = AsyncMock(
        return_value=_make_provider_response(
            text="accuracy: 4\ncompleteness: 3\nformat: 3\nrationale: Good response."
        )
    )

    result = await score_response(
        prompt="Test prompt",
        response="Test response",
        judge_provider=provider,
        judge_model="qwen2.5:7b",
    )

    assert not result.parse_failed
    assert result.score == 1.0
    assert provider.generate.call_count == 1
    call_kwargs = provider.generate.call_args[1]
    assert call_kwargs.get("thinking") is False


@pytest.mark.asyncio
async def test_judge_empty_response_retries_with_thinking():
    """Empty first response triggers retry with thinking=True."""
    provider = AsyncMock()
    provider.generate = AsyncMock(
        side_effect=[
            _make_provider_response(text=""),
            _make_provider_response(
                text="accuracy: 3\ncompleteness: 2\nformat: 2\nrationale: Partial answer."
            ),
        ]
    )

    result = await score_response(
        prompt="Test prompt",
        response="Test response",
        judge_provider=provider,
        judge_model="qwen3.6:35b-a3b",
    )

    assert not result.parse_failed
    assert result.score == 0.7
    assert provider.generate.call_count == 2
    second_call = provider.generate.call_args_list[1][1]
    assert second_call.get("thinking") is True


@pytest.mark.asyncio
async def test_judge_falls_back_to_thinking_text():
    """Both response fields empty, but thinking_text contains parseable score."""
    provider = AsyncMock()
    provider.generate = AsyncMock(
        side_effect=[
            _make_provider_response(text=""),
            _make_provider_response(
                text="",
                thinking_text=(
                    "accuracy: 2\ncompleteness: 1\nformat: 1\nrationale: Weak answer with errors."
                ),
            ),
        ]
    )

    result = await score_response(
        prompt="Test prompt",
        response="Test response",
        judge_provider=provider,
        judge_model="qwen3.6:27b",
    )

    assert not result.parse_failed
    assert result.score == 0.4
    assert provider.generate.call_count == 2


@pytest.mark.asyncio
async def test_judge_all_fallbacks_fail_gracefully():
    """All three attempts produce unparseable output — returns parse_failed with 0.5 score."""
    provider = AsyncMock()
    provider.generate = AsyncMock(
        side_effect=[
            _make_provider_response(text=""),
            _make_provider_response(
                text="I cannot score this.", thinking_text="hmm let me think..."
            ),
        ]
    )

    result = await score_response(
        prompt="Test prompt",
        response="Test response",
        judge_provider=provider,
        judge_model="broken-model",
    )

    assert result.parse_failed
    assert result.score == 0.5


@pytest.mark.asyncio
async def test_judge_thinking_model_works_first_try():
    """Thinking model that correctly responds with thinking=False (no fallback needed)."""
    provider = AsyncMock()
    provider.generate = AsyncMock(
        return_value=_make_provider_response(
            text="accuracy: 4\ncompleteness: 3\nformat: 3\nrationale: Correct and comprehensive.",
        )
    )

    result = await score_response(
        prompt="Write a Sigma rule",
        response="title: Detect PsExec\nstatus: experimental\n...",
        judge_provider=provider,
        judge_model="qwen3.6:35b-a3b",
    )

    assert not result.parse_failed
    assert result.score == 1.0
    assert provider.generate.call_count == 1


@pytest.mark.asyncio
async def test_judge_provider_exception_handled():
    """Provider exception on first call doesn't crash — returns parse_failed."""
    provider = AsyncMock()
    provider.generate = AsyncMock(side_effect=Exception("Connection refused"))

    result = await score_response(
        prompt="Test prompt",
        response="Test response",
        judge_provider=provider,
        judge_model="unreachable-model",
    )

    assert result.parse_failed
    assert "Judge call failed" in result.rationale
    assert result.score == 0.5


@pytest.mark.asyncio
async def test_rag_and_multiturn_judges_share_empty_thinking_fallback_and_strip_think_tags():
    """RAG and multiturn judges must share the empty->thinking retry and <think> stripping."""
    from atomics.eval.multiturn.judge import score_conversation, score_turn
    from atomics.eval.rag.fixtures import ALL_RAG_FIXTURES
    from atomics.eval.rag.judge import score_rag_response

    rag_judge = AsyncMock()
    rag_judge.generate = AsyncMock(
        side_effect=[
            _make_provider_response(text=""),
            _make_provider_response(
                text=(
                    "<think>GROUNDING: 0</think>\n"
                    "GROUNDING: 4\nFAITHFULNESS: 3\nABSTENTION: 3\nRATIONALE: Grounded."
                )
            ),
        ]
    )
    rag_res = await score_rag_response("ans", ALL_RAG_FIXTURES[0], rag_judge)
    assert not rag_res.parse_failed
    assert rag_res.score == 1.0
    assert rag_judge.generate.call_count == 2

    turn_judge = AsyncMock()
    turn_judge.generate = AsyncMock(
        side_effect=[
            _make_provider_response(text=""),
            _make_provider_response(
                text="",
                thinking_text="ACCURACY: 4\nCONTEXT_USE: 3\nCOHERENCE: 3\nRATIONALE: Coherent.",
            ),
        ]
    )
    turn_res = await score_turn("hist", "hi", "reply", "greet", turn_judge)
    assert not turn_res.parse_failed
    assert turn_res.score == 1.0

    conv_judge = AsyncMock()
    conv_judge.generate = AsyncMock(
        return_value=_make_provider_response(
            text="<think>RETENTION: 0\nCONSISTENCY: 0\nINSTRUCTION: 0\nRATIONALE: inner</think>\n"
            "RETENTION: 4\nCONSISTENCY: 3\nINSTRUCTION: 3\nRATIONALE: Solid."
        )
    )
    conv_res = await score_conversation("hist", ["c1"], conv_judge)
    assert not conv_res.parse_failed
    assert conv_res.score == 1.0
