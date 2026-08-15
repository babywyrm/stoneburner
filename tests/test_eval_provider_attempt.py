"""Tests for building typed provider attempts."""

from __future__ import annotations

from atomics.eval.outcomes import (
    JudgeOutcome,
    JudgeOutcomeStatus,
    ProviderOutcome,
    ProviderOutcomeKind,
)
from atomics.eval.provider_attempt import build_attempt, provider_outcome_from_response
from atomics.providers.base import ProviderResponse


def _response(
    text: str,
    *,
    outcome: ProviderOutcome | None = None,
    thinking_tokens: int = 0,
    thinking_text: str = "",
) -> ProviderResponse:
    return ProviderResponse(
        text=text,
        input_tokens=2,
        output_tokens=3,
        total_tokens=5,
        model="qwen",
        latency_ms=10.0,
        estimated_cost_usd=0.25,
        thinking_tokens=thinking_tokens,
        thinking_text=thinking_text,
        outcome=outcome,
        finish_reason=outcome.finish_reason if outcome is not None else "stop",
    )


def _scored_judge() -> JudgeOutcome:
    return JudgeOutcome(
        status=JudgeOutcomeStatus.SCORED,
        score=1.0,
        label="pass",
        judge_model="judge",
        judge_scores=(1.0,),
        judge_cost_usd=0.5,
        judges_expected=1,
        judges_scored=1,
    )


def test_provider_outcome_from_response_marks_empty_text() -> None:
    outcome = provider_outcome_from_response(_response(""))

    assert outcome.kind is ProviderOutcomeKind.EMPTY
    assert outcome.finish_reason == "stop"


def test_empty_visible_with_thinking_tokens_is_budget_spent_on_reasoning() -> None:
    """A silent answer that burned tokens on hidden reasoning is not a dead provider."""
    outcome = provider_outcome_from_response(_response("", thinking_tokens=768))

    assert outcome.kind is ProviderOutcomeKind.THINKING_BUDGET
    assert outcome.is_scorable is False
    assert outcome.is_infrastructure_invalid is False


def test_empty_visible_with_thinking_text_is_budget_spent_on_reasoning() -> None:
    outcome = provider_outcome_from_response(
        _response("", thinking_text="the model reasoned here")
    )

    assert outcome.kind is ProviderOutcomeKind.THINKING_BUDGET


def test_provider_outcome_from_response_marks_visible_text_completed() -> None:
    outcome = provider_outcome_from_response(_response("answer"))

    assert outcome.kind is ProviderOutcomeKind.COMPLETED
    assert outcome.finish_reason == "stop"


def test_provider_outcome_from_response_preserves_normalized_outcome() -> None:
    normalized = ProviderOutcome(
        ProviderOutcomeKind.SAFETY_BLOCKED,
        finish_reason="content_filter",
        safety_reason="content_filter",
    )

    assert provider_outcome_from_response(_response("", outcome=normalized)) is normalized


def test_build_attempt_includes_response_usage_and_judge_cost() -> None:
    response = _response("answer", thinking_tokens=1)
    attempt = build_attempt(
        attempt_index=0,
        outcome=ProviderOutcome(ProviderOutcomeKind.COMPLETED),
        response=response,
        judge=_scored_judge(),
    )

    assert attempt.response_text == "answer"
    assert attempt.latency_ms == 10.0
    assert attempt.estimated_cost_usd == 0.75
    assert attempt.input_tokens == 2
    assert attempt.output_tokens == 3
    assert attempt.thinking_tokens == 1


def test_build_attempt_without_response_has_zero_metrics() -> None:
    outcome = ProviderOutcome(
        ProviderOutcomeKind.TIMEOUT,
        error_class="ReadTimeout",
        error_message="timed out",
    )
    attempt = build_attempt(
        attempt_index=0,
        outcome=outcome,
        response=None,
        judge=None,
    )

    assert attempt.provider is outcome
    assert attempt.response_text == ""
    assert attempt.latency_ms == 0.0
    assert attempt.estimated_cost_usd == 0.0
    assert attempt.input_tokens == 0
    assert attempt.output_tokens == 0
