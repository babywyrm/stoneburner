"""Tests for shared evaluation-attempt serialization."""

from __future__ import annotations

from atomics.eval.attempt_serialization import (
    attempt_to_dict,
    generation_summary,
    has_parse_failure,
    integrity_to_dict,
    judge_summary,
    representative_error,
    summarize_statuses,
)
from atomics.eval.outcomes import (
    AttemptResult,
    JudgeCallResult,
    JudgeOutcome,
    JudgeOutcomeStatus,
    ProviderOutcome,
    ProviderOutcomeKind,
    RunIntegrity,
)


def _attempt(
    *,
    provider_kind: ProviderOutcomeKind = ProviderOutcomeKind.COMPLETED,
    judge_status: JudgeOutcomeStatus = JudgeOutcomeStatus.SCORED,
    provider_error: str | None = None,
    judge_error: str | None = None,
) -> AttemptResult:
    score = 1.0 if judge_status is JudgeOutcomeStatus.SCORED else None
    call = JudgeCallResult(
        status=judge_status,
        judge_model="judge-v1",
        response_text="CLASS: COMPLIED",
        error_class="JudgeError" if judge_error else None,
        error_message=judge_error,
        input_tokens=12,
        output_tokens=4,
        thinking_tokens=0,
        latency_ms=20.0,
        estimated_cost_usd=0.02,
        score=score,
        label="comply" if score is not None else None,
        rationale="matched expected behavior",
        effective_text="CLASS: COMPLIED",
    )
    judge = JudgeOutcome(
        status=judge_status,
        score=score,
        label="comply" if score is not None else None,
        rationale="matched expected behavior",
        judge_model="judge-v1",
        judge_scores=(score,) if score is not None else (),
        judge_cost_usd=0.02,
        calls=(call,),
        judges_expected=1,
        judges_scored=1 if score is not None else 0,
    )
    return AttemptResult(
        attempt_index=0,
        provider=ProviderOutcome(
            provider_kind,
            finish_reason="stop",
            error_class="ProviderError" if provider_error else None,
            error_message=provider_error,
        ),
        response_text="model response",
        latency_ms=40.0,
        estimated_cost_usd=0.03,
        input_tokens=8,
        output_tokens=6,
        thinking_tokens=1,
        judge=judge,
    )


def test_attempt_to_dict_retains_provider_and_judge_evidence() -> None:
    payload = attempt_to_dict(_attempt())

    assert payload["provider_kind"] == "completed"
    assert payload["provider_finish_reason"] == "stop"
    assert payload["response_text"] == "model response"
    assert payload["judge_status"] == "scored"
    assert payload["judge_score"] == 1.0
    calls = payload["judge_calls"]
    assert isinstance(calls, list)
    assert calls[0]["response_text"] == "CLASS: COMPLIED"
    assert calls[0]["input_tokens"] == 12


def test_integrity_to_dict_reports_coverage() -> None:
    integrity = RunIntegrity.from_fixture_attempts([[_attempt()]])

    assert integrity_to_dict(integrity) == {
        "status": "complete",
        "fixtures_total": 1,
        "fixtures_scored": 1,
        "attempts_total": 1,
        "attempts_scorable": 1,
        "attempts_scored": 1,
        "generation_failures": 0,
        "infrastructure_failures": 0,
        "judge_failures": 0,
        "fixture_coverage": 1.0,
        "attempt_coverage": 1.0,
        "infrastructure_failure_rate": 0.0,
        "judge_failure_rate": 0.0,
        "should_exit_nonzero": False,
    }


def test_status_summaries_report_mixed_and_skipped() -> None:
    completed = _attempt()
    timeout = AttemptResult(
        attempt_index=1,
        provider=ProviderOutcome(ProviderOutcomeKind.TIMEOUT),
        response_text="",
        latency_ms=0.0,
        estimated_cost_usd=0.0,
        input_tokens=0,
        output_tokens=0,
        thinking_tokens=0,
    )

    assert summarize_statuses([]) == ("not_attempted", {})
    assert generation_summary([completed, timeout]) == (
        "mixed",
        {"completed": 1, "timeout": 1},
    )
    assert judge_summary([completed, timeout]) == (
        "mixed",
        {"scored": 1, "skipped": 1},
    )


def test_representative_error_prefers_provider_and_sanitizes_secret() -> None:
    error_class, error_message = representative_error(
        [_attempt(provider_error="api_key=secret-value")]
    )

    assert error_class == "ProviderError"
    assert "secret-value" not in error_message
    assert "[REDACTED]" in error_message


def test_has_parse_failure_checks_judge_status() -> None:
    assert has_parse_failure(
        [_attempt(judge_status=JudgeOutcomeStatus.PARSE_FAILED)]
    )
    assert not has_parse_failure([_attempt()])
