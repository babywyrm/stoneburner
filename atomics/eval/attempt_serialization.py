"""Shared serialization helpers for typed evaluation attempts."""

from __future__ import annotations

from collections.abc import Sequence

from atomics.eval.outcomes import (
    AttemptResult,
    JudgeOutcomeStatus,
    RunIntegrity,
)
from atomics.validation import sanitize_error


def summarize_statuses(statuses: Sequence[str]) -> tuple[str, dict[str, int]]:
    """Return one aggregate status and counts for a status sequence."""
    counts: dict[str, int] = {}
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    if not statuses:
        return "not_attempted", counts
    return (statuses[0] if len(counts) == 1 else "mixed"), counts


def generation_summary(
    attempts: Sequence[AttemptResult],
) -> tuple[str, dict[str, int]]:
    """Summarize provider outcomes for one fixture."""
    return summarize_statuses([attempt.provider.kind.value for attempt in attempts])


def judge_summary(
    attempts: Sequence[AttemptResult],
) -> tuple[str, dict[str, int]]:
    """Summarize judge outcomes for one fixture."""
    return summarize_statuses(
        [
            (
                attempt.judge.status.value
                if attempt.judge is not None
                else JudgeOutcomeStatus.SKIPPED.value
            )
            for attempt in attempts
        ]
    )


def attempt_to_dict(attempt: AttemptResult) -> dict[str, object]:
    """Serialize one provider attempt and its complete judge-call ledger."""
    judge = attempt.judge
    return {
        "attempt_index": attempt.attempt_index,
        "generation_status": attempt.provider.kind.value,
        "provider_kind": attempt.provider.kind.value,
        "provider_finish_reason": attempt.provider.finish_reason,
        "provider_safety_reason": attempt.provider.safety_reason,
        "provider_error_class": attempt.provider.error_class,
        "provider_error_message": _sanitize_error_message(
            attempt.provider.error_message
        ),
        "response_text": attempt.response_text,
        "latency_ms": attempt.latency_ms,
        "estimated_cost_usd": attempt.estimated_cost_usd,
        "input_tokens": attempt.input_tokens,
        "output_tokens": attempt.output_tokens,
        "thinking_tokens": attempt.thinking_tokens,
        "judge_status": (
            judge.status.value if judge is not None else JudgeOutcomeStatus.SKIPPED.value
        ),
        "judge_score": judge.score if judge is not None else None,
        "judge_label": judge.label if judge is not None else None,
        "judge_rationale": judge.rationale if judge is not None else "",
        "judge_model": judge.judge_model if judge is not None else "",
        "judge_scores": list(judge.judge_scores) if judge is not None else [],
        "judges_expected": judge.judges_expected if judge is not None else 0,
        "judges_scored": judge.judges_scored if judge is not None else 0,
        "panel_complete": judge.panel_complete if judge is not None else True,
        "judge_cost_usd": (
            round(judge.judge_cost_usd, 6) if judge is not None else 0.0
        ),
        "judge_calls": (
            [
                {
                    "status": call.status.value,
                    "judge_model": call.judge_model,
                    "response_text": call.response_text,
                    "thinking_text": call.thinking_text,
                    "effective_text": call.effective_text,
                    "error_class": call.error_class,
                    "error_message": _sanitize_error_message(call.error_message),
                    "input_tokens": call.input_tokens,
                    "output_tokens": call.output_tokens,
                    "thinking_tokens": call.thinking_tokens,
                    "latency_ms": call.latency_ms,
                    "estimated_cost_usd": call.estimated_cost_usd,
                    "score": call.score,
                    "label": call.label,
                    "rationale": call.rationale,
                }
                for call in judge.calls
            ]
            if judge is not None
            else []
        ),
    }


def representative_error(
    attempts: Sequence[AttemptResult],
) -> tuple[str, str]:
    """Return the first sanitized provider or judge-call error."""
    for attempt in attempts:
        if attempt.provider.error_class or attempt.provider.error_message:
            return (
                attempt.provider.error_class or "",
                _sanitize_error_message(attempt.provider.error_message) or "",
            )
        if attempt.judge is None:
            continue
        for call in attempt.judge.calls:
            if call.error_class or call.error_message:
                return (
                    call.error_class or "",
                    _sanitize_error_message(call.error_message) or "",
                )
    return "", ""


def has_parse_failure(attempts: Sequence[AttemptResult]) -> bool:
    """Return whether any retained judge operation failed to parse."""
    return any(
        attempt.judge is not None
        and attempt.judge.status is JudgeOutcomeStatus.PARSE_FAILED
        for attempt in attempts
    )


def integrity_to_dict(integrity: RunIntegrity) -> dict[str, object]:
    """Serialize run-integrity coverage and failure rates."""
    return {
        "status": integrity.status.value,
        "fixtures_total": integrity.fixtures_total,
        "fixtures_scored": integrity.fixtures_scored,
        "attempts_total": integrity.attempts_total,
        "attempts_scorable": integrity.attempts_scorable,
        "attempts_scored": integrity.attempts_scored,
        "generation_failures": integrity.generation_failures,
        "infrastructure_failures": integrity.infrastructure_failures,
        "judge_failures": integrity.judge_failures,
        "fixture_coverage": integrity.fixture_coverage,
        "attempt_coverage": integrity.attempt_coverage,
        "infrastructure_failure_rate": integrity.infrastructure_failure_rate,
        "judge_failure_rate": integrity.judge_failure_rate,
        "should_exit_nonzero": integrity.should_exit_nonzero,
    }


def _sanitize_error_message(message: str | None) -> str | None:
    return sanitize_error(Exception(message)) if message else message
