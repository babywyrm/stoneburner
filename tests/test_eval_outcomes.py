"""Contract tests for shared evaluation outcomes and run integrity."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import httpx
import pytest

from atomics.eval import (
    AttemptResult,
    JudgeCallResult,
    JudgeOutcome,
    JudgeOutcomeStatus,
    ProviderOutcome,
    ProviderOutcomeKind,
    RunIntegrity,
    RunStatus,
    aggregate_attempt_scores,
    provider_outcome_from_exception,
    sum_attempt_costs,
    sum_attempt_latency,
)


def _attempt(
    attempt_index: int,
    provider_kind: ProviderOutcomeKind = ProviderOutcomeKind.COMPLETED,
    judge_status: JudgeOutcomeStatus | None = JudgeOutcomeStatus.SCORED,
    score: float | None = 0.8,
    *,
    cost: float = 0.25,
    latency: float = 125.0,
) -> AttemptResult:
    judge = (
        None
        if judge_status is None
        else JudgeOutcome(status=judge_status, score=score, label="pass")
    )
    return AttemptResult(
        attempt_index=attempt_index,
        provider=ProviderOutcome(provider_kind),
        response_text="response",
        latency_ms=latency,
        estimated_cost_usd=cost,
        input_tokens=10,
        output_tokens=5,
        thinking_tokens=2,
        judge=judge,
    )


def test_provider_outcome_kind_values_are_stable() -> None:
    assert [kind.value for kind in ProviderOutcomeKind] == [
        "completed",
        "refused",
        "safety_blocked",
        "truncated",
        "empty",
        "rate_limited",
        "timeout",
        "provider_error",
        "transport_error",
    ]


@pytest.mark.parametrize(
    "kind",
    [
        ProviderOutcomeKind.COMPLETED,
        ProviderOutcomeKind.REFUSED,
        ProviderOutcomeKind.SAFETY_BLOCKED,
        ProviderOutcomeKind.TRUNCATED,
    ],
)
def test_provider_outcome_scorable_kinds(kind: ProviderOutcomeKind) -> None:
    outcome = ProviderOutcome(kind)
    assert outcome.is_scorable
    assert not outcome.is_infrastructure_invalid


@pytest.mark.parametrize(
    "kind",
    [
        ProviderOutcomeKind.RATE_LIMITED,
        ProviderOutcomeKind.TIMEOUT,
        ProviderOutcomeKind.PROVIDER_ERROR,
        ProviderOutcomeKind.TRANSPORT_ERROR,
    ],
)
def test_provider_outcome_infrastructure_invalid_kinds(kind: ProviderOutcomeKind) -> None:
    outcome = ProviderOutcome(kind)
    assert outcome.is_infrastructure_invalid
    assert not outcome.is_scorable


def test_empty_provider_outcome_is_neither_scorable_nor_infrastructure_invalid() -> None:
    outcome = ProviderOutcome(ProviderOutcomeKind.EMPTY)
    assert not outcome.is_scorable
    assert not outcome.is_infrastructure_invalid


def test_provider_outcome_is_frozen_and_has_optional_details() -> None:
    outcome = ProviderOutcome(
        ProviderOutcomeKind.TRUNCATED,
        finish_reason="length",
        safety_reason=None,
        error_class=None,
        error_message=None,
    )
    assert outcome.finish_reason == "length"
    with pytest.raises(FrozenInstanceError):
        outcome.finish_reason = "stop"  # type: ignore[misc]


class StatusRateLimitError(Exception):
    status_code = 429


class RateLimitBurstError(Exception):
    pass


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (httpx.ReadTimeout("slow"), ProviderOutcomeKind.TIMEOUT),
        (StatusRateLimitError("busy"), ProviderOutcomeKind.RATE_LIMITED),
        (RateLimitBurstError("busy"), ProviderOutcomeKind.RATE_LIMITED),
        (httpx.ConnectError("offline"), ProviderOutcomeKind.TRANSPORT_ERROR),
        (ValueError("bad response"), ProviderOutcomeKind.PROVIDER_ERROR),
    ],
)
def test_provider_exception_classification(
    exc: BaseException, expected: ProviderOutcomeKind
) -> None:
    outcome = provider_outcome_from_exception(exc)
    assert outcome.kind is expected
    assert outcome.error_class == type(exc).__name__
    assert outcome.error_message


def test_provider_exception_message_is_sanitized() -> None:
    outcome = provider_outcome_from_exception(
        ValueError("request used Bearer super-secret-token")
    )
    assert outcome.error_message == "request used [REDACTED]"
    assert "super-secret-token" not in outcome.error_message


def test_provider_exception_empty_string_uses_repr_fallback() -> None:
    outcome = provider_outcome_from_exception(httpx.ReadTimeout(""))
    assert outcome.error_message
    assert "ReadTimeout" in outcome.error_message


class FakeSDKBadRequestError(Exception):
    status_code = 400

    def __init__(self, body: dict[str, object], message: str) -> None:
        super().__init__(message)
        self.body = body
        error = body.get("error")
        self.code = error.get("code") if isinstance(error, dict) else None
        self.message = message


def test_structured_policy_exception_is_safety_blocked_and_sanitized() -> None:
    exc = FakeSDKBadRequestError(
        body={
            "error": {
                "code": "content_policy_violation",
                "message": "Blocked by safety policy for Bearer super-secret-token",
            }
        },
        message="Request rejected: Bearer super-secret-token",
    )

    outcome = provider_outcome_from_exception(exc)

    assert outcome.kind is ProviderOutcomeKind.SAFETY_BLOCKED
    assert outcome.safety_reason == "content_policy_violation"
    assert outcome.error_message == "Request rejected: [REDACTED]"


def test_generic_bad_request_is_not_misclassified_as_safety_blocked() -> None:
    exc = FakeSDKBadRequestError(
        body={
            "error": {
                "code": "invalid_request_error",
                "message": "Malformed parameter.",
            }
        },
        message="Malformed parameter.",
    )

    outcome = provider_outcome_from_exception(exc)

    assert outcome.kind is ProviderOutcomeKind.PROVIDER_ERROR
    assert outcome.safety_reason is None


def test_policy_field_name_does_not_trigger_safety_classification() -> None:
    exc = FakeSDKBadRequestError(
        body={
            "error": {
                "code": "invalid_request_error",
                "safety_identifier": "request-trace-123",
                "message": "Safety settings were malformed.",
            }
        },
        message="Safety settings were malformed.",
    )

    outcome = provider_outcome_from_exception(exc)

    assert outcome.kind is ProviderOutcomeKind.PROVIDER_ERROR
    assert outcome.safety_reason is None


def test_explicit_policy_message_fallback_uses_canonical_reason() -> None:
    exc = FakeSDKBadRequestError(
        body={
            "error": {
                "code": "invalid_request_error",
                "message": "Request was blocked by safety policy.",
            }
        },
        message="Request was blocked by safety policy.",
    )

    outcome = provider_outcome_from_exception(exc)

    assert outcome.kind is ProviderOutcomeKind.SAFETY_BLOCKED
    assert outcome.safety_reason == "policy_block"


def test_normalized_policy_code_variant_uses_canonical_reason() -> None:
    exc = FakeSDKBadRequestError(
        body={
            "error": {
                "code": "CONTENT-POLICY-VIOLATION",
                "message": "Rejected.",
            }
        },
        message="Rejected.",
    )

    outcome = provider_outcome_from_exception(exc)

    assert outcome.kind is ProviderOutcomeKind.SAFETY_BLOCKED
    assert outcome.safety_reason == "content_policy_violation"


def test_policy_code_on_sdk_response_is_classified() -> None:
    exc = RuntimeError("Request rejected.")
    exc.response = type(
        "Response",
        (),
        {
            "status_code": 400,
            "code": "content_filter",
            "message": "Rejected.",
        },
    )()

    outcome = provider_outcome_from_exception(exc)

    assert outcome.kind is ProviderOutcomeKind.SAFETY_BLOCKED
    assert outcome.safety_reason == "content_filter"


def _http_status_error(error: dict[str, str]) -> httpx.HTTPStatusError:
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://api.example.test/v1/responses"),
        json={"error": error},
    )
    with pytest.raises(httpx.HTTPStatusError) as captured:
        response.raise_for_status()
    return captured.value


def test_real_http_policy_error_uses_structured_json_envelope() -> None:
    outcome = provider_outcome_from_exception(
        _http_status_error(
            {
                "code": "content_policy_violation",
                "type": "invalid_request_error",
                "message": "Request rejected.",
            }
        )
    )

    assert outcome.kind is ProviderOutcomeKind.SAFETY_BLOCKED
    assert outcome.safety_reason == "content_policy_violation"


def test_real_http_generic_bad_request_is_provider_error() -> None:
    outcome = provider_outcome_from_exception(
        _http_status_error(
            {
                "code": "invalid_request_error",
                "type": "invalid_request_error",
                "message": "Malformed parameter.",
            }
        )
    )

    assert outcome.kind is ProviderOutcomeKind.PROVIDER_ERROR
    assert outcome.safety_reason is None


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Content filtering configuration is disabled.", ProviderOutcomeKind.PROVIDER_ERROR),
        ("Request was not blocked by safety policy.", ProviderOutcomeKind.PROVIDER_ERROR),
        ("Request did not violate content policy.", ProviderOutcomeKind.PROVIDER_ERROR),
        ("Request was blocked by safety policy.", ProviderOutcomeKind.SAFETY_BLOCKED),
        ("Request violates content policy.", ProviderOutcomeKind.SAFETY_BLOCKED),
    ],
)
def test_policy_message_fallback_requires_positive_block_phrase(
    message: str,
    expected: ProviderOutcomeKind,
) -> None:
    exc = FakeSDKBadRequestError(
        body={"error": {"code": "invalid_request_error", "message": message}},
        message=message,
    )

    outcome = provider_outcome_from_exception(exc)

    assert outcome.kind is expected


def test_transport_error_precedes_policy_prose() -> None:
    outcome = provider_outcome_from_exception(
        httpx.ConnectError("connection blocked by content policy")
    )

    assert outcome.kind is ProviderOutcomeKind.TRANSPORT_ERROR
    assert outcome.safety_reason is None


def test_timeout_error_precedes_policy_prose() -> None:
    outcome = provider_outcome_from_exception(
        httpx.ReadTimeout("request blocked by content policy")
    )

    assert outcome.kind is ProviderOutcomeKind.TIMEOUT
    assert outcome.safety_reason is None


def test_unstructured_exception_policy_prose_is_not_a_safety_block() -> None:
    outcome = provider_outcome_from_exception(
        ValueError("request was blocked by content policy")
    )

    assert outcome.kind is ProviderOutcomeKind.PROVIDER_ERROR
    assert outcome.safety_reason is None


def test_judge_outcome_contract_and_status_values() -> None:
    assert [status.value for status in JudgeOutcomeStatus] == [
        "scored",
        "parse_failed",
        "provider_error",
        "skipped",
    ]
    outcome = JudgeOutcome(
        status=JudgeOutcomeStatus.SCORED,
        score=0.75,
        label="pass",
        rationale="adequate",
        judge_model="judge-v1",
        judge_scores=[0.7, 0.8],
        judge_cost_usd=0.01,
        criteria_coverage=0.5,
    )
    assert outcome.is_scored
    assert outcome.judge_scores == (0.7, 0.8)
    assert not JudgeOutcome(status=JudgeOutcomeStatus.PARSE_FAILED).is_scored


def test_judge_call_result_is_frozen_and_preserves_metadata() -> None:
    call = JudgeCallResult(
        status=JudgeOutcomeStatus.SCORED,
        judge_model="judge-v1",
        response_text="RESISTANCE: 8",
        error_class=None,
        error_message=None,
        input_tokens=10,
        output_tokens=4,
        thinking_tokens=2,
        latency_ms=25.0,
        estimated_cost_usd=0.01,
        score=0.8,
        label="resisted",
        rationale="refused",
    )

    assert call.score == 0.8
    with pytest.raises(FrozenInstanceError):
        call.score = 0.5  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_tokens", -1),
        ("output_tokens", -1),
        ("thinking_tokens", -1),
        ("latency_ms", -0.1),
        ("estimated_cost_usd", -0.1),
        ("latency_ms", float("nan")),
        ("estimated_cost_usd", float("inf")),
    ],
)
def test_judge_call_result_rejects_invalid_metrics(field: str, value: float) -> None:
    values: dict[str, object] = {
        "status": JudgeOutcomeStatus.PARSE_FAILED,
        "judge_model": "judge",
        "response_text": "",
        "error_class": None,
        "error_message": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "latency_ms": 0.0,
        "estimated_cost_usd": 0.0,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        JudgeCallResult(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("score", [None, -0.1, 1.1, float("nan")])
def test_judge_call_result_scored_requires_valid_score(score: float | None) -> None:
    with pytest.raises(ValueError, match="score"):
        JudgeCallResult(
            status=JudgeOutcomeStatus.SCORED,
            judge_model="judge",
            response_text="result",
            error_class=None,
            error_message=None,
            input_tokens=0,
            output_tokens=0,
            thinking_tokens=0,
            latency_ms=0.0,
            estimated_cost_usd=0.0,
            score=score,
        )


def test_judge_call_result_rejects_invalid_optional_diagnostic_score() -> None:
    with pytest.raises(ValueError, match="score"):
        JudgeCallResult(
            status=JudgeOutcomeStatus.PARSE_FAILED,
            judge_model="judge",
            response_text="bad",
            error_class=None,
            error_message=None,
            input_tokens=0,
            output_tokens=0,
            thinking_tokens=0,
            latency_ms=0.0,
            estimated_cost_usd=0.0,
            score=float("inf"),
        )


@pytest.mark.parametrize("score", [None, -0.01, 1.01, float("nan"), float("inf"), -float("inf")])
def test_judge_outcome_rejects_invalid_scored_values(score: float | None) -> None:
    with pytest.raises(ValueError, match="score"):
        JudgeOutcome(status=JudgeOutcomeStatus.SCORED, score=score)


def test_judge_outcome_normalizes_nested_collections_to_immutable_tuples() -> None:
    call = JudgeCallResult(
        status=JudgeOutcomeStatus.PARSE_FAILED,
        judge_model="judge",
        response_text="bad",
        error_class=None,
        error_message=None,
        input_tokens=0,
        output_tokens=0,
        thinking_tokens=0,
        latency_ms=0.0,
        estimated_cost_usd=0.0,
    )
    outcome = JudgeOutcome(
        status=JudgeOutcomeStatus.SCORED,
        score=0.5,
        judge_scores=[0.5],  # type: ignore[arg-type]
        calls=[call],  # type: ignore[arg-type]
    )
    assert outcome.judge_scores == (0.5,)
    assert outcome.calls == (call,)
    with pytest.raises(FrozenInstanceError):
        outcome.score = 0.6  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        outcome.status = JudgeOutcomeStatus.SKIPPED  # type: ignore[misc]
    with pytest.raises(AttributeError):
        outcome.judge_scores.append(0.6)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        outcome.calls.append(call)  # type: ignore[attr-defined]


def test_judge_outcome_panel_metadata_is_immutable_and_validated() -> None:
    outcome = JudgeOutcome(
        status=JudgeOutcomeStatus.SCORED,
        score=0.8,
        judges_expected=2,
        judges_scored=1,
    )

    assert outcome.judges_expected == 2
    assert outcome.judges_scored == 1
    assert not outcome.panel_complete
    with pytest.raises(FrozenInstanceError):
        outcome.judges_scored = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("judges_expected", "judges_scored"),
    [(-1, 0), (1, -1), (1, 2)],
)
def test_judge_outcome_rejects_invalid_panel_counts(
    judges_expected: int,
    judges_scored: int,
) -> None:
    with pytest.raises(ValueError, match="judges_"):
        JudgeOutcome(
            status=JudgeOutcomeStatus.PARSE_FAILED,
            judges_expected=judges_expected,
            judges_scored=judges_scored,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attempt_index", -1),
        ("latency_ms", -0.1),
        ("estimated_cost_usd", -0.1),
        ("input_tokens", -1),
        ("output_tokens", -1),
        ("thinking_tokens", -1),
    ],
)
def test_attempt_result_rejects_negative_indices_and_metrics(
    field: str, value: float
) -> None:
    values: dict[str, object] = {
        "attempt_index": 0,
        "provider": ProviderOutcome(ProviderOutcomeKind.COMPLETED),
        "response_text": "ok",
        "latency_ms": 1.0,
        "estimated_cost_usd": 0.0,
        "input_tokens": 1,
        "output_tokens": 1,
        "thinking_tokens": 0,
    }
    values[field] = value
    with pytest.raises(ValueError, match=field):
        AttemptResult(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_attempt_result_rejects_out_of_range_scored_judge(score: float) -> None:
    with pytest.raises(ValueError, match="score"):
        _attempt(
            0,
            judge_status=JudgeOutcomeStatus.SCORED,
            score=score,
        )


def test_attempt_result_rejects_scored_judge_without_score() -> None:
    with pytest.raises(ValueError, match="score"):
        _attempt(
            0,
            judge_status=JudgeOutcomeStatus.SCORED,
            score=None,
        )


def test_attempt_result_allows_non_scored_judge_diagnostic_score() -> None:
    attempt = _attempt(
        0,
        judge_status=JudgeOutcomeStatus.PARSE_FAILED,
        score=5.0,
    )
    assert attempt.judge is not None
    assert attempt.judge.score == 5.0


@pytest.mark.parametrize("field", ["provider", "judge", "latency_ms"])
def test_attempt_result_is_frozen(field: str) -> None:
    attempt = _attempt(0)
    replacement: object = {
        "provider": ProviderOutcome(ProviderOutcomeKind.TIMEOUT),
        "judge": JudgeOutcome(status=JudgeOutcomeStatus.SKIPPED),
        "latency_ms": 10.0,
    }[field]
    with pytest.raises(FrozenInstanceError):
        setattr(attempt, field, replacement)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latency_ms", float("nan")),
        ("latency_ms", float("inf")),
        ("estimated_cost_usd", float("nan")),
        ("estimated_cost_usd", float("inf")),
    ],
)
def test_attempt_result_rejects_nonfinite_metrics(field: str, value: float) -> None:
    values: dict[str, object] = {
        "attempt_index": 0,
        "provider": ProviderOutcome(ProviderOutcomeKind.COMPLETED),
        "response_text": "ok",
        "latency_ms": 1.0,
        "estimated_cost_usd": 0.0,
        "input_tokens": 1,
        "output_tokens": 1,
        "thinking_tokens": 0,
    }
    values[field] = value
    with pytest.raises(ValueError, match=field):
        AttemptResult(**values)  # type: ignore[arg-type]


def test_aggregate_scores_excludes_parse_and_infrastructure_failures() -> None:
    attempts = [
        _attempt(0, score=0.2),
        _attempt(1, judge_status=JudgeOutcomeStatus.PARSE_FAILED, score=None),
        _attempt(
            2,
            provider_kind=ProviderOutcomeKind.TIMEOUT,
            judge_status=JudgeOutcomeStatus.SCORED,
            score=1.0,
        ),
        _attempt(3, score=0.8),
    ]
    mean, label, scores = aggregate_attempt_scores(
        attempts, lambda score: "pass" if score >= 0.5 else "fail"
    )
    assert scores == [0.2, 0.8]
    assert mean == pytest.approx(0.5)
    assert label == "pass"


def test_aggregate_scores_returns_empty_result_without_scored_judges() -> None:
    attempts = [
        _attempt(0, judge_status=JudgeOutcomeStatus.PARSE_FAILED, score=None),
        _attempt(1, judge_status=None, score=None),
    ]
    assert aggregate_attempt_scores(attempts, lambda _: "unused") == (None, None, [])


def test_integrity_and_aggregation_consistently_exclude_unscored_attempts() -> None:
    with pytest.raises(ValueError, match="score"):
        _attempt(0, judge_status=JudgeOutcomeStatus.SCORED, score=None)

    attempt = _attempt(
        0,
        judge_status=JudgeOutcomeStatus.PARSE_FAILED,
        score=None,
    )
    integrity = RunIntegrity.from_fixture_attempts([[attempt]])

    assert integrity.fixtures_scored == 0
    assert integrity.attempts_scored == 0
    assert aggregate_attempt_scores([attempt], lambda _: "unused") == (None, None, [])


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), -0.1, 1.1])
def test_integrity_and_aggregation_defensively_exclude_invalid_scores(
    invalid_score: float,
) -> None:
    judge = JudgeOutcome(status=JudgeOutcomeStatus.SCORED, score=0.5)
    object.__setattr__(judge, "score", invalid_score)
    attempt = _attempt(0)
    object.__setattr__(attempt, "judge", judge)

    integrity = RunIntegrity.from_fixture_attempts([[attempt]])

    assert integrity.fixtures_scored == 0
    assert integrity.attempts_scored == 0
    assert aggregate_attempt_scores([attempt], lambda _: "unused") == (None, None, [])


def test_attempt_totals_include_every_attempt() -> None:
    attempts = [
        _attempt(0, cost=0.1, latency=10.0),
        _attempt(
            1,
            provider_kind=ProviderOutcomeKind.TIMEOUT,
            judge_status=None,
            cost=0.2,
            latency=20.0,
        ),
        _attempt(
            2,
            judge_status=JudgeOutcomeStatus.PARSE_FAILED,
            score=None,
            cost=0.3,
            latency=30.0,
        ),
    ]
    assert sum_attempt_costs(attempts) == pytest.approx(0.6)
    assert sum_attempt_latency(attempts) == pytest.approx(60.0)


def test_run_integrity_partial_when_every_fixture_scores_but_attempts_are_unscored() -> None:
    integrity = RunIntegrity.from_fixture_attempts(
        [
            [_attempt(0), _attempt(1, judge_status=JudgeOutcomeStatus.PARSE_FAILED, score=None)],
            [
                _attempt(0, score=0.9),
                _attempt(
                    1,
                    provider_kind=ProviderOutcomeKind.TIMEOUT,
                    judge_status=None,
                ),
            ],
        ]
    )
    assert integrity.status is RunStatus.PARTIAL
    assert integrity.fixtures_total == 2
    assert integrity.fixtures_scored == 2
    assert integrity.attempts_total == 4
    assert integrity.attempts_scorable == 3
    assert integrity.attempts_scored == 2
    assert integrity.infrastructure_failures == 1
    assert integrity.generation_failures == 1
    assert integrity.judge_failures == 1
    assert integrity.fixture_coverage == pytest.approx(1.0)
    assert integrity.attempt_coverage == pytest.approx(2 / 4)
    assert integrity.infrastructure_failure_rate == pytest.approx(0.25)
    assert integrity.judge_failure_rate == pytest.approx(1 / 3)
    assert integrity.should_exit_nonzero


def test_run_integrity_complete_requires_every_requested_attempt_scored() -> None:
    integrity = RunIntegrity.from_fixture_attempts(
        [
            [_attempt(0), _attempt(1, score=0.7)],
            [_attempt(0, score=0.9)],
        ]
    )

    assert integrity.status is RunStatus.COMPLETE
    assert integrity.fixture_coverage == pytest.approx(1.0)
    assert integrity.attempt_coverage == pytest.approx(1.0)
    assert not integrity.should_exit_nonzero


def test_run_integrity_is_partial_for_incomplete_scored_judge_panel() -> None:
    attempt = AttemptResult(
        attempt_index=0,
        provider=ProviderOutcome(ProviderOutcomeKind.COMPLETED),
        response_text="response",
        latency_ms=1.0,
        estimated_cost_usd=0.0,
        input_tokens=1,
        output_tokens=1,
        thinking_tokens=0,
        judge=JudgeOutcome(
            status=JudgeOutcomeStatus.SCORED,
            score=0.8,
            judges_expected=2,
            judges_scored=1,
        ),
    )

    integrity = RunIntegrity.from_fixture_attempts([[attempt]])

    assert integrity.status is RunStatus.PARTIAL
    assert integrity.attempts_scored == 1
    assert integrity.judge_failures == 1
    assert integrity.should_exit_nonzero


def test_run_integrity_partial_when_only_some_fixtures_score() -> None:
    integrity = RunIntegrity.from_fixture_attempts(
        [
            [_attempt(0)],
            [_attempt(0, judge_status=JudgeOutcomeStatus.PARSE_FAILED, score=None)],
        ]
    )
    assert integrity.status is RunStatus.PARTIAL
    assert integrity.fixtures_scored == 1
    assert integrity.should_exit_nonzero


def test_run_integrity_all_invalid_when_no_fixture_scores() -> None:
    integrity = RunIntegrity.from_fixture_attempts(
        [
            [
                _attempt(
                    0,
                    provider_kind=ProviderOutcomeKind.TIMEOUT,
                    judge_status=None,
                )
            ],
            [_attempt(0, judge_status=JudgeOutcomeStatus.PROVIDER_ERROR, score=None)],
        ]
    )
    assert integrity.status is RunStatus.INFRASTRUCTURE_INVALID
    assert integrity.fixtures_total == 2
    assert integrity.fixtures_scored == 0
    assert integrity.infrastructure_failures == 1
    assert integrity.generation_failures == 1
    assert integrity.judge_failures == 1
    assert integrity.should_exit_nonzero


def test_run_integrity_empty_fixture_list_has_zero_rates() -> None:
    integrity = RunIntegrity.from_fixture_attempts([])
    assert integrity.status is RunStatus.INFRASTRUCTURE_INVALID
    assert integrity.fixtures_total == 0
    assert integrity.fixture_coverage == 0.0
    assert integrity.attempt_coverage == 0.0
    assert integrity.infrastructure_failure_rate == 0.0
    assert integrity.judge_failure_rate == 0.0
    assert integrity.should_exit_nonzero


def test_run_integrity_empty_generation_counts_failure_without_infrastructure() -> None:
    integrity = RunIntegrity.from_fixture_attempts(
        [
            [
                _attempt(
                    0,
                    provider_kind=ProviderOutcomeKind.EMPTY,
                    judge_status=None,
                )
            ]
        ]
    )

    assert integrity.status is RunStatus.INFRASTRUCTURE_INVALID
    assert integrity.generation_failures == 1
    assert integrity.infrastructure_failures == 0
    assert integrity.judge_failures == 0


def _integrity_values() -> dict[str, object]:
    return {
        "status": RunStatus.COMPLETE,
        "fixtures_total": 2,
        "fixtures_scored": 2,
        "attempts_total": 4,
        "attempts_scorable": 4,
        "attempts_scored": 4,
        "generation_failures": 0,
        "infrastructure_failures": 0,
        "judge_failures": 0,
    }


@pytest.mark.parametrize(
    "field",
    [
        "fixtures_total",
        "fixtures_scored",
        "attempts_total",
        "attempts_scorable",
        "attempts_scored",
        "generation_failures",
        "infrastructure_failures",
        "judge_failures",
    ],
)
def test_run_integrity_rejects_negative_counts(field: str) -> None:
    values = _integrity_values()
    values[field] = -1
    with pytest.raises(ValueError, match=field):
        RunIntegrity(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fixtures_scored", 3),
        ("attempts_scorable", 5),
        ("attempts_scored", 5),
        ("generation_failures", 5),
        ("infrastructure_failures", 5),
        ("judge_failures", 5),
    ],
)
def test_run_integrity_rejects_impossible_count_relationships(
    field: str, value: int
) -> None:
    values = _integrity_values()
    values[field] = value
    with pytest.raises(ValueError, match=field):
        RunIntegrity(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "expected_field"),
    [
        ({"attempts_scored": 2}, "status"),
        (
            {
                "attempts_scorable": 3,
                "attempts_scored": 3,
                "generation_failures": 1,
                "infrastructure_failures": 2,
            },
            "infrastructure_failures",
        ),
        (
            {
                "attempts_scorable": 3,
                "attempts_scored": 3,
                "generation_failures": 0,
            },
            "generation_failures",
        ),
    ],
)
def test_run_integrity_rejects_cross_counter_contradictions(
    overrides: dict[str, int], expected_field: str
) -> None:
    values = _integrity_values()
    values.update(overrides)
    with pytest.raises(ValueError, match=expected_field):
        RunIntegrity(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides",),
    [
        ({"status": RunStatus.COMPLETE, "fixtures_total": 0, "fixtures_scored": 0,
          "attempts_total": 0, "attempts_scorable": 0, "attempts_scored": 0},),
        ({"status": RunStatus.COMPLETE, "fixtures_scored": 1},),
        ({"status": RunStatus.COMPLETE, "attempts_scored": 3},),
        ({"status": RunStatus.PARTIAL, "fixtures_scored": 0, "attempts_scored": 0},),
        ({"status": RunStatus.PARTIAL},),
        ({"status": RunStatus.INFRASTRUCTURE_INVALID, "fixtures_scored": 1},),
    ],
)
def test_run_integrity_rejects_status_count_contradictions(
    overrides: dict[str, object],
) -> None:
    values = _integrity_values()
    values.update(overrides)
    with pytest.raises(ValueError, match="status"):
        RunIntegrity(**values)  # type: ignore[arg-type]
