"""Shared provider, judge, attempt, and run-integrity contracts."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any

import httpx

from atomics.validation import sanitize_error


class ProviderOutcomeKind(StrEnum):
    """Normalized outcome of one provider attempt."""

    COMPLETED = "completed"
    REFUSED = "refused"
    SAFETY_BLOCKED = "safety_blocked"
    TRUNCATED = "truncated"
    EMPTY = "empty"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"
    TRANSPORT_ERROR = "transport_error"


_SCORABLE_PROVIDER_OUTCOMES = frozenset(
    {
        ProviderOutcomeKind.COMPLETED,
        ProviderOutcomeKind.REFUSED,
        ProviderOutcomeKind.SAFETY_BLOCKED,
        ProviderOutcomeKind.TRUNCATED,
    }
)
_INFRASTRUCTURE_INVALID_PROVIDER_OUTCOMES = frozenset(
    {
        ProviderOutcomeKind.RATE_LIMITED,
        ProviderOutcomeKind.TIMEOUT,
        ProviderOutcomeKind.PROVIDER_ERROR,
        ProviderOutcomeKind.TRANSPORT_ERROR,
    }
)


@dataclass(frozen=True)
class ProviderOutcome:
    """Normalized provider result and optional diagnostic details."""

    kind: ProviderOutcomeKind
    finish_reason: str | None = None
    safety_reason: str | None = None
    error_class: str | None = None
    error_message: str | None = None

    @property
    def is_scorable(self) -> bool:
        return self.kind in _SCORABLE_PROVIDER_OUTCOMES

    @property
    def is_infrastructure_invalid(self) -> bool:
        return self.kind in _INFRASTRUCTURE_INVALID_PROVIDER_OUTCOMES


def provider_outcome_from_exception(exc: BaseException) -> ProviderOutcome:
    """Normalize a provider exception without persisting secrets."""

    response = getattr(exc, "response", None)
    if isinstance(exc, httpx.TimeoutException):
        kind = ProviderOutcomeKind.TIMEOUT
    elif _is_rate_limit_exception(exc):
        kind = ProviderOutcomeKind.RATE_LIMITED
    elif isinstance(exc, httpx.TransportError):
        kind = ProviderOutcomeKind.TRANSPORT_ERROR
    elif safety_reason := policy_block_reason(
        code=getattr(exc, "code", None),
        message=(
            getattr(exc, "message", None) or str(exc)
            if _has_structured_provider_context(exc)
            else None
        ),
        body=(
            getattr(exc, "body", None),
            {
                "code": getattr(response, "code", None),
                "message": getattr(response, "message", None),
            },
            getattr(response, "body", None),
            _structured_response_json(response),
        ),
    ):
        kind = ProviderOutcomeKind.SAFETY_BLOCKED
    else:
        kind = ProviderOutcomeKind.PROVIDER_ERROR

    return ProviderOutcome(
        kind=kind,
        safety_reason=safety_reason if kind is ProviderOutcomeKind.SAFETY_BLOCKED else None,
        error_class=type(exc).__name__,
        error_message=sanitize_error(exc),
    )


def _is_rate_limit_exception(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return status_code == 429 or response_status == 429 or "ratelimit" in type(exc).__name__.lower()


def _has_structured_provider_context(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    return (
        isinstance(exc, httpx.HTTPStatusError)
        or isinstance(getattr(exc, "status_code", None), int)
        or isinstance(getattr(response, "status_code", None), int)
        or getattr(exc, "body", None) is not None
        or getattr(exc, "code", None) is not None
    )


def _structured_response_json(response: object) -> object:
    if response is None:
        return None
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            parsed = json_method()
        except (TypeError, ValueError):
            pass
        else:
            return parsed if isinstance(parsed, (dict, list)) else None

    text = getattr(response, "text", None)
    if not isinstance(text, str):
        return None
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, (dict, list)) else None


_POLICY_CODE_ALIASES = {
    "content_filter": "content_filter",
    "content_filtered": "content_filter",
    "content_policy_violation": "content_policy_violation",
    "content_policy_violated": "content_policy_violation",
    "content_policy_blocked": "content_policy_violation",
    "image_content_policy_violation": "image_content_policy_violation",
    "image_content_policy_blocked": "image_content_policy_violation",
    "image_policy_violation": "image_content_policy_violation",
    "safety_policy_violation": "policy_block",
    "responsible_ai_policy_violation": "policy_block",
    "responsibleaipolicyviolation": "policy_block",
    "policy_block": "policy_block",
    "policy_blocked": "policy_block",
    "blocked_by_policy": "policy_block",
    "cybersecurity_risk": "cybersecurity_risk",
    "cyber_security_risk": "cybersecurity_risk",
}
_POLICY_MESSAGE_PATTERNS = (
    (
        re.compile(r"\bblocked by (?:the )?(?:safety|content) policy\b", re.IGNORECASE),
        "policy_block",
    ),
    (
        re.compile(r"\bviolat(?:e|es|ed|ing) (?:the )?content policy\b", re.IGNORECASE),
        "content_policy_violation",
    ),
    (
        re.compile(r"\bcybersecurity risk\b", re.IGNORECASE),
        "cybersecurity_risk",
    ),
)
_NEGATED_POLICY_MESSAGE = re.compile(
    r"\b(?:not|never)\s+blocked by (?:the )?(?:safety|content) policy\b"
    r"|\b(?:did|does|do)\s+not\s+violate (?:the )?content policy\b",
    re.IGNORECASE,
)


def policy_block_reason(
    *,
    code: object = None,
    message: object = None,
    body: object = None,
) -> str | None:
    """Return a canonical policy reason from structured codes or explicit prose."""

    codes: list[str] = []
    messages: list[str] = []
    if code is not None:
        codes.append(str(code))
    if message is not None:
        messages.append(str(message))
    _collect_policy_fields(body, codes=codes, messages=messages)

    for candidate in codes:
        normalized = re.sub(r"[^a-z0-9]+", "_", candidate.lower()).strip("_")
        if canonical := _POLICY_CODE_ALIASES.get(normalized):
            return canonical

    for candidate in messages:
        if _NEGATED_POLICY_MESSAGE.search(candidate):
            continue
        for pattern, canonical in _POLICY_MESSAGE_PATTERNS:
            if pattern.search(candidate):
                return canonical
    return None


def _collect_policy_fields(
    value: Any,
    *,
    codes: list[str],
    messages: list[str],
) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = str(key).lower()
            if normalized_key in {"code", "reason", "type"} and isinstance(
                nested, (str, int)
            ):
                codes.append(str(nested))
            elif normalized_key in {"message", "detail", "error_description"} and isinstance(
                nested, str
            ):
                messages.append(nested)
            if isinstance(nested, (dict, list, tuple)):
                _collect_policy_fields(nested, codes=codes, messages=messages)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _collect_policy_fields(nested, codes=codes, messages=messages)


class JudgeOutcomeStatus(StrEnum):
    """Normalized status of one judge operation."""

    SCORED = "scored"
    PARSE_FAILED = "parse_failed"
    PROVIDER_ERROR = "provider_error"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class JudgeCallResult:
    """Immutable record of one actual judge provider call."""

    status: JudgeOutcomeStatus
    judge_model: str
    response_text: str
    error_class: str | None
    error_message: str | None
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    latency_ms: float
    estimated_cost_usd: float
    score: float | None = None
    label: str | None = None
    rationale: str = ""
    thinking_text: str = ""
    effective_text: str = ""

    def __post_init__(self) -> None:
        counts = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
        }
        for name, value in counts.items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")

        metrics = {
            "latency_ms": self.latency_ms,
            "estimated_cost_usd": self.estimated_cost_usd,
        }
        for name, metric_value in metrics.items():
            if not isfinite(metric_value) or metric_value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

        if self.status is JudgeOutcomeStatus.SCORED:
            if not _is_valid_score(self.score):
                raise ValueError("scored judge call must include a valid score")
        elif self.score is not None and not _is_valid_score(self.score):
            raise ValueError("judge call score must be finite and between 0 and 1")


@dataclass(frozen=True)
class JudgeOutcome:
    """Judge result shared across evaluation runners."""

    status: JudgeOutcomeStatus
    score: float | None = None
    label: str | None = None
    rationale: str = ""
    judge_model: str = ""
    judge_scores: tuple[float, ...] = ()
    judge_cost_usd: float = 0.0
    criteria_coverage: float | None = None
    calls: tuple[JudgeCallResult, ...] = ()
    judges_expected: int = 0
    judges_scored: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "judge_scores", tuple(self.judge_scores))
        object.__setattr__(self, "calls", tuple(self.calls))
        if self.judges_expected < 0:
            raise ValueError("judges_expected must be nonnegative")
        if self.judges_scored < 0:
            raise ValueError("judges_scored must be nonnegative")
        if self.judges_scored > self.judges_expected:
            raise ValueError("judges_scored cannot exceed judges_expected")
        if self.status is JudgeOutcomeStatus.SCORED and not _is_valid_score(self.score):
            raise ValueError("scored judge must include a finite score between 0 and 1")

    @property
    def is_scored(self) -> bool:
        return self.status is JudgeOutcomeStatus.SCORED

    @property
    def panel_complete(self) -> bool:
        return self.judges_expected == self.judges_scored


@dataclass(frozen=True)
class AttemptResult:
    """Provider and optional judge result for one evaluation attempt."""

    attempt_index: int
    provider: ProviderOutcome
    response_text: str
    latency_ms: float
    estimated_cost_usd: float
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    judge: JudgeOutcome | None = None

    def __post_init__(self) -> None:
        counts = {
            "attempt_index": self.attempt_index,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
        }
        for name, value in counts.items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")

        metrics = {
            "latency_ms": self.latency_ms,
            "estimated_cost_usd": self.estimated_cost_usd,
        }
        for name, metric_value in metrics.items():
            if not isfinite(metric_value) or metric_value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

        if self.judge is not None and self.judge.is_scored:
            if not _is_valid_score(self.judge.score):
                raise ValueError("scored judge must include a finite score between 0 and 1")


class RunStatus(StrEnum):
    """Integrity status for an evaluation run."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    INFRASTRUCTURE_INVALID = "infrastructure_invalid"


@dataclass(frozen=True)
class RunIntegrity:
    """Coverage and failure counts for a collection of fixture attempts."""

    status: RunStatus
    fixtures_total: int
    fixtures_scored: int
    attempts_total: int
    attempts_scorable: int
    attempts_scored: int
    generation_failures: int
    infrastructure_failures: int
    judge_failures: int

    def __post_init__(self) -> None:
        counts = {
            "fixtures_total": self.fixtures_total,
            "fixtures_scored": self.fixtures_scored,
            "attempts_total": self.attempts_total,
            "attempts_scorable": self.attempts_scorable,
            "attempts_scored": self.attempts_scored,
            "generation_failures": self.generation_failures,
            "infrastructure_failures": self.infrastructure_failures,
            "judge_failures": self.judge_failures,
        }
        for name, value in counts.items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")

        relationships = [
            ("fixtures_scored", self.fixtures_scored, self.fixtures_total),
            ("attempts_scorable", self.attempts_scorable, self.attempts_total),
            ("attempts_scored", self.attempts_scored, self.attempts_scorable),
            ("generation_failures", self.generation_failures, self.attempts_total),
            ("infrastructure_failures", self.infrastructure_failures, self.attempts_total),
            (
                "infrastructure_failures",
                self.infrastructure_failures,
                self.generation_failures,
            ),
            ("judge_failures", self.judge_failures, self.attempts_scorable),
            ("fixtures_scored", self.fixtures_scored, self.attempts_scored),
            (
                "infrastructure_failures",
                self.infrastructure_failures,
                self.attempts_total - self.attempts_scorable,
            ),
        ]
        for name, value, upper_bound in relationships:
            if value > upper_bound:
                raise ValueError(f"{name} exceeds its denominator")
        if self.generation_failures != self.attempts_total - self.attempts_scorable:
            raise ValueError("generation_failures contradicts attempt counts")

        complete = (
            self.fixtures_total > 0
            and self.fixtures_scored == self.fixtures_total
            and self.attempts_total > 0
            and self.attempts_scored == self.attempts_total
            and self.judge_failures == 0
        )
        if self.status is RunStatus.COMPLETE:
            valid_status = complete
        elif self.status is RunStatus.PARTIAL:
            valid_status = self.fixtures_scored > 0 and not complete
        else:
            valid_status = self.fixtures_scored == 0
        if not valid_status:
            raise ValueError("status contradicts fixture counts")

    @property
    def fixture_coverage(self) -> float:
        return _safe_ratio(self.fixtures_scored, self.fixtures_total)

    @property
    def attempt_coverage(self) -> float:
        return _safe_ratio(self.attempts_scored, self.attempts_total)

    @property
    def infrastructure_failure_rate(self) -> float:
        return _safe_ratio(self.infrastructure_failures, self.attempts_total)

    @property
    def judge_failure_rate(self) -> float:
        return _safe_ratio(self.judge_failures, self.attempts_scorable)

    @property
    def should_exit_nonzero(self) -> bool:
        return self.status is not RunStatus.COMPLETE

    @classmethod
    def from_fixture_attempts(
        cls, per_fixture_attempts: Sequence[Sequence[AttemptResult]]
    ) -> RunIntegrity:
        attempts = [attempt for fixture in per_fixture_attempts for attempt in fixture]
        fixtures_scored = sum(
            any(_is_scored_attempt(attempt) for attempt in fixture)
            for fixture in per_fixture_attempts
        )
        fixtures_total = len(per_fixture_attempts)

        attempts_scored = sum(_is_scored_attempt(attempt) for attempt in attempts)
        judge_failures = sum(
            attempt.provider.is_scorable
            and (
                attempt.judge is None
                or attempt.judge.status
                in {
                    JudgeOutcomeStatus.PARSE_FAILED,
                    JudgeOutcomeStatus.PROVIDER_ERROR,
                }
                or not attempt.judge.panel_complete
            )
            for attempt in attempts
        )
        if (
            fixtures_total > 0
            and fixtures_scored == fixtures_total
            and attempts
            and attempts_scored == len(attempts)
            and judge_failures == 0
        ):
            status = RunStatus.COMPLETE
        elif fixtures_scored > 0:
            status = RunStatus.PARTIAL
        else:
            status = RunStatus.INFRASTRUCTURE_INVALID

        return cls(
            status=status,
            fixtures_total=fixtures_total,
            fixtures_scored=fixtures_scored,
            attempts_total=len(attempts),
            attempts_scorable=sum(attempt.provider.is_scorable for attempt in attempts),
            attempts_scored=attempts_scored,
            generation_failures=sum(
                not attempt.provider.is_scorable for attempt in attempts
            ),
            infrastructure_failures=sum(
                attempt.provider.is_infrastructure_invalid for attempt in attempts
            ),
            judge_failures=judge_failures,
        )


def aggregate_attempt_scores(
    attempts: Sequence[AttemptResult], label_fn: Callable[[float], str]
) -> tuple[float | None, str | None, list[float]]:
    """Aggregate valid scored attempts and derive a label from their mean."""

    scores = [
        attempt.judge.score
        for attempt in attempts
        if _is_scored_attempt(attempt)
        and attempt.judge is not None
        and attempt.judge.score is not None
    ]
    if not scores:
        return None, None, []
    mean_score = sum(scores) / len(scores)
    return mean_score, label_fn(mean_score), scores


def sum_attempt_costs(attempts: Sequence[AttemptResult]) -> float:
    """Return provider costs across every attempt."""

    return sum(attempt.estimated_cost_usd for attempt in attempts)


def sum_attempt_latency(attempts: Sequence[AttemptResult]) -> float:
    """Return provider latency across every attempt."""

    return sum(attempt.latency_ms for attempt in attempts)


def _is_scored_attempt(attempt: AttemptResult) -> bool:
    return (
        attempt.provider.is_scorable
        and attempt.judge is not None
        and attempt.judge.is_scored
        and _is_valid_score(attempt.judge.score)
    )


def _is_valid_score(score: float | None) -> bool:
    return score is not None and isfinite(score) and 0.0 <= score <= 1.0


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
