"""Judge, attempt, and run-integrity contracts for evaluation runs.

The provider-attempt contract moved to `atomics.providers.outcomes` so the
providers layer no longer depends on this one. Those names are re-exported here
because the eval suites, the CLI, and the tests all import them from this path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from atomics.providers.outcomes import (
    ProviderOutcome,
    ProviderOutcomeKind,
    policy_block_reason,
    provider_outcome_from_exception,
)

__all__ = [
    "AttemptResult",
    "FixtureOutcome",
    "JudgeCallResult",
    "JudgeOutcome",
    "JudgeOutcomeStatus",
    "ProviderOutcome",
    "ProviderOutcomeKind",
    "RunIntegrity",
    "RunStatus",
    "aggregate_attempt_scores",
    "policy_block_reason",
    "provider_outcome_from_exception",
    "sum_attempt_costs",
    "sum_attempt_latency",
]


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
class FixtureOutcome:
    """One fixture's contribution to run integrity, in suite-neutral terms.

    For suites that never adopted `AttemptResult`. It carries only what
    integrity accounting needs — did the provider produce something scorable,
    and did a judge score it — using the same enums as the typed path so no
    suite has to invent a placeholder score to be counted.
    """

    generation: ProviderOutcomeKind
    judge: JudgeOutcomeStatus


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
        """Derive integrity from the typed attempt records a suite produced."""
        return _count_integrity(
            [
                [_view_of_attempt(attempt) for attempt in fixture]
                for fixture in per_fixture_attempts
            ]
        )

    @classmethod
    def from_fixture_outcomes(
        cls, outcomes: Sequence[FixtureOutcome]
    ) -> RunIntegrity:
        """Derive integrity for a suite whose results are not `AttemptResult`.

        Several suites — multi-turn, RAG, red/blue, codegen, tool-call — grew
        their own result and judge types before this contract existed, and
        rewriting them onto `AttemptResult` would mean rewriting their judges
        too. They describe each fixture in the neutral terms below instead, and
        share the counting with `from_fixture_attempts` so the two paths cannot
        report the same run differently.

        One fixture is one attempt here. Suites that retry a fixture use the
        attempt-based constructor.
        """
        return _count_integrity([[_view_of_outcome(o)] for o in outcomes])


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


@dataclass(frozen=True)
class _AttemptView:
    """What integrity counting needs from one attempt, whatever produced it."""

    scorable: bool
    infrastructure_invalid: bool
    scored: bool
    judge_failed: bool


def _view_of_attempt(attempt: AttemptResult) -> _AttemptView:
    judge = attempt.judge
    return _AttemptView(
        scorable=attempt.provider.is_scorable,
        infrastructure_invalid=attempt.provider.is_infrastructure_invalid,
        scored=_is_scored_attempt(attempt),
        # A missing judge counts as a failure: the attempt was scorable and
        # nothing scored it. An explicit SKIPPED does not, since the suite said
        # so deliberately.
        judge_failed=attempt.provider.is_scorable
        and (
            judge is None
            or judge.status
            in {JudgeOutcomeStatus.PARSE_FAILED, JudgeOutcomeStatus.PROVIDER_ERROR}
            or not judge.panel_complete
        ),
    )


def _view_of_outcome(outcome: FixtureOutcome) -> _AttemptView:
    provider = ProviderOutcome(kind=outcome.generation)
    return _AttemptView(
        scorable=provider.is_scorable,
        infrastructure_invalid=provider.is_infrastructure_invalid,
        scored=provider.is_scorable and outcome.judge is JudgeOutcomeStatus.SCORED,
        judge_failed=provider.is_scorable
        and outcome.judge
        in {JudgeOutcomeStatus.PARSE_FAILED, JudgeOutcomeStatus.PROVIDER_ERROR},
    )


def _count_integrity(
    per_fixture: Sequence[Sequence[_AttemptView]],
) -> RunIntegrity:
    """The one place run integrity is counted, for every suite."""
    views = [view for fixture in per_fixture for view in fixture]
    fixtures_total = len(per_fixture)
    fixtures_scored = sum(
        any(view.scored for view in fixture) for fixture in per_fixture
    )
    attempts_scored = sum(view.scored for view in views)
    judge_failures = sum(view.judge_failed for view in views)

    if (
        fixtures_total > 0
        and fixtures_scored == fixtures_total
        and views
        and attempts_scored == len(views)
        and judge_failures == 0
    ):
        status = RunStatus.COMPLETE
    elif fixtures_scored > 0:
        status = RunStatus.PARTIAL
    else:
        status = RunStatus.INFRASTRUCTURE_INVALID

    return RunIntegrity(
        status=status,
        fixtures_total=fixtures_total,
        fixtures_scored=fixtures_scored,
        attempts_total=len(views),
        attempts_scorable=sum(view.scorable for view in views),
        attempts_scored=attempts_scored,
        generation_failures=sum(not view.scorable for view in views),
        infrastructure_failures=sum(view.infrastructure_invalid for view in views),
        judge_failures=judge_failures,
    )


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
