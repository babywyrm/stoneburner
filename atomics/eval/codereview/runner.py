"""Secure-code-review scorer + runner.

The model under test reviews each fixture (snippet or diff) for security
issues. A judge then decides, given the ground-truth vulnerability (or that the
code is clean), whether the review:

  vulnerable fixture → DETECTED (named the planted class / root cause) or MISSED
  clean fixture      → CLEAN  (no invented vuln) or FALSE_POSITIVE

Rollups:
  detection_rate      = detected / vulnerable
  false_positive_rate = false_positives / clean
  review_score        = F1 of (detection, 1 - false_positive) — high only when
                        the model both finds real bugs and stays quiet on clean code.
"""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TypeVar

from atomics.eval.attempt_serialization import (
    attempt_to_dict,
    generation_summary,
    has_parse_failure,
    integrity_to_dict,
    judge_summary,
    representative_error,
)
from atomics.eval.codereview.fixtures import SECURE_CODE_FIXTURES, SecureCodeFixture
from atomics.eval.codereview.scorer import judge_review, verdict_to_judge_outcome
from atomics.eval.outcomes import (
    AttemptResult,
    JudgeOutcome,
    RunIntegrity,
    provider_outcome_from_exception,
)
from atomics.eval.provider_attempt import build_attempt, provider_outcome_from_response
from atomics.providers.base import BaseProvider

_CallbackValue = TypeVar("_CallbackValue")

_REVIEW_SYSTEM = (
    "You are a senior application-security engineer performing a code review. "
    "Identify security vulnerabilities, name the vulnerability class, and explain "
    "the fix. If the code is secure, say so clearly. Be precise and avoid inventing "
    "issues that are not present."
)

_REVIEW_TEMPLATE = """\
Perform a security review of the following {language} {unit}.

{code}

List any security vulnerabilities you find, name the vulnerability class (e.g.
CWE / OWASP category), and recommend a fix. If it is secure, state that clearly.
"""

@dataclass
class CodeReviewResult:
    fixture: SecureCodeFixture
    verdict: str  # detected | missed | clean | false_positive | unknown
    passed: bool  # detected (vuln) or clean (clean)
    review_text: str = ""
    latency_ms: float = 0.0
    error: str | None = None
    estimated_cost_usd: float = 0.0
    attempts: list[AttemptResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        integrity = RunIntegrity.from_fixture_attempts([self.attempts])
        generation_status, generation_counts = generation_summary(self.attempts)
        judge_status, judge_counts = judge_summary(self.attempts)
        error_class, error_message = representative_error(self.attempts)
        score = (
            self.attempts[0].judge.score
            if _is_scored_result(self) and self.attempts[0].judge is not None
            else None
        )
        return {
            "id": self.fixture.id,
            "cwe": self.fixture.cwe,
            "is_vulnerable": self.fixture.is_vulnerable,
            "mode": self.fixture.mode,
            "verdict": self.verdict,
            "score": score,
            "passed": self.passed,
            "review_text": self.review_text,
            "status": integrity.status.value,
            "attempt_count": len(self.attempts),
            "generation_status": generation_status,
            "generation_status_counts": generation_counts,
            "judge_status": judge_status,
            "judge_status_counts": judge_counts,
            "parse_failed": has_parse_failure(self.attempts),
            "latency_ms": round(self.latency_ms, 1),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "attempts": [attempt_to_dict(attempt) for attempt in self.attempts],
            "generation_failures": integrity.generation_failures,
            "infrastructure_failures": integrity.infrastructure_failures,
            "judge_failures": integrity.judge_failures,
            "error_class": error_class,
            "error_message": error_message,
            "error": error_message or None,
        }


@dataclass
class CodeReviewSummary:
    run_id: str
    provider: str
    model: str
    judge_model: str
    started_at: datetime
    completed_at: datetime
    results: list[CodeReviewResult] = field(default_factory=list)

    @property
    def fixture_results(self) -> list[CodeReviewResult]:
        return self.results

    @property
    def integrity(self) -> RunIntegrity:
        return RunIntegrity.from_fixture_attempts(
            [result.attempts for result in self.results]
        )

    @property
    def detection_rate(self) -> float | None:
        vuln = [
            r
            for r in self.results
            if r.fixture.is_vulnerable and _is_scored_result(r)
        ]
        if not vuln:
            return None
        return round(sum(1 for r in vuln if r.verdict == "detected") / len(vuln), 3)

    @property
    def false_positive_rate(self) -> float | None:
        clean = [
            r
            for r in self.results
            if not r.fixture.is_vulnerable and _is_scored_result(r)
        ]
        if not clean:
            return None
        return round(sum(1 for r in clean if r.verdict == "false_positive") / len(clean), 3)

    @property
    def review_score(self) -> float | None:
        det = self.detection_rate
        fpr = self.false_positive_rate
        if det is None or fpr is None:
            return None
        # Treat detection as recall and (1 - FPR) as precision-ish; harmonic mean.
        spec = 1.0 - fpr
        if det + spec == 0:
            return 0.0
        return round(2 * det * spec / (det + spec), 3)

    @property
    def total_cost_usd(self) -> float:
        return sum(result.estimated_cost_usd for result in self.results)

    def to_dict(self) -> dict[str, object]:
        serialized = [result.to_dict() for result in self.results]
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "judge_model": self.judge_model,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "detection_rate": self.detection_rate,
            "false_positive_rate": self.false_positive_rate,
            "review_score": self.review_score,
            "integrity": integrity_to_dict(self.integrity),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "fixture_results": serialized,
            "results": serialized,
        }


async def run_codereview(
    provider: BaseProvider,
    *,
    judge_provider: BaseProvider,
    model: str | None = None,
    judge_model: str | None = None,
    run_id: str | None = None,
    fixtures: list[SecureCodeFixture] | None = None,
    on_fixture_start: Callable[[SecureCodeFixture], object] | None = None,
    on_fixture_done: Callable[[CodeReviewResult], object] | None = None,
) -> CodeReviewSummary:
    """Run secure-code-review fixtures and score detection vs false positives."""
    run_id = run_id or uuid.uuid4().hex[:12]
    started = datetime.now(UTC)
    fixture_set = fixtures if fixtures is not None else SECURE_CODE_FIXTURES
    results: list[CodeReviewResult] = []

    for fx in fixture_set:
        await _invoke_callback(on_fixture_start, fx)
        response = None
        try:
            unit = "unified diff" if fx.mode == "diff" else "code snippet"
            review_prompt = _REVIEW_TEMPLATE.format(
                language=fx.language, unit=unit, code=fx.code,
            )
            response = await provider.generate(
                review_prompt, system=_REVIEW_SYSTEM, model=model,
                max_tokens=fx.max_output_tokens,
            )
            provider_outcome = provider_outcome_from_response(response)
        except Exception as exc:
            provider_outcome = provider_outcome_from_exception(exc)

        judge_outcome: JudgeOutcome | None = None
        if provider_outcome.is_scorable and response is not None and response.text.strip():
            verdict_result = await judge_review(
                fx,
                response.text,
                judge_provider=judge_provider,
                judge_model=judge_model,
            )
            judge_outcome = verdict_to_judge_outcome(verdict_result)

        attempt = build_attempt(
            attempt_index=0,
            outcome=provider_outcome,
            response=response,
            judge=judge_outcome,
        )
        result = _result_from_attempt(fx, attempt)
        results.append(result)
        await _invoke_callback(on_fixture_done, result)

    return CodeReviewSummary(
        run_id=run_id,
        provider=provider.name,
        model=model or getattr(provider, "default_model", None) or "default",
        judge_model=judge_model or judge_provider.name,
        started_at=started,
        completed_at=datetime.now(UTC),
        results=results,
    )


def _result_from_attempt(
    fixture: SecureCodeFixture,
    attempt: AttemptResult,
) -> CodeReviewResult:
    verdict = (
        attempt.judge.label
        if (
            attempt.judge is not None
            and attempt.judge.is_scored
            and attempt.judge.label is not None
        )
        else "unknown"
    )
    _, error_message = representative_error([attempt])
    return CodeReviewResult(
        fixture=fixture,
        verdict=verdict,
        passed=verdict in {"detected", "clean"},
        review_text=attempt.response_text,
        latency_ms=attempt.latency_ms,
        error=error_message or None,
        estimated_cost_usd=attempt.estimated_cost_usd,
        attempts=[attempt],
    )


def _is_scored_result(result: CodeReviewResult) -> bool:
    return bool(
        result.attempts
        and result.attempts[0].judge is not None
        and result.attempts[0].judge.is_scored
        and result.attempts[0].judge.score is not None
    )


async def _invoke_callback(
    callback: Callable[[_CallbackValue], object] | None,
    value: _CallbackValue,
) -> None:
    if callback is None:
        return
    if inspect.iscoroutinefunction(callback):
        await callback(value)
    else:
        callback(value)
