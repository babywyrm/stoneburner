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

With `runs` above 1 each fixture is reviewed again. The rates count every
scored run, and `review_score_stdev` is the spread of each run's F1.
"""

from __future__ import annotations

import logging
import statistics
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from atomics.eval.attempt_serialization import (
    attempt_ledger,
    integrity_to_dict,
    representative_error,
    scored_runs,
)
from atomics.eval.codereview.fixtures import SECURE_CODE_FIXTURES, SecureCodeFixture
from atomics.eval.codereview.scorer import (
    ReviewVerdictResult,
    _score_for_verdict,
    judge_review,
    verdict_to_judge_outcome,
)
from atomics.eval.consensus import (
    CategoricalConsensus,
    CategoricalVote,
    combine_categorical,
)
from atomics.eval.judge import detect_self_judge
from atomics.eval.outcomes import (
    AttemptResult,
    JudgeOutcome,
    JudgeOutcomeStatus,
    RunIntegrity,
    provider_outcome_from_exception,
    sum_attempt_costs,
    sum_attempt_latency,
)
from atomics.eval.provider_attempt import build_attempt, provider_outcome_from_response
from atomics.eval.runner import _call_hook
from atomics.eval.suite_integrity import headline_rate
from atomics.providers.base import BaseProvider

logger = logging.getLogger("atomics.eval.codereview.runner")

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
    judge_agreement: float | None = None

    @property
    def run_scores(self) -> list[float]:
        return [score for _, score in scored_runs(self.attempts)]

    @property
    def run_labels(self) -> list[str]:
        return [label for label, _ in scored_runs(self.attempts)]

    def to_dict(self) -> dict[str, object]:
        ledger = attempt_ledger(self.attempts)
        scores = self.run_scores
        score = sum(scores) / len(scores) if scores else None
        return {
            "id": self.fixture.id,
            "cwe": self.fixture.cwe,
            "is_vulnerable": self.fixture.is_vulnerable,
            "mode": self.fixture.mode,
            "verdict": self.verdict,
            "score": score,
            "passed": self.passed,
            "review_text": self.review_text,
            "latency_ms": round(self.latency_ms, 1),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            **ledger,
            "error": ledger["error_message"] or None,
            "judge_agreement": self.judge_agreement,
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
    runs: int = 1

    @property
    def fixture_results(self) -> list[CodeReviewResult]:
        return self.results

    @property
    def integrity(self) -> RunIntegrity:
        return RunIntegrity.from_fixture_attempts([result.attempts for result in self.results])

    @property
    def detection_rate(self) -> float | None:
        return _detection_rate(self._verdicts())

    @property
    def false_positive_rate(self) -> float | None:
        return _false_positive_rate(self._verdicts())

    @property
    def review_score(self) -> float | None:
        return _review_f1(self._verdicts())

    @property
    def review_score_stdev(self) -> float | None:
        """Sample stdev of each run's F1. None unless every fixture scored every run."""
        if self.runs < 2 or any(len(r.run_labels) != self.runs for r in self.results):
            return None
        per_run = [_review_f1(self._verdicts(run)) for run in range(self.runs)]
        if any(score is None for score in per_run):
            return None
        return round(statistics.stdev(s for s in per_run if s is not None), 3)

    def _verdicts(self, run: int | None = None) -> list[tuple[bool, str]]:
        return [
            (r.fixture.is_vulnerable, label)
            for r in self.results
            for k, label in enumerate(r.run_labels)
            if run is None or k == run
        ]

    @property
    def total_cost_usd(self) -> float:
        return sum(result.estimated_cost_usd for result in self.results)

    def to_dict(self) -> dict[str, object]:
        serialized = [result.to_dict() for result in self.results]
        integrity = self.integrity
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "judge_model": self.judge_model,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "runs": self.runs,
            "detection_rate": headline_rate(self.detection_rate, integrity),
            "false_positive_rate": headline_rate(self.false_positive_rate, integrity),
            "review_score": headline_rate(self.review_score, integrity),
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
    extra_judges: list[tuple[BaseProvider, str | None]] | None = None,
    thinking: bool | None = None,
    thinking_budget: int | None = None,
    effort: str | None = None,
    reasoning_mode: str | None = None,
    run_id: str | None = None,
    fixtures: list[SecureCodeFixture] | None = None,
    on_fixture_start: Callable[[SecureCodeFixture], object] | None = None,
    on_fixture_done: Callable[[CodeReviewResult], object] | None = None,
    on_phase: Callable[..., object] | None = None,
    runs: int = 1,
) -> CodeReviewSummary:
    """Run secure-code-review fixtures and score detection vs false positives."""
    if runs < 1:
        raise ValueError("runs must be at least 1")
    extra_judges = extra_judges or []
    collisions = detect_self_judge(
        provider,
        model,
        [(judge_provider, judge_model), *extra_judges],
    )
    if collisions:
        logger.warning(
            "Self-judging detected: model under test is also a judge (%s). "
            "Scores are biased upward by self-preference — use a "
            "different judge model for a fair evaluation.",
            ", ".join(collisions),
        )
    run_id = run_id or uuid.uuid4().hex[:12]
    started = datetime.now(UTC)
    fixture_set = fixtures if fixtures is not None else SECURE_CODE_FIXTURES
    results: list[CodeReviewResult] = []

    async def run_once(fx: SecureCodeFixture, run: int) -> tuple[AttemptResult, float | None]:
        response = None
        generate_model = model or getattr(provider, "default_model", None)
        await _call_hook(on_phase, fx.id, "generate", generate_model)
        try:
            unit = "unified diff" if fx.mode == "diff" else "code snippet"
            review_prompt = _REVIEW_TEMPLATE.format(
                language=fx.language,
                unit=unit,
                code=fx.code,
            )
            response = await provider.generate(
                review_prompt,
                system=_REVIEW_SYSTEM,
                model=model,
                max_tokens=fx.max_output_tokens,
                thinking=thinking,
                thinking_budget=thinking_budget,
                effort=effort,
                reasoning_mode=reasoning_mode,
            )
            provider_outcome = provider_outcome_from_response(response)
        except Exception as exc:
            provider_outcome = provider_outcome_from_exception(exc)

        judge_outcome: JudgeOutcome | None = None
        judge_agreement: float | None = None
        if provider_outcome.is_scorable and response is not None and response.text.strip():
            judge_tag = judge_model or getattr(judge_provider, "default_model", None)
            await _call_hook(on_phase, fx.id, "judge", judge_tag)
            judge_outcome, judge_agreement = await _judge_with_panel(
                fx,
                response.text,
                judge_provider=judge_provider,
                judge_model=judge_model,
                extra_judges=extra_judges,
            )

        attempt = build_attempt(
            attempt_index=run,
            outcome=provider_outcome,
            response=response,
            judge=judge_outcome,
        )
        return attempt, judge_agreement

    for fx in fixture_set:
        await _call_hook(on_fixture_start, fx)
        passes = [await run_once(fx, run) for run in range(runs)]
        agreements = [agreement for _, agreement in passes if agreement is not None]
        result = _result_from_attempts(
            fx,
            [attempt for attempt, _ in passes],
            judge_agreement=sum(agreements) / len(agreements) if agreements else None,
        )
        results.append(result)
        await _call_hook(on_fixture_done, result)

    return CodeReviewSummary(
        run_id=run_id,
        provider=provider.name,
        model=model or getattr(provider, "default_model", None) or "default",
        judge_model=judge_model or judge_provider.name,
        started_at=started,
        completed_at=datetime.now(UTC),
        results=results,
        runs=runs,
    )


async def _judge_with_panel(
    fixture: SecureCodeFixture,
    review: str,
    *,
    judge_provider: BaseProvider,
    judge_model: str | None,
    extra_judges: list[tuple[BaseProvider, str | None]],
) -> tuple[JudgeOutcome, float | None]:
    """Grade with the primary judge, then majority-vote extras if any."""
    primary = await judge_review(
        fixture,
        review,
        judge_provider=judge_provider,
        judge_model=judge_model,
    )
    if not extra_judges:
        return verdict_to_judge_outcome(primary), None

    panel = [primary]
    for extra_provider, extra_model in extra_judges:
        panel.append(
            await judge_review(
                fixture,
                review,
                judge_provider=extra_provider,
                judge_model=extra_model,
            )
        )
    combined = combine_categorical(
        [
            CategoricalVote(
                label=result.verdict,
                parse_failed=result.status is not JudgeOutcomeStatus.SCORED,
                judge_model=result.judge_model,
                rationale=result.rationale,
            )
            for result in panel
        ]
    )
    return _outcome_from_categorical(panel, combined), combined.agreement


def _outcome_from_categorical(
    panel: list[ReviewVerdictResult],
    combined: CategoricalConsensus,
) -> JudgeOutcome:
    calls = tuple(call for result in panel for call in result.calls)
    cost = sum(call.estimated_cost_usd for call in calls)
    if combined.parse_failed:
        return verdict_to_judge_outcome(panel[0])
    if combined.unresolved or combined.label is None:
        return JudgeOutcome(
            status=JudgeOutcomeStatus.SKIPPED,
            score=None,
            label=None,
            rationale=combined.rationale,
            judge_model=combined.judge_model,
            judge_cost_usd=cost,
            calls=calls,
            judges_expected=len(panel),
            judges_scored=combined.n_judges,
        )
    score = _score_for_verdict(combined.label)
    return JudgeOutcome(
        status=JudgeOutcomeStatus.SCORED,
        score=score,
        label=combined.label,
        rationale=combined.rationale,
        judge_model=combined.judge_model,
        judge_scores=(score,),
        judge_cost_usd=cost,
        calls=calls,
        judges_expected=len(panel),
        judges_scored=combined.n_judges,
    )


def _result_from_attempts(
    fixture: SecureCodeFixture,
    attempts: list[AttemptResult],
    *,
    judge_agreement: float | None = None,
) -> CodeReviewResult:
    labels = [label for label, _ in scored_runs(attempts)]
    verdict = Counter(labels).most_common(1)[0][0] if labels else "unknown"
    _, error_message = representative_error(attempts)
    return CodeReviewResult(
        fixture=fixture,
        verdict=verdict,
        passed=verdict in {"detected", "clean"},
        review_text=attempts[0].response_text if attempts else "",
        latency_ms=sum_attempt_latency(attempts),
        error=error_message or None,
        estimated_cost_usd=sum_attempt_costs(attempts),
        attempts=attempts,
        judge_agreement=judge_agreement,
    )


def _detection_rate(verdicts: list[tuple[bool, str]]) -> float | None:
    vuln = [label for vulnerable, label in verdicts if vulnerable]
    return round(vuln.count("detected") / len(vuln), 3) if vuln else None


def _false_positive_rate(verdicts: list[tuple[bool, str]]) -> float | None:
    clean = [label for vulnerable, label in verdicts if not vulnerable]
    return round(clean.count("false_positive") / len(clean), 3) if clean else None


def _review_f1(verdicts: list[tuple[bool, str]]) -> float | None:
    det = _detection_rate(verdicts)
    fpr = _false_positive_rate(verdicts)
    if det is None or fpr is None:
        return None
    # Treat detection as recall and (1 - FPR) as precision-ish; harmonic mean.
    spec = 1.0 - fpr
    if det + spec == 0:
        return 0.0
    return round(2 * det * spec / (det + spec), 3)
