"""Red/Blue team capability eval runner.

Reuses the existing quality judge (score_response) — same scoring as eval.py
but run against security-domain fixtures tagged as red or blue team.
"""

from __future__ import annotations

from atomics.validation import sanitize_error

import inspect
import logging
import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from atomics.eval.judge import (
    JudgeResult,
    char_budget_for_tokens,
    detect_self_judge,
    score_response,
)
from atomics.eval.redblue.fixtures import ALL_FIXTURES, BLUE_FIXTURES, RED_FIXTURES, RedBlueFixture
from atomics.model_classes import supports_thinking
from atomics.models import TaskCategory, TaskResult, TaskStatus
from atomics.providers.base import BaseProvider

logger = logging.getLogger("atomics.eval.redblue.runner")

# When thinking is enabled, reasoning models spend most of their output budget on
# hidden reasoning before the visible answer. The fixture's max_output_tokens
# (1024) is sized for the visible answer; without headroom the answer is truncated
# and scored as a capability gap rather than reflecting the model's real ability.
_THINKING_MIN_OUTPUT_TOKENS = 4096


def _output_budget(fixture: RedBlueFixture, *, thinking: bool | None, model: str | None) -> int:
    """Resolve the output-token budget, giving thinking models room to reason."""
    base = fixture.max_output_tokens
    use_thinking = thinking if thinking is not None else (
        supports_thinking(model) if model else False
    )
    if use_thinking:
        return max(base, _THINKING_MIN_OUTPUT_TOKENS)
    return base


@dataclass
class RedBlueFixtureResult:
    fixture: RedBlueFixture
    task_result: TaskResult
    judge: JudgeResult | None
    # Per-run judge scores when runs > 1 (mean is written to task_result/judge).
    run_scores: list[float] = field(default_factory=list)


@dataclass
class RedBlueSummary:
    run_id: str
    provider: str
    model: str
    mode: str
    started_at: datetime
    completed_at: datetime
    runs: int = 1
    results: list[RedBlueFixtureResult] = field(default_factory=list)

    @property
    def fixture_results(self) -> list[RedBlueFixtureResult]:
        """Alias for `results` — the convergent name used across eval suites.

        See ARCHITECTURE.md "known divergences". New code should prefer
        `fixture_results`; `results` remains for back-compat.
        """
        return self.results

    @property
    def total_fixtures(self) -> int:
        return len(self.results)

    @property
    def overall_quality(self) -> float | None:
        scored = [r.judge.score for r in self.results if r.judge and not r.judge.parse_failed]
        return round(sum(scored) / len(scored), 3) if scored else None

    @property
    def quality_stddev(self) -> float | None:
        """Stddev across per-run scores — only meaningful when runs > 1."""
        if self.runs <= 1:
            return None
        all_scores: list[float] = []
        for r in self.results:
            all_scores.extend(r.run_scores)
        if len(all_scores) < 2:
            return None
        mean = sum(all_scores) / len(all_scores)
        variance = sum((s - mean) ** 2 for s in all_scores) / len(all_scores)
        return round(math.sqrt(variance), 3)

    @property
    def category_scores(self) -> dict[str, float]:
        cats: dict[str, list[float]] = {}
        for r in self.results:
            if r.judge and not r.judge.parse_failed:
                cats.setdefault(r.fixture.category, []).append(r.judge.score)
        return {cat: round(sum(s) / len(s), 3) for cat, s in cats.items()}

    @property
    def avg_latency_ms(self) -> float:
        lats = [r.task_result.latency_ms for r in self.results]
        return round(sum(lats) / len(lats), 1) if lats else 0.0

    @property
    def total_cost_usd(self) -> float:
        return sum(r.task_result.estimated_cost_usd for r in self.results)

    def to_dict(self) -> dict:
        """Machine-readable view of the run for --json-out / dashboards / CI."""
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "mode": self.mode,
            "runs": self.runs,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "overall_quality": self.overall_quality,
            "quality_stddev": self.quality_stddev,
            "total_fixtures": self.total_fixtures,
            "category_scores": self.category_scores,
            "avg_latency_ms": self.avg_latency_ms,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "fixtures": [
                {
                    "id": r.fixture.id,
                    "team": r.fixture.team,
                    "category": r.fixture.category,
                    "complexity": getattr(r.fixture, "complexity", None),
                    "status": r.task_result.status.value,
                    "score": r.judge.score if r.judge else None,
                    "parse_failed": r.judge.parse_failed if r.judge else True,
                    "run_scores": r.run_scores,
                    "rationale": r.judge.rationale if r.judge else "",
                    "criteria_coverage": (
                        r.judge.criteria_coverage if r.judge else None
                    ),
                    "latency_ms": round(r.task_result.latency_ms, 1),
                    "output_tokens": r.task_result.output_tokens,
                    "thinking_tokens": r.task_result.thinking_tokens,
                    "estimated_cost_usd": round(r.task_result.estimated_cost_usd, 6),
                    "error": r.task_result.error_message or None,
                }
                for r in self.results
            ],
        }


async def run_redblue(
    provider: BaseProvider,
    *,
    judge_provider: BaseProvider,
    mode: str = "all",
    model: str | None = None,
    judge_model: str | None = None,
    runs: int = 1,
    run_id: str | None = None,
    thinking: bool | None = None,
    thinking_budget: int | None = None,
    on_fixture_start: Callable[..., object] | None = None,
    on_fixture_done: Callable[..., object] | None = None,
) -> RedBlueSummary:
    """Run red/blue fixtures against provider, judge with quality scorer.

    runs>1 re-generates and re-scores each fixture N times; the mean score is
    written to the result and per-run scores are kept for variance reporting.
    """
    runs = max(1, runs)
    if detect_self_judge(provider, model, [(judge_provider, judge_model)]):
        logger.warning(
            "Self-judging detected: the model under test is also the judge. "
            "Scores are biased upward by self-preference — use a different "
            "judge model for a fair evaluation.",
        )

    run_id = run_id or uuid.uuid4().hex[:12]
    started_at = datetime.now(UTC)

    fixture_set: list[RedBlueFixture] = {
        "red": RED_FIXTURES,
        "blue": BLUE_FIXTURES,
        "all": ALL_FIXTURES,
    }.get(mode, ALL_FIXTURES)

    results: list[RedBlueFixtureResult] = []

    for idx, fixture in enumerate(fixture_set):
        if on_fixture_start:
            if inspect.iscoroutinefunction(on_fixture_start):
                await on_fixture_start(idx, fixture)
            else:
                on_fixture_start(idx, fixture)

        logger.info(
            "[redblue] %s (%s/%s) %s",
            fixture.id, fixture.team, fixture.category, fixture.prompt[:60],
        )
        run_scores: list[float] = []
        last_task_result: TaskResult | None = None
        last_judge: JudgeResult | None = None
        failed_task: TaskResult | None = None

        for run_num in range(runs):
            task_result = TaskResult(
                run_id=run_id,
                category=TaskCategory.GENERAL_QA,
                task_name=fixture.id,
                provider=provider.name,
                model=model or "default",
                status=TaskStatus.PENDING,
                thinking_enabled=bool(thinking),
            )

            try:
                resp = await provider.generate(
                    fixture.prompt,
                    system="You are a highly knowledgeable security engineering assistant.",
                    model=model,
                    max_tokens=_output_budget(fixture, thinking=thinking, model=model),
                    thinking=thinking,
                    thinking_budget=thinking_budget,
                )
                task_result.status = TaskStatus.SUCCESS
                task_result.response = resp.text
                task_result.prompt = fixture.prompt
                task_result.input_tokens = resp.input_tokens
                task_result.output_tokens = resp.output_tokens
                task_result.total_tokens = resp.total_tokens
                task_result.thinking_tokens = resp.thinking_tokens
                task_result.latency_ms = resp.latency_ms
                task_result.estimated_cost_usd = resp.estimated_cost_usd
                task_result.tokens_per_second = resp.tokens_per_second
                task_result.completed_at = datetime.now(UTC)
            except Exception as exc:
                # Fall back to repr for exceptions with an empty str (e.g. ReadTimeout).
                err = sanitize_error(exc)
                logger.warning(
                    "[redblue] %s run %d generate failed: %s",
                    fixture.id, run_num + 1, err,
                )
                task_result.status = TaskStatus.FAILED
                task_result.error_class = type(exc).__name__
                task_result.error_message = err
                task_result.completed_at = datetime.now(UTC)
                failed_task = task_result
                continue

            judge = await score_response(
                fixture.prompt,
                resp.text,
                judge_provider=judge_provider,
                judge_model=judge_model,
                gold_criteria=fixture.gold_criteria,
                max_response_chars=char_budget_for_tokens(fixture.max_output_tokens),
            )
            task_result.accuracy_score = judge.score
            task_result.judge_model = judge.judge_model
            task_result.quality_rationale = judge.rationale
            task_result.criteria_coverage = judge.criteria_coverage
            if not judge.parse_failed:
                run_scores.append(judge.score)
            last_task_result = task_result
            last_judge = judge

            logger.info(
                "[redblue] %s run %d/%d → %.3f — %s",
                fixture.id, run_num + 1, runs, judge.score, judge.rationale[:80],
            )

        if last_task_result is None:
            # Every run failed — record the last failure.
            fr = RedBlueFixtureResult(
                fixture=fixture,
                task_result=failed_task,  # type: ignore[arg-type]
                judge=None,
            )
        else:
            # Write the mean score across runs to the persisted result.
            if run_scores:
                mean_score = round(sum(run_scores) / len(run_scores), 3)
                last_task_result.accuracy_score = mean_score
                if last_judge:
                    last_judge.score = mean_score
            fr = RedBlueFixtureResult(
                fixture=fixture,
                task_result=last_task_result,
                judge=last_judge,
                run_scores=run_scores,
            )
        results.append(fr)

        if on_fixture_done:
            if inspect.iscoroutinefunction(on_fixture_done):
                await on_fixture_done(fr)
            else:
                on_fixture_done(fr)

    return RedBlueSummary(
        run_id=run_id,
        provider=provider.name,
        model=model or "default",
        mode=mode,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        runs=runs,
        results=results,
    )
