"""Red/Blue team capability eval runner.

Reuses the existing quality judge (score_response) — same scoring as eval.py
but run against security-domain fixtures tagged as red or blue team.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from atomics.eval.judge import (
    JudgeResult,
    char_budget_for_tokens,
    detect_self_judge,
    score_response,
)
from atomics.eval.redblue.fixtures import ALL_FIXTURES, BLUE_FIXTURES, RED_FIXTURES, RedBlueFixture
from atomics.models import TaskCategory, TaskResult, TaskStatus
from atomics.providers.base import BaseProvider

logger = logging.getLogger("atomics.eval.redblue.runner")


@dataclass
class RedBlueFixtureResult:
    fixture: RedBlueFixture
    task_result: TaskResult
    judge: JudgeResult | None


@dataclass
class RedBlueSummary:
    run_id: str
    provider: str
    model: str
    mode: str
    started_at: datetime
    completed_at: datetime
    results: list[RedBlueFixtureResult] = field(default_factory=list)

    @property
    def total_fixtures(self) -> int:
        return len(self.results)

    @property
    def overall_quality(self) -> float | None:
        scored = [r.judge.score for r in self.results if r.judge and not r.judge.parse_failed]
        return round(sum(scored) / len(scored), 3) if scored else None

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


async def run_redblue(
    provider: BaseProvider,
    *,
    judge_provider: BaseProvider,
    mode: str = "all",
    model: str | None = None,
    judge_model: str | None = None,
    run_id: str | None = None,
    thinking: bool | None = None,
    thinking_budget: int | None = None,
    on_fixture_done: object | None = None,
) -> RedBlueSummary:
    """Run red/blue fixtures against provider, judge with quality scorer."""
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

    for fixture in fixture_set:
        logger.info(
            "[redblue] %s (%s/%s) %s",
            fixture.id, fixture.team, fixture.category, fixture.prompt[:60],
        )
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
                max_tokens=fixture.max_output_tokens,
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
            err = (str(exc) or repr(exc))[:500]
            logger.warning("[redblue] %s generate failed: %s", fixture.id, err)
            task_result.status = TaskStatus.FAILED
            task_result.error_class = type(exc).__name__
            task_result.error_message = err
            task_result.completed_at = datetime.now(UTC)
            fr = RedBlueFixtureResult(fixture=fixture, task_result=task_result, judge=None)
            results.append(fr)
            if on_fixture_done:
                if inspect.iscoroutinefunction(on_fixture_done):
                    await on_fixture_done(fr)
                else:
                    on_fixture_done(fr)
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

        logger.info(
            "[redblue] %s → %.3f — %s",
            fixture.id, judge.score, judge.rationale[:80],
        )

        fr = RedBlueFixtureResult(fixture=fixture, task_result=task_result, judge=judge)
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
        results=results,
    )
