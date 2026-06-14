"""Eval runner — executes the fixed fixture set against a provider and scores results."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from atomics.eval.fixtures import EVAL_FIXTURES, EvalFixture
from atomics.eval.judge import (
    JudgeResult,
    char_budget_for_tokens,
    score_consensus,
    score_response,
)
from atomics.models import TaskCategory, TaskResult, TaskStatus
from atomics.providers.base import BaseProvider

logger = logging.getLogger("atomics.eval.runner")


@dataclass
class FixtureResult:
    fixture: EvalFixture
    task_result: TaskResult
    judge: JudgeResult | None


@dataclass
class EvalRunSummary:
    run_id: str
    provider: str
    model: str
    judge_provider: str
    judge_model: str
    started_at: datetime
    completed_at: datetime
    fixture_results: list[FixtureResult]

    @property
    def overall_accuracy(self) -> float | None:
        scored = [r.judge.score for r in self.fixture_results if r.judge and not r.judge.parse_failed]
        return round(sum(scored) / len(scored), 3) if scored else None

    @property
    def total_cost_usd(self) -> float:
        return sum(r.task_result.estimated_cost_usd for r in self.fixture_results)

    @property
    def avg_latency_ms(self) -> float:
        lats = [r.task_result.latency_ms for r in self.fixture_results if r.task_result.latency_ms]
        return round(sum(lats) / len(lats), 1) if lats else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.task_result.total_tokens for r in self.fixture_results)

    @property
    def value_score(self) -> float | None:
        """Accuracy per $0.001 cost (floor prevents div-by-zero for free local runs)."""
        acc = self.overall_accuracy
        if acc is None:
            return None
        cost_per_1k = (self.total_cost_usd / self.total_tokens * 1000) if self.total_tokens else 0.0
        eps = 0.001
        return round(acc / max(cost_per_1k, eps), 1)


async def run_eval(
    provider: BaseProvider,
    *,
    judge_provider: BaseProvider,
    model: str | None = None,
    judge_model: str | None = None,
    run_id: str | None = None,
    on_fixture_done: object | None = None,
    thinking: bool | None = None,
    thinking_budget: int | None = None,
    fixtures: list[EvalFixture] | None = None,
    extra_judges: list[tuple[BaseProvider, str | None]] | None = None,
) -> EvalRunSummary:
    """Run eval fixtures against provider, score each with judge_provider.

    Args:
        provider: The model under test.
        judge_provider: Provider used for scoring (typically local Ollama for $0 cost).
        model: Model override for the provider under test.
        judge_model: Model override for the judge.
        run_id: Optional run ID (auto-generated if omitted).
        on_fixture_done: Optional async callable(fixture_result) called after each fixture.
        fixtures: Optional subset of fixtures to run (default: all EVAL_FIXTURES).
        extra_judges: Optional (provider, model) pairs that, together with the
            primary judge, form a consensus panel. When supplied, each fixture is
            scored by every judge and the mean score plus inter-judge stdev is
            recorded.
    """
    extra_judges = extra_judges or []
    run_id = run_id or uuid.uuid4().hex[:12]
    started_at = datetime.now(UTC)
    fixture_results: list[FixtureResult] = []
    effective_fixtures = fixtures if fixtures is not None else EVAL_FIXTURES

    for fixture in effective_fixtures:
        logger.info("[eval] %s (%s) — %s", fixture.id, fixture.complexity.value, fixture.prompt[:60])
        task_result = TaskResult(
            run_id=run_id,
            category=TaskCategory.GENERAL_QA,
            task_name=fixture.id,
            provider=provider.name,
            model=model or "default",
            prompt=fixture.prompt,
        )

        try:
            task_result.status = TaskStatus.RUNNING
            resp = await provider.generate(
                fixture.prompt,
                system="You are a knowledgeable technical assistant. Be accurate and concise.",
                model=model,
                max_tokens=fixture.max_output_tokens,
                thinking=thinking,
                thinking_budget=thinking_budget,
            )
            task_result.status = TaskStatus.SUCCESS
            task_result.response = resp.text
            task_result.input_tokens = resp.input_tokens
            task_result.output_tokens = resp.output_tokens
            task_result.total_tokens = resp.total_tokens
            task_result.thinking_tokens = resp.thinking_tokens
            task_result.cache_read_tokens = resp.cache_read_tokens
            task_result.cache_write_tokens = resp.cache_write_tokens
            task_result.model = resp.model
            task_result.latency_ms = resp.latency_ms
            task_result.estimated_cost_usd = resp.estimated_cost_usd
            task_result.tokens_per_second = resp.tokens_per_second
            task_result.tps_basis = resp.tps_basis
            task_result.thinking_enabled = thinking is True
        except Exception as exc:
            task_result.status = TaskStatus.FAILED
            task_result.error_class = type(exc).__name__
            task_result.error_message = str(exc)[:500]
            task_result.completed_at = datetime.now(UTC)
            fr = FixtureResult(fixture=fixture, task_result=task_result, judge=None)
            fixture_results.append(fr)
            logger.warning("[eval] %s failed: %s", fixture.id, exc)
            if on_fixture_done is not None:
                import asyncio
                if asyncio.iscoroutinefunction(on_fixture_done):
                    await on_fixture_done(fr)
                else:
                    on_fixture_done(fr)
            continue

        task_result.completed_at = datetime.now(UTC)

        # Judge the full intended answer, not a fixed-cap truncation, so long
        # HEAVY responses aren't unfairly marked down on completeness.
        char_budget = char_budget_for_tokens(fixture.max_output_tokens)
        if extra_judges:
            judge = await score_consensus(
                fixture.prompt,
                task_result.response,
                primary_judge=judge_provider,
                primary_model=judge_model,
                extra_judges=extra_judges,
                gold_criteria=fixture.gold_criteria,
                max_response_chars=char_budget,
            )
        else:
            judge = await score_response(
                fixture.prompt,
                task_result.response,
                judge_provider=judge_provider,
                judge_model=judge_model,
                gold_criteria=fixture.gold_criteria,
                max_response_chars=char_budget,
            )

        task_result.accuracy_score = judge.score
        task_result.judge_model = judge.judge_model
        task_result.quality_rationale = judge.rationale
        task_result.criteria_coverage = judge.criteria_coverage
        task_result.judge_score_stdev = judge.score_stdev

        fr = FixtureResult(fixture=fixture, task_result=task_result, judge=judge)
        fixture_results.append(fr)

        logger.info(
            "[eval] %s scored %.3f — %s",
            fixture.id, judge.score, judge.rationale[:80],
        )

        if on_fixture_done is not None:
            import asyncio
            if asyncio.iscoroutinefunction(on_fixture_done):
                await on_fixture_done(fr)
            else:
                on_fixture_done(fr)

    return EvalRunSummary(
        run_id=run_id,
        provider=provider.name,
        model=model or fixture_results[0].task_result.model if fixture_results else "unknown",
        judge_provider=judge_provider.name,
        judge_model=judge_model or "default",
        started_at=started_at,
        completed_at=datetime.now(UTC),
        fixture_results=fixture_results,
    )
