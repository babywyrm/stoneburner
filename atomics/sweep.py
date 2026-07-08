"""Multi-model sweep orchestrator.

Runs eval fixtures across a list of models sequentially, collecting
per-model quality/latency/cost summaries for comparison. Designed for
the local-vs-cloud thesis: sweep all models on a GPU host and compare
against a single cloud run.
"""

from __future__ import annotations

from atomics.validation import sanitize_error

import logging
from collections.abc import Callable
from dataclasses import dataclass

from atomics.eval.fixtures import EVAL_FIXTURES, EvalFixture
from atomics.eval.runner import EvalRunSummary, run_eval
from atomics.providers.base import BaseProvider

logger = logging.getLogger("atomics.sweep")


@dataclass
class ModelSweepResult:
    model: str
    fixtures_run: int
    overall_quality: float | None
    avg_latency_ms: float
    total_tokens: int
    total_cost_usd: float
    value_score: float | None
    eval_summary: EvalRunSummary | None
    # Representative failure reason (e.g. "404 Not Found" when a model tag
    # isn't pulled on the host). None when nothing failed.
    error: str | None = None


def _representative_error(summary: EvalRunSummary) -> str | None:
    """Most common per-fixture error message in a summary, or None."""
    from collections import Counter

    msgs = [
        fr.task_result.error_message
        for fr in summary.fixture_results
        if fr.task_result.error_message
    ]
    if not msgs:
        return None
    return Counter(msgs).most_common(1)[0][0][:200]


def _filter_fixtures(fixture_ids: list[str] | None) -> list[EvalFixture]:
    """Return fixtures matching the given IDs, or all if None."""
    if fixture_ids is None:
        return list(EVAL_FIXTURES)
    id_set = set(fixture_ids)
    return [f for f in EVAL_FIXTURES if f.id in id_set]


async def run_model_sweep(
    *,
    provider_factory: Callable[[str], BaseProvider],
    judge_provider: BaseProvider,
    models: list[str],
    fixture_ids: list[str] | None = None,
    judge_model: str | None = None,
    thinking: bool | None = None,
    thinking_budget: int | None = None,
    on_model_done: Callable[[ModelSweepResult], None] | None = None,
    on_fixture_done: Callable | None = None,
) -> list[ModelSweepResult]:
    """Run eval fixtures against each model in sequence.

    Args:
        provider_factory: Callable that takes a model name and returns a configured provider.
        judge_provider: Provider for quality scoring (typically local Ollama).
        models: List of model identifiers to sweep.
        fixture_ids: Optional subset of fixture IDs to run (default: all).
        judge_model: Model override for the judge.
        thinking: Force thinking on/off for capable models.
        thinking_budget: Max thinking tokens.
        on_model_done: Optional callback after each model completes.
        on_fixture_done: Optional callback after each fixture completes (for live verbose output).
    """
    fixtures = _filter_fixtures(fixture_ids)
    results: list[ModelSweepResult] = []

    for model in models:
        logger.info("[sweep] Starting model %s (%d fixtures)", model, len(fixtures))
        provider = provider_factory(model)

        try:
            summary = await run_eval(
                provider,
                judge_provider=judge_provider,
                model=model,
                judge_model=judge_model,
                thinking=thinking,
                thinking_budget=thinking_budget,
                fixtures=fixtures,
                on_fixture_done=on_fixture_done,
            )
            result = ModelSweepResult(
                model=model,
                fixtures_run=len(summary.fixture_results),
                overall_quality=summary.overall_accuracy,
                avg_latency_ms=summary.avg_latency_ms,
                total_tokens=summary.total_tokens,
                total_cost_usd=summary.total_cost_usd,
                value_score=summary.value_score,
                eval_summary=summary,
                # If nothing scored (e.g. every fixture 404'd because the tag
                # isn't pulled), capture why so the caller needn't dig the logs.
                error=_representative_error(summary) if summary.overall_accuracy is None else None,
            )
        except Exception as exc:
            err = sanitize_error(exc)[:200]
            logger.warning("[sweep] Model %s failed: %s", model, err)
            result = ModelSweepResult(
                model=model,
                fixtures_run=0,
                overall_quality=None,
                avg_latency_ms=0.0,
                total_tokens=0,
                total_cost_usd=0.0,
                value_score=None,
                eval_summary=None,
                error=err,
            )

        results.append(result)
        if on_model_done is not None:
            on_model_done(result)

    return results
