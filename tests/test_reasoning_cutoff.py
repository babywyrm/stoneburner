"""A reply cut off during hidden reasoning is not judged, in any suite.

gpt-oss spent 761 of 768 tokens reasoning and left a heading. Judges graded
the heading. Each suite now records `thinking_budget` and skips the judge.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from atomics.models import TaskStatus
from atomics.providers.base import ProviderResponse
from atomics.providers.outcomes import ProviderOutcome, ProviderOutcomeKind


def _cutoff() -> ProviderResponse:
    return ProviderResponse(
        text="**Plan**",
        input_tokens=40,
        output_tokens=768,
        total_tokens=808,
        model="cutoff",
        latency_ms=1.0,
        estimated_cost_usd=0.0,
        thinking_tokens=760,
        outcome=ProviderOutcome(ProviderOutcomeKind.THINKING_BUDGET, finish_reason="length"),
    )


def _provider() -> AsyncMock:
    provider = AsyncMock()
    provider.name = "ollama"
    provider.default_model = "cutoff"
    provider.generate = AsyncMock(return_value=_cutoff())
    return provider


def _judge() -> AsyncMock:
    judge = AsyncMock()
    judge.name = "judge"
    judge.default_model = "judge"
    return judge


@pytest.mark.asyncio
async def test_eval_skips_the_judge() -> None:
    from atomics.eval.fixtures import EVAL_FIXTURES
    from atomics.eval.runner import run_eval

    judge = _judge()
    summary = await run_eval(
        _provider(), judge_provider=judge, fixtures=EVAL_FIXTURES[:1], quiet=True
    )

    judge.generate.assert_not_called()
    task = summary.fixture_results[0].task_result
    assert task.status is TaskStatus.FAILED
    assert task.error_class == "thinking_budget"
    assert summary.overall_accuracy is None


@pytest.mark.asyncio
async def test_rag_skips_the_judge() -> None:
    from atomics.eval.rag.fixtures import ALL_RAG_FIXTURES
    from atomics.eval.rag.runner import run_rag

    judge = _judge()
    summary = await run_rag(_provider(), judge_provider=judge, fixtures=ALL_RAG_FIXTURES[:1])

    judge.generate.assert_not_called()
    assert summary.fixture_results[0].task_result.error_class == "thinking_budget"
    assert summary.integrity.infrastructure_failures == 0
    assert summary.integrity.generation_failures == 1


@pytest.mark.asyncio
async def test_codegen_does_not_run_tests_on_a_fragment() -> None:
    from atomics.eval.codegen.fixtures import ALL_CODEGEN_FIXTURES
    from atomics.eval.codegen.runner import run_codegen

    summary = await run_codegen(_provider(), fixtures=ALL_CODEGEN_FIXTURES[:1])

    assert summary.fixture_results[0].task_result.error_class == "thinking_budget"
    assert summary.integrity.infrastructure_failures == 0
    assert summary.integrity.generation_failures == 1


@pytest.mark.asyncio
async def test_multiturn_stops_the_conversation_unjudged() -> None:
    from atomics.eval.multiturn.fixtures import ALL_MULTITURN_FIXTURES
    from atomics.eval.multiturn.runner import run_multiturn

    judge = _judge()
    summary = await run_multiturn(
        _provider(), judge_provider=judge, fixtures=ALL_MULTITURN_FIXTURES[:1]
    )

    judge.generate.assert_not_called()
    assert summary.conversation_results[0].task_result.error_class == "thinking_budget"
    assert summary.integrity.infrastructure_failures == 0
    assert summary.integrity.generation_failures == 1


@pytest.mark.asyncio
async def test_agreement_study_casts_no_votes_on_a_fragment() -> None:
    from atomics.eval.agreement import run_agreement_study

    first, second = _judge(), _judge()
    summary = await run_agreement_study(
        suite="redblue",
        provider=_provider(),
        judges=[(first, None), (second, None)],
        fixture_ids=["rb-r01"],
    )

    first.generate.assert_not_called()
    second.generate.assert_not_called()
    assert summary.fixtures[0].votes == []
