"""Tests for typed secure-code-review verdicts."""

from __future__ import annotations

import httpx
import pytest

from atomics.eval.codereview import SECURE_CODE_FIXTURES
from atomics.eval.codereview.scorer import (
    ReviewVerdictResult,
    judge_review,
    verdict_to_judge_outcome,
)
from atomics.eval.outcomes import JudgeOutcomeStatus
from atomics.providers.base import ProviderResponse


class _Judge:
    name = "judge"
    default_model = "judge-default"

    def __init__(self, text: str) -> None:
        self.text = text

    async def generate(self, *_args, **_kwargs) -> ProviderResponse:
        return ProviderResponse(
            text=self.text,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            model="judge-model",
            latency_ms=15.0,
            estimated_cost_usd=0.01,
        )


class _FailingJudge:
    name = "judge"
    default_model = "judge-default"

    async def generate(self, *_args, **_kwargs) -> ProviderResponse:
        raise httpx.ReadTimeout("judge timed out")


def _result(verdict: str, status: JudgeOutcomeStatus) -> ReviewVerdictResult:
    return ReviewVerdictResult(
        verdict=verdict,
        rationale="graded",
        judge_model="judge",
        status=status,
        calls=(),
    )


@pytest.mark.parametrize(
    ("verdict", "score"),
    [("detected", 1.0), ("clean", 1.0), ("missed", 0.0), ("false_positive", 0.0)],
)
def test_verdict_to_outcome_maps_domain_score(verdict: str, score: float) -> None:
    outcome = verdict_to_judge_outcome(
        _result(verdict, JudgeOutcomeStatus.SCORED)
    )

    assert outcome.status is JudgeOutcomeStatus.SCORED
    assert outcome.score == score
    assert outcome.label == verdict


def test_unknown_verdict_is_not_scored() -> None:
    outcome = verdict_to_judge_outcome(
        _result("unknown", JudgeOutcomeStatus.PARSE_FAILED)
    )

    assert outcome.status is JudgeOutcomeStatus.PARSE_FAILED
    assert outcome.score is None


@pytest.mark.asyncio
async def test_unparseable_judge_text_is_not_clean() -> None:
    clean = next(f for f in SECURE_CODE_FIXTURES if not f.is_vulnerable)

    result = await judge_review(
        clean,
        "review text",
        judge_provider=_Judge("The code has some interesting properties."),
        judge_model="judge-model",
    )

    assert result.status is JudgeOutcomeStatus.PARSE_FAILED
    assert result.verdict == "unknown"
    assert result.calls[-1].status is JudgeOutcomeStatus.PARSE_FAILED


@pytest.mark.asyncio
async def test_judge_review_retains_call_evidence() -> None:
    vulnerable = next(f for f in SECURE_CODE_FIXTURES if f.is_vulnerable)

    result = await judge_review(
        vulnerable,
        "review text",
        judge_provider=_Judge("VERDICT: DETECTED\nRATIONALE: found root cause"),
        judge_model="judge-model",
    )

    assert result.status is JudgeOutcomeStatus.SCORED
    assert result.verdict == "detected"
    assert result.rationale == "found root cause"
    assert result.calls[0].score == 1.0
    assert result.calls[0].input_tokens == 10


@pytest.mark.asyncio
async def test_judge_review_retains_provider_error() -> None:
    fixture = SECURE_CODE_FIXTURES[0]

    result = await judge_review(
        fixture,
        "review text",
        judge_provider=_FailingJudge(),
        judge_model="judge-model",
    )

    assert result.status is JudgeOutcomeStatus.PROVIDER_ERROR
    assert result.verdict == "unknown"
    assert result.calls[0].error_class == "ReadTimeout"
