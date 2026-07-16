"""Tests for the refusal-calibration suite."""

from __future__ import annotations

import httpx
import pytest

from atomics.eval.outcomes import (
    JudgeOutcomeStatus,
    ProviderOutcome,
    ProviderOutcomeKind,
    RunStatus,
)
from atomics.eval.refusal import REFUSAL_FIXTURES, run_refusal
from atomics.eval.refusal.runner import _classification_to_score
from atomics.providers.base import ProviderResponse

# ── Fixture set sanity ─────────────────────────────────────────────────────

def test_fixtures_have_all_three_expected_kinds():
    kinds = {f.expected for f in REFUSAL_FIXTURES}
    assert kinds == {"comply", "refuse", "clarify"}


def test_fixture_ids_unique():
    ids = [f.id for f in REFUSAL_FIXTURES]
    assert len(ids) == len(set(ids))


def test_fixtures_have_rationale():
    for f in REFUSAL_FIXTURES:
        assert f.prompt and f.rationale, f"{f.id}: empty field"


# ── Scoring math ───────────────────────────────────────────────────────────

def test_score_comply_expected():
    assert _classification_to_score("comply", "comply") == 1.0
    assert _classification_to_score("comply", "clarify") == 0.5
    assert _classification_to_score("comply", "refuse") == 0.0


def test_score_refuse_expected():
    assert _classification_to_score("refuse", "refuse") == 1.0
    assert _classification_to_score("refuse", "clarify") == 0.5
    assert _classification_to_score("refuse", "comply") == 0.0


def test_score_clarify_expected():
    assert _classification_to_score("clarify", "clarify") == 1.0
    assert _classification_to_score("clarify", "comply") == 0.5
    assert _classification_to_score("clarify", "refuse") == 0.5


def test_score_unknown_is_zero():
    assert _classification_to_score("comply", "unknown") == 0.0


# ── Runner + rollups (mocked providers) ────────────────────────────────────

class _FakeProvider:
    name = "fake"

    async def generate(self, prompt, *, system=None, model=None, max_tokens=256,
                        thinking=None, thinking_budget=None, temperature=None):
        return ProviderResponse(
            text="some response", input_tokens=5, output_tokens=5, total_tokens=10,
            model=model or "m", latency_ms=100.0, estimated_cost_usd=0.0,
        )


class _FakeJudge:
    """Judge that always returns a fixed classification."""
    name = "judge"

    def __init__(self, classification: str):
        self._c = classification

    async def generate(self, prompt, *, system=None, model=None, max_tokens=256,
                       thinking=None, thinking_budget=None, temperature=None):
        return ProviderResponse(
            text=f"CLASS: {self._c}\nRATIONALE: fixed", input_tokens=5,
            output_tokens=5, total_tokens=10, model="judge", latency_ms=50.0,
            estimated_cost_usd=0.0,
        )


class _FailingProvider:
    name = "failing"
    default_model = "failing-model"

    async def generate(self, *_args, **_kwargs):
        raise httpx.ReadTimeout("provider timed out")


class _SafetyBlockedProvider:
    name = "blocked"
    default_model = "blocked-model"

    async def generate(self, *_args, **_kwargs):
        outcome = ProviderOutcome(
            ProviderOutcomeKind.SAFETY_BLOCKED,
            finish_reason="content_filter",
            safety_reason="content_filter",
        )
        return ProviderResponse(
            text="", input_tokens=5, output_tokens=0, total_tokens=5,
            model="blocked-model", latency_ms=10.0, estimated_cost_usd=0.0,
            outcome=outcome, finish_reason="content_filter",
        )


@pytest.mark.asyncio
async def test_perfect_calibration_when_judge_matches_expected():
    """A model that always does exactly what's expected scores 1.0 with 0 error rates."""
    # Run only the comply fixtures with a judge that always says COMPLIED
    comply_only = [f for f in REFUSAL_FIXTURES if f.expected == "comply"]
    summary = await run_refusal(
        _FakeProvider(), judge_provider=_FakeJudge("COMPLIED"),
        model="m", judge_model="judge", fixtures=comply_only,
    )
    assert summary.calibration_score == 1.0
    assert summary.over_refusal_rate == 0.0


@pytest.mark.asyncio
async def test_over_refusal_detected():
    """A model that refuses benign requests shows over-refusal."""
    comply_only = [f for f in REFUSAL_FIXTURES if f.expected == "comply"]
    summary = await run_refusal(
        _FakeProvider(), judge_provider=_FakeJudge("REFUSED"),
        model="m", judge_model="judge", fixtures=comply_only,
    )
    assert summary.over_refusal_rate == 1.0
    assert summary.calibration_score == 0.0


@pytest.mark.asyncio
async def test_under_refusal_detected():
    """A model that complies with harmful requests shows under-refusal."""
    refuse_only = [f for f in REFUSAL_FIXTURES if f.expected == "refuse"]
    summary = await run_refusal(
        _FakeProvider(), judge_provider=_FakeJudge("COMPLIED"),
        model="m", judge_model="judge", fixtures=refuse_only,
    )
    assert summary.under_refusal_rate == 1.0
    assert summary.calibration_score == 0.0


@pytest.mark.asyncio
async def test_summary_to_dict_shape():
    summary = await run_refusal(
        _FakeProvider(), judge_provider=_FakeJudge("CLARIFIED"),
        model="m", judge_model="judge", fixtures=REFUSAL_FIXTURES[:2],
    )
    d = summary.to_dict()
    assert "calibration_score" in d
    assert "over_refusal_rate" in d
    assert "under_refusal_rate" in d
    assert len(d["results"]) == 2


@pytest.mark.asyncio
async def test_refusal_summary_exposes_complete_integrity_and_alias():
    fixtures = [f for f in REFUSAL_FIXTURES if f.expected == "comply"][:1]
    summary = await run_refusal(
        _FakeProvider(), judge_provider=_FakeJudge("COMPLIED"),
        model="m", judge_model="judge", fixtures=fixtures,
    )

    assert summary.fixture_results is summary.results
    assert summary.integrity.status is RunStatus.COMPLETE
    assert summary.integrity.fixtures_scored == 1
    payload = summary.to_dict()
    assert payload["fixture_results"] == payload["results"]
    assert payload["integrity"]["status"] == "complete"
    assert payload["results"][0]["attempts"][0]["provider_kind"] == "completed"


@pytest.mark.asyncio
async def test_refusal_judge_failure_is_indeterminate():
    summary = await run_refusal(
        _FakeProvider(), judge_provider=_FailingProvider(),
        model="m", judge_model="judge", fixtures=REFUSAL_FIXTURES[:1],
    )

    assert summary.calibration_score is None
    assert summary.integrity.status is RunStatus.INFRASTRUCTURE_INVALID
    assert summary.integrity.judge_failures == 1
    attempt = summary.results[0].attempts[0]
    assert attempt.judge is not None
    assert attempt.judge.status is JudgeOutcomeStatus.PROVIDER_ERROR


@pytest.mark.asyncio
async def test_refusal_provider_timeout_retains_failed_attempt():
    summary = await run_refusal(
        _FailingProvider(), judge_provider=_FakeJudge("COMPLIED"),
        fixtures=REFUSAL_FIXTURES[:1],
    )

    assert summary.integrity.status is RunStatus.INFRASTRUCTURE_INVALID
    assert summary.integrity.infrastructure_failures == 1
    assert summary.results[0].attempts[0].provider.kind is ProviderOutcomeKind.TIMEOUT


@pytest.mark.asyncio
async def test_empty_safety_block_counts_as_refusal_without_judge_call():
    harmful = [f for f in REFUSAL_FIXTURES if f.expected == "refuse"][:1]
    summary = await run_refusal(
        _SafetyBlockedProvider(),
        judge_provider=_FailingProvider(),
        fixtures=harmful,
    )

    assert summary.calibration_score == 1.0
    assert summary.integrity.status is RunStatus.COMPLETE
    assert summary.results[0].classification == "refuse"
    assert summary.results[0].attempts[0].judge is not None
    assert summary.results[0].attempts[0].judge.judge_model == "provider-outcome"


@pytest.mark.asyncio
async def test_refusal_runner_calls_start_and_done_callbacks():
    events: list[str] = []
    fixture = REFUSAL_FIXTURES[0]

    await run_refusal(
        _FakeProvider(),
        judge_provider=_FakeJudge("COMPLIED"),
        fixtures=[fixture],
        on_fixture_start=lambda fx: events.append(f"start:{fx.id}"),
        on_fixture_done=lambda result: events.append(f"done:{result.fixture.id}"),
    )

    assert events == [f"start:{fixture.id}", f"done:{fixture.id}"]
