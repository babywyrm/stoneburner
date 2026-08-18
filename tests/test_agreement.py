"""Tests for judge-agreement math and the refusal study loop."""

from __future__ import annotations

import asyncio

import pytest

from atomics.eval.adversarial.scorer import _label_from_score
from atomics.eval.agreement import (
    load_study_fixtures,
    majority_flip,
    numeric_flip,
    pairwise_agreement,
    run_agreement_study,
)
from atomics.eval.outcomes import JudgeOutcomeStatus
from atomics.eval.refusal.fixtures import REFUSAL_FIXTURES
from atomics.eval.refusal.scorer import ClassificationResult
from atomics.providers.base import ProviderResponse


def test_pairwise_agreement_all_match() -> None:
    assert pairwise_agreement(["refuse", "refuse", "refuse"]) == 1.0


def test_pairwise_agreement_two_of_three() -> None:
    assert pairwise_agreement(["refuse", "refuse", "comply"]) == pytest.approx(1 / 3)


def test_pairwise_agreement_single_label_is_none() -> None:
    assert pairwise_agreement(["refuse"]) is None


def test_flip_when_primary_loses_majority() -> None:
    assert majority_flip(primary="comply", majority="refuse") is True
    assert majority_flip(primary="refuse", majority="refuse") is False
    assert majority_flip(primary="refuse", majority=None) is False


def test_numeric_flip_rounds_to_three_decimals() -> None:
    assert numeric_flip(0.701, 0.7014) is False
    assert numeric_flip(0.701, 0.704) is True


def test_numeric_flip_uses_label_fn() -> None:
    assert numeric_flip(1.0, 0.4, label_fn=_label_from_score) is True
    assert numeric_flip(1.0, 0.8, label_fn=_label_from_score) is False


class _Named:
    def __init__(self, name: str) -> None:
        self.name = name
        self.default_model = name
        self.generates = 0

    async def generate(self, *_args, **_kwargs) -> ProviderResponse:
        self.generates += 1
        return ProviderResponse(
            text="model response",
            input_tokens=5,
            output_tokens=5,
            total_tokens=10,
            model=self.name,
            latency_ms=10.0,
            estimated_cost_usd=0.01,
        )


@pytest.mark.asyncio
async def test_refusal_study_generates_once_and_classifies_n_times(monkeypatch):
    classifies: list[str] = []

    async def fake_classify(prompt, text, *, expected, judge_provider, judge_model):
        classifies.append(judge_provider.name)
        return ClassificationResult(
            classification="refuse" if judge_provider.name == "j1" else "comply",
            rationale="ok",
            judge_model=judge_provider.name,
            status=JudgeOutcomeStatus.SCORED,
            calls=(),
        )

    monkeypatch.setattr("atomics.eval.agreement.classify_response", fake_classify)

    provider = _Named("model")
    j1 = _Named("j1")
    j2 = _Named("j2")
    fixture = [f for f in REFUSAL_FIXTURES if f.expected == "refuse"][:1]
    summary = await run_agreement_study(
        suite="refusal",
        provider=provider,
        model="m",
        judges=[(j1, None), (j2, None)],
        fixture_ids=[fixture[0].id],
    )

    assert provider.generates == 1
    assert classifies == ["j1", "j2"]
    assert summary.n_unresolved == 1
    assert summary.n_flipped == 0
    assert summary.fixtures[0].unresolved is True
    assert summary.fixtures[0].flipped is False
    assert summary.fixtures[0].agreement == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_refusal_study_majority_flip(monkeypatch):
    async def fake_classify(prompt, text, *, expected, judge_provider, judge_model):
        label = "comply" if judge_provider.name == "primary" else "refuse"
        return ClassificationResult(
            classification=label,
            rationale="ok",
            judge_model=judge_provider.name,
            status=JudgeOutcomeStatus.SCORED,
            calls=(),
        )

    monkeypatch.setattr("atomics.eval.agreement.classify_response", fake_classify)

    fixture = [f for f in REFUSAL_FIXTURES if f.expected == "refuse"][:1]
    summary = await run_agreement_study(
        suite="refusal",
        provider=_Named("model"),
        judges=[(_Named("primary"), None), (_Named("a"), None), (_Named("b"), None)],
        fixture_ids=[fixture[0].id],
    )

    assert summary.n_flipped == 1
    assert summary.flip_rate == 1.0
    assert summary.fixtures[0].combined_label == "refuse"
    assert summary.fixtures[0].flipped is True


def test_study_rejects_unknown_suite_and_one_judge() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        asyncio.run(
            run_agreement_study(
                suite="probe",
                provider=_Named("model"),
                judges=[(_Named("a"), None), (_Named("b"), None)],
            )
        )
    with pytest.raises(ValueError, match="at least two"):
        asyncio.run(
            run_agreement_study(
                suite="refusal",
                provider=_Named("model"),
                judges=[(_Named("a"), None)],
            )
        )


def test_load_study_fixtures_rejects_unknown_ids() -> None:
    with pytest.raises(ValueError, match="Unknown fixture"):
        load_study_fixtures("refusal", ["no-such-fixture"])


def test_load_study_fixtures_skips_tool_channel_only() -> None:
    fixtures = load_study_fixtures("toolcall", None)
    assert fixtures
    assert all(not getattr(fx, "tool_channel_only", False) for fx in fixtures)


@pytest.mark.asyncio
async def test_generate_failure_is_not_a_flip():
    class _Boom(_Named):
        async def generate(self, *_args, **_kwargs):
            raise RuntimeError("provider down")

    fixture = [f for f in REFUSAL_FIXTURES if f.expected == "refuse"][:1]
    summary = await run_agreement_study(
        suite="refusal",
        provider=_Boom("model"),
        judges=[(_Named("j1"), None), (_Named("j2"), None)],
        fixture_ids=[fixture[0].id],
    )
    assert summary.fixtures[0].votes == []
    assert summary.fixtures[0].flipped is False
    assert summary.n_flipped == 0


@pytest.mark.asyncio
async def test_codereview_study_majority(monkeypatch):
    from atomics.eval.codereview.fixtures import SECURE_CODE_FIXTURES
    from atomics.eval.codereview.scorer import ReviewVerdictResult

    async def fake_review(fixture, text, *, judge_provider, judge_model):
        verdict = "detected" if judge_provider.name != "miss" else "missed"
        return ReviewVerdictResult(
            verdict=verdict,
            rationale="ok",
            judge_model=judge_provider.name,
            status=JudgeOutcomeStatus.SCORED,
            calls=(),
        )

    monkeypatch.setattr("atomics.eval.agreement.judge_review", fake_review)
    fixture = [f for f in SECURE_CODE_FIXTURES if f.is_vulnerable][:1]
    summary = await run_agreement_study(
        suite="codereview",
        provider=_Named("model"),
        judges=[
            (_Named("hit"), None),
            (_Named("hit2"), None),
            (_Named("miss"), None),
        ],
        fixture_ids=[fixture[0].id],
    )
    assert summary.fixtures[0].combined_label == "detected"
    assert summary.fixtures[0].agreement == pytest.approx(2 / 3)
    assert summary.n_flipped == 0


@pytest.mark.asyncio
async def test_redblue_study_numeric_flip(monkeypatch):
    from atomics.eval.judge import JudgeResult
    from atomics.eval.redblue.fixtures import ALL_FIXTURES

    async def fake_score(prompt, text, *, judge_provider, judge_model, gold_criteria=None):
        score = 0.9 if judge_provider.name == "high" else 0.3
        return JudgeResult(
            score=score,
            accuracy=4,
            completeness=3,
            format_score=3,
            rationale="ok",
            judge_model=judge_provider.name,
        )

    monkeypatch.setattr("atomics.eval.judge.score_response", fake_score)
    summary = await run_agreement_study(
        suite="redblue",
        provider=_Named("model"),
        judges=[(_Named("high"), None), (_Named("low"), None)],
        fixture_ids=[ALL_FIXTURES[0].id],
    )
    row = summary.fixtures[0]
    assert row.combined_score == pytest.approx(0.6)
    assert row.flipped is True
    assert row.score_stdev == pytest.approx(0.3)
    assert summary.mean_stdev == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_toolcall_study_uses_resistance_labels(monkeypatch):
    from atomics.eval.adversarial.scorer import ResistanceResult
    from atomics.eval.toolcall.fixtures import ALL_FIXTURES

    async def fake_resistance(prompt, text, **kwargs):
        judge = kwargs["judge_provider"]
        score = 1.0 if judge.name == "resist" else 0.2
        return ResistanceResult(
            score=score,
            label="resisted" if score >= 0.7 else "complied",
            rationale="ok",
            judge_model=judge.name,
        )

    monkeypatch.setattr("atomics.eval.agreement.score_resistance", fake_resistance)
    prose = next(fx for fx in ALL_FIXTURES if not fx.tool_channel_only)
    summary = await run_agreement_study(
        suite="toolcall",
        provider=_Named("model"),
        judges=[(_Named("resist"), None), (_Named("comply"), None)],
        fixture_ids=[prose.id],
    )
    row = summary.fixtures[0]
    assert row.combined_label == "partial"
    assert row.flipped is True


@pytest.mark.asyncio
async def test_adversarial_study_label_flip(monkeypatch):
    from atomics.eval.adversarial import ALL_FIXTURES
    from atomics.eval.adversarial.scorer import ResistanceResult

    async def fake_resistance(prompt, text, **kwargs):
        judge = kwargs["judge_provider"]
        score = 1.0 if judge.name == "resist" else 0.2
        return ResistanceResult(
            score=score,
            label="resisted" if score >= 0.7 else "complied",
            rationale="ok",
            judge_model=judge.name,
        )

    monkeypatch.setattr("atomics.eval.agreement.score_resistance", fake_resistance)
    summary = await run_agreement_study(
        suite="adversarial",
        provider=_Named("model"),
        judges=[(_Named("resist"), None), (_Named("comply"), None)],
        fixture_ids=[ALL_FIXTURES[0].id],
    )
    assert summary.fixtures[0].combined_label == "partial"
    assert summary.fixtures[0].flipped is True


@pytest.mark.asyncio
async def test_multiturn_study_numeric_flip(monkeypatch):
    from atomics.eval.multiturn.fixtures import ALL_MULTITURN_FIXTURES
    from atomics.eval.multiturn.judge import ConversationJudgeResult

    async def fake_conversation(transcript, criteria, judge, *, judge_model=None):
        score = 0.9 if judge.name == "high" else 0.3
        return ConversationJudgeResult(4, 3, 3, score, "ok")

    monkeypatch.setattr(
        "atomics.eval.multiturn.judge.score_conversation", fake_conversation
    )
    summary = await run_agreement_study(
        suite="multiturn",
        provider=_Named("model"),
        judges=[(_Named("high"), None), (_Named("low"), None)],
        fixture_ids=[ALL_MULTITURN_FIXTURES[0].id],
    )
    row = summary.fixtures[0]
    assert row.combined_score == pytest.approx(0.6)
    assert row.flipped is True


@pytest.mark.asyncio
async def test_rag_study_numeric_flip(monkeypatch):
    from atomics.eval.rag.fixtures import ALL_RAG_FIXTURES
    from atomics.eval.rag.judge import RAGJudgeResult

    async def fake_rag(response, fixture, judge, *, judge_model=None, max_response_chars=4000):
        score = 0.9 if judge.name == "high" else 0.3
        return RAGJudgeResult(
            grounding=4 if score > 0.5 else 1,
            faithfulness=3 if score > 0.5 else 1,
            abstention=3 if score > 0.5 else 1,
            score=score,
            rationale="ok",
        )

    monkeypatch.setattr("atomics.eval.rag.judge.score_rag_response", fake_rag)
    provider = _Named("model")
    summary = await run_agreement_study(
        suite="rag",
        provider=provider,
        judges=[(_Named("high"), None), (_Named("low"), None)],
        fixture_ids=[ALL_RAG_FIXTURES[0].id],
    )
    row = summary.fixtures[0]
    assert provider.generates == 1
    assert row.combined_score == pytest.approx(0.6)
    assert row.flipped is True
