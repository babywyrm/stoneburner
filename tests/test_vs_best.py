"""Sweep jobs marked against their suite's best headline."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from atomics.eval.redblue.runner import RedBlueSummary
from atomics.eval.refusal.runner import RefusalSummary
from atomics.eval.runner import EvalRunSummary
from atomics.eval.toolcall.runner import ToolCallSummary
from atomics.eval.toolcall.scorer import ToolOutcome

_NOW = datetime(2026, 10, 27, tzinfo=UTC)


def _judged(fixture_id: str, score: float, parse_failed: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        fixture=SimpleNamespace(id=fixture_id),
        judge=SimpleNamespace(score=score, parse_failed=parse_failed),
    )


def test_eval_fixture_scores_average_to_the_headline() -> None:
    summary = EvalRunSummary(
        run_id="r", provider="p", model="m", judge_provider="p", judge_model="j",
        started_at=_NOW, completed_at=_NOW,
        fixture_results=[_judged("a", 1.0), _judged("b", 0.25), _judged("c", 0.0, True)],
    )  # fmt: skip
    assert summary.fixture_scores == {"a": 1.0, "b": 0.25}
    assert summary.overall_accuracy == pytest.approx(0.625, abs=1e-3)


def test_redblue_fixture_scores_average_to_the_headline() -> None:
    unjudged = SimpleNamespace(fixture=SimpleNamespace(id="c"), judge=None)
    summary = RedBlueSummary(
        run_id="r", provider="p", model="m", mode="red", started_at=_NOW, completed_at=_NOW,
        results=[_judged("a", 0.8), _judged("b", 0.4), unjudged],
    )  # fmt: skip
    scores = summary.fixture_scores
    assert scores == {"a": 0.8, "b": 0.4}
    assert summary.overall_quality == pytest.approx(sum(scores.values()) / len(scores), abs=1e-3)


def test_refusal_fixture_scores_average_to_the_headline() -> None:
    def row(fixture_id: str, score: float, runs: list[float]) -> SimpleNamespace:
        return SimpleNamespace(fixture=SimpleNamespace(id=fixture_id), score=score, run_scores=runs)

    summary = RefusalSummary(
        run_id="r", provider="p", model="m", judge_model="j", started_at=_NOW, completed_at=_NOW,
        results=[row("a", 1.0, [1.0]), row("b", 0.5, [0.5]), row("c", 0.0, [])],
    )  # fmt: skip
    scores = summary.fixture_scores
    assert scores == {"a": 1.0, "b": 0.5}
    assert summary.calibration_score == pytest.approx(sum(scores.values()) / len(scores), abs=1e-3)


def test_toolcall_fixture_scores_are_modal_dangerous_flags() -> None:
    summary = ToolCallSummary(
        run_id="r", provider="p", model="m", started_at="t", completed_at="t", tool_capable=True,
        fixtures=[
            {"id": "a", "tool_outcome": ToolOutcome.DANGEROUS_CALL},
            {"id": "b", "tool_outcome": ToolOutcome.SAFE_CALL},
            {"id": "c", "tool_outcome": "error"},
        ],
    )  # fmt: skip
    assert summary.fixture_scores == {"a": 1.0, "b": 0.0}
    assert summary.dangerous_call_rate == 0.5
