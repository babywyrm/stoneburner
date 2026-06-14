"""Judge calibration regression tests.

The deterministic tests inject a scoring function so they exercise the
calibration harness's ranking logic without a live model. The live test
(opt-in via ATOMICS_LIVE_JUDGE) runs the real quality judge against a local
Ollama and asserts it ranks graded answers correctly.
"""

from __future__ import annotations

import os

import pytest

from atomics.eval.calibration import (
    CALIBRATION_CASES,
    CalibrationCase,
    calibrate_judge,
)
from atomics.eval.judge import JudgeResult


def _score_fn_from_map(score_map: dict[str, float]):
    """Build a score_fn that returns a fixed score keyed by the answer text."""
    async def _fn(prompt, response, *, judge_provider=None, judge_model=None,
                  gold_criteria=None, max_response_chars=3000):
        return JudgeResult(
            score=score_map[response], accuracy=0, completeness=0, format_score=0,
            rationale="", judge_model="fake",
        )
    return _fn


def _ordered_map(case: CalibrationCase, scores: list[float]) -> dict[str, float]:
    return dict(zip(case.graded_answers, scores))


@pytest.mark.asyncio
async def test_calibration_passes_when_ranking_is_correct():
    case = CALIBRATION_CASES[0]
    score_fn = _score_fn_from_map(_ordered_map(case, [0.2, 0.6, 0.9]))
    report = await calibrate_judge(None, cases=[case], score_fn=score_fn)
    assert report.passed is True
    assert report.results[0].monotonic is True
    assert report.results[0].separated is True


@pytest.mark.asyncio
async def test_calibration_detects_inverted_ranking():
    """A judge that ranks best-as-worst must fail calibration."""
    case = CALIBRATION_CASES[0]
    score_fn = _score_fn_from_map(_ordered_map(case, [0.9, 0.6, 0.2]))
    report = await calibrate_judge(None, cases=[case], score_fn=score_fn)
    assert report.passed is False
    assert report.results[0].monotonic is False


@pytest.mark.asyncio
async def test_calibration_detects_insufficient_separation():
    """Flat scoring is monotonic but not separated → not calibrated."""
    case = CALIBRATION_CASES[0]
    score_fn = _score_fn_from_map(_ordered_map(case, [0.5, 0.5, 0.5]))
    report = await calibrate_judge(None, cases=[case], score_fn=score_fn)
    assert report.results[0].monotonic is True
    assert report.results[0].separated is False
    assert report.passed is False


@pytest.mark.asyncio
async def test_calibration_pass_rate_partial():
    good, bad = CALIBRATION_CASES[0], CALIBRATION_CASES[1]
    score_map = {
        **_ordered_map(good, [0.2, 0.6, 0.9]),   # calibrated
        **_ordered_map(bad, [0.8, 0.5, 0.3]),    # inverted
    }
    report = await calibrate_judge(
        None, cases=[good, bad], score_fn=_score_fn_from_map(score_map),
    )
    assert report.pass_rate == 0.5
    assert report.passed is False


@pytest.mark.skipif(
    not os.getenv("ATOMICS_LIVE_JUDGE"),
    reason="set ATOMICS_LIVE_JUDGE=1 to run live judge calibration against Ollama",
)
@pytest.mark.asyncio
async def test_live_judge_is_calibrated():
    from atomics.config import load_settings
    from atomics.providers.ollama import OllamaProvider

    settings = load_settings()
    judge = OllamaProvider(host=settings.ollama_host, default_model=settings.ollama_model)
    report = await calibrate_judge(judge)
    assert report.passed, [(r.case_id, r.scores) for r in report.results]
