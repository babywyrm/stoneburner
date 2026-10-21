"""Judge calibration against a human-graded set: math, loader, scoring."""

from __future__ import annotations

from pathlib import Path

import pytest

from atomics.eval.calibration import (
    CalibrationStatus,
    GradedItem,
    GradedSet,
    ItemCalibration,
    JudgeCalibration,
    _status_from_rows,
    calibration_label,
    default_set_path,
    judge_calibration_status,
    load_graded_set,
    mean_abs_error,
    mean_bias,
    pair_order_agreement,
    run_judge_calibration,
    save_calibration,
)
from atomics.eval.judge import JudgeResult
from atomics.eval.outcomes import RunStatus
from atomics.storage import MetricsRepository

PAIRS = [(0.3, 0.3), (0.8, 0.6), (0.5, 0.7)]  # (human, judge)


def test_error_bias_and_order_math():
    assert mean_abs_error(PAIRS) == pytest.approx(0.133, abs=1e-3)
    assert mean_bias(PAIRS) == pytest.approx(0.0, abs=1e-3)
    # The judge flips the 0.5 / 0.8 pair and keeps the other two.
    assert pair_order_agreement(PAIRS) == pytest.approx(2 / 3, abs=1e-3)


def test_math_on_nothing_is_none():
    assert mean_abs_error([]) is None
    assert mean_bias([]) is None
    assert pair_order_agreement([]) is None


def test_pair_order_skips_human_ties_and_counts_judge_ties_as_misses():
    assert pair_order_agreement([(0.5, 0.1), (0.5, 0.9)]) is None
    assert pair_order_agreement([(0.2, 0.5), (0.8, 0.5)]) == 0.0


GOOD = """
name: t
items:
  - id: A
    task: do x
    answer: did x
    gold_criteria: [x]
    human: {accuracy: 4, completeness: 3, format: 3}
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "set.yaml"
    path.write_text(body)
    return path


def test_loader_normalises_human_grade(tmp_path):
    graded = load_graded_set(_write(tmp_path, GOOD))
    assert graded.name == "t"
    assert len(graded.digest) == 12
    assert graded.items[0].human == 1.0
    assert graded.items[0].gold_criteria == ("x",)


def test_loader_digest_tracks_content(tmp_path):
    first = load_graded_set(_write(tmp_path, GOOD)).digest
    second = load_graded_set(_write(tmp_path, GOOD.replace("did x", "did y"))).digest
    assert first != second


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("items: []", "non-empty items"),
        ("- 1", "non-empty items"),
        ("items: [1]", "item 1: not a mapping"),
        (GOOD.replace("    answer: did x\n", ""), "missing answer"),
        (GOOD.replace("accuracy: 4", "accuracy: 5"), "accuracy must be an integer 0-4"),
        (GOOD.replace("format: 3", "format: true"), "format must be an integer 0-3"),
        (GOOD.replace("human: {", "human: [").replace("3}", "3]"), "human must be a mapping"),
        (GOOD.replace("[x]", "[1]"), "gold_criteria must be a list of strings"),
        (GOOD + GOOD.split("items:\n")[1], "duplicate id A"),
    ],
)
def test_loader_rejects_bad_sets(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        load_graded_set(_write(tmp_path, body))


SET = GradedSet(
    name="t",
    digest="abc",
    items=(GradedItem("A", "t", "a", 0.3), GradedItem("B", "t", "b", 0.8)),
)


def _scores(*values: float | None):
    it = iter(values)

    async def score_fn(prompt, response, **kwargs):
        assert kwargs["max_response_chars"] > len(response)
        value = next(it)
        return JudgeResult(
            score=0.5 if value is None else value,
            accuracy=0,
            completeness=0,
            format_score=0,
            rationale="",
            judge_model="j",
            parse_failed=value is None,
        )

    return score_fn


async def _run(*values: float | None, max_error: float = 0.15):
    return await run_judge_calibration(
        object(),
        provider_name="ollama",
        judge_model="j",
        graded=SET,
        max_error=max_error,
        score_fn=_scores(*values),
    )


@pytest.mark.asyncio
async def test_passes_at_threshold_and_fails_just_above():
    assert (await _run(0.45, 0.8)).passed
    assert (await _run(0.6, 0.8)).passed
    assert not (await _run(0.61, 0.8)).passed


@pytest.mark.asyncio
async def test_bias_sign_is_judge_minus_human():
    result = await _run(0.5, 1.0)
    assert result.bias == pytest.approx(0.2)
    assert result.to_dict()["bias"] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_unscored_item_makes_run_partial_and_never_passes():
    result = await _run(0.3, None)
    assert result.integrity.status is RunStatus.PARTIAL
    assert result.mae == 0.0
    assert not result.passed
    assert result.to_dict()["items"][1] == {"id": "B", "human": 0.8, "judge": None}


# Judge scores the 2026-10-03 study recorded for the shipped blue items.
STUDY = {
    "gemma4:26b": {"G01": 0.3, "G05": 0.9, "G07": 0.7, "G09": 0.6, "G11": 0.2, "G12": 0.3},
    "granite4.2:8b": {"G01": 0.6, "G05": 1.0, "G07": 0.8, "G09": 0.9, "G11": 0.7, "G12": 0.6},
    "qwen3.8:27b": {"G01": 0.0, "G05": 0.9, "G07": 0.8, "G09": 0.7, "G11": 0.1, "G12": 0.1},
}


def test_shipped_set_loads_the_six_blue_items():
    graded = load_graded_set(default_set_path())
    assert [i.id for i in graded.items] == ["G01", "G05", "G07", "G09", "G11", "G12"]
    assert all(i.gold_criteria and i.answer for i in graded.items)


@pytest.mark.parametrize(
    ("judge", "mae", "bias", "passes"),
    [
        ("gemma4:26b", 0.133, -0.067, True),
        ("granite4.2:8b", 0.2, 0.2, False),
        ("qwen3.8:27b", 0.233, -0.133, False),
    ],
)
@pytest.mark.asyncio
async def test_study_replay_matches_the_human_grade(judge, mae, bias, passes):
    graded = load_graded_set(default_set_path())
    result = await run_judge_calibration(
        object(),
        provider_name="ollama",
        judge_model=judge,
        graded=graded,
        score_fn=_scores(*(STUDY[judge][i.id] for i in graded.items)),
    )
    assert (result.mae, result.bias, result.passed) == (mae, bias, passes)


def _calibration(model: str = "j", judge: float = 0.3) -> JudgeCalibration:
    return JudgeCalibration(
        set_name="t",
        set_digest="abc",
        judge_provider="ollama",
        judge_model=model,
        max_error=0.15,
        items=[ItemCalibration("A", 0.3, judge), ItemCalibration("B", 0.8, 0.8)],
    )


def test_lookup_is_none_and_creates_nothing_when_db_missing(tmp_path):
    db = tmp_path / "nope.db"
    assert judge_calibration_status(db, "ollama", "j") is None
    assert not db.exists()


def test_save_then_lookup_newest_wins_per_judge(tmp_path):
    db = tmp_path / "m.db"
    repo = MetricsRepository(db)
    save_calibration(repo, "r1", _calibration(judge=0.9))
    save_calibration(repo, "r2", _calibration(judge=0.3))
    save_calibration(repo, "r3", _calibration(model="other", judge=0.9))
    repo.close()

    status = judge_calibration_status(db, "ollama", "j")
    assert status is not None
    assert (status.passed, status.mae, status.bias, status.n) == (True, 0.0, 0.0, 2)
    assert status.set_name == "t"
    other = judge_calibration_status(db, "ollama", "other")
    assert other is not None and not other.passed and other.mae == 0.3
    assert judge_calibration_status(db, "ollama", "missing") is None
    assert judge_calibration_status(db, "claude", "j") is None


def test_saved_calibration_is_a_completed_run(tmp_path):
    repo = MetricsRepository(tmp_path / "m.db")
    save_calibration(repo, "r1", _calibration())
    run = repo.get_run("r1")
    rows = repo.get_evaluation_results(run_id="r1")
    repo.close()
    assert run is not None and run["completed_at"] and run["model"] == "j"
    assert [r["fixture_id"] for r in rows] == ["A", "B"]
    assert rows[0]["result_json"]["human"] == 0.3


def test_lookup_swallows_an_unreadable_db(tmp_path):
    db = tmp_path / "bad.db"
    db.write_bytes(b"not sqlite at all, just bytes" * 10)
    assert judge_calibration_status(db, "ollama", "j") is None


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "no rows"),
        ([{"result_json": None, "score": 0.3}], "no result_json"),
        ([{"result_json": {"human": "0.3", "max_error": 0.15}, "score": 0.3}], "got str"),
        ([{"result_json": {"human": True, "max_error": 0.15}, "score": 0.3}], "got bool"),
    ],
)
def test_corrupt_stored_rows_are_rejected(rows, message):
    with pytest.raises(ValueError, match=message):
        _status_from_rows(rows)


def test_labels_for_each_state():
    assert calibration_label(None) == "uncalibrated"
    passed = CalibrationStatus(True, 0.133, -0.067, 6, "t", 0.15, "2026-10-20T01:02:03")
    assert calibration_label(passed) == "calibrated · error 0.13 · bias -0.07 · t · 2026-10-20"
    failed = CalibrationStatus(False, 0.2, 0.2, 6, "t", 0.15, "2026-10-20T01:02:03")
    assert failed.label() == "failed calibration · error 0.20 · bias +0.20 · t · 2026-10-20"
