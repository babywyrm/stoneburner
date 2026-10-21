"""`atomics judge-calibrate`: exit codes, output, and what it stores."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from atomics.cli import cli
from atomics.config import load_settings
from atomics.eval.calibration import CALIBRATE_SUITE, judge_calibration_status
from atomics.providers.base import BaseProvider, ProviderResponse
from atomics.storage import MetricsRepository

SET = """
name: tiny
items:
  - id: A
    task: task a
    answer: answer-a
    human: {accuracy: 4, completeness: 3, format: 3}
  - id: B
    task: task b
    answer: answer-b
    human: {accuracy: 1, completeness: 1, format: 1}
"""


class RubricJudge(BaseProvider):
    """Replies with a fixed rubric per answer; `None` replies garbage."""

    def __init__(self, grades: dict[str, tuple[int, int, int] | None]) -> None:
        self.grades = grades
        self.calls = 0

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def default_model(self) -> str | None:
        return "judge-m"

    async def generate(self, prompt, **_kwargs):
        self.calls += 1
        reply = "no scores here"
        for answer, grade in self.grades.items():
            if answer in prompt and grade is not None:
                acc, comp, fmt = grade
                reply = f"Accuracy: {acc}\nCompleteness: {comp}\nFormat: {fmt}\nRationale: ok"
        return ProviderResponse(
            text=reply,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            model="judge-m",
            latency_ms=1.0,
            estimated_cost_usd=0.0,
        )

    async def health_check(self):
        return True


EXACT = {"answer-a": (4, 3, 3), "answer-b": (1, 1, 1)}


@pytest.fixture
def set_file(tmp_path: Path) -> Path:
    path = tmp_path / "set.yaml"
    path.write_text(SET)
    return path


def _invoke(monkeypatch, judge: RubricJudge, *args: str):
    monkeypatch.setattr("atomics.commands.calibration._make_provider", lambda *_a, **_k: judge)
    return CliRunner().invoke(cli, ["judge-calibrate", *args])


def _stored_rows() -> list[dict[str, object]]:
    db = load_settings().db_path
    if not db.exists():
        return []
    repo = MetricsRepository(db)
    try:
        return repo.get_evaluation_results(suite=CALIBRATE_SUITE)
    finally:
        repo.close()


def test_matching_judge_passes_and_is_stored(monkeypatch, set_file):
    result = _invoke(monkeypatch, RubricJudge(EXACT), "--set", str(set_file))
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output
    assert "mean error   0.000" in result.output
    status = judge_calibration_status(load_settings().db_path, "ollama", "judge-m")
    assert status is not None and status.passed and status.set_name == "tiny"


def test_lenient_judge_fails_exit_one_and_is_stored_as_failed(monkeypatch, set_file):
    lenient = {"answer-a": (4, 3, 3), "answer-b": (4, 3, 3)}
    result = _invoke(monkeypatch, RubricJudge(lenient), "--set", str(set_file))
    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert "bias         +0.350" in result.output
    status = judge_calibration_status(load_settings().db_path, "ollama", "judge-m")
    assert status is not None and not status.passed and status.bias == 0.35


def test_unparseable_item_is_incomplete_and_never_stored(monkeypatch, set_file):
    judge = RubricJudge({"answer-a": (4, 3, 3), "answer-b": None})
    result = _invoke(monkeypatch, judge, "--set", str(set_file))
    assert result.exit_code == 1
    assert "INCOMPLETE" in result.output
    assert "1/2 scored" in result.output
    assert _stored_rows() == []


def test_no_save_stores_nothing(monkeypatch, set_file):
    result = _invoke(monkeypatch, RubricJudge(EXACT), "--set", str(set_file), "--no-save")
    assert result.exit_code == 0, result.output
    assert _stored_rows() == []


def test_max_error_moves_the_verdict(monkeypatch, set_file):
    lenient = {"answer-a": (4, 3, 3), "answer-b": (4, 3, 3)}
    result = _invoke(
        monkeypatch, RubricJudge(lenient), "--set", str(set_file), "--max-error", "0.4"
    )
    assert result.exit_code == 0, result.output


def test_bad_set_names_the_item(monkeypatch, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(SET.replace("accuracy: 1", "accuracy: 9"))
    result = _invoke(monkeypatch, RubricJudge(EXACT), "--set", str(bad))
    assert result.exit_code == 2
    assert "item 2" in result.output and "accuracy" in result.output


def test_json_out(monkeypatch, set_file, tmp_path):
    out = tmp_path / "cal.json"
    result = _invoke(monkeypatch, RubricJudge(EXACT), "--set", str(set_file), "-o", str(out))
    assert result.exit_code == 0, result.output
    body = json.loads(out.read_text())
    assert (body["mae"], body["bias"], body["passed"]) == (0.0, 0.0, True)
    assert body["judge_model"] == "judge-m"


def test_default_set_is_the_shipped_six(monkeypatch):
    judge = RubricJudge({})
    result = _invoke(monkeypatch, judge, "--no-save")
    assert "items: 6" in result.output
    assert result.exit_code == 1


def test_help_lists_every_option():
    result = CliRunner().invoke(cli, ["judge-calibrate", "--help"])
    assert result.exit_code == 0
    for flag in (
        "--judge-provider",
        "--judge-model",
        "--judge-host",
        "--set",
        "--max-error",
        "--save",
        "--json-out",
        "--budget",
    ):
        assert flag in result.output
