"""Judged suites label their judge with its stored calibration; the label never changes a run."""

from __future__ import annotations

import importlib
import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from atomics.api import _runners as runners
from atomics.api.models import EvalRequest
from atomics.cli import cli
from atomics.config import load_settings
from atomics.eval.calibration import ItemCalibration, JudgeCalibration, save_calibration
from atomics.storage import MetricsRepository
from tests.test_cli_refusal import _patch_refusal


def _store(model: str, *, judge: float = 0.3) -> None:
    repo = MetricsRepository(load_settings().db_path)
    try:
        save_calibration(
            repo,
            f"cal-{model}",
            JudgeCalibration(
                set_name="t",
                set_digest="abc",
                judge_provider="ollama",
                judge_model=model,
                max_error=0.15,
                items=[ItemCalibration("A", 0.3, judge), ItemCalibration("B", 0.8, 0.8)],
            ),
        )
    finally:
        repo.close()


def _refusal(tmp_path, name: str):
    out = tmp_path / f"{name}.json"
    result = CliRunner().invoke(cli, ["refusal", "--no-save", "--json-out", str(out)])
    return result, json.loads(out.read_text())


def test_refusal_labels_its_judge_and_the_label_changes_nothing_else(monkeypatch, tmp_path):
    _patch_refusal(monkeypatch)
    before, before_json = _refusal(tmp_path, "before")
    assert "Judge calibration: uncalibrated" in before.output

    _store("mock-model")
    after, after_json = _refusal(tmp_path, "after")
    assert "Judge calibration: calibrated · error 0.00 · bias +0.00 · t" in after.output
    assert after.exit_code == before.exit_code == 0
    assert after_json == before_json


def test_failed_calibration_is_labelled_failed(monkeypatch, tmp_path):
    _patch_refusal(monkeypatch)
    _store("mock-model", judge=0.9)
    result, _ = _refusal(tmp_path, "failed")
    assert "Judge calibration: failed calibration · error 0.30 · bias +0.30" in result.output
    assert result.exit_code == 0


def test_an_unreadable_db_labels_uncalibrated_and_the_run_still_passes(monkeypatch, tmp_path):
    _patch_refusal(monkeypatch)
    db = load_settings().db_path
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_bytes(b"not a database " * 20)
    result, _ = _refusal(tmp_path, "corrupt")
    assert result.exit_code == 0, result.output
    assert "Judge calibration: uncalibrated" in result.output


@pytest.mark.parametrize(
    "module",
    [
        "atomics.commands.eval",
        "atomics.commands.probe",
        "atomics.commands.security.cmd_adversarial",
        "atomics.commands.security.cmd_codereview",
        "atomics.commands.security.cmd_multiturn",
        "atomics.commands.security.cmd_redblue",
        "atomics.commands.security.cmd_refusal",
    ],
)
def test_every_judged_suite_prints_its_judge_calibration(module):
    assert "print_judge_calibration(" in inspect.getsource(importlib.import_module(module))


@pytest.mark.asyncio
async def test_api_eval_result_carries_the_judge_calibration():
    summary = SimpleNamespace(
        overall_accuracy=1.0,
        fixtures=[],
        fixture_results=[],
        total_tokens=0,
        total_cost_usd=0.0,
    )
    payload = EvalRequest(suite="refusal", provider="ollama", model="m1", judge_model="j1")

    async def run() -> dict[str, object]:
        with (
            patch.object(runners, "_provider_for", side_effect=[MagicMock(), MagicMock()]),
            patch.object(runners, "run_refusal", new_callable=AsyncMock, return_value=summary),
        ):
            return await runners.run_eval_suite(payload)

    assert (await run())["judge_calibration"] is None
    _store("j1")
    label = (await run())["judge_calibration"]
    assert isinstance(label, dict)
    assert (label["passed"], label["mae"], label["set_name"]) == (True, 0.0, "t")
