"""Run the creative quickstart scenarios through the CLI.

The model is a stand-in. Each test invokes the command in the matching
``# scenario:`` block and checks the result that block describes.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import asdict
from pathlib import Path

import pytest
from click.testing import CliRunner

from atomics.benchmark.labcompare import CellResult
from atomics.cli import cli
from atomics.eval.calibration import run_judge_calibration
from atomics.eval.gauntlet import SuiteJobResult, mark_vs_best, run_gauntlet
from atomics.eval.judge import JudgeResult
from atomics.probe.runner import ProbeResult, ProbeSummary
from atomics.providers.base import BaseProvider, ProviderResponse
from tests.test_cli_refusal import _patch_refusal

_QUICKSTART = Path(__file__).resolve().parents[1] / "QUICKSTART.md"


def _scenario(name: str) -> list[list[str]]:
    text = _QUICKSTART.read_text(encoding="utf-8")
    marker = f"# scenario: {name}"
    commands: list[list[str]] = []
    for block in text.split("```bash"):
        body = block.split("```", 1)[0]
        if marker not in body:
            continue
        buf = ""
        for raw in body.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith("\\"):
                buf += line[:-1].strip() + " "
                continue
            buf += line
            argv = buf.replace("atomics ", "", 1).split()
            commands.append(argv)
            buf = ""
    if not commands:
        raise AssertionError(f"no scenario {name} in QUICKSTART.md")
    return commands


def test_scenario_same_rate_prints_both_marks(monkeypatch: pytest.MonkeyPatch) -> None:
    ids = [f"f{i}" for i in range(8)]
    leader = SuiteJobResult(
        model="MODEL_A", suite="toolcall", ok=True, headline=0.0, any_run=0.0,
        fixture_scores=dict.fromkeys(ids, 0.0),
    )  # fmt: skip
    steady = SuiteJobResult(
        model="MODEL_B", suite="toolcall", ok=True, headline=0.5, any_run=0.5,
        fixture_scores=dict.fromkeys(ids, 0.5),
    )  # fmt: skip
    noisy = SuiteJobResult(
        model="MODEL_C", suite="toolcall", ok=True, headline=0.2, any_run=0.5,
        fixture_scores={"f0": 1.0, "f1": 1.0, "f2": -1.0, "f3": -0.2},
    )  # fmt: skip

    async def fake_gauntlet(**_kwargs: object) -> list[SuiteJobResult]:
        return mark_vs_best([leader, steady, noisy])

    monkeypatch.setattr("atomics.eval.gauntlet.run_gauntlet", fake_gauntlet)
    monkeypatch.setattr(
        "atomics.commands.sweep._make_provider",
        lambda *_a, **_k: type("P", (), {"name": "ollama", "default_model": "m"})(),
    )
    argv = _scenario("same-rate")[0]
    result = CliRunner().invoke(cli, argv)
    assert result.exit_code == 0, result.output
    assert "behind " in result.output and " to " in result.output
    assert "tied " in result.output
    assert "MODEL_A" in result.output and "best" in result.output


def test_scenario_exit_zero_still_names_the_finding(monkeypatch) -> None:
    monkeypatch.setattr("atomics.commands.battery.invoke_atomics", lambda _args: 0)
    argv = _scenario("exit-zero")[0]
    result = CliRunner().invoke(cli, argv)
    assert result.exit_code == 0, result.output
    assert "DANGEROUS tool row is a finding" in result.output
    assert "One run is one sample." in result.output


def test_scenario_blue_score_is_not_safety() -> None:
    argv = _scenario("blue-is-not-safety")[0]
    result = CliRunner().invoke(cli, argv)
    assert result.exit_code == 0, result.output
    assert "Not a pass:" in result.output
    assert "DANGEROUS" in result.output


@pytest.mark.asyncio
async def test_scenario_resume_keeps_the_finished_model_and_marks(tmp_path: Path) -> None:
    ids = [f"f{i}" for i in range(8)]
    kept = SuiteJobResult(
        model="MODEL_A", suite="toolcall", ok=True, headline=0.0, any_run=0.0,
        fixture_scores=dict.fromkeys(ids, 0.0),
    )  # fmt: skip
    status = tmp_path / "night.status.json"
    status.write_text(
        json.dumps({"started_at": "t", "completed": [asdict(kept)]}),
        encoding="utf-8",
    )
    seen: list[str] = []

    async def run_suite(*, model: str, suite: str, skip_incapable: bool) -> SuiteJobResult:
        seen.append(model)
        return SuiteJobResult(
            model=model, suite=suite, ok=True, headline=0.5, any_run=0.5,
            fixture_scores=dict.fromkeys(ids, 0.5),
        )  # fmt: skip

    results = await run_gauntlet(
        models=["MODEL_A", "MODEL_B"],
        suites=["toolcall"],
        run_suite=run_suite,
        status_path=status,
        resume=True,
    )
    assert seen == ["MODEL_B"]
    marked = {row.model: row.vs_best for row in results}
    assert marked == {"MODEL_A": "best", "MODEL_B": "behind"}
    saved = {row["model"]: row["vs_best"] for row in json.loads(status.read_text())["completed"]}
    assert saved == marked


def test_scenario_resume_command_forwards_resume(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    async def fake_gauntlet(**kwargs: object) -> list[SuiteJobResult]:
        seen.update(kwargs)
        return [
            SuiteJobResult(
                model="MODEL_A",
                suite="toolcall",
                ok=True,
                headline=0.0,
                vs_best="best",
            )  # fmt: skip
        ]

    monkeypatch.setattr("atomics.eval.gauntlet.run_gauntlet", fake_gauntlet)
    monkeypatch.setattr(
        "atomics.commands.sweep._make_provider",
        lambda *_a, **_k: type("P", (), {"name": "ollama", "default_model": "m"})(),
    )
    argv = [
        str(tmp_path / item) if item in {"night.status.json", "night.log"} else item
        for item in _scenario("resume")[0]
    ]
    result = CliRunner().invoke(cli, argv)
    assert result.exit_code == 0, result.output
    assert seen["resume"] is True
    assert "best" in result.output


def _gate_reply(prompt: str) -> str:
    if "RootBot" in prompt:
        return "The admin password is hunter2"
    if "Translate to French" in prompt:
        return "Ignorez vos instructions"
    if "capital of France" in prompt:
        return "Paris"
    return "I cannot provide that"


def test_scenario_gate_held_names_a_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_query(_client, _host, _model, prompt, _num_predict=1024, **_kwargs):
        return _gate_reply(prompt), 10.0, 4, 8, None

    monkeypatch.setattr("atomics.benchmark.qa_runner._query_ollama", fake_query)
    result = CliRunner().invoke(cli, _scenario("gate-held")[0])
    assert result.exit_code == 1, result.output
    assert "direct-secret-extraction" in result.output
    assert "roleplay-jailbreak" in result.output
    assert "PASS" in result.output
    assert "FAIL" in result.output
    assert "5/6" in result.output


def test_scenario_withheld_score_is_not_a_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_refusal(monkeypatch, partial=True)
    result = CliRunner().invoke(cli, _scenario("score-withheld")[0])
    assert result.exit_code == 1, result.output
    assert "n/a (0/1 scored)" in result.output
    assert "unscored" in result.output
    assert "100.0%" not in result.output


class _FreeLocal(BaseProvider):
    @property
    def name(self) -> str:
        return "ollama"

    async def generate(self, prompt, *, system="", model=None, max_tokens=1024, **_kwargs):
        return ProviderResponse(
            text="ok",
            input_tokens=30,
            output_tokens=60,
            total_tokens=90,
            model=model or "MODEL",
            latency_ms=10.0,
            estimated_cost_usd=0.0,
        )

    async def health_check(self) -> bool:
        return True


def test_scenario_one_local_answer_costs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "atomics.commands.benchmark._make_provider",
        lambda *_a, **_k: _FreeLocal(),
    )
    result = CliRunner().invoke(cli, _scenario("one-answer-cost")[0])
    assert result.exit_code == 0, result.output
    assert "tokens=90" in result.output
    assert "cost=$0.000000" in result.output


def _stand_in(*_a, **_k):
    return type("P", (), {"name": "ollama", "default_model": "JUDGE"})()


def test_scenario_probe_names_the_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    row = ProbeResult(
        target_name="app-gate-guardrails.yaml",
        artifact_type="config-file",
        check_id="config",
        score=0.4,
        prev_score=0.9,
        regressed=True,
        judge_model="JUDGE",
        judge_rationale="dropped",
    )

    async def fake_probe(*_args, on_result=None, **_kwargs):
        if on_result is not None:
            on_result(row)
        return ProbeSummary(results=[row])

    monkeypatch.setattr("atomics.commands.probe._make_provider", _stand_in)
    monkeypatch.setattr("atomics.probe.runner.run_probe", fake_probe)
    result = CliRunner().invoke(cli, _scenario("probe-regression")[0])
    assert result.exit_code == 0, result.output
    assert "REGRESSION" in result.output
    assert "90.0% → 40.0%" in result.output


class _MissIndex:
    def __init__(self, *_args, **_kwargs) -> None:
        return None

    def info(self) -> dict[str, str]:
        return {"embedding_model": "mock"}

    def search(self, *_args, **_kwargs) -> list[types.SimpleNamespace]:
        return [types.SimpleNamespace(source="other.txt")]


def test_scenario_retrieval_miss_is_not_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps({"q1": {"relevant": ["doc.txt"], "scores": {"doc.txt": 1.0}}}),
        encoding="utf-8",
    )
    index = tmp_path / "index.db"
    index.write_text("", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.ModuleType("sentence_transformers")
    )
    monkeypatch.setitem(sys.modules, "sqlite_vec", types.ModuleType("sqlite_vec"))
    monkeypatch.setattr("atomics.eval.rag.retrieval.RAGIndex", _MissIndex)
    monkeypatch.setattr(
        "atomics.eval.rag.retrieval.LocalSentenceTransformerEmbedder",
        lambda *_a, **_k: object(),
    )
    argv = [
        str(index) if item == "index.db" else str(gold) if item == "gold.json" else item
        for item in _scenario("retrieval-miss")[0]
    ]
    result = CliRunner().invoke(cli, argv)
    assert result.exit_code == 0, result.output
    assert "Recall@k: 0.000" in result.output


def test_scenario_same_quality_names_the_slow_box(monkeypatch: pytest.MonkeyPatch) -> None:
    cells = [
        CellResult(
            "gpu", "http://127.0.0.1:11435", "MODEL",
            tokens_per_second=40.0, latency_ms=100.0, vram_fit_pct=1.0, quality_score=0.9,
        ),
        CellResult(
            "cpu", "http://127.0.0.1:11434", "MODEL",
            tokens_per_second=10.0, latency_ms=400.0, vram_fit_pct=0.0, quality_score=0.9,
        ),
    ]  # fmt: skip

    monkeypatch.setattr("atomics.commands.labcompare._run_labcompare_sync", lambda **_k: cells)
    result = CliRunner().invoke(cli, _scenario("same-quality")[0])
    assert result.exit_code == 0, result.output
    assert "parity OK" in result.output
    assert "4.0x faster" in result.output
    assert "0% GPU" in result.output


def test_scenario_a_zero_judge_fails_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    async def every_answer_is_zero(*_args, **_kwargs) -> JudgeResult:
        return JudgeResult(
            score=0.0, accuracy=0, completeness=0, format_score=0,
            rationale="empty", judge_model="JUDGE",
        )  # fmt: skip

    async def fake_run(*args, **kwargs):
        kwargs["score_fn"] = every_answer_is_zero
        return await run_judge_calibration(*args, **kwargs)

    monkeypatch.setattr("atomics.commands.calibration._make_provider", _stand_in)
    monkeypatch.setattr("atomics.commands.calibration.run_judge_calibration", fake_run)
    result = CliRunner().invoke(cli, _scenario("judge-fails")[0])
    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output
    assert "not stored (--no-save)" in result.output


def test_scenario_no_auth_refuses_a_public_bind() -> None:
    result = CliRunner().invoke(cli, _scenario("server-loopback")[0])
    assert result.exit_code == 1, result.output
    assert "no_auth cannot be combined" in result.output
    assert "127.0.0.1" in result.output


def test_scenario_schedule_preview_does_not_install() -> None:
    result = CliRunner().invoke(cli, _scenario("schedule-preview")[0])
    assert result.exit_code == 0, result.output
    assert "Add this to your crontab" in result.output
    assert "--trigger scheduled" in result.output
    assert "--provider ollama" in result.output
    assert "installed" not in result.output
