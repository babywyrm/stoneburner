"""Tests for atomics.scenario — mixed-workload scenario runner."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from atomics.scenario import run_scenario
from atomics.scenario_models import (
    ScenarioResult,
    WorkloadResult,
    WorkloadSpec,
    load_scenario_yaml,
    parse_workload_flag,
)
from atomics.scenario_prompts import (
    BUILTIN_PROMPTS,
    EVAL_PROMPTS,
    GATE_PROMPTS,
    load_custom_prompts,
    resolve_prompts,
)


# ── WorkloadSpec ──────────────────────────────────────────────────────────────

class TestWorkloadSpec:
    def test_valid_gate(self) -> None:
        spec = WorkloadSpec(name="test", type="gate", model="m:1b", concurrency=2)
        assert spec.num_predict == 32

    def test_valid_eval(self) -> None:
        spec = WorkloadSpec(name="test", type="eval", model="m:1b", concurrency=1)
        assert spec.num_predict == 256

    def test_invalid_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown workload type"):
            WorkloadSpec(name="x", type="chat", model="m", concurrency=1)

    def test_invalid_concurrency(self) -> None:
        with pytest.raises(ValueError, match="Concurrency must be >= 1"):
            WorkloadSpec(name="x", type="gate", model="m", concurrency=0)

    def test_custom_num_predict(self) -> None:
        spec = WorkloadSpec(name="x", type="gate", model="m", concurrency=1, num_predict=64)
        assert spec.num_predict == 64


# ── WorkloadResult properties ─────────────────────────────────────────────────

class TestWorkloadResult:
    def _make(self, **kwargs) -> WorkloadResult:
        spec = WorkloadSpec(name="t", type="gate", model="m", concurrency=1, sla_ms=1000.0)
        return WorkloadResult(spec=spec, **kwargs)

    def test_p50_empty(self) -> None:
        wr = self._make()
        assert wr.p50_ms == 0.0

    def test_p50_values(self) -> None:
        wr = self._make(latencies=[100.0, 200.0, 300.0, 400.0, 500.0])
        assert wr.p50_ms == 300.0

    def test_p95(self) -> None:
        wr = self._make(latencies=[float(i) for i in range(1, 101)])
        assert wr.p95_ms == pytest.approx(95.05, abs=0.1)

    def test_avg_tps_empty(self) -> None:
        wr = self._make()
        assert wr.avg_tps == 0.0

    def test_avg_tps(self) -> None:
        wr = self._make(per_request_tps=[10.0, 20.0, 30.0])
        assert wr.avg_tps == 20.0

    def test_sla_violations(self) -> None:
        wr = self._make(latencies=[500.0, 1500.0, 800.0, 2000.0])
        assert wr.sla_violations == 2

    def test_sla_compliance(self) -> None:
        wr = self._make(latencies=[500.0, 1500.0, 800.0, 2000.0])
        assert wr.sla_compliance_pct == 50.0

    def test_sla_no_threshold(self) -> None:
        spec = WorkloadSpec(name="t", type="gate", model="m", concurrency=1)
        wr = WorkloadResult(spec=spec, latencies=[5000.0])
        assert wr.sla_violations == 0
        assert wr.sla_compliance_pct == 100.0


# ── parse_workload_flag ───────────────────────────────────────────────────────

class TestParseWorkloadFlag:
    def test_basic_three_parts(self) -> None:
        spec = parse_workload_flag("gate:mymodel:3")
        assert spec.type == "gate"
        assert spec.model == "mymodel"
        assert spec.concurrency == 3
        assert spec.sla_ms is None

    def test_with_sla(self) -> None:
        spec = parse_workload_flag("eval:big:1:15000")
        assert spec.type == "eval"
        assert spec.model == "big"
        assert spec.concurrency == 1
        assert spec.sla_ms == 15000.0

    def test_model_with_colon(self) -> None:
        spec = parse_workload_flag("gate:qwen2.5:3b:2:5000")
        assert spec.model == "qwen2.5:3b"
        assert spec.concurrency == 2
        assert spec.sla_ms == 5000.0

    def test_model_with_colon_no_sla(self) -> None:
        spec = parse_workload_flag("gate:qwen2.5:3b:2")
        assert spec.model == "qwen2.5:3b"
        assert spec.concurrency == 2
        assert spec.sla_ms is None

    def test_too_few_parts(self) -> None:
        with pytest.raises(ValueError, match="Invalid workload format"):
            parse_workload_flag("gate:model")

    def test_invalid_concurrency(self) -> None:
        with pytest.raises(ValueError, match="Invalid concurrency"):
            parse_workload_flag("gate:model:abc")


# ── load_scenario_yaml ────────────────────────────────────────────────────────

class TestLoadScenarioYaml:
    def test_valid_yaml(self, tmp_path: Path) -> None:
        f = tmp_path / "scenario.yaml"
        f.write_text(
            "workloads:\n"
            "  - name: gk\n"
            "    type: gate\n"
            "    model: qwen2.5:3b\n"
            "    concurrency: 2\n"
            "    sla_ms: 5000\n"
            "  - name: review\n"
            "    type: eval\n"
            "    model: qwen2.5:7b\n"
            "    concurrency: 1\n"
        )
        specs = load_scenario_yaml(str(f))
        assert len(specs) == 2
        assert specs[0].name == "gk"
        assert specs[0].type == "gate"
        assert specs[0].model == "qwen2.5:3b"
        assert specs[0].sla_ms == 5000
        assert specs[1].name == "review"
        assert specs[1].sla_ms is None

    def test_missing_workloads_key(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("something: else\n")
        with pytest.raises(ValueError, match="workloads"):
            load_scenario_yaml(str(f))

    def test_missing_model(self, tmp_path: Path) -> None:
        f = tmp_path / "bad2.yaml"
        f.write_text(
            "workloads:\n"
            "  - name: x\n"
            "    type: gate\n"
            "    concurrency: 1\n"
        )
        with pytest.raises(ValueError, match="missing required 'model'"):
            load_scenario_yaml(str(f))


# ── Prompt fixtures ───────────────────────────────────────────────────────────

class TestPromptFixtures:
    def test_gate_prompts_count(self) -> None:
        assert len(GATE_PROMPTS) == 8

    def test_eval_prompts_count(self) -> None:
        assert len(EVAL_PROMPTS) == 8

    def test_gate_prompts_not_empty(self) -> None:
        for p in GATE_PROMPTS:
            assert len(p) > 50

    def test_eval_prompts_not_empty(self) -> None:
        for p in EVAL_PROMPTS:
            assert len(p) > 50

    def test_resolve_gate(self) -> None:
        prompts = resolve_prompts("gate")
        assert prompts == GATE_PROMPTS

    def test_resolve_eval(self) -> None:
        prompts = resolve_prompts("eval")
        assert prompts == EVAL_PROMPTS

    def test_resolve_unknown(self) -> None:
        with pytest.raises(ValueError, match="No built-in prompts"):
            resolve_prompts("stream")

    def test_custom_prompts(self, tmp_path: Path) -> None:
        f = tmp_path / "custom.txt"
        f.write_text("prompt one\n---\nprompt two\n---\nprompt three\n")
        prompts = load_custom_prompts(str(f))
        assert len(prompts) == 3
        assert prompts[0] == "prompt one"

    def test_custom_prompts_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("\n\n")
        with pytest.raises(ValueError, match="No prompts found"):
            load_custom_prompts(str(f))

    def test_resolve_custom_file(self, tmp_path: Path) -> None:
        f = tmp_path / "custom.txt"
        f.write_text("p1\n---\np2\n")
        prompts = resolve_prompts("gate", prompts_file=str(f))
        assert len(prompts) == 2


# ── ScenarioResult ────────────────────────────────────────────────────────────

class TestScenarioResult:
    def test_defaults(self) -> None:
        sr = ScenarioResult()
        assert sr.duration_seconds == 0.0
        assert sr.workloads == []
        assert sr.baselines == {}
        assert sr.interference == {}


# ── Runner (mocked) ──────────────────────────────────────────────────────────

async def _fake_single_request(client, host, model, prompt, num_predict):
    """Simulate a fast Ollama response."""
    await asyncio.sleep(0.001)
    return (20, 100, 500.0, 40.0)


class TestRunner:
    def test_single_workload(self) -> None:
        spec = WorkloadSpec(
            name="test-gate", type="gate", model="m:1b", concurrency=1,
            sla_ms=2000.0,
        )
        with patch("atomics.scenario._single_request", side_effect=_fake_single_request):
            result = asyncio.run(run_scenario(
                host="http://fake:11434",
                specs=[spec],
                duration_seconds=1.0,
                skip_baseline=True,
            ))
        assert len(result.workloads) == 1
        wr = result.workloads[0]
        assert wr.requests > 0
        assert wr.failed == 0
        assert wr.sla_compliance_pct == 100.0

    def test_multiple_workloads_concurrent(self) -> None:
        specs = [
            WorkloadSpec(name="g1", type="gate", model="m:1b", concurrency=1),
            WorkloadSpec(name="e1", type="eval", model="m:3b", concurrency=1),
        ]
        with patch("atomics.scenario._single_request", side_effect=_fake_single_request):
            result = asyncio.run(run_scenario(
                host="http://fake:11434",
                specs=specs,
                duration_seconds=1.0,
                skip_baseline=True,
            ))
        assert len(result.workloads) == 2
        assert result.total_requests > 0
        assert all(wr.requests > 0 for wr in result.workloads)

    def test_interference_scoring(self) -> None:
        spec = WorkloadSpec(
            name="g1", type="gate", model="m:1b", concurrency=1,
        )
        with patch("atomics.scenario._single_request", side_effect=_fake_single_request):
            result = asyncio.run(run_scenario(
                host="http://fake:11434",
                specs=[spec],
                duration_seconds=1.0,
                skip_baseline=False,
            ))
        assert "g1" in result.baselines
        assert "g1" in result.interference
        assert result.interference["g1"] > 0

    def test_sla_violations_counted(self) -> None:
        spec = WorkloadSpec(
            name="strict", type="gate", model="m:1b", concurrency=1,
            sla_ms=100.0,
        )
        with patch("atomics.scenario._single_request", side_effect=_fake_single_request):
            result = asyncio.run(run_scenario(
                host="http://fake:11434",
                specs=[spec],
                duration_seconds=1.0,
                skip_baseline=True,
            ))
        wr = result.workloads[0]
        assert wr.sla_violations == wr.requests
        assert wr.sla_compliance_pct == 0.0

    def test_callbacks_called(self) -> None:
        spec = WorkloadSpec(name="cb", type="gate", model="m:1b", concurrency=1)
        baselines_seen: list[str] = []
        workloads_seen: list[str] = []

        def on_bl(name: str, p50: float) -> None:
            baselines_seen.append(name)

        def on_wr(wr) -> None:
            workloads_seen.append(wr.spec.name)

        with patch("atomics.scenario._single_request", side_effect=_fake_single_request):
            asyncio.run(run_scenario(
                host="http://fake:11434",
                specs=[spec],
                duration_seconds=1.0,
                skip_baseline=False,
                on_baseline_done=on_bl,
                on_workload_done=on_wr,
            ))
        assert "cb" in baselines_seen
        assert "cb" in workloads_seen


# ── CLI integration ───────────────────────────────────────────────────────────

class TestCLI:
    def test_scenario_help(self) -> None:
        from click.testing import CliRunner
        from atomics.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["scenario", "--help"])
        assert result.exit_code == 0
        assert "mixed-workload" in result.output.lower() or "scenario" in result.output.lower()

    def test_scenario_no_args(self) -> None:
        from click.testing import CliRunner
        from atomics.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["scenario"])
        assert result.exit_code != 0

    def test_scenario_mutual_exclusion(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from atomics.cli import cli

        f = tmp_path / "s.yaml"
        f.write_text("workloads:\n  - name: x\n    type: gate\n    model: m\n    concurrency: 1\n")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "scenario", "--file", str(f), "--workload", "gate:m:1",
        ])
        assert result.exit_code != 0
        assert "Cannot use both" in result.output


# ── DB persistence ────────────────────────────────────────────────────────────

class TestDBPersistence:
    def test_save_scenario_result(self, tmp_path: Path) -> None:
        from atomics.storage.repository import MetricsRepository

        db = tmp_path / "test.db"
        repo = MetricsRepository(db)

        spec = WorkloadSpec(name="g1", type="gate", model="m:1b", concurrency=2, sla_ms=5000.0)
        wr = WorkloadResult(
            spec=spec, requests=10, failed=1,
            latencies=[500.0] * 10, per_request_tps=[40.0] * 10,
            total_output_tokens=200,
        )
        sr = ScenarioResult(
            duration_seconds=60.0,
            workloads=[wr],
            baselines={"g1": 400.0},
            interference={"g1": 1.25},
            total_requests=10,
            total_failed=1,
        )

        repo.save_scenario_result(sr)

        rows = repo._conn.execute("SELECT * FROM scenario_results").fetchall()
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["total_requests"] == 10
        assert row["workload_count"] == 1
        assert row["max_interference"] == pytest.approx(1.25, abs=0.01)

        repo.close()
