"""Tests for atomics soak — long-duration stability testing."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from atomics.soak import (
    SoakResult,
    SoakSample,
    _compute_verdict,
    _drift_pct,
    _linear_slope,
    parse_duration,
)


# ── Duration parsing ──────────────────────────────────────────────────────────


class TestParseDuration:
    def test_minutes_only(self):
        assert parse_duration("30m") == 1800

    def test_hours_only(self):
        assert parse_duration("2h") == 7200

    def test_hours_and_minutes(self):
        assert parse_duration("1h30m") == 5400

    def test_bare_number_is_minutes(self):
        assert parse_duration("90") == 5400

    def test_single_minute(self):
        assert parse_duration("1m") == 60

    def test_case_insensitive(self):
        assert parse_duration("2H30M") == 9000

    def test_whitespace_stripped(self):
        assert parse_duration("  30m  ") == 1800

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Empty"):
            parse_duration("")

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            parse_duration("abc")

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            parse_duration("0m")

    def test_zero_h_zero_m_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            parse_duration("0h0m")


# ── Linear regression ─────────────────────────────────────────────────────────


class TestLinearSlope:
    def test_flat_line(self):
        assert _linear_slope([0, 1, 2, 3], [5, 5, 5, 5]) == 0.0

    def test_positive_slope(self):
        slope = _linear_slope([0, 1, 2, 3], [0, 1, 2, 3])
        assert slope == pytest.approx(1.0)

    def test_negative_slope(self):
        slope = _linear_slope([0, 1, 2, 3], [6, 4, 2, 0])
        assert slope == pytest.approx(-2.0)

    def test_single_point(self):
        assert _linear_slope([0], [5]) == 0.0

    def test_empty(self):
        assert _linear_slope([], []) == 0.0

    def test_noisy_positive(self):
        slope = _linear_slope([0, 1, 2, 3, 4], [10, 12, 11, 14, 13])
        assert slope > 0


# ── Drift percentage ──────────────────────────────────────────────────────────


class TestDriftPct:
    def test_flat_series(self):
        assert _drift_pct([100, 100, 100, 100]) == pytest.approx(0.0)

    def test_degrading_throughput(self):
        drift = _drift_pct([100, 90, 80, 70])
        assert drift < 0
        assert drift == pytest.approx(-30.0)

    def test_improving_latency(self):
        drift = _drift_pct([100, 110, 120, 130])
        assert drift > 0
        assert drift == pytest.approx(30.0)

    def test_single_value(self):
        assert _drift_pct([42]) == 0.0

    def test_empty(self):
        assert _drift_pct([]) == 0.0

    def test_zero_start(self):
        assert _drift_pct([0, 10, 20]) == 0.0


# ── Verdict computation ──────────────────────────────────────────────────────


class TestComputeVerdict:
    def test_stable(self):
        assert _compute_verdict(-3.0, 5.0, 0.001) == "STABLE"

    def test_degraded_throughput(self):
        assert _compute_verdict(-8.0, 5.0, 0.001) == "DEGRADED"

    def test_degraded_latency(self):
        assert _compute_verdict(-2.0, 15.0, 0.001) == "DEGRADED"

    def test_degraded_errors(self):
        assert _compute_verdict(-2.0, 5.0, 0.01) == "DEGRADED"

    def test_unstable_throughput(self):
        assert _compute_verdict(-20.0, 5.0, 0.001) == "UNSTABLE"

    def test_unstable_latency(self):
        assert _compute_verdict(-2.0, 30.0, 0.001) == "UNSTABLE"

    def test_unstable_errors(self):
        assert _compute_verdict(-2.0, 5.0, 0.10) == "UNSTABLE"

    def test_boundary_stable(self):
        assert _compute_verdict(-4.99, 9.99, 0.0049) == "STABLE"

    def test_boundary_degraded(self):
        assert _compute_verdict(-5.0, 10.0, 0.005) == "DEGRADED"

    def test_boundary_unstable(self):
        assert _compute_verdict(-15.0, 25.0, 0.05) == "UNSTABLE"


# ── SoakSample defaults ──────────────────────────────────────────────────────


class TestSoakSample:
    def test_defaults(self):
        s = SoakSample()
        assert s.elapsed_seconds == 0.0
        assert s.requests == 0
        assert s.failed == 0
        assert s.aggregate_tps == 0.0
        assert s.vram_used_mb is None

    def test_custom_values(self):
        s = SoakSample(elapsed_seconds=30.0, requests=10, aggregate_tps=42.5)
        assert s.elapsed_seconds == 30.0
        assert s.requests == 10
        assert s.aggregate_tps == 42.5


# ── SoakResult defaults ──────────────────────────────────────────────────────


class TestSoakResult:
    def test_defaults(self):
        r = SoakResult()
        assert r.model == ""
        assert r.concurrency == 4
        assert r.verdict == "STABLE"
        assert r.samples == []
        assert r.total_cost_usd == 0.0

    def test_with_samples(self):
        r = SoakResult(
            model="qwen2.5:7b",
            samples=[SoakSample(aggregate_tps=40), SoakSample(aggregate_tps=38)],
        )
        assert len(r.samples) == 2
        assert r.model == "qwen2.5:7b"


# ── Runner (mocked) ──────────────────────────────────────────────────────────


async def _fake_single_request(client, host, model, prompt, num_predict):
    await asyncio.sleep(0.001)
    return (100, 50, 500.0, 200.0)


class TestRunSoak:
    @pytest.mark.asyncio
    async def test_basic_run(self):
        with patch("atomics.stress._get_vram_used_mb", return_value=8000.0), \
             patch("atomics.stress._single_request", side_effect=_fake_single_request):
            from atomics.soak import run_soak
            result = await run_soak(
                host="http://localhost:11434",
                model="test-model",
                concurrency=2,
                duration_seconds=3,
                sample_interval=1,
            )
            assert result.model == "test-model"
            assert result.concurrency == 2
            assert result.total_requests > 0
            assert result.actual_duration_seconds >= 2.5
            assert len(result.samples) >= 2
            assert result.verdict in ("STABLE", "DEGRADED", "UNSTABLE")

    @pytest.mark.asyncio
    async def test_samples_have_data(self):
        with patch("atomics.stress._get_vram_used_mb", return_value=None), \
             patch("atomics.stress._single_request", side_effect=_fake_single_request):
            from atomics.soak import run_soak
            result = await run_soak(
                host="http://localhost:11434",
                model="test-model",
                concurrency=1,
                duration_seconds=2.5,
                sample_interval=1,
            )
            for sample in result.samples:
                assert sample.requests > 0
                assert sample.aggregate_tps > 0

    @pytest.mark.asyncio
    async def test_callback_called(self):
        samples_received: list[SoakSample] = []

        def on_sample(s: SoakSample) -> None:
            samples_received.append(s)

        with patch("atomics.stress._get_vram_used_mb", return_value=None), \
             patch("atomics.stress._single_request", side_effect=_fake_single_request):
            from atomics.soak import run_soak
            await run_soak(
                host="http://localhost:11434",
                model="test-model",
                concurrency=1,
                duration_seconds=2.5,
                sample_interval=1,
                on_sample=on_sample,
            )
            assert len(samples_received) >= 2

    @pytest.mark.asyncio
    async def test_failure_handling(self):
        call_count = 0

        async def _failing_request(client, host, model, prompt, num_predict):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                raise ConnectionError("Simulated failure")
            await asyncio.sleep(0.001)
            return (50, 25, 300.0, 166.0)

        with patch("atomics.stress._get_vram_used_mb", return_value=None), \
             patch("atomics.stress._single_request", side_effect=_failing_request):
            from atomics.soak import run_soak
            result = await run_soak(
                host="http://localhost:11434",
                model="test-model",
                concurrency=1,
                duration_seconds=2.5,
                sample_interval=1,
            )
            assert result.total_failed > 0
            assert result.error_rate > 0


# ── DB persistence ────────────────────────────────────────────────────────────


class TestSoakStorage:
    def test_save_and_retrieve(self):
        from atomics.storage.repository import MetricsRepository

        tmp = tempfile.mktemp(suffix=".db")
        repo = MetricsRepository(Path(tmp))

        sr = SoakResult(
            model="qwen2.5:7b",
            host="http://localhost:11434",
            provider="ollama",
            concurrency=4,
            duration_seconds=1800,
            actual_duration_seconds=1798.5,
            sample_interval=30,
            total_requests=500,
            total_failed=2,
            total_tokens=100000,
            throughput_drift_pct=-3.2,
            latency_drift_pct=8.1,
            vram_start_mb=8200.0,
            vram_end_mb=8400.0,
            vram_drift_mb=200.0,
            avg_tps=42.0,
            peak_tps=44.0,
            min_tps=39.0,
            avg_p95_ms=2800.0,
            error_rate=0.004,
            verdict="STABLE",
            samples=[
                SoakSample(elapsed_seconds=30, requests=25, aggregate_tps=42.0, p95_latency_ms=2800),
                SoakSample(elapsed_seconds=60, requests=24, aggregate_tps=41.5, p95_latency_ms=2900),
            ],
        )
        repo.save_soak_result(sr)
        rows = repo.get_soak_results()
        assert len(rows) == 1
        assert rows[0]["model"] == "qwen2.5:7b"
        assert rows[0]["verdict"] == "STABLE"
        assert rows[0]["throughput_drift_pct"] == -3.2

        samples = json.loads(rows[0]["samples_json"])
        assert len(samples) == 2
        assert samples[0]["aggregate_tps"] == 42.0

        repo.close()

    def test_filter_by_model(self):
        from atomics.storage.repository import MetricsRepository

        tmp = tempfile.mktemp(suffix=".db")
        repo = MetricsRepository(Path(tmp))

        for model in ["qwen2.5:7b", "mistral:7b", "qwen2.5:7b"]:
            sr = SoakResult(model=model, host="http://localhost:11434", verdict="STABLE")
            repo.save_soak_result(sr)

        rows = repo.get_soak_results(model="qwen2.5:7b")
        assert len(rows) == 2

        rows = repo.get_soak_results(model="mistral:7b")
        assert len(rows) == 1

        repo.close()


# ── CLI integration ───────────────────────────────────────────────────────────


class TestSoakCLI:
    def test_help_renders(self):
        from click.testing import CliRunner
        from atomics.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["soak", "--help"])
        assert result.exit_code == 0
        assert "--duration" in result.output
        assert "--concurrency" in result.output
        assert "--sample-interval" in result.output

    def test_duration_flag_parsing(self):
        from click.testing import CliRunner
        from atomics.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["soak", "--duration", "invalid_xyz"])
        assert result.exit_code != 0

    def test_no_args_shows_error(self):
        from click.testing import CliRunner
        from atomics.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["soak"])
        assert result.exit_code != 0 or "--model" in result.output or "Missing" in result.output


# ── Schema version ────────────────────────────────────────────────────────────


class TestSchemaVersion:
    def test_soak_table_exists(self):
        from atomics.storage.schema import SCHEMA_VERSION, init_db

        tmp = tempfile.mktemp(suffix=".db")
        conn = init_db(Path(tmp))
        conn.execute(
            "INSERT INTO soak_results "
            "(result_id, model, host, concurrency, duration_seconds, actual_duration_seconds, "
            "sample_interval, total_requests, total_failed, total_tokens, avg_tps, peak_tps, "
            "min_tps, throughput_drift_pct, latency_drift_pct, avg_p95_ms, error_rate, "
            "verdict, samples_json, timestamp) "
            "VALUES ('r1','m','h',4,60,59,30,100,0,10000,42,44,40,-1.0,2.0,2800,0.0,'STABLE','[]','2026-06-01')"
        )
        row = conn.execute("SELECT verdict FROM soak_results WHERE result_id='r1'").fetchone()
        assert row[0] == "STABLE"
        conn.close()

    def test_schema_version_bumped(self):
        from atomics.storage.schema import SCHEMA_VERSION
        assert SCHEMA_VERSION == 10
