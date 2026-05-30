"""Tests for the GPU stress testing module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from atomics.stress import (
    ConcurrencyResult,
    StressResult,
    _percentile,
    _run_phase,
    run_stress,
)


def test_percentile_empty():
    assert _percentile([], 50) == 0.0


def test_percentile_single():
    assert _percentile([42.0], 50) == 42.0


def test_percentile_p50():
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0


def test_percentile_p95():
    values = list(range(1, 101))
    p95 = _percentile([float(v) for v in values], 95)
    assert 94 <= p95 <= 96


def test_concurrency_result_defaults():
    r = ConcurrencyResult(concurrency=4)
    assert r.concurrency == 4
    assert r.requests == 0
    assert r.aggregate_tps == 0.0
    assert r.latencies == []


def test_stress_result_defaults():
    r = StressResult(model="test", host="http://fake:11434")
    assert r.model == "test"
    assert r.phases == []
    assert r.peak_tps == 0.0
    assert r.saturation_concurrency == 0


def test_stress_prompts_are_populated():
    from atomics.stress import STRESS_PROMPTS

    assert len(STRESS_PROMPTS) >= 6
    for p in STRESS_PROMPTS:
        assert len(p) > 50


@pytest.mark.asyncio
async def test_run_stress_with_mock():
    """Verify the ramp logic works end-to-end with a mocked HTTP client."""
    import httpx
    from atomics.stress import _run_phase

    call_count = 0

    async def mock_post(url, *, json, timeout):
        nonlocal call_count
        call_count += 1
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "response": "test " * 50,
            "eval_count": 100,
            "prompt_eval_count": 30,
            "eval_duration": 800_000_000,
            "total_duration": 900_000_000,
        }
        return mock_resp

    mock_client = AsyncMock()
    mock_client.post = mock_post

    result = await _run_phase(
        mock_client, "http://fake:11434", "test-model",
        concurrency=2, duration_seconds=1.0, num_predict=512,
    )

    assert result.concurrency == 2
    assert result.requests >= 2
    assert result.total_output_tokens >= 200
    assert result.aggregate_tps > 0
    assert result.avg_request_tps > 0
    assert len(result.latencies) >= 2


@pytest.mark.asyncio
async def test_run_stress_e2e_with_mock(monkeypatch):
    """Full run_stress ramp with mocked HTTP and GPU helpers."""

    async def mock_post(url, *, json, timeout):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "response": "test " * 50,
            "eval_count": 80,
            "prompt_eval_count": 20,
            "eval_duration": 500_000_000,
            "total_duration": 600_000_000,
        }
        return mock_resp

    mock_client = AsyncMock()
    mock_client.post = mock_post
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    monkeypatch.setattr("atomics.stress._get_gpu_info", lambda: ("FakeGPU", 12288.0))
    monkeypatch.setattr("atomics.stress._get_vram_used_mb", lambda: 4000.0)

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    phase_log: list[int] = []

    result = await run_stress(
        host="http://fake:11434",
        model="test-model",
        max_concurrency=4,
        phase_seconds=0.5,
        num_predict=128,
        on_phase=lambda p: phase_log.append(p.concurrency),
    )

    assert result.model == "test-model"
    assert result.host == "http://fake:11434"
    assert result.gpu_name == "FakeGPU"
    assert result.vram_total_mb == 12288.0
    assert result.vram_peak_mb == 4000.0
    assert len(result.phases) == 3  # 1, 2, 4
    assert result.peak_tps > 0
    assert result.saturation_concurrency >= 1
    assert result.total_requests > 0
    assert result.duration_seconds > 0
    assert phase_log == [1, 2, 4]


@pytest.mark.asyncio
async def test_run_phase_handles_failures():
    """Failed requests should be counted, not crash the phase."""
    call_count = 0

    async def flaky_post(url, *, json, timeout):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 0:
            raise ConnectionError("refused")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "response": "ok",
            "eval_count": 10,
            "prompt_eval_count": 5,
            "eval_duration": 100_000_000,
            "total_duration": 120_000_000,
        }
        return mock_resp

    mock_client = AsyncMock()
    mock_client.post = flaky_post

    result = await _run_phase(
        mock_client, "http://fake:11434", "test-model",
        concurrency=2, duration_seconds=0.5, num_predict=64,
    )

    assert result.requests >= 1
    assert result.failed >= 1
    assert result.concurrency == 2


def test_cli_stress_command_with_save(monkeypatch, tmp_path):
    """atomics stress --save should persist results to database."""
    from click.testing import CliRunner
    from atomics.cli import cli

    async def fake_run_stress(**kwargs):
        on_phase = kwargs.get("on_phase")
        phase = ConcurrencyResult(
            concurrency=1, requests=5, total_output_tokens=500,
            aggregate_tps=25.0, avg_request_tps=25.0,
            avg_latency_ms=200.0, p95_latency_ms=300.0,
        )
        if on_phase:
            on_phase(phase)
        return StressResult(
            model=kwargs.get("model", "test"),
            host=kwargs.get("host", "http://fake:11434"),
            phases=[phase],
            peak_tps=25.0, saturation_concurrency=1,
            duration_seconds=15.0, total_tokens=500, total_requests=5,
        )

    monkeypatch.setattr("atomics.stress.run_stress", fake_run_stress)
    monkeypatch.setenv("ATOMICS_DB_PATH", str(tmp_path / "test.db"))

    runner = CliRunner()
    result = runner.invoke(cli, [
        "stress", "--model", "test-model", "--max-concurrency", "1",
        "--phase-seconds", "1", "--save",
    ])
    assert result.exit_code == 0
    assert "Peak throughput" in result.output or "peak" in result.output.lower()
    assert "saved" in result.output.lower()


def test_cli_stress_command_no_save(monkeypatch, tmp_path):
    """atomics stress --no-save should skip database persistence."""
    from click.testing import CliRunner
    from atomics.cli import cli

    async def fake_run_stress(**kwargs):
        phase = ConcurrencyResult(
            concurrency=1, requests=5, total_output_tokens=500,
            aggregate_tps=25.0, avg_request_tps=25.0,
            avg_latency_ms=200.0, p95_latency_ms=300.0,
        )
        on_phase = kwargs.get("on_phase")
        if on_phase:
            on_phase(phase)
        return StressResult(
            model=kwargs.get("model", "test"),
            host=kwargs.get("host", "http://fake:11434"),
            phases=[phase],
            peak_tps=25.0, saturation_concurrency=1,
            duration_seconds=15.0, total_tokens=500, total_requests=5,
        )

    monkeypatch.setattr("atomics.stress.run_stress", fake_run_stress)
    monkeypatch.setenv("ATOMICS_DB_PATH", str(tmp_path / "test.db"))

    runner = CliRunner()
    result = runner.invoke(cli, [
        "stress", "--model", "test-model", "--max-concurrency", "1",
        "--phase-seconds", "1", "--no-save",
    ])
    assert result.exit_code == 0
    assert "saved" not in result.output.lower()


def test_cli_stress_throttling_detected(monkeypatch, tmp_path):
    """Stress CLI should detect throttling when final phase TPS drops."""
    from click.testing import CliRunner
    from atomics.cli import cli

    async def fake_run_stress(**kwargs):
        phases = [
            ConcurrencyResult(
                concurrency=1, requests=5, total_output_tokens=500,
                aggregate_tps=50.0, avg_request_tps=50.0,
                avg_latency_ms=200.0, p95_latency_ms=300.0,
            ),
            ConcurrencyResult(
                concurrency=2, requests=10, total_output_tokens=800,
                aggregate_tps=40.0, avg_request_tps=20.0,
                avg_latency_ms=400.0, p95_latency_ms=600.0,
            ),
        ]
        on_phase = kwargs.get("on_phase")
        if on_phase:
            for p in phases:
                on_phase(p)
        return StressResult(
            model=kwargs.get("model", "test"),
            host=kwargs.get("host", "http://fake:11434"),
            phases=phases,
            peak_tps=50.0, saturation_concurrency=1,
            duration_seconds=30.0, total_tokens=1300, total_requests=15,
        )

    monkeypatch.setattr("atomics.stress.run_stress", fake_run_stress)
    monkeypatch.setenv("ATOMICS_DB_PATH", str(tmp_path / "test.db"))

    runner = CliRunner()
    result = runner.invoke(cli, [
        "stress", "--model", "test-model", "--max-concurrency", "2",
        "--phase-seconds", "1", "--no-save",
    ])
    assert result.exit_code == 0
    assert "Possible" in result.output  # throttling detected


def test_cli_stress_help():
    from click.testing import CliRunner

    from atomics.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["stress", "--help"])
    assert result.exit_code == 0
    assert "saturation" in result.output.lower()
    assert "--max-concurrency" in result.output
    assert "--phase-seconds" in result.output
    assert "--num-predict" in result.output
