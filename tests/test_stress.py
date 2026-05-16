"""Tests for the GPU stress testing module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from atomics.stress import (
    ConcurrencyResult,
    StressResult,
    _percentile,
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
