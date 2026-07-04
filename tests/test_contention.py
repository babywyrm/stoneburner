"""Tests for atomics contention — multi-model VRAM contention testing."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from atomics.contention import (
    ContentionModelResult,
    ContentionResult,
    _percentile,
    run_contention,
)

# ── ContentionModelResult ─────────────────────────────────────────────────────


class TestContentionModelResult:
    def test_avg_tps_empty(self):
        r = ContentionModelResult(model="test")
        assert r.avg_tps == 0.0

    def test_avg_tps_computed(self):
        r = ContentionModelResult(model="test", per_request_tps=[10.0, 20.0, 30.0])
        assert r.avg_tps == pytest.approx(20.0)

    def test_p95_ms(self):
        r = ContentionModelResult(model="t", latencies=[100.0, 200.0, 300.0, 400.0, 500.0])
        assert r.p95_ms > 400.0

    def test_avg_latency_ms(self):
        r = ContentionModelResult(model="t", latencies=[100.0, 200.0, 300.0])
        assert r.avg_latency_ms == pytest.approx(200.0)

    def test_error_rate_zero(self):
        r = ContentionModelResult(model="t", requests=10)
        assert r.error_rate == 0.0

    def test_error_rate_partial(self):
        r = ContentionModelResult(model="t", requests=9, failed=1)
        assert r.error_rate == pytest.approx(0.1)

    def test_error_rate_no_requests(self):
        r = ContentionModelResult(model="t")
        assert r.error_rate == 0.0


# ── ContentionResult ──────────────────────────────────────────────────────────


class TestContentionResult:
    def _make_result(self, solo_a=100.0, mixed_a=70.0) -> ContentionResult:
        r = ContentionResult(host="http://h", models=["a", "b"], phase_seconds=20.0)
        r.solo_tps = {"a": solo_a, "b": 80.0}
        ra = ContentionModelResult(model="a", per_request_tps=[mixed_a])
        rb = ContentionModelResult(model="b", per_request_tps=[60.0])
        r.contention_results = [ra, rb]
        return r

    def test_contention_factor_below_one_means_degradation(self):
        r = self._make_result(solo_a=100.0, mixed_a=70.0)
        assert r.contention_factor("a") == pytest.approx(0.7)

    def test_contention_factor_one_means_no_degradation(self):
        r = self._make_result(solo_a=100.0, mixed_a=100.0)
        assert r.contention_factor("a") == pytest.approx(1.0)

    def test_contention_factor_missing_model(self):
        r = self._make_result()
        assert r.contention_factor("nonexistent") is None

    def test_contention_factor_zero_solo(self):
        r = self._make_result(solo_a=0.0, mixed_a=50.0)
        assert r.contention_factor("a") is None

    def test_models_list_preserved(self):
        r = ContentionResult(host="h", models=["x", "y"], phase_seconds=10.0)
        assert r.models == ["x", "y"]


# ── Percentile helper ─────────────────────────────────────────────────────────


class TestPercentile:
    def test_empty(self):
        assert _percentile([], 95) == 0.0

    def test_single(self):
        assert _percentile([42.0], 95) == pytest.approx(42.0)

    def test_p50(self):
        vals = sorted([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _percentile(vals, 50) == pytest.approx(3.0)

    def test_p95_five_values(self):
        vals = sorted([100.0, 200.0, 300.0, 400.0, 500.0])
        assert _percentile(vals, 95) > 400.0


# ── run_contention (mocked) ───────────────────────────────────────────────────


class TestRunContention:
    @pytest.mark.asyncio
    async def test_solo_phase_runs_each_model(self):
        call_models: list[str] = []

        async def _fake_request(client, host, model, prompt, num_predict):
            call_models.append(model)
            await asyncio.sleep(0.001)
            return (50, 10, 200.0, 100.0)

        with patch("atomics.stress._single_request", side_effect=_fake_request):
            result = await run_contention(
                host="http://fake:11434",
                models=["alpha", "beta"],
                concurrency=1,
                phase_seconds=0.05,
                num_predict=16,
            )

        assert "alpha" in result.solo_tps
        assert "beta" in result.solo_tps
        assert len(result.contention_results) == 2

    @pytest.mark.asyncio
    async def test_contention_factor_computed(self):
        call_count: dict[str, int] = {}

        async def _fake_request(client, host, model, prompt, num_predict):
            call_count[model] = call_count.get(model, 0) + 1
            await asyncio.sleep(0.001)
            return (50, 10, 200.0, 80.0)

        with patch("atomics.stress._single_request", side_effect=_fake_request):
            result = await run_contention(
                host="http://fake:11434",
                models=["m1", "m2"],
                concurrency=1,
                phase_seconds=0.05,
                num_predict=16,
            )

        for model in ["m1", "m2"]:
            factor = result.contention_factor(model)
            assert factor is not None
            assert factor > 0

    @pytest.mark.asyncio
    async def test_duration_recorded(self):
        async def _fake_request(client, host, model, prompt, num_predict):
            await asyncio.sleep(0.001)
            return (50, 10, 200.0, 80.0)

        with patch("atomics.stress._single_request", side_effect=_fake_request):
            result = await run_contention(
                host="http://fake:11434",
                models=["m"],
                phase_seconds=0.05,
                num_predict=16,
            )

        assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_handles_request_failures_gracefully(self):
        call_count = 0

        async def _flaky_request(client, host, model, prompt, num_predict):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise ConnectionError("simulated failure")
            return (50, 10, 200.0, 80.0)

        with patch("atomics.stress._single_request", side_effect=_flaky_request):
            result = await run_contention(
                host="http://fake:11434",
                models=["m"],
                phase_seconds=0.05,
                num_predict=16,
            )

        assert result.duration_seconds > 0
        mr = result.contention_results[0]
        assert mr.failed >= 0


# ── CLI ───────────────────────────────────────────────────────────────────────


class TestContentionCLI:
    def test_stress_models_flag_in_help(self):
        from click.testing import CliRunner

        from atomics.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["stress", "--help"])
        assert "--models" in result.output

    def test_models_flag_triggers_contention_path(self):
        from click.testing import CliRunner

        from atomics.cli import cli
        from atomics.contention import ContentionResult

        captured: dict = {}

        async def _fake_contention(**kwargs):
            captured.update(kwargs)
            r = ContentionResult(
                host="http://fake:11434",
                models=["a", "b"],
                phase_seconds=5.0,
            )
            return r

        runner = CliRunner()
        with patch("atomics.contention.run_contention", side_effect=_fake_contention):
            result = runner.invoke(cli, [
                "stress",
                "--models", "qwen2.5:3b,qwen2.5:7b",
                "--ollama-host", "http://fake:11434",
                "--no-save",
            ])

        assert result.exit_code == 0
        assert captured.get("models") == ["qwen2.5:3b", "qwen2.5:7b"]


# ── avg_latency_ms empty branch + contention_factor None branches ─────────────

def test_contention_model_result_avg_latency_empty():
    from atomics.contention import ContentionModelResult
    r = ContentionModelResult(model="m", requests=0, failed=0, latencies=[])
    assert r.avg_latency_ms == 0.0


def test_contention_factor_mixed_none():
    from atomics.contention import ContentionModelResult, ContentionResult
    cr = ContentionResult(host="h", models=["a", "b"], phase_seconds=5.0)
    cr.solo_tps = {"a": 10.0, "b": 5.0}
    # "a" not in contention_results → mixed avg_tps = 0 (no per_request_tps),
    # but the next() lookup won't find "a" via per_request_tps, and avg_tps
    # is computed as 0.0 — which is not None, so let's use a model not present
    cr.contention_results = [ContentionModelResult(model="b", requests=10,
                                                    failed=0, latencies=[100.0])]
    # "a" has no entry → next() returns None → factor is None
    assert cr.contention_factor("a") is None


def test_contention_factor_solo_zero():
    from atomics.contention import ContentionModelResult, ContentionResult
    cr = ContentionResult(host="h", models=["a"], phase_seconds=5.0)
    cr.solo_tps = {"a": 0.0}  # solo == 0 → factor is None
    r = ContentionModelResult(model="a", requests=5, failed=0, latencies=[80.0])
    r.per_request_tps = [3.0]
    cr.contention_results = [r]
    assert cr.contention_factor("a") is None
