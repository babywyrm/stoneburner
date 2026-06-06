"""Tests for the capacity projection simulator."""

from __future__ import annotations

import pytest

from atomics.capacity import (
    CapacityScenario,
    CapacityProjection,
    LoadProfile,
    project_capacity,
    interpolate_latency,
    estimate_concurrency,
)


# ── Stress data fixtures ─────────────────────────────────────────────────────

GPU_HOST_PHASES = [
    {"concurrency": 1, "aggregate_tps": 83.7, "avg_latency_ms": 14900, "p95_latency_ms": 20500},
    {"concurrency": 2, "aggregate_tps": 84.9, "avg_latency_ms": 20900, "p95_latency_ms": 24200},
    {"concurrency": 4, "aggregate_tps": 84.5, "avg_latency_ms": 41400, "p95_latency_ms": 60400},
    {"concurrency": 8, "aggregate_tps": 105.9, "avg_latency_ms": 48900, "p95_latency_ms": 73700},
    {"concurrency": 16, "aggregate_tps": 107.0, "avg_latency_ms": 103800, "p95_latency_ms": 161900},
]


# ── estimate_concurrency (Little's Law) ──────────────────────────────────────

def test_estimate_concurrency_basic():
    """200 users, 5 min think time, 15s avg response."""
    conc = estimate_concurrency(users=200, think_time_s=300.0, avg_response_s=15.0)
    assert 9.0 <= conc <= 10.0


def test_estimate_concurrency_fast_responses():
    """Fast 2s responses with long think time = very low concurrency."""
    conc = estimate_concurrency(users=100, think_time_s=600.0, avg_response_s=2.0)
    assert conc < 1.0


def test_estimate_concurrency_high_load():
    """Many users with short think time = high concurrency."""
    conc = estimate_concurrency(users=500, think_time_s=60.0, avg_response_s=10.0)
    assert conc > 50


def test_estimate_concurrency_zero_users():
    conc = estimate_concurrency(users=0, think_time_s=300.0, avg_response_s=15.0)
    assert conc == 0.0


# ── interpolate_latency ──────────────────────────────────────────────────────

def test_interpolate_latency_exact_match():
    """Exact concurrency level should return exact data."""
    lat = interpolate_latency(concurrency=1.0, phases=GPU_HOST_PHASES)
    assert lat == 14900


def test_interpolate_latency_between_points():
    """Concurrency 3 should interpolate between 2 and 4."""
    lat = interpolate_latency(concurrency=3.0, phases=GPU_HOST_PHASES)
    assert 20900 < lat < 41400


def test_interpolate_latency_below_min():
    """Concurrency below 1 should use the lowest phase."""
    lat = interpolate_latency(concurrency=0.5, phases=GPU_HOST_PHASES)
    assert lat == 14900


def test_interpolate_latency_above_max():
    """Concurrency above max should extrapolate linearly."""
    lat = interpolate_latency(concurrency=32.0, phases=GPU_HOST_PHASES)
    assert lat > 103800


def test_interpolate_latency_p95():
    """Should also support P95 interpolation."""
    lat = interpolate_latency(concurrency=1.0, phases=GPU_HOST_PHASES, percentile="p95")
    assert lat == 20500


# ── project_capacity ─────────────────────────────────────────────────────────

def test_project_capacity_returns_scenarios():
    profile = LoadProfile(users=200, think_time_s=300.0, response_tokens=400)
    result = project_capacity(profile=profile, phases=GPU_HOST_PHASES, peak_tps=107.0)
    assert isinstance(result, CapacityProjection)
    assert len(result.scenarios) >= 3
    assert result.model or result.model == ""


def test_project_capacity_scenario_fields():
    profile = LoadProfile(users=200, think_time_s=300.0, response_tokens=400)
    result = project_capacity(profile=profile, phases=GPU_HOST_PHASES, peak_tps=107.0)
    for s in result.scenarios:
        assert isinstance(s, CapacityScenario)
        assert s.name
        assert s.concurrent >= 0
        assert s.p50_latency_ms >= 0
        assert s.verdict in ("OK", "CAUTION", "SLOW", "OVERLOAD")


def test_project_capacity_normal_50_users_5min():
    """50 users at 5 min think time on the GPU host should be manageable."""
    profile = LoadProfile(users=50, think_time_s=300.0, response_tokens=400)
    result = project_capacity(profile=profile, phases=GPU_HOST_PHASES, peak_tps=107.0)
    normal = next(s for s in result.scenarios if s.name == "Normal")
    assert normal.concurrent < 10
    assert normal.verdict in ("OK", "CAUTION")


def test_project_capacity_200_users_is_heavy():
    """200 users at 5 min think time on a single GPU will be slow."""
    profile = LoadProfile(users=200, think_time_s=300.0, response_tokens=400)
    result = project_capacity(profile=profile, phases=GPU_HOST_PHASES, peak_tps=107.0)
    normal = next(s for s in result.scenarios if s.name == "Normal")
    assert normal.verdict in ("SLOW", "OVERLOAD")


def test_project_capacity_burst_is_worse():
    """Burst scenario should always have higher latency than normal."""
    profile = LoadProfile(users=200, think_time_s=300.0, response_tokens=400)
    result = project_capacity(profile=profile, phases=GPU_HOST_PHASES, peak_tps=107.0)
    normal = next(s for s in result.scenarios if s.name == "Normal")
    burst = next(s for s in result.scenarios if "Burst" in s.name)
    assert burst.p50_latency_ms > normal.p50_latency_ms


def test_project_capacity_small_user_count():
    """10 users should be easily OK."""
    profile = LoadProfile(users=10, think_time_s=300.0, response_tokens=400)
    result = project_capacity(profile=profile, phases=GPU_HOST_PHASES, peak_tps=107.0)
    normal = next(s for s in result.scenarios if s.name == "Normal")
    assert normal.verdict == "OK"


def test_project_capacity_overload():
    """1000 users at 30s think time should overload a single GPU."""
    profile = LoadProfile(users=1000, think_time_s=30.0, response_tokens=400)
    result = project_capacity(profile=profile, phases=GPU_HOST_PHASES, peak_tps=107.0)
    normal = next(s for s in result.scenarios if s.name == "Normal")
    assert normal.verdict in ("SLOW", "OVERLOAD")


def test_project_capacity_recommendation():
    """Projection should include a recommendation string."""
    profile = LoadProfile(users=200, think_time_s=300.0, response_tokens=400)
    result = project_capacity(profile=profile, phases=GPU_HOST_PHASES, peak_tps=107.0)
    assert result.recommendation
    assert len(result.recommendation) > 10


def test_project_capacity_with_cloud_latency():
    """Should work with cloud-style latency data (higher latency, no scaling)."""
    cloud_phases = [
        {"concurrency": 1, "aggregate_tps": 50.0, "avg_latency_ms": 3000, "p95_latency_ms": 5000},
    ]
    profile = LoadProfile(users=50, think_time_s=300.0, response_tokens=400)
    result = project_capacity(
        profile=profile, phases=cloud_phases, peak_tps=50.0, model="claude-sonnet-4-6",
    )
    assert result.model == "claude-sonnet-4-6"
    assert len(result.scenarios) >= 3


def test_project_capacity_custom_burst():
    """Custom burst factor should scale concurrency."""
    profile = LoadProfile(users=100, think_time_s=300.0, response_tokens=400, burst_factor=0.5)
    result = project_capacity(profile=profile, phases=GPU_HOST_PHASES, peak_tps=107.0)
    burst = next(s for s in result.scenarios if "Burst" in s.name)
    assert burst.concurrent >= 50 * 0.3  # at least some spike


# ── CLI tests ────────────────────────────────────────────────────────────────

def test_cli_capacity_from_manual_params(monkeypatch, tmp_path):
    """atomics capacity with manual --peak-tps and --single-latency should work."""
    from click.testing import CliRunner
    from atomics.cli import cli

    monkeypatch.setenv("ATOMICS_DB_PATH", str(tmp_path / "test.db"))
    runner = CliRunner()
    result = runner.invoke(cli, [
        "capacity", "--users", "100", "--think-time", "300",
        "--peak-tps", "107", "--single-latency", "15000",
    ])
    assert result.exit_code == 0
    assert "Normal" in result.output
    assert "Burst" in result.output
    assert "Recommendation" in result.output


def test_cli_capacity_from_db(monkeypatch, tmp_path):
    """atomics capacity --model should pull stress data from DB."""
    from click.testing import CliRunner
    from atomics.cli import cli
    from atomics.storage.repository import MetricsRepository
    from atomics.stress import StressResult, ConcurrencyResult

    repo = MetricsRepository(tmp_path / "test.db")
    sr = StressResult(
        model="qwen2.5:7b", host="http://localhost:11434",
        peak_tps=107.0, saturation_concurrency=16,
        duration_seconds=60.0, total_tokens=5000, total_requests=20,
        phases=[
            ConcurrencyResult(
                concurrency=1, requests=3, total_output_tokens=1000,
                aggregate_tps=84.0, avg_request_tps=84.0,
                avg_latency_ms=15000.0, p95_latency_ms=20000.0,
            ),
            ConcurrencyResult(
                concurrency=4, requests=6, total_output_tokens=2000,
                aggregate_tps=85.0, avg_request_tps=21.0,
                avg_latency_ms=40000.0, p95_latency_ms=60000.0,
            ),
            ConcurrencyResult(
                concurrency=16, requests=18, total_output_tokens=5000,
                aggregate_tps=107.0, avg_request_tps=7.0,
                avg_latency_ms=100000.0, p95_latency_ms=160000.0,
            ),
        ],
    )
    repo.save_stress_result(sr)
    repo.close()

    monkeypatch.setenv("ATOMICS_DB_PATH", str(tmp_path / "test.db"))
    runner = CliRunner()
    result = runner.invoke(cli, [
        "capacity", "--users", "50", "--model", "qwen2.5:7b",
    ])
    assert result.exit_code == 0
    assert "qwen2.5:7b" in result.output
    assert "Normal" in result.output


def test_cli_capacity_help():
    from click.testing import CliRunner
    from atomics.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["capacity", "--help"])
    assert result.exit_code == 0
    assert "--users" in result.output
    assert "--think-time" in result.output
    assert "--peak-tps" in result.output
