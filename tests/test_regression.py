"""Tests for atomics regression — baseline tracking and comparison."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from atomics.regression import (
    BaselineRecord,
    RegressionReport,
    _pct_delta,
    compute_regression,
    list_baselines,
    load_baseline,
    save_baseline,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_db() -> sqlite3.Connection:
    from atomics.storage.schema import init_db
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    return init_db(path)


def _stable_baseline(**kwargs) -> BaselineRecord:
    defaults = dict(
        name="test", suite="soak", model="qwen2.5:3b", host="http://gpu:11434",
        avg_tps=150.0, peak_tps=200.0, avg_p95_ms=10000.0,
        error_rate=0.0, verdict="STABLE", concurrency=4, notes="",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    defaults.update(kwargs)
    return BaselineRecord(**defaults)


# ── _pct_delta ────────────────────────────────────────────────────────────────


class TestPctDelta:
    def test_no_change(self):
        assert _pct_delta(100.0, 100.0) == 0.0

    def test_improvement(self):
        assert _pct_delta(120.0, 100.0) == pytest.approx(20.0)

    def test_regression(self):
        assert _pct_delta(80.0, 100.0) == pytest.approx(-20.0)

    def test_zero_baseline_returns_zero(self):
        assert _pct_delta(50.0, 0.0) == 0.0

    def test_full_drop(self):
        assert _pct_delta(0.0, 100.0) == pytest.approx(-100.0)


# ── compute_regression ────────────────────────────────────────────────────────


class TestComputeRegression:
    def test_stable_no_change(self):
        bl = _stable_baseline()
        report = compute_regression(bl, 150.0, 200.0, 10000.0, 0.0, "STABLE")
        assert report.status == "STABLE"
        assert report.avg_tps_delta_pct == pytest.approx(0.0)
        assert report.p95_delta_pct == pytest.approx(0.0)
        assert not report.verdict_changed

    def test_regressed_throughput_drop(self):
        bl = _stable_baseline(avg_tps=150.0)
        report = compute_regression(bl, 120.0, 180.0, 10000.0, 0.0, "STABLE")
        assert report.status == "REGRESSED"
        assert report.avg_tps_delta_pct == pytest.approx(-20.0)

    def test_regressed_latency_spike(self):
        bl = _stable_baseline(avg_p95_ms=10000.0)
        report = compute_regression(bl, 150.0, 200.0, 13000.0, 0.0, "STABLE")
        assert report.status == "REGRESSED"
        assert report.p95_delta_pct == pytest.approx(30.0)

    def test_regressed_verdict_degraded_to_unstable(self):
        bl = _stable_baseline(verdict="STABLE")
        report = compute_regression(bl, 150.0, 200.0, 10000.0, 0.0, "UNSTABLE")
        assert report.status == "REGRESSED"
        assert report.verdict_changed

    def test_improved(self):
        bl = _stable_baseline(avg_tps=100.0, avg_p95_ms=15000.0)
        report = compute_regression(bl, 120.0, 160.0, 12000.0, 0.0, "STABLE")
        assert report.status == "IMPROVED"
        assert report.avg_tps_delta_pct == pytest.approx(20.0)

    def test_error_rate_delta_computed(self):
        bl = _stable_baseline(error_rate=0.01)
        report = compute_regression(bl, 150.0, 200.0, 10000.0, 0.05, "STABLE")
        assert report.error_rate_delta == pytest.approx(0.04)

    def test_verdict_unchanged_flag(self):
        bl = _stable_baseline(verdict="DEGRADED")
        report = compute_regression(bl, 150.0, 200.0, 10000.0, 0.0, "DEGRADED")
        assert not report.verdict_changed

    def test_small_regression_stays_stable(self):
        """Sub-threshold drop (-5%) does not trigger REGRESSED."""
        bl = _stable_baseline(avg_tps=150.0)
        report = compute_regression(bl, 143.0, 190.0, 10000.0, 0.0, "STABLE")
        assert report.status == "STABLE"

    def test_boundary_tps_exactly_at_threshold(self):
        """Exactly -10% avg_tps triggers REGRESSED."""
        bl = _stable_baseline(avg_tps=150.0)
        report = compute_regression(bl, 135.0, 200.0, 10000.0, 0.0, "STABLE")
        assert report.status == "REGRESSED"


# ── DB persistence ────────────────────────────────────────────────────────────


class TestSaveAndLoadBaseline:
    def test_save_and_load(self):
        conn = _make_db()
        bl = save_baseline(
            conn, name="gpu-host-3b", suite="soak",
            model="qwen2.5:3b", host="http://gpu:11434",
            avg_tps=140.0, peak_tps=160.0, avg_p95_ms=14000.0,
            error_rate=0.0, verdict="STABLE", concurrency=2,
        )
        loaded = load_baseline(conn, "gpu-host-3b", "soak")
        assert loaded is not None
        assert loaded.name == "gpu-host-3b"
        assert loaded.avg_tps == pytest.approx(140.0)
        assert loaded.peak_tps == pytest.approx(160.0)
        assert loaded.verdict == "STABLE"
        assert loaded.model == "qwen2.5:3b"
        conn.close()

    def test_load_missing_returns_none(self):
        conn = _make_db()
        assert load_baseline(conn, "nonexistent", "soak") is None
        conn.close()

    def test_upsert_overwrites(self):
        conn = _make_db()
        save_baseline(conn, "bl", "soak", "m", "h", 100.0, 120.0, 10000.0, 0.0, "STABLE", 2)
        save_baseline(conn, "bl", "soak", "m", "h", 200.0, 240.0, 8000.0, 0.0, "STABLE", 4)
        loaded = load_baseline(conn, "bl", "soak")
        assert loaded.avg_tps == pytest.approx(200.0)
        assert loaded.concurrency == 4
        conn.close()

    def test_different_suites_are_independent(self):
        conn = _make_db()
        save_baseline(conn, "x", "soak", "m", "h", 100.0, 120.0, 10000.0, 0.0, "STABLE", 2)
        save_baseline(conn, "x", "stress", "m", "h", 300.0, 350.0, 5000.0, 0.0, "STABLE", 8)
        soak_bl = load_baseline(conn, "x", "soak")
        stress_bl = load_baseline(conn, "x", "stress")
        assert soak_bl.avg_tps == pytest.approx(100.0)
        assert stress_bl.avg_tps == pytest.approx(300.0)
        conn.close()

    def test_list_baselines_empty(self):
        conn = _make_db()
        assert list_baselines(conn) == []
        conn.close()

    def test_list_baselines_multiple(self):
        conn = _make_db()
        save_baseline(conn, "b", "soak", "m", "h", 100.0, 120.0, 10000.0, 0.0, "STABLE", 2)
        save_baseline(conn, "a", "soak", "m", "h", 200.0, 240.0, 8000.0, 0.0, "STABLE", 4)
        save_baseline(conn, "c", "stress", "m", "h", 300.0, 350.0, 5000.0, 0.0, "STABLE", 8)
        records = list_baselines(conn)
        assert len(records) == 3
        names = [r.name for r in records]
        assert "a" in names and "b" in names and "c" in names
        conn.close()

    def test_notes_field_persisted(self):
        conn = _make_db()
        save_baseline(conn, "n", "soak", "m", "h", 100.0, 120.0, 10000.0, 0.0, "STABLE", 2,
                      notes="post-tuning gpu-host")
        loaded = load_baseline(conn, "n", "soak")
        assert loaded.notes == "post-tuning gpu-host"
        conn.close()


# ── Schema version ────────────────────────────────────────────────────────────


class TestSchemaVersion:
    def test_schema_version_is_current(self):
        from atomics.storage.schema import SCHEMA_VERSION
        assert SCHEMA_VERSION == 13

    def test_baselines_table_exists(self):
        conn = _make_db()
        conn.execute(
            "INSERT INTO baselines (baseline_id, name, suite, model, host, timestamp) "
            "VALUES ('x','y','soak','m','h','2026-01-01')"
        )
        conn.commit()
        row = conn.execute("SELECT name FROM baselines WHERE baseline_id='x'").fetchone()
        assert row[0] == "y"
        conn.close()
