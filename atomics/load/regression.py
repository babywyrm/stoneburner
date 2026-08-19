"""Baseline regression tracking for soak and stress results.

Usage:
    atomics soak --model qwen2.5:3b -d 10m --save-baseline gpu-host-3b
    atomics soak --model qwen2.5:3b -d 10m --compare-baseline gpu-host-3b
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass
class BaselineRecord:
    name: str
    suite: str
    model: str
    host: str
    avg_tps: float
    peak_tps: float
    avg_p95_ms: float
    error_rate: float
    verdict: str
    concurrency: int
    notes: str
    timestamp: str


@dataclass
class RegressionReport:
    baseline: BaselineRecord
    avg_tps_delta_pct: float
    peak_tps_delta_pct: float
    p95_delta_pct: float
    error_rate_delta: float
    verdict_changed: bool
    current_verdict: str
    status: str  # "IMPROVED" | "STABLE" | "REGRESSED"


_REGRESSION_THRESHOLD_TPS: float = -10.0   # >10% throughput drop = regression
_REGRESSION_THRESHOLD_P95: float = 20.0    # >20% latency increase = regression
_IMPROVEMENT_THRESHOLD_TPS: float = 10.0   # >10% throughput gain = improvement


def _pct_delta(current: float, baseline: float) -> float:
    """Percentage change from baseline to current. Returns 0 if baseline is 0."""
    if baseline == 0:
        return 0.0
    return ((current - baseline) / baseline) * 100.0


def compute_regression(
    baseline: BaselineRecord,
    current_avg_tps: float,
    current_peak_tps: float,
    current_avg_p95_ms: float,
    current_error_rate: float,
    current_verdict: str,
) -> RegressionReport:
    """Compare current run metrics against a stored baseline."""
    tps_delta = _pct_delta(current_avg_tps, baseline.avg_tps)
    peak_delta = _pct_delta(current_peak_tps, baseline.peak_tps)
    p95_delta = _pct_delta(current_avg_p95_ms, baseline.avg_p95_ms)
    err_delta = current_error_rate - baseline.error_rate

    regressed = (
        tps_delta <= _REGRESSION_THRESHOLD_TPS
        or p95_delta >= _REGRESSION_THRESHOLD_P95
        or current_verdict == "UNSTABLE" and baseline.verdict != "UNSTABLE"
    )
    improved = (
        tps_delta >= _IMPROVEMENT_THRESHOLD_TPS
        and p95_delta <= 0
        and not regressed
    )

    if regressed:
        status = "REGRESSED"
    elif improved:
        status = "IMPROVED"
    else:
        status = "STABLE"

    return RegressionReport(
        baseline=baseline,
        avg_tps_delta_pct=round(tps_delta, 2),
        peak_tps_delta_pct=round(peak_delta, 2),
        p95_delta_pct=round(p95_delta, 2),
        error_rate_delta=round(err_delta, 4),
        verdict_changed=current_verdict != baseline.verdict,
        current_verdict=current_verdict,
        status=status,
    )


def save_baseline(
    conn: sqlite3.Connection,
    name: str,
    suite: str,
    model: str,
    host: str,
    avg_tps: float,
    peak_tps: float,
    avg_p95_ms: float,
    error_rate: float,
    verdict: str,
    concurrency: int,
    notes: str = "",
) -> BaselineRecord:
    """Upsert a named baseline into the DB. Overwrites if name+suite already exists."""
    now = datetime.now(UTC).isoformat()
    bid = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO baselines
            (baseline_id, name, suite, model, host, avg_tps, peak_tps,
             avg_p95_ms, error_rate, verdict, concurrency, notes, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name, suite) DO UPDATE SET
            baseline_id = excluded.baseline_id,
            model       = excluded.model,
            host        = excluded.host,
            avg_tps     = excluded.avg_tps,
            peak_tps    = excluded.peak_tps,
            avg_p95_ms  = excluded.avg_p95_ms,
            error_rate  = excluded.error_rate,
            verdict     = excluded.verdict,
            concurrency = excluded.concurrency,
            notes       = excluded.notes,
            timestamp   = excluded.timestamp
        """,
        (bid, name, suite, model, host, avg_tps, peak_tps,
         avg_p95_ms, error_rate, verdict, concurrency, notes, now),
    )
    conn.commit()
    return BaselineRecord(
        name=name, suite=suite, model=model, host=host,
        avg_tps=avg_tps, peak_tps=peak_tps, avg_p95_ms=avg_p95_ms,
        error_rate=error_rate, verdict=verdict, concurrency=concurrency,
        notes=notes, timestamp=now,
    )


def load_baseline(conn: sqlite3.Connection, name: str, suite: str = "soak") -> BaselineRecord | None:
    """Load a named baseline from the DB. Returns None if not found."""
    row = conn.execute(
        "SELECT * FROM baselines WHERE name = ? AND suite = ?",
        (name, suite),
    ).fetchone()
    if row is None:
        return None
    return BaselineRecord(
        name=row["name"],
        suite=row["suite"],
        model=row["model"],
        host=row["host"],
        avg_tps=row["avg_tps"],
        peak_tps=row["peak_tps"],
        avg_p95_ms=row["avg_p95_ms"],
        error_rate=row["error_rate"],
        verdict=row["verdict"],
        concurrency=row["concurrency"],
        notes=row["notes"] or "",
        timestamp=row["timestamp"],
    )


def list_baselines(conn: sqlite3.Connection) -> list[BaselineRecord]:
    """Return all stored baselines ordered by suite, name."""
    rows = conn.execute(
        "SELECT * FROM baselines ORDER BY suite, name"
    ).fetchall()
    return [
        BaselineRecord(
            name=r["name"], suite=r["suite"], model=r["model"], host=r["host"],
            avg_tps=r["avg_tps"], peak_tps=r["peak_tps"], avg_p95_ms=r["avg_p95_ms"],
            error_rate=r["error_rate"], verdict=r["verdict"], concurrency=r["concurrency"],
            notes=r["notes"] or "", timestamp=r["timestamp"],
        )
        for r in rows
    ]
