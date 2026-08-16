"""Stress, soak, sweep, scenario, and labcompare results."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from atomics.storage.repository._base import RepositoryBase

if TYPE_CHECKING:
    from atomics.scenario_models import ScenarioResult
    from atomics.soak import SoakResult
    from atomics.stress import StressResult
    from atomics.sweep import ModelSweepResult


class LoadMixin(RepositoryBase):
    def save_stress_result(self, sr: StressResult) -> None:
        """Persist a StressResult from atomics.stress."""
        import json as _json

        result_id = uuid.uuid4().hex[:12]
        now = datetime.now(UTC).isoformat()

        phases_data = [
            {
                "concurrency": p.concurrency,
                "requests": p.requests,
                "failed": p.failed,
                "total_output_tokens": p.total_output_tokens,
                "aggregate_tps": round(p.aggregate_tps, 2),
                "avg_request_tps": round(p.avg_request_tps, 2),
                "avg_latency_ms": round(p.avg_latency_ms, 2),
                "p95_latency_ms": round(p.p95_latency_ms, 2),
            }
            for p in sr.phases
        ]

        self._conn.execute(
            """
            INSERT INTO stress_results
            (result_id, model, host, peak_tps, saturation_concurrency,
             duration_seconds, total_tokens, total_requests, total_failed,
             total_phases, gpu_name, vram_total_mb, vram_peak_mb,
             phases_json, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result_id, sr.model, sr.host,
                round(sr.peak_tps, 2), sr.saturation_concurrency,
                round(sr.duration_seconds, 2), sr.total_tokens,
                sr.total_requests, sr.total_failed,
                len(sr.phases), sr.gpu_name or "",
                sr.vram_total_mb, sr.vram_peak_mb,
                _json.dumps(phases_data), now,
            ),
        )
        self._conn.commit()

    def get_stress_results(self, *, model: str | None = None) -> list[dict]:
        """Retrieve stress results, optionally filtered by model."""
        if model:
            rows = self._conn.execute(
                "SELECT * FROM stress_results WHERE model = ? ORDER BY timestamp DESC",
                (model,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM stress_results ORDER BY timestamp DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def save_sweep_result(self, sr: ModelSweepResult) -> None:
        """Persist a ModelSweepResult to the sweep_results table."""
        import uuid
        now = datetime.now(UTC).isoformat()
        result_id = str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT OR REPLACE INTO sweep_results
            (result_id, model, provider, quality, avg_latency_ms,
             total_tokens, total_cost_usd, fixtures_run, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result_id,
                getattr(sr, "model", ""),
                getattr(sr, "provider", ""),
                round(getattr(sr, "overall_quality", None) or 0.0, 4),
                round(getattr(sr, "avg_latency_ms", 0.0), 2),
                getattr(sr, "total_tokens", 0),
                round(getattr(sr, "total_cost_usd", 0.0), 6),
                getattr(sr, "fixtures_run", 0),
                now,
            ),
        )
        self._conn.commit()

    def get_sweep_results(self, *, model: str | None = None) -> list[dict]:
        """Retrieve sweep results, optionally filtered by model."""
        if model:
            rows = self._conn.execute(
                "SELECT * FROM sweep_results WHERE model = ? ORDER BY timestamp DESC",
                (model,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM sweep_results ORDER BY timestamp DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Soak results ──────────────────────────────────────

    def save_soak_result(self, sr: SoakResult) -> None:
        """Persist a SoakResult from atomics.soak."""
        import json as _json

        result_id = uuid.uuid4().hex[:12]
        now = datetime.now(UTC).isoformat()

        samples_data = [
            {
                "elapsed_seconds": s.elapsed_seconds,
                "requests": s.requests,
                "failed": s.failed,
                "total_output_tokens": s.total_output_tokens,
                "aggregate_tps": round(s.aggregate_tps, 2),
                "avg_latency_ms": round(s.avg_latency_ms, 2),
                "p95_latency_ms": round(s.p95_latency_ms, 2),
                "vram_used_mb": s.vram_used_mb,
            }
            for s in sr.samples
        ]

        self._conn.execute(
            """
            INSERT INTO soak_results
            (result_id, model, host, provider, concurrency, duration_seconds,
             actual_duration_seconds, sample_interval, total_requests, total_failed,
             total_tokens, avg_tps, peak_tps, min_tps,
             throughput_drift_pct, latency_drift_pct, avg_p95_ms,
             vram_start_mb, vram_end_mb, vram_drift_mb,
             error_rate, verdict, total_cost_usd, samples_json, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result_id, sr.model, sr.host, sr.provider,
                sr.concurrency, round(sr.duration_seconds, 2),
                round(sr.actual_duration_seconds, 2), sr.sample_interval,
                sr.total_requests, sr.total_failed, sr.total_tokens,
                round(sr.avg_tps, 2), round(sr.peak_tps, 2), round(sr.min_tps, 2),
                round(sr.throughput_drift_pct, 2), round(sr.latency_drift_pct, 2),
                round(sr.avg_p95_ms, 2),
                sr.vram_start_mb, sr.vram_end_mb, sr.vram_drift_mb,
                round(sr.error_rate, 6), sr.verdict,
                round(sr.total_cost_usd, 6),
                _json.dumps(samples_data), now,
            ),
        )
        self._conn.commit()

    def get_soak_results(self, *, model: str | None = None) -> list[dict]:
        """Retrieve soak results, optionally filtered by model."""
        if model:
            rows = self._conn.execute(
                "SELECT * FROM soak_results WHERE model = ? ORDER BY timestamp DESC",
                (model,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM soak_results ORDER BY timestamp DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Scenario results ──────────────────────────────────

    def save_scenario_result(self, sr: ScenarioResult) -> None:
        """Persist a ScenarioResult from atomics.scenario."""
        import json as _json

        result_id = uuid.uuid4().hex[:12]
        now = datetime.now(UTC).isoformat()

        workloads_data = [
            {
                "name": wr.spec.name,
                "type": wr.spec.type,
                "model": wr.spec.model,
                "concurrency": wr.spec.concurrency,
                "requests": wr.requests,
                "failed": wr.failed,
                "p50_ms": round(wr.p50_ms, 2),
                "p95_ms": round(wr.p95_ms, 2),
                "avg_tps": round(wr.avg_tps, 2),
                "sla_ms": wr.spec.sla_ms,
                "sla_compliance_pct": round(wr.sla_compliance_pct, 2),
            }
            for wr in sr.workloads
        ]

        max_intf = max(sr.interference.values()) if sr.interference else None

        self._conn.execute(
            """
            INSERT INTO scenario_results
            (result_id, duration_seconds, total_requests, total_failed,
             workload_count, max_interference,
             workloads_json, interference_json, baselines_json, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result_id,
                round(sr.duration_seconds, 2),
                sr.total_requests,
                sr.total_failed,
                len(sr.workloads),
                round(max_intf, 4) if max_intf is not None else None,
                _json.dumps(workloads_data),
                _json.dumps({k: round(v, 4) for k, v in sr.interference.items()}),
                _json.dumps({k: round(v, 2) for k, v in sr.baselines.items()}),
                now,
            ),
        )
        self._conn.commit()

    def save_labcompare_result(
        self,
        *,
        comparison_run_id: str,
        host_name: str,
        host_url: str,
        model: str,
        tokens_per_second: float | None,
        latency_ms: float | None,
        prompt_eval_rate: float | None,
        vram_fit_pct: float | None,
        gpu_name: str | None,
        quality_score: float | None,
        quality_suite: str | None,
        judge_model: str | None,
        dimensions: str,
    ) -> None:
        """Persist one host × model cell from a labcompare run."""
        self._conn.execute(
            """
            INSERT INTO labcompare_results (
                comparison_run_id, created_at, host_name, host_url, model,
                tokens_per_second, latency_ms, prompt_eval_rate, vram_fit_pct,
                gpu_name, quality_score, quality_suite, judge_model, dimensions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                comparison_run_id, datetime.now(UTC).isoformat(), host_name,
                host_url, model, tokens_per_second, latency_ms, prompt_eval_rate,
                vram_fit_pct, gpu_name, quality_score, quality_suite, judge_model,
                dimensions,
            ),
        )
        self._conn.commit()

    def get_labcompare_run(self, comparison_run_id: str) -> list[dict]:
        """Return all rows for one labcompare invocation, ordered model/host."""
        rows = self._conn.execute(
            "SELECT * FROM labcompare_results WHERE comparison_run_id = ? "
            "ORDER BY model, host_name",
            (comparison_run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

