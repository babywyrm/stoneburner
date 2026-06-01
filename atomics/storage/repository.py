"""Repository for persisting and querying run/task metrics."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from atomics.models import RunSummary, TaskResult, TaskStatus
from atomics.storage.schema import init_db


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Compute a percentile from a pre-sorted list."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_values):
        return sorted_values[f]
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


class MetricsRepository:
    def __init__(self, db_path: Path) -> None:
        self._conn = init_db(db_path)

    def close(self) -> None:
        self._conn.close()

    # ── Runs ──────────────────────────────────────────────

    def create_run(
        self,
        run_id: str,
        *,
        tier: str = "baseline",
        provider: str = "claude",
        model: str = "",
        trigger: str = "manual",
    ) -> None:
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "INSERT INTO runs (run_id, started_at, tier, provider, model, trigger) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, now, tier, provider, model, trigger),
        )
        self._conn.commit()

    def complete_run(self, run_id: str) -> RunSummary:
        now = datetime.now(UTC).isoformat()
        rows = self._conn.execute(
            """
            SELECT
                COUNT(*) as total,
                COALESCE(SUM(CASE WHEN status = ? THEN 1 ELSE 0 END), 0) as success,
                COALESCE(SUM(CASE WHEN status = ? THEN 1 ELSE 0 END), 0) as failed,
                COALESCE(SUM(input_tokens), 0) as inp,
                COALESCE(SUM(output_tokens), 0) as outp,
                COALESCE(SUM(total_tokens), 0) as tot,
                COALESCE(SUM(estimated_cost_usd), 0.0) as cost,
                COALESCE(AVG(latency_ms), 0.0) as avg_lat
            FROM task_results WHERE run_id = ?
            """,
            (TaskStatus.SUCCESS.value, TaskStatus.FAILED.value, run_id),
        ).fetchone()

        self._conn.execute(
            """
            UPDATE runs SET
                completed_at = ?,
                total_tasks = ?, successful_tasks = ?, failed_tasks = ?,
                total_input_tokens = ?, total_output_tokens = ?, total_tokens = ?,
                total_cost_usd = ?, avg_latency_ms = ?
            WHERE run_id = ?
            """,
            (now, rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7], run_id),
        )
        self._conn.commit()

        return RunSummary(
            run_id=run_id,
            started_at=datetime.fromisoformat(
                self._conn.execute(
                    "SELECT started_at FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()[0]
            ),
            completed_at=datetime.fromisoformat(now),
            total_tasks=rows[0],
            successful_tasks=rows[1],
            failed_tasks=rows[2],
            total_input_tokens=rows[3],
            total_output_tokens=rows[4],
            total_tokens=rows[5],
            total_cost_usd=rows[6],
            avg_latency_ms=rows[7],
        )

    # ── Task results ──────────────────────────────────────

    def save_task_result(self, result: TaskResult, *, suite: str = "eval") -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO task_results (
                task_id, run_id, category, task_name, provider, model, status,
                suite,
                prompt, response, input_tokens, output_tokens, total_tokens,
                thinking_tokens,
                latency_ms, estimated_cost_usd, tokens_per_second,
                thinking_enabled,
                error_class, error_message,
                started_at, completed_at,
                accuracy_score, judge_model, quality_rationale
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.task_id,
                result.run_id,
                result.category.value,
                result.task_name,
                result.provider,
                result.model,
                result.status.value,
                suite,
                result.prompt,
                result.response,
                result.input_tokens,
                result.output_tokens,
                result.total_tokens,
                result.thinking_tokens,
                result.latency_ms,
                result.estimated_cost_usd,
                result.tokens_per_second,
                int(result.thinking_enabled),
                result.error_class,
                result.error_message,
                result.started_at.isoformat(),
                result.completed_at.isoformat() if result.completed_at else None,
                result.accuracy_score,
                result.judge_model,
                result.quality_rationale,
            ),
        )
        self._conn.commit()

    def save_adversarial_result(
        self,
        run_id: str,
        result: object,
        *,
        thinking_enabled: bool = False,
    ) -> None:
        r = result
        res = r.resistance  # type: ignore[attr-defined]
        self._conn.execute(
            """
            INSERT OR REPLACE INTO adversarial_results (
                result_id, run_id, fixture_id, category, severity,
                provider, model, prompt, response, attack_goal,
                resistance_score, resistance_label, judge_model, judge_rationale,
                thinking_enabled, thinking_tokens, latency_ms, estimated_cost_usd,
                timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                run_id,
                r.fixture.id,  # type: ignore[attr-defined]
                r.fixture.category,  # type: ignore[attr-defined]
                r.fixture.severity,  # type: ignore[attr-defined]
                "",
                "",
                r.fixture.prompt,  # type: ignore[attr-defined]
                r.response,  # type: ignore[attr-defined]
                r.fixture.attack_goal,  # type: ignore[attr-defined]
                res.score if res else None,
                res.label if res else None,
                res.judge_model if res else None,
                res.rationale if res else None,
                int(thinking_enabled),
                r.thinking_tokens,  # type: ignore[attr-defined]
                r.latency_ms,  # type: ignore[attr-defined]
                r.estimated_cost_usd,  # type: ignore[attr-defined]
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()

    def save_probe_result(self, run_id: str, result: object) -> None:
        r = result
        self._conn.execute(
            """
            INSERT OR REPLACE INTO probe_results (
                result_id, run_id, target_name, artifact_type, check_id,
                score, prev_score, regressed,
                provider, model, judge_model, judge_rationale,
                thinking_enabled, thinking_tokens, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                run_id,
                r.target_name,  # type: ignore[attr-defined]
                r.artifact_type,  # type: ignore[attr-defined]
                r.check_id,  # type: ignore[attr-defined]
                r.score,  # type: ignore[attr-defined]
                r.prev_score,  # type: ignore[attr-defined]
                int(r.regressed),  # type: ignore[attr-defined]
                "",
                "",
                r.judge_model,  # type: ignore[attr-defined]
                r.judge_rationale,  # type: ignore[attr-defined]
                int(r.thinking_enabled),  # type: ignore[attr-defined]
                r.thinking_tokens,  # type: ignore[attr-defined]
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()

    # ── Queries ───────────────────────────────────────────

    def get_recent_runs(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_run_tasks(self, run_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM task_results WHERE run_id = ? ORDER BY started_at", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_token_usage_by_hour(self, hours: int = 24) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT
                strftime('%Y-%m-%d %H:00', started_at) as hour,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(estimated_cost_usd) as cost,
                COUNT(*) as task_count
            FROM task_results
            WHERE started_at >= datetime('now', ? || ' hours')
            GROUP BY hour ORDER BY hour
            """,
            (f"-{hours}",),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_usage_by_category(self) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT
                category,
                COUNT(*) as task_count,
                SUM(total_tokens) as total_tokens,
                SUM(estimated_cost_usd) as total_cost,
                AVG(latency_ms) as avg_latency
            FROM task_results
            GROUP BY category ORDER BY total_tokens DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def get_hourly_token_rate(self) -> float:
        """Tokens consumed in the last complete hour."""
        row = self._conn.execute(
            """
            SELECT COALESCE(SUM(total_tokens), 0)
            FROM task_results
            WHERE started_at >= datetime('now', '-1 hour')
            """
        ).fetchone()
        return float(row[0])

    def query_task_results(
        self,
        *,
        since_hours: float | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Return task rows for export, newest first."""
        clauses: list[str] = []
        params: list = []
        if since_hours is not None:
            clauses.append("started_at >= datetime('now', ?)")
            params.append(f"-{since_hours} hours")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM task_results {where} ORDER BY started_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ── Provider comparison ───────────────────────────────

    def compare_providers(
        self,
        *,
        since_hours: float | None = None,
        tier: str | None = None,
        category: str | None = None,
        group_by: str = "provider",
    ) -> list[dict]:
        """Aggregate task metrics grouped by provider or model.

        Returns dicts with aggregates plus latency percentiles (p50, p95)
        and cost_per_1k_tokens for fairer cross-model comparison.
        """
        clauses: list[str] = []
        params: list = []
        if since_hours is not None:
            clauses.append("started_at >= datetime('now', ?)")
            params.append(f"-{since_hours} hours")
        if tier is not None:
            clauses.append(
                "run_id IN (SELECT run_id FROM runs WHERE tier = ?)"
            )
            params.append(tier)
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        col = "provider" if group_by == "provider" else "model"
        other = "model" if group_by == "provider" else "provider"
        sql = f"""
            SELECT
                {col} as group_key,
                GROUP_CONCAT(DISTINCT {other}) as models_used,
                COUNT(*) as task_count,
                COALESCE(SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END), 0) as successes,
                COALESCE(AVG(total_tokens), 0) as avg_tokens,
                COALESCE(AVG(latency_ms), 0) as avg_latency_ms,
                COALESCE(AVG(estimated_cost_usd), 0) as avg_cost_per_task,
                COALESCE(SUM(estimated_cost_usd), 0) as total_cost,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                AVG(tokens_per_second) as avg_tokens_per_second,
                AVG(accuracy_score) as avg_accuracy_score,
                COUNT(accuracy_score) as scored_tasks
            FROM task_results {where}
            GROUP BY {col}
            ORDER BY avg_cost_per_task ASC
        """
        agg_rows = self._conn.execute(sql, params).fetchall()

        detail_sql = f"""
            SELECT {col} as group_key, latency_ms, estimated_cost_usd, total_tokens
            FROM task_results {where}
            ORDER BY {col}
        """
        detail_rows = self._conn.execute(detail_sql, params).fetchall()

        from collections import defaultdict
        latencies: dict[str, list[float]] = defaultdict(list)
        costs_tokens: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))
        for dr in detail_rows:
            key = dr["group_key"]
            latencies[key].append(dr["latency_ms"])
            cost, toks = costs_tokens[key]
            costs_tokens[key] = (cost + dr["estimated_cost_usd"], toks + dr["total_tokens"])

        results = []
        for row in agg_rows:
            d = dict(row)
            key = d["group_key"]
            lats = sorted(latencies.get(key, []))
            d["p50_latency_ms"] = _percentile(lats, 50)
            d["p95_latency_ms"] = _percentile(lats, 95)
            total_cost, total_toks = costs_tokens.get(key, (0.0, 0))
            d["cost_per_1k_tokens"] = (
                (total_cost / total_toks * 1000) if total_toks > 0 else 0.0
            )
            # value_score = accuracy / cost_per_1k (ε prevents div-by-zero for free local runs)
            acc = d.get("avg_accuracy_score")
            if acc is not None:
                eps = 0.001  # ~$1 per million tokens as floor so local isn't literally infinite
                d["value_score"] = acc / max(d["cost_per_1k_tokens"], eps)
            else:
                d["value_score"] = None
            results.append(d)
        return results

    def get_runs_by_provider(
        self,
        *,
        since_hours: float | None = None,
    ) -> list[dict]:
        """Aggregate run-level metrics grouped by provider."""
        clauses: list[str] = []
        params: list = []
        if since_hours is not None:
            clauses.append("started_at >= datetime('now', ?)")
            params.append(f"-{since_hours} hours")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT
                provider,
                COUNT(*) as run_count,
                COALESCE(SUM(total_tasks), 0) as total_tasks,
                COALESCE(SUM(successful_tasks), 0) as successful_tasks,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                COALESCE(SUM(total_cost_usd), 0) as total_cost,
                COALESCE(AVG(avg_latency_ms), 0) as avg_latency_ms
            FROM runs {where}
            GROUP BY provider
            ORDER BY total_cost DESC
        """
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ── Schedule registry ─────────────────────────────────

    def save_schedule(
        self,
        *,
        schedule_id: str,
        format: str,
        tier: str,
        provider: str,
        model: str | None,
        interval_minutes: int,
        max_iterations: int,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO schedules
                (schedule_id, format, tier, provider, model,
                 interval_minutes, max_iterations, installed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (schedule_id, format, tier, provider, model,
             interval_minutes, max_iterations, now),
        )
        self._conn.commit()

    def remove_schedule(self, schedule_id: str) -> None:
        self._conn.execute(
            "DELETE FROM schedules WHERE schedule_id = ?", (schedule_id,)
        )
        self._conn.commit()

    def get_schedules(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM schedules ORDER BY installed_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_schedule_last_run(
        self, schedule_id: str, status: str
    ) -> None:
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE schedules SET last_run_at = ?, last_status = ? "
            "WHERE schedule_id = ?",
            (now, status, schedule_id),
        )
        self._conn.commit()

    # ── Stress results ─────────────────────────────────────

    def save_stress_result(self, sr: object) -> None:
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

    def save_sweep_result(self, sr: object) -> None:
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

    # ── Scenario results ──────────────────────────────────

    def save_scenario_result(self, sr: object) -> None:
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
