"""Repository for persisting and querying run/task metrics."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from atomics.models import RunSummary, TaskResult, TaskStatus
from atomics.stats import percentile as _percentile
from atomics.storage.records import EvaluationResultRecord
from atomics.storage.schema import init_db
from atomics.validation import sanitize_error

if TYPE_CHECKING:
    from atomics.archreview.models import ArchReviewResult
    from atomics.eval.adversarial.runner import AdversarialFixtureResult
    from atomics.probe.runner import ProbeResult
    from atomics.scenario_models import ScenarioResult
    from atomics.soak import SoakResult
    from atomics.stress import StressResult
    from atomics.sweep import ModelSweepResult


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

    def save_evaluation_result(self, record: EvaluationResultRecord) -> None:
        """Upsert one logical generic evaluation fixture."""
        error_message = record.error_message
        if error_message and "[REDACTED]" not in error_message:
            error_message = sanitize_error(Exception(error_message))
        self._conn.execute(
            """
            INSERT INTO evaluation_results (
                result_id, run_id, suite, fixture_id, status, score,
                generation_status, judge_status, latency_ms, estimated_cost_usd,
                input_tokens, output_tokens, total_tokens, thinking_tokens,
                attempt_count, generation_failures, infrastructure_failures,
                judge_failures, parse_failed, provider, model, error_class,
                error_message, result_json, timestamp
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?
            )
            ON CONFLICT(run_id, suite, fixture_id) DO UPDATE SET
                status = excluded.status,
                score = excluded.score,
                generation_status = excluded.generation_status,
                judge_status = excluded.judge_status,
                latency_ms = excluded.latency_ms,
                estimated_cost_usd = excluded.estimated_cost_usd,
                input_tokens = excluded.input_tokens,
                output_tokens = excluded.output_tokens,
                total_tokens = excluded.total_tokens,
                thinking_tokens = excluded.thinking_tokens,
                attempt_count = excluded.attempt_count,
                generation_failures = excluded.generation_failures,
                infrastructure_failures = excluded.infrastructure_failures,
                judge_failures = excluded.judge_failures,
                parse_failed = excluded.parse_failed,
                provider = excluded.provider,
                model = excluded.model,
                error_class = excluded.error_class,
                error_message = excluded.error_message,
                result_json = excluded.result_json,
                timestamp = excluded.timestamp
            """,
            (
                uuid.uuid4().hex[:12],
                record.run_id,
                record.suite,
                record.fixture_id,
                record.status,
                record.score,
                record.generation_status,
                record.judge_status,
                record.latency_ms,
                record.estimated_cost_usd,
                record.input_tokens,
                record.output_tokens,
                record.total_tokens,
                record.thinking_tokens,
                record.attempt_count,
                record.generation_failures,
                record.infrastructure_failures,
                record.judge_failures,
                int(record.parse_failed),
                record.provider,
                record.model,
                record.error_class,
                error_message,
                json.dumps(record.result_json),
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()

    def get_evaluation_results(
        self,
        *,
        run_id: str | None = None,
        suite: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        """Return decoded generic evaluation rows in timestamp order."""
        if limit is not None and limit < 0:
            raise ValueError("limit must be nonnegative")
        query = "SELECT * FROM evaluation_results"
        clauses: list[str] = []
        params: list[object] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if suite is not None:
            clauses.append("suite = ?")
            params.append(suite)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY timestamp, fixture_id"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        results: list[dict[str, object]] = []
        for row in rows:
            item = dict(row)
            item["result_json"] = json.loads(str(item["result_json"]))
            item["parse_failed"] = bool(item["parse_failed"])
            results.append(item)
        return results

    def complete_evaluation_run(self, run_id: str) -> RunSummary:
        """Finalize a generic evaluation parent from persisted fixture rows."""
        now = datetime.now(UTC).isoformat()
        rows = self._conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                COALESCE(SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END), 0)
                    AS success,
                COALESCE(SUM(CASE WHEN status != 'complete' THEN 1 ELSE 0 END), 0)
                    AS failed,
                COALESCE(SUM(input_tokens), 0) AS inp,
                COALESCE(SUM(output_tokens), 0) AS outp,
                COALESCE(SUM(total_tokens), 0) AS tot,
                COALESCE(SUM(estimated_cost_usd), 0.0) AS cost,
                COALESCE(AVG(latency_ms), 0.0) AS avg_lat
            FROM evaluation_results WHERE run_id = ?
            """,
            (run_id,),
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
            (
                now,
                rows[0],
                rows[1],
                rows[2],
                rows[3],
                rows[4],
                rows[5],
                rows[6],
                rows[7],
                run_id,
            ),
        )
        self._conn.commit()
        started_row = self._conn.execute(
            "SELECT started_at FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return RunSummary(
            run_id=run_id,
            started_at=datetime.fromisoformat(started_row[0]),
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
                thinking_tokens, cache_read_tokens, cache_write_tokens,
                latency_ms, estimated_cost_usd, tokens_per_second, tps_basis,
                thinking_enabled,
                error_class, error_message,
                started_at, completed_at,
                accuracy_score, judge_model, quality_rationale, criteria_coverage,
                judge_score_stdev
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
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
                result.cache_read_tokens,
                result.cache_write_tokens,
                result.latency_ms,
                result.estimated_cost_usd,
                result.tokens_per_second,
                result.tps_basis,
                int(result.thinking_enabled),
                result.error_class,
                result.error_message,
                result.started_at.isoformat(),
                result.completed_at.isoformat() if result.completed_at else None,
                result.accuracy_score,
                result.judge_model,
                result.quality_rationale,
                result.criteria_coverage,
                result.judge_score_stdev,
            ),
        )
        self._conn.commit()

    def save_adversarial_result(
        self,
        run_id: str,
        result: AdversarialFixtureResult,
        *,
        thinking_enabled: bool = False,
        provider: str = "",
        model: str = "",
    ) -> None:
        r = result
        res = r.resistance
        serialized = r.to_dict()
        # Parent task token semantics track only the model under test. Judge-call
        # tokens remain in attempts_json as cost/evidence and are not rolled up.
        input_tokens = sum(attempt.input_tokens for attempt in r.attempts)
        output_tokens = sum(attempt.output_tokens for attempt in r.attempts)
        total_tokens = input_tokens + output_tokens
        representative_error = str(serialized["error_message"])
        sanitized_error = (
            (
                representative_error
                if "[REDACTED]" in representative_error
                else sanitize_error(Exception(representative_error))
            )
            if representative_error
            else ""
        )
        self._conn.execute(
            """
            INSERT INTO adversarial_results (
                result_id, run_id, fixture_id, category, severity,
                provider, model, prompt, response, attack_goal,
                resistance_score, resistance_label, judge_model, judge_rationale,
                thinking_enabled, thinking_tokens, latency_ms, estimated_cost_usd,
                timestamp, status, generation_status, judge_status, attempt_count,
                input_tokens, output_tokens, total_tokens, attempts_json,
                run_scores_json, generation_failures,
                infrastructure_failures, judge_failures, parse_failed,
                error_class, error_message
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(run_id, fixture_id) DO UPDATE SET
                category = excluded.category,
                severity = excluded.severity,
                provider = excluded.provider,
                model = excluded.model,
                prompt = excluded.prompt,
                response = excluded.response,
                attack_goal = excluded.attack_goal,
                resistance_score = excluded.resistance_score,
                resistance_label = excluded.resistance_label,
                judge_model = excluded.judge_model,
                judge_rationale = excluded.judge_rationale,
                thinking_enabled = excluded.thinking_enabled,
                thinking_tokens = excluded.thinking_tokens,
                latency_ms = excluded.latency_ms,
                estimated_cost_usd = excluded.estimated_cost_usd,
                timestamp = excluded.timestamp,
                status = excluded.status,
                generation_status = excluded.generation_status,
                judge_status = excluded.judge_status,
                attempt_count = excluded.attempt_count,
                input_tokens = excluded.input_tokens,
                output_tokens = excluded.output_tokens,
                total_tokens = excluded.total_tokens,
                attempts_json = excluded.attempts_json,
                run_scores_json = excluded.run_scores_json,
                generation_failures = excluded.generation_failures,
                infrastructure_failures = excluded.infrastructure_failures,
                judge_failures = excluded.judge_failures,
                parse_failed = excluded.parse_failed,
                error_class = excluded.error_class,
                error_message = excluded.error_message
            """,
            (
                uuid.uuid4().hex,
                run_id,
                r.fixture.id,
                r.fixture.category,
                r.fixture.severity,
                provider,
                model,
                r.fixture.prompt,
                r.response,
                r.fixture.attack_goal,
                res.score if res else None,
                res.label if res else "",
                res.judge_model if res else "",
                res.rationale if res else "",
                int(thinking_enabled),
                r.thinking_tokens,
                r.latency_ms,
                r.estimated_cost_usd,
                datetime.now(UTC).isoformat(),
                serialized["status"],
                serialized["generation_status"],
                serialized["judge_status"],
                serialized["attempt_count"],
                input_tokens,
                output_tokens,
                total_tokens,
                json.dumps(serialized["attempts"]),
                json.dumps(serialized["run_scores"]),
                serialized["generation_failures"],
                serialized["infrastructure_failures"],
                serialized["judge_failures"],
                int(bool(serialized["parse_failed"])),
                serialized["error_class"],
                sanitized_error,
            ),
        )
        self._conn.commit()

    def complete_adversarial_run(self, run_id: str) -> None:
        """Finalize a run whose results live in adversarial_results (not task_results).

        The generic `complete_run` aggregates `task_results`, so adversarial runs
        need their own completion that reads the right table. Sets completed_at
        and rolls up counts, provider-attempt tokens, cost, and latency for
        `atomics report`-style listing.
        """
        now = datetime.now(UTC).isoformat()
        row = self._conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                COALESCE(SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END), 0)
                    AS success,
                COALESCE(SUM(CASE WHEN status <> 'complete' THEN 1 ELSE 0 END), 0)
                    AS failed,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(total_tokens), 0) AS total_tokens,
                COALESCE(SUM(estimated_cost_usd), 0.0) AS cost,
                COALESCE(AVG(latency_ms), 0.0) AS avg_lat
            FROM adversarial_results WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        self._conn.execute(
            """
            UPDATE runs SET
                completed_at = ?, total_tasks = ?, successful_tasks = ?,
                failed_tasks = ?, total_input_tokens = ?,
                total_output_tokens = ?, total_tokens = ?,
                total_cost_usd = ?, avg_latency_ms = ?
            WHERE run_id = ?
            """,
            (
                now,
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
                row[7],
                run_id,
            ),
        )
        self._conn.commit()

    def complete_probe_run(self, run_id: str) -> None:
        """Finalize a run whose results live in probe_results.

        Mirrors complete_adversarial_run but reads probe_results (which has no
        cost/latency columns), so it just records completion + row count.
        """
        now = datetime.now(UTC).isoformat()
        row = self._conn.execute(
            "SELECT COUNT(*) FROM probe_results WHERE run_id = ?", (run_id,)
        ).fetchone()
        self._conn.execute(
            """
            UPDATE runs SET
                completed_at = ?, total_tasks = ?, successful_tasks = ?
            WHERE run_id = ?
            """,
            (now, row[0], row[0], run_id),
        )
        self._conn.commit()

    def complete_archreview_run(self, run_id: str) -> None:
        """Finalize a run whose results live in archreview_results."""
        now = datetime.now(UTC).isoformat()
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS total,
                   COALESCE(SUM(cost_usd), 0.0) AS cost
            FROM archreview_results WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        self._conn.execute(
            """
            UPDATE runs SET completed_at = ?, total_tasks = ?, successful_tasks = ?,
                total_cost_usd = ?
            WHERE run_id = ?
            """,
            (now, row[0], row[0], row[1], run_id),
        )
        self._conn.commit()

    def get_adversarial_results(
        self, *, limit: int | None = None, run_id: str | None = None
    ) -> list[dict]:
        """Return adversarial result rows for export, newest first."""
        clauses: list[str] = []
        params: list = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM adversarial_results {where} ORDER BY timestamp DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def save_probe_result(self, run_id: str, result: ProbeResult) -> None:
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
                r.target_name,
                r.artifact_type,
                r.check_id,
                r.score,
                r.prev_score,
                int(r.regressed),
                "",
                "",
                r.judge_model,
                r.judge_rationale,
                int(r.thinking_enabled),
                r.thinking_tokens,
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
        suite: str | None = None,
        suite_prefix: str | None = None,
    ) -> list[dict]:
        """Return task rows for export, newest first.

        `suite` matches an exact suite value; `suite_prefix` matches with a
        trailing wildcard (e.g. "redblue-" selects redblue-red + redblue-blue),
        so callers can isolate a suite instead of getting all task_results mixed.
        """
        clauses: list[str] = []
        params: list = []
        if since_hours is not None:
            clauses.append("started_at >= datetime('now', ?)")
            params.append(f"-{since_hours} hours")
        if suite is not None:
            clauses.append("suite = ?")
            params.append(suite)
        if suite_prefix is not None:
            clauses.append("suite LIKE ?")
            params.append(f"{suite_prefix}%")
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
                COALESCE(SUM(cache_read_tokens), 0) as total_cache_read_tokens,
                COALESCE(SUM(cache_write_tokens), 0) as total_cache_write_tokens,
                COALESCE(AVG(thinking_tokens), 0) as avg_thinking_tokens,
                GROUP_CONCAT(DISTINCT tps_basis) as tps_bases,
                AVG(tokens_per_second) as avg_tokens_per_second,
                AVG(accuracy_score) as avg_accuracy_score,
                COUNT(accuracy_score) as scored_tasks,
                AVG(criteria_coverage) as avg_criteria_coverage,
                AVG(judge_score_stdev) as avg_judge_score_stdev
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

    def save_archreview_result(self, r: ArchReviewResult) -> None:
        """Persist an ArchReviewResult from atomics.archreview."""
        import json as _json

        result_id = uuid.uuid4().hex[:12]
        now = datetime.now(UTC).isoformat()

        findings_data = [
            {"category": f.category, "location": f.location,
             "severity": f.severity, "rationale": f.rationale}
            for f in r.findings
        ]

        self._conn.execute(
            """
            INSERT INTO archreview_results
            (result_id, run_id, repo, tier, model, provider, round,
             objective_recall, objective_precision, objective_f, judge_score,
             judge_rematch_recall, finding_count, parse_failed,
             tokens_in, tokens_out, cost_usd, latency_ms, judge_model,
             pack_hash, findings_json, matched_categories_json,
             error_class, error_message, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?)
            """,
            (
                result_id, r.run_id, r.repo, r.tier, r.model, r.provider, r.round,
                r.objective_recall, r.objective_precision, r.objective_f,
                r.judge_score, r.judge_rematch_recall, len(r.findings),
                1 if r.parse_failed else 0, r.tokens_in, r.tokens_out,
                r.cost_usd, r.latency_ms, r.judge_model, r.pack_hash,
                _json.dumps(findings_data),
                _json.dumps(r.matched_categories),
                r.error_class or "", r.error_message or "", now,
            ),
        )
        self._conn.commit()

    # ── LabCompare ────────────────────────────────────────

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
