"""Aggregates, trends, and provider comparison."""

from __future__ import annotations

from atomics.reporting.stats import percentile as _percentile
from atomics.storage.repository._base import RepositoryBase


class AnalyticsMixin(RepositoryBase):
    def get_token_usage_by_hour(self, hours: int = 24) -> list[dict]:
        """Hourly tokens across benchmark tasks and eval/adversarial fixtures.

        The window is an ISO `T` cutoff, not SQLite `datetime()`, so a stored
        `2026-08-15T14:00:00+00:00` is not compared to `2026-08-15 15:00:00`
        (where `T` sorts after space and a 25-hour-old row looks recent).
        """
        cutoff = f"-{hours}"
        rows = self._conn.execute(
            """
            SELECT
                hour,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(cost) as cost,
                SUM(task_count) as task_count
            FROM (
                SELECT
                    strftime('%Y-%m-%d %H:00', started_at) as hour,
                    input_tokens, output_tokens, total_tokens,
                    estimated_cost_usd as cost,
                    1 as task_count
                FROM task_results
                WHERE started_at >= strftime('%Y-%m-%dT%H:%M:%S', 'now', ? || ' hours')
                UNION ALL
                SELECT
                    strftime('%Y-%m-%d %H:00', timestamp) as hour,
                    input_tokens, output_tokens, total_tokens,
                    estimated_cost_usd as cost,
                    1 as task_count
                FROM evaluation_results
                WHERE timestamp >= strftime('%Y-%m-%dT%H:%M:%S', 'now', ? || ' hours')
                UNION ALL
                SELECT
                    strftime('%Y-%m-%d %H:00', timestamp) as hour,
                    input_tokens, output_tokens, total_tokens,
                    estimated_cost_usd as cost,
                    1 as task_count
                FROM adversarial_results
                WHERE timestamp >= strftime('%Y-%m-%dT%H:%M:%S', 'now', ? || ' hours')
            )
            GROUP BY hour ORDER BY hour
            """,
            (cutoff, cutoff, cutoff),
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
        clauses.append(
            "COALESCE((SELECT pass_count FROM runs WHERE run_id = task_results.run_id), 1) = ("
            "SELECT MAX(COALESCE(r2.pass_count, 1)) "
            "FROM task_results t2 "
            "JOIN runs r2 ON r2.run_id = t2.run_id "
            "WHERE t2.model = task_results.model)"
        )
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

