"""Generic evaluation and judge-agreement rows."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from atomics.models import RunSummary
from atomics.storage.records import EvaluationResultRecord
from atomics.storage.repository._base import RepositoryBase
from atomics.validation import sanitize_error


class EvaluationMixin(RepositoryBase):
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
                error_message, result_json, timestamp, judge_agreement
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?
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
                timestamp = excluded.timestamp,
                judge_agreement = excluded.judge_agreement
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
                record.judge_agreement,
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

    def save_agreement_result(
        self,
        *,
        run_id: str,
        suite: str,
        fixture_id: str,
        votes: dict[str, object],
        agreement: float | None,
        flipped: bool,
    ) -> None:
        """Upsert one judge-agreement study row. No parent `runs` row required."""
        self._conn.execute(
            """
            INSERT INTO judge_agreement_results (
                result_id, run_id, suite, fixture_id, votes_json,
                agreement, flipped, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, suite, fixture_id) DO UPDATE SET
                votes_json = excluded.votes_json,
                agreement = excluded.agreement,
                flipped = excluded.flipped,
                created_at = excluded.created_at
            """,
            (
                uuid.uuid4().hex[:12],
                run_id,
                suite,
                fixture_id,
                json.dumps(votes),
                agreement,
                int(flipped),
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()

    def get_agreement_results(
        self,
        *,
        run_id: str | None = None,
        suite: str | None = None,
    ) -> list[dict[str, object]]:
        """Return study rows, newest first within a run."""
        query = "SELECT * FROM judge_agreement_results"
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
        query += " ORDER BY created_at, fixture_id"
        rows = self._conn.execute(query, params).fetchall()
        results: list[dict[str, object]] = []
        for row in rows:
            item = dict(row)
            item["votes_json"] = json.loads(str(item["votes_json"]))
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

