"""Adversarial, probe, and archreview persistence."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from atomics.storage.repository._base import RepositoryBase
from atomics.validation import sanitize_error

if TYPE_CHECKING:
    from atomics.archreview.models import ArchReviewResult
    from atomics.eval.adversarial.runner import AdversarialFixtureResult
    from atomics.probe.runner import ProbeResult


class SecurityMixin(RepositoryBase):
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

    def save_archreview_result(self, r: ArchReviewResult) -> None:
        """Persist an ArchReviewResult from atomics.archreview."""
        import json as _json

        result_id = uuid.uuid4().hex[:12]
        now = datetime.now(UTC).isoformat()

        findings_data = [
            {
                "category": f.category,
                "location": f.location,
                "severity": f.severity,
                "rationale": f.rationale,
            }
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
                result_id,
                r.run_id,
                r.repo,
                r.tier,
                r.model,
                r.provider,
                r.round,
                r.objective_recall,
                r.objective_precision,
                r.objective_f,
                r.judge_score,
                r.judge_rematch_recall,
                len(r.findings),
                1 if r.parse_failed else 0,
                r.tokens_in,
                r.tokens_out,
                r.cost_usd,
                r.latency_ms,
                r.judge_model,
                r.pack_hash,
                _json.dumps(findings_data),
                _json.dumps(r.matched_categories),
                r.error_class or "",
                r.error_message or "",
                now,
            ),
        )
        self._conn.commit()
