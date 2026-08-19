"""Benchmark task_results rows."""

from __future__ import annotations

from atomics.models import TaskResult
from atomics.storage.repository._base import RepositoryBase


class TasksMixin(RepositoryBase):
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
