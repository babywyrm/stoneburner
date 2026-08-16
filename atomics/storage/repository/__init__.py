"""Repository for persisting and querying run/task metrics.

Split along domain seams. Callers still import `MetricsRepository` from
this package; the public method set is unchanged.
"""

from atomics.stats import percentile as _percentile
from atomics.storage.repository.analytics import AnalyticsMixin
from atomics.storage.repository.evaluation import EvaluationMixin
from atomics.storage.repository.load import LoadMixin
from atomics.storage.repository.runs import RunsMixin
from atomics.storage.repository.schedules import SchedulesMixin
from atomics.storage.repository.security import SecurityMixin
from atomics.storage.repository.tasks import TasksMixin

__all__ = ["MetricsRepository", "_percentile"]


class MetricsRepository(
    RunsMixin,
    EvaluationMixin,
    TasksMixin,
    SecurityMixin,
    AnalyticsMixin,
    SchedulesMixin,
    LoadMixin,
):
    """SQLite metrics store. Mixins own domain methods; this class owns the connection."""

    def get_run_detail(self, run_id: str) -> dict | None:
        """Parent run plus fixture rows, with prompts and raw JSON stripped.

        The dashboard and `GET /api/v1/runs/{id}` share this so a future
        export path cannot accidentally reuse the raw getters. Lives on the
        composed class because it reads evaluation, adversarial, and task rows.
        """
        run = self.get_run(run_id)
        if run is None:
            return None
        fixtures: list[dict[str, object]] = []
        for row in self.get_evaluation_results(run_id=run_id):
            fixtures.append(
                {
                    "id": row["fixture_id"],
                    "kind": "evaluation",
                    "suite": row["suite"],
                    "score": row["score"],
                    "label": None,
                    "status": row["status"],
                    "generation_status": row["generation_status"],
                    "latency_ms": row["latency_ms"],
                    "cost_usd": row["estimated_cost_usd"],
                }
            )
        for row in self.get_adversarial_results(run_id=run_id):
            fixtures.append(
                {
                    "id": row["fixture_id"],
                    "kind": "adversarial",
                    "suite": "adversarial",
                    "score": row["resistance_score"],
                    "label": row["resistance_label"],
                    "status": row["status"],
                    "generation_status": row["generation_status"],
                    "latency_ms": row["latency_ms"],
                    "cost_usd": row["estimated_cost_usd"],
                }
            )
        for row in self.get_run_tasks(run_id):
            fixtures.append(
                {
                    "id": row["task_name"],
                    "kind": "task",
                    "suite": row["suite"],
                    "score": row["accuracy_score"],
                    "label": None,
                    "status": row["status"],
                    "generation_status": row["status"],
                    "latency_ms": row["latency_ms"],
                    "cost_usd": row["estimated_cost_usd"],
                }
            )
        return {"run": run, "fixtures": fixtures}
