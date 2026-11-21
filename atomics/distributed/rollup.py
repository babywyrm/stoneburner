"""Per-worker aggregation of distributed run results.

A finished fleet run should answer which host was faster, cheaper, and more
reliable. Workers already serialize a full ``TaskResult`` per assignment, so every
number here is a rollup of data the coordinator has stored rather than new
measurement.

Kept free of database access so the arithmetic can be tested against known inputs,
the same split the eval suites use between scoring and running.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AssignmentRecord:
    """One assignment's outcome, as the coordinator has it on disk.

    ``result`` is the worker's serialized TaskResult, absent when the assignment
    never produced one.
    """

    worker_id: str
    labels: dict[str, str] = field(default_factory=dict)
    status: str = "completed"
    result: dict[str, Any] | None = None


_INTEGRITY_STATUSES = frozenset({"complete", "partial", "unscored", "infrastructure_invalid"})


def _combine_integrity(
    statuses: Sequence[str],
    *,
    failed_assignments: int,
) -> dict[str, Any] | None:
    """Roll up suite integrity across worker slices using RunIntegrity semantics."""
    if not statuses:
        return None
    combined = [s for s in statuses if s in _INTEGRITY_STATUSES]
    if not combined:
        return None
    all_slices = combined + ["infrastructure_invalid"] * max(0, failed_assignments)
    if all(s == "complete" for s in all_slices):
        status = "complete"
    elif any(s in ("complete", "partial") for s in all_slices):
        status = "partial"
    elif any(s == "infrastructure_invalid" for s in all_slices):
        status = "infrastructure_invalid"
    else:
        status = "unscored"
    return {
        "status": status,
        "should_exit_nonzero": status != "complete",
    }


@dataclass(frozen=True)
class WorkerRollup:
    """What one host did with its share of the run."""

    worker_id: str
    labels: dict[str, str]
    completed: int
    failed: int
    input_tokens: int
    output_tokens: int
    mean_latency_ms: float
    p95_latency_ms: float
    mean_tokens_per_second: float
    estimated_cost_usd: float
    provider: str
    model: str
    suite: str = ""
    overall_score: float | None = None
    integrity: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "worker_id": self.worker_id,
            "labels": dict(self.labels),
            "completed": self.completed,
            "failed": self.failed,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "mean_latency_ms": self.mean_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "mean_tokens_per_second": self.mean_tokens_per_second,
            "estimated_cost_usd": self.estimated_cost_usd,
            "provider": self.provider,
            "model": self.model,
        }
        if self.suite:
            payload["suite"] = self.suite
        if self.overall_score is not None:
            payload["overall_score"] = self.overall_score
        if self.integrity is not None:
            payload["integrity"] = self.integrity
        return payload


@dataclass(frozen=True)
class FleetRollup:
    """Per-host results plus job-wide totals."""

    workers: list[WorkerRollup]
    completed: int
    failed: int
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float
    integrity: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "workers": [worker.to_dict() for worker in self.workers],
            "completed": self.completed,
            "failed": self.failed,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
        }
        if self.integrity is not None:
            payload["integrity"] = dict(self.integrity)
        return payload


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile.

    Chosen over interpolation because a fleet slice is often only a handful of
    samples, where an interpolated value reports a latency no request actually saw.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return round(ordered[rank - 1], 2)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def _number(result: Mapping[str, Any], key: str, *, nonneg: bool = False) -> float:
    value = result.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return 0.0
    num = float(value)
    return max(0.0, num) if nonneg else num


def assignment_cost_usd(result_json: str | None) -> float:
    """Extract finite non-negative USD spend from a serialized worker assignment result."""
    if not result_json:
        return 0.0
    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        return 0.0
    if not isinstance(data, dict):
        return 0.0
    if "estimated_cost_usd" in data:
        return _number(data, "estimated_cost_usd", nonneg=True)
    if "total_cost_usd" in data:
        return _number(data, "total_cost_usd", nonneg=True)
    summary = data.get("summary")
    if isinstance(summary, dict):
        return _number(summary, "total_cost_usd", nonneg=True)
    return 0.0


def build_rollup(records: Sequence[AssignmentRecord]) -> FleetRollup:
    """Group assignment outcomes by worker and summarize each host.

    Workers appear in first-seen order so repeated runs over the same data produce
    identical output.
    """
    order: list[str] = []
    grouped: dict[str, list[AssignmentRecord]] = {}
    for record in records:
        if record.worker_id not in grouped:
            grouped[record.worker_id] = []
            order.append(record.worker_id)
        grouped[record.worker_id].append(record)

    workers: list[WorkerRollup] = []
    job_eval_statuses: list[str] = []
    for worker_id in order:
        entries = grouped[worker_id]
        latencies: list[float] = []
        throughputs: list[float] = []
        scores: list[float] = []
        worker_eval_statuses: list[str] = []
        input_tokens = 0
        output_tokens = 0
        cost = 0.0
        completed = 0
        failed = 0
        provider = ""
        model = ""
        suite = ""
        labels: dict[str, str] = {}
        for entry in entries:
            labels = labels or entry.labels
            if entry.status != "completed" or entry.result is None:
                failed += 1
                continue
            completed += 1
            result = entry.result
            integrity_obj = result.get("integrity")
            if isinstance(integrity_obj, dict) or "fixtures_run" in result:
                cost += _number(result, "total_cost_usd", nonneg=True)
                provider = provider or str(result.get("provider", "") or "")
                model = model or str(result.get("model", "") or "")
                suite = suite or str(result.get("suite", "") or "")
                for score_key in ("overall_score", "overall_accuracy"):
                    raw_score = result.get(score_key)
                    if (
                        not isinstance(raw_score, bool)
                        and isinstance(raw_score, int | float)
                        and math.isfinite(raw_score)
                    ):
                        scores.append(float(raw_score))
                        break
                fixtures = result.get("fixtures")
                if isinstance(fixtures, list):
                    for f_row in fixtures:
                        if isinstance(f_row, dict):
                            lat = _number(f_row, "latency_ms", nonneg=True)
                            if lat > 0.0:
                                latencies.append(lat)
                            input_tokens += int(_number(f_row, "input_tokens", nonneg=True))
                            output_tokens += int(_number(f_row, "output_tokens", nonneg=True))
                if isinstance(integrity_obj, dict):
                    raw_st = integrity_obj.get("status")
                    if isinstance(raw_st, str) and raw_st in _INTEGRITY_STATUSES:
                        worker_eval_statuses.append(raw_st)
                        job_eval_statuses.append(raw_st)
                continue

            summary = result.get("summary")
            if isinstance(summary, dict) and "estimated_cost_usd" not in result:
                latencies.append(_number(summary, "avg_latency_ms", nonneg=True))
                cost += _number(summary, "total_cost_usd", nonneg=True)
                task_rows = result.get("task_results")
                if isinstance(task_rows, list):
                    for row in task_rows:
                        if isinstance(row, dict):
                            input_tokens += int(_number(row, "input_tokens", nonneg=True))
                            output_tokens += int(_number(row, "output_tokens", nonneg=True))
                provider = provider or str(summary.get("provider", ""))
                model = model or str(summary.get("model", ""))
                continue

            latencies.append(_number(result, "latency_ms", nonneg=True))
            input_tokens += int(_number(result, "input_tokens", nonneg=True))
            output_tokens += int(_number(result, "output_tokens", nonneg=True))
            cost += _number(result, "estimated_cost_usd", nonneg=True)
            # Absent on providers that do not report it; averaging in a zero
            # would understate a host that simply did not measure throughput.
            throughput = result.get("tokens_per_second")
            if (
                not isinstance(throughput, bool)
                and isinstance(throughput, int | float)
                and math.isfinite(throughput)
            ):
                throughputs.append(float(throughput))
            provider = provider or str(result.get("provider", ""))
            model = model or str(result.get("model", ""))

        worker_integrity_dict = _combine_integrity(worker_eval_statuses, failed_assignments=failed)
        workers.append(
            WorkerRollup(
                worker_id=worker_id,
                labels=dict(labels),
                completed=completed,
                failed=failed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                mean_latency_ms=_mean(latencies),
                p95_latency_ms=_percentile(latencies, 0.95),
                mean_tokens_per_second=_mean(throughputs),
                estimated_cost_usd=round(cost, 6),
                provider=provider,
                model=model,
                suite=suite,
                overall_score=round(sum(scores) / len(scores), 3) if scores else None,
                integrity=(
                    str(worker_integrity_dict["status"])
                    if worker_integrity_dict is not None
                    else None
                ),
            )
        )

    total_failed = sum(worker.failed for worker in workers)
    return FleetRollup(
        workers=workers,
        completed=sum(worker.completed for worker in workers),
        failed=total_failed,
        total_input_tokens=sum(worker.input_tokens for worker in workers),
        total_output_tokens=sum(worker.output_tokens for worker in workers),
        estimated_cost_usd=round(sum(worker.estimated_cost_usd for worker in workers), 6),
        integrity=_combine_integrity(job_eval_statuses, failed_assignments=total_failed),
    )
