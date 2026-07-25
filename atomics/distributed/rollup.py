"""Per-worker aggregation of distributed run results.

A finished fleet run should answer which host was faster, cheaper, and more
reliable. Workers already serialize a full ``TaskResult`` per assignment, so every
number here is a rollup of data the coordinator has stored rather than new
measurement.

Kept free of database access so the arithmetic can be tested against known inputs,
the same split the eval suites use between scoring and running.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
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

    def to_dict(self) -> dict[str, Any]:
        return {
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


@dataclass(frozen=True)
class FleetRollup:
    """Per-host results plus job-wide totals."""

    workers: list[WorkerRollup]
    completed: int
    failed: int
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "workers": [worker.to_dict() for worker in self.workers],
            "completed": self.completed,
            "failed": self.failed,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


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


def _number(result: dict[str, Any], key: str) -> float:
    value = result.get(key)
    return float(value) if isinstance(value, int | float) else 0.0


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
    for worker_id in order:
        entries = grouped[worker_id]
        latencies: list[float] = []
        throughputs: list[float] = []
        input_tokens = 0
        output_tokens = 0
        cost = 0.0
        completed = 0
        failed = 0
        provider = ""
        model = ""
        labels: dict[str, str] = {}
        for entry in entries:
            labels = labels or entry.labels
            if entry.status != "completed" or entry.result is None:
                failed += 1
                continue
            completed += 1
            result = entry.result
            latencies.append(_number(result, "latency_ms"))
            input_tokens += int(_number(result, "input_tokens"))
            output_tokens += int(_number(result, "output_tokens"))
            cost += _number(result, "estimated_cost_usd")
            # Absent on providers that do not report it; averaging in a zero
            # would understate a host that simply did not measure throughput.
            throughput = result.get("tokens_per_second")
            if isinstance(throughput, int | float):
                throughputs.append(float(throughput))
            provider = provider or str(result.get("provider", ""))
            model = model or str(result.get("model", ""))

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
            )
        )

    return FleetRollup(
        workers=workers,
        completed=sum(worker.completed for worker in workers),
        failed=sum(worker.failed for worker in workers),
        total_input_tokens=sum(worker.input_tokens for worker in workers),
        total_output_tokens=sum(worker.output_tokens for worker in workers),
        estimated_cost_usd=round(
            sum(worker.estimated_cost_usd for worker in workers), 6
        ),
    )
