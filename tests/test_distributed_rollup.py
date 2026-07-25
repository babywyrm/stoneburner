"""Tests for per-worker aggregation of distributed run results.

The math lives in a pure module so it can be checked against known inputs without
standing up a coordinator or a worker, mirroring how the eval suites keep scoring
separate from running.
"""

from __future__ import annotations

from atomics.distributed.rollup import AssignmentRecord, build_rollup


def _record(
    worker_id: str,
    *,
    labels: dict[str, str] | None = None,
    status: str = "completed",
    latency_ms: float = 100.0,
    input_tokens: int = 10,
    output_tokens: int = 20,
    cost: float = 0.001,
    tokens_per_second: float | None = 50.0,
    provider: str = "ollama",
    model: str = "qwen3:14b",
) -> AssignmentRecord:
    result = None
    if status == "completed":
        result = {
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": cost,
            "tokens_per_second": tokens_per_second,
            "provider": provider,
            "model": model,
        }
    return AssignmentRecord(
        worker_id=worker_id,
        labels=labels or {},
        status=status,
        result=result,
    )


def test_rollup_groups_results_by_worker():
    rollup = build_rollup(
        [
            _record("host-a", labels={"gpu": "4090"}),
            _record("host-a"),
            _record("host-b", labels={"gpu": "3060"}),
        ]
    )

    assert [w.worker_id for w in rollup.workers] == ["host-a", "host-b"]
    assert rollup.workers[0].completed == 2
    assert rollup.workers[1].completed == 1
    assert rollup.workers[0].labels == {"gpu": "4090"}


def test_rollup_reports_mean_and_p95_latency_from_known_inputs():
    rollup = build_rollup(
        [_record("host-a", latency_ms=value) for value in (10.0, 20.0, 30.0, 40.0)]
    )

    worker = rollup.workers[0]
    assert worker.mean_latency_ms == 25.0
    # Nearest-rank: ceil(0.95 * 4) = 4, so the fourth-slowest sample.
    assert worker.p95_latency_ms == 40.0


def test_p95_picks_the_nearest_rank_over_twenty_samples():
    rollup = build_rollup(
        [_record("host-a", latency_ms=float(value)) for value in range(1, 21)]
    )

    # ceil(0.95 * 20) = 19, so the 19th value.
    assert rollup.workers[0].p95_latency_ms == 19.0


def test_rollup_counts_failures_per_worker():
    rollup = build_rollup(
        [
            _record("host-a"),
            _record("host-a", status="failed"),
            _record("host-b", status="failed"),
        ]
    )

    by_id = {w.worker_id: w for w in rollup.workers}
    assert (by_id["host-a"].completed, by_id["host-a"].failed) == (1, 1)
    assert (by_id["host-b"].completed, by_id["host-b"].failed) == (0, 1)
    assert rollup.completed == 1
    assert rollup.failed == 2


def test_rollup_totals_tokens_and_cost():
    rollup = build_rollup(
        [
            _record("host-a", input_tokens=10, output_tokens=20, cost=0.5),
            _record("host-b", input_tokens=5, output_tokens=7, cost=0.25),
        ]
    )

    assert rollup.total_input_tokens == 15
    assert rollup.total_output_tokens == 27
    assert rollup.estimated_cost_usd == 0.75
    by_id = {w.worker_id: w for w in rollup.workers}
    assert by_id["host-a"].output_tokens == 20
    assert by_id["host-b"].input_tokens == 5


def test_a_worker_with_only_failures_reports_zeros_not_an_error():
    """A host that failed everything is exactly the case worth reporting on."""
    rollup = build_rollup(
        [_record("host-a", status="failed"), _record("host-a", status="failed")]
    )

    worker = rollup.workers[0]
    assert worker.completed == 0
    assert worker.failed == 2
    assert worker.mean_latency_ms == 0.0
    assert worker.p95_latency_ms == 0.0
    assert worker.mean_tokens_per_second == 0.0


def test_rollup_ignores_a_missing_tokens_per_second():
    """Providers that do not report throughput must not skew the mean."""
    rollup = build_rollup(
        [
            _record("host-a", tokens_per_second=40.0),
            _record("host-a", tokens_per_second=None),
            _record("host-a", tokens_per_second=60.0),
        ]
    )

    assert rollup.workers[0].mean_tokens_per_second == 50.0


def test_rollup_records_the_provider_and_model_each_host_ran():
    rollup = build_rollup(
        [
            _record("host-a", provider="vllm", model="qwen3:32b"),
            _record("host-b", provider="ollama", model="qwen3:14b"),
        ]
    )

    by_id = {w.worker_id: w for w in rollup.workers}
    assert (by_id["host-a"].provider, by_id["host-a"].model) == ("vllm", "qwen3:32b")
    assert by_id["host-b"].provider == "ollama"


def test_to_dict_is_json_ready_and_keeps_every_worker():
    rollup = build_rollup([_record("host-a"), _record("host-b", status="failed")])

    payload = rollup.to_dict()

    assert payload["completed"] == 1
    assert payload["failed"] == 1
    assert [w["worker_id"] for w in payload["workers"]] == ["host-a", "host-b"]
    assert isinstance(payload["workers"][0]["labels"], dict)


def test_an_empty_run_rolls_up_to_nothing():
    rollup = build_rollup([])

    assert rollup.workers == []
    assert rollup.completed == 0
    assert rollup.estimated_cost_usd == 0.0
