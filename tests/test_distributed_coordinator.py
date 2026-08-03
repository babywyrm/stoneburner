import json
from datetime import UTC, datetime, timedelta

import pytest

from atomics.distributed.coordinator import Coordinator
from atomics.distributed.models import (
    AssignmentStatus,
    DistributedRunRequest,
    JobMode,
    JobStatus,
    WorkerRegisterRequest,
    WorkerStatus,
)
from atomics.storage.schema import init_db


@pytest.fixture
def coordinator(tmp_path):
    db = tmp_path / "test.db"
    conn = init_db(db)
    return Coordinator(conn)


def test_register_worker(coordinator):
    req = WorkerRegisterRequest(labels={"provider": "ollama"})
    w = coordinator.register_worker(req, api_key_hint="1234")
    assert w.labels["provider"] == "ollama"
    assert w.api_key_hint == "1234"
    assert w.status.value == "online"


def test_heartbeat_updates_last_seen(coordinator):
    w = coordinator.register_worker(WorkerRegisterRequest())
    w2 = coordinator.heartbeat(w.worker_id)
    assert w2 is not None
    assert w2.status.value == "online"


def test_create_split_job_creates_assignments(coordinator):
    req = DistributedRunRequest(mode=JobMode.SPLIT, run_request={"iterations": 3})
    job = coordinator.create_split_job(req, [{"i": 1}, {"i": 2}, {"i": 3}])
    assert job.mode == JobMode.SPLIT
    rows = coordinator._conn.execute(
        "SELECT COUNT(*) FROM distributed_assignments WHERE job_id = ?",
        (job.job_id,),
    ).fetchone()
    assert rows[0] == 3


def test_claim_assignment(coordinator):
    job = coordinator.create_split_job(
        DistributedRunRequest(mode=JobMode.SPLIT), [{"i": 1}]
    )
    w = coordinator.register_worker(WorkerRegisterRequest())
    a = coordinator.claim_assignment(w.worker_id)
    assert a is not None
    assert a.worker_id == w.worker_id
    assert a.status.value == "assigned"
    assert a.job_id == job.job_id


def _assignments(coordinator, job_id: str) -> list[tuple[str, str]]:
    """(target_worker_id, task_spec) for every assignment of a job."""
    return [
        (row[0], row[1])
        for row in coordinator._conn.execute(
            "SELECT target_worker_id, task_spec FROM distributed_assignments "
            "WHERE job_id = ? ORDER BY assignment_id",
            (job_id,),
        )
    ]


def _offline(coordinator, worker_id: str) -> None:
    coordinator._conn.execute(
        "UPDATE workers SET status = 'offline' WHERE worker_id = ?", (worker_id,)
    )
    coordinator._conn.commit()


def test_matching_workers_requires_every_selector_pair(coordinator):
    both = coordinator.register_worker(
        WorkerRegisterRequest(labels={"gpu": "4090", "site": "lab"})
    )
    coordinator.register_worker(WorkerRegisterRequest(labels={"gpu": "4090"}))
    coordinator.register_worker(WorkerRegisterRequest(labels={"site": "lab"}))

    matched = coordinator.matching_workers({"gpu": "4090", "site": "lab"})

    assert [w.worker_id for w in matched] == [both.worker_id]


def test_an_empty_selector_matches_every_online_worker(coordinator):
    first = coordinator.register_worker(WorkerRegisterRequest(labels={"gpu": "4090"}))
    second = coordinator.register_worker(WorkerRegisterRequest())

    matched = coordinator.matching_workers({})

    assert {w.worker_id for w in matched} == {first.worker_id, second.worker_id}


def test_offline_workers_are_excluded_from_the_snapshot(coordinator):
    online = coordinator.register_worker(WorkerRegisterRequest(labels={"gpu": "4090"}))
    gone = coordinator.register_worker(WorkerRegisterRequest(labels={"gpu": "4090"}))
    _offline(coordinator, gone.worker_id)

    matched = coordinator.matching_workers({"gpu": "4090"})

    assert [w.worker_id for w in matched] == [online.worker_id]


def test_fleet_job_creates_one_assignment_per_worker_per_task(coordinator):
    workers = [
        coordinator.register_worker(WorkerRegisterRequest(labels={"gpu": "4090"}))
        for _ in range(3)
    ]
    specs = [{"prompt": "a"}, {"prompt": "b"}]

    job = coordinator.create_fleet_job(
        DistributedRunRequest(mode=JobMode.FLEET), specs, workers
    )

    assert job.mode == JobMode.FLEET
    rows = _assignments(coordinator, job.job_id)
    assert len(rows) == 6
    # Every assignment is pinned, and each worker got exactly len(specs) of them.
    pinned_counts: dict[str, int] = {}
    for target, _spec in rows:
        assert target is not None
        pinned_counts[target] = pinned_counts.get(target, 0) + 1
    assert pinned_counts == {w.worker_id: 2 for w in workers}


def test_every_worker_receives_the_identical_task_set(coordinator):
    """The comparison is meaningless if hosts run different prompts.

    Task specs are sampled randomly per call upstream, so building them per worker
    would silently give each host different work while still producing the right
    number of assignments.
    """
    workers = [
        coordinator.register_worker(WorkerRegisterRequest()) for _ in range(3)
    ]
    specs = [{"prompt": "alpha"}, {"prompt": "beta"}]

    job = coordinator.create_fleet_job(
        DistributedRunRequest(mode=JobMode.FLEET), specs, workers
    )

    per_worker: dict[str, list[str]] = {}
    for target, spec in _assignments(coordinator, job.job_id):
        per_worker.setdefault(target, []).append(spec)
    task_sets = {frozenset(specs) for specs in per_worker.values()}
    assert len(task_sets) == 1, f"hosts received different task sets: {per_worker}"


def test_fleet_assignments_are_only_claimable_by_their_own_worker(coordinator):
    first = coordinator.register_worker(WorkerRegisterRequest())
    second = coordinator.register_worker(WorkerRegisterRequest())
    coordinator.create_fleet_job(
        DistributedRunRequest(mode=JobMode.FLEET), [{"prompt": "a"}], [first]
    )

    assert coordinator.claim_assignment(second.worker_id) is None
    claimed = coordinator.claim_assignment(first.worker_id)
    assert claimed is not None
    assert claimed.target_worker_id == first.worker_id


def _pin(coordinator, job_id: str, worker_id: str) -> None:
    """Pin every assignment of a job to one worker, as fleet mode will."""
    coordinator._conn.execute(
        "UPDATE distributed_assignments SET target_worker_id = ? WHERE job_id = ?",
        (worker_id, job_id),
    )
    coordinator._conn.commit()


def test_a_pinned_assignment_is_not_claimable_by_another_worker(coordinator):
    """Fleet mode's core invariant: work aimed at one host stays on that host."""
    owner = coordinator.register_worker(WorkerRegisterRequest(labels={"host": "a"}))
    other = coordinator.register_worker(WorkerRegisterRequest(labels={"host": "b"}))
    job = coordinator.create_split_job(
        DistributedRunRequest(mode=JobMode.SPLIT), [{"i": 1}]
    )
    _pin(coordinator, job.job_id, owner.worker_id)

    assert coordinator.claim_assignment(other.worker_id) is None


def test_a_worker_claims_its_own_pinned_assignment(coordinator):
    owner = coordinator.register_worker(WorkerRegisterRequest())
    job = coordinator.create_split_job(
        DistributedRunRequest(mode=JobMode.SPLIT), [{"i": 1}]
    )
    _pin(coordinator, job.job_id, owner.worker_id)

    claimed = coordinator.claim_assignment(owner.worker_id)
    assert claimed is not None
    assert claimed.worker_id == owner.worker_id
    assert claimed.target_worker_id == owner.worker_id


def test_unpinned_assignments_remain_claimable_by_any_worker(coordinator):
    """Split mode must not change: an unpinned assignment goes to whoever asks."""
    coordinator.register_worker(WorkerRegisterRequest(labels={"host": "a"}))
    other = coordinator.register_worker(WorkerRegisterRequest(labels={"host": "b"}))
    coordinator.create_split_job(DistributedRunRequest(mode=JobMode.SPLIT), [{"i": 1}])

    claimed = coordinator.claim_assignment(other.worker_id)
    assert claimed is not None
    assert claimed.target_worker_id is None


def test_a_pinned_assignment_does_not_block_an_unpinned_one(coordinator):
    """A worker skipped over someone else's pinned row must still get its own work.

    The claim query selects a single row, so an ORDER BY that reaches the pinned
    row first could starve a worker that has perfectly good unpinned work waiting.
    """
    owner = coordinator.register_worker(WorkerRegisterRequest())
    other = coordinator.register_worker(WorkerRegisterRequest())
    pinned_job = coordinator.create_split_job(
        DistributedRunRequest(mode=JobMode.SPLIT), [{"pinned": True}]
    )
    _pin(coordinator, pinned_job.job_id, owner.worker_id)
    coordinator.create_split_job(
        DistributedRunRequest(mode=JobMode.SPLIT), [{"pinned": False}]
    )

    claimed = coordinator.claim_assignment(other.worker_id)
    assert claimed is not None
    assert claimed.task_spec == {"pinned": False}


def _backdate(coordinator, assignment_id: str, seconds: int = 3600) -> None:
    """Push started_at into the past so the assignment reads as timed out."""
    past = (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat()
    coordinator._conn.execute(
        "UPDATE distributed_assignments SET started_at = ? WHERE assignment_id = ?",
        (past, assignment_id),
    )
    coordinator._conn.commit()


def test_pinned_assignment_fails_when_its_worker_goes_offline(coordinator):
    owner = coordinator.register_worker(WorkerRegisterRequest())
    coordinator.create_fleet_job(
        DistributedRunRequest(mode=JobMode.FLEET), [{"i": 1}], [owner]
    )
    claimed = coordinator.claim_assignment(owner.worker_id)
    assert claimed is not None

    _offline(coordinator, owner.worker_id)
    coordinator._requeue_stale_assignments()

    after = coordinator.get_assignment(claimed.assignment_id)
    assert after is not None
    assert after.status == AssignmentStatus.FAILED


def test_a_dead_hosts_tasks_are_never_given_to_another_worker(coordinator):
    """The regression this design exists to prevent.

    If a dead host's slice migrated to a live one, the job would still report
    completed while the per-host comparison had quietly become a blend of two
    machines — a wrong answer that looks like a right one.
    """
    dead = coordinator.register_worker(WorkerRegisterRequest())
    alive = coordinator.register_worker(WorkerRegisterRequest())
    coordinator.create_fleet_job(
        DistributedRunRequest(mode=JobMode.FLEET),
        [{"i": 1}, {"i": 2}],
        [dead],
    )
    # One assignment in flight, one still pending, then the host disappears.
    assert coordinator.claim_assignment(dead.worker_id) is not None
    _offline(coordinator, dead.worker_id)

    for _ in range(5):
        assert coordinator.claim_assignment(alive.worker_id) is None


def test_pinned_assignment_that_timed_out_keeps_its_pin_while_the_worker_lives(
    coordinator,
):
    owner = coordinator.register_worker(WorkerRegisterRequest())
    other = coordinator.register_worker(WorkerRegisterRequest())
    coordinator.create_fleet_job(
        DistributedRunRequest(mode=JobMode.FLEET, timeout_seconds=1),
        [{"i": 1}],
        [owner],
    )
    claimed = coordinator.claim_assignment(owner.worker_id)
    assert claimed is not None
    _backdate(coordinator, claimed.assignment_id)

    coordinator._requeue_stale_assignments()

    after = coordinator.get_assignment(claimed.assignment_id)
    assert after is not None
    assert after.status == AssignmentStatus.PENDING
    assert after.target_worker_id == owner.worker_id
    # A slow host gets another try; a different host still cannot take the work.
    assert coordinator.claim_assignment(other.worker_id) is None
    assert coordinator.claim_assignment(owner.worker_id) is not None


def test_unpinned_stale_assignments_still_requeue_to_anyone(coordinator):
    """Split mode's recovery path must not change."""
    first = coordinator.register_worker(WorkerRegisterRequest())
    second = coordinator.register_worker(WorkerRegisterRequest())
    coordinator.create_split_job(
        DistributedRunRequest(mode=JobMode.SPLIT, timeout_seconds=1), [{"i": 1}]
    )
    claimed = coordinator.claim_assignment(first.worker_id)
    assert claimed is not None
    _backdate(coordinator, claimed.assignment_id)

    reclaimed = coordinator.claim_assignment(second.worker_id)
    assert reclaimed is not None
    assert reclaimed.assignment_id == claimed.assignment_id


def test_an_unpinned_assignment_from_an_offline_worker_still_requeues(coordinator):
    """Split mode again: an offline worker's task goes back in the shared pool."""
    first = coordinator.register_worker(WorkerRegisterRequest())
    second = coordinator.register_worker(WorkerRegisterRequest())
    coordinator.create_split_job(DistributedRunRequest(mode=JobMode.SPLIT), [{"i": 1}])
    claimed = coordinator.claim_assignment(first.worker_id)
    assert claimed is not None
    _offline(coordinator, first.worker_id)

    reclaimed = coordinator.claim_assignment(second.worker_id)
    assert reclaimed is not None
    assert reclaimed.assignment_id == claimed.assignment_id
    assert reclaimed.status == AssignmentStatus.ASSIGNED


def test_a_departed_hosts_pending_work_fails_with_its_slice(coordinator):
    """Otherwise the job waits forever on tasks only a departed host could claim."""
    dead = coordinator.register_worker(WorkerRegisterRequest())
    job = coordinator.create_fleet_job(
        DistributedRunRequest(mode=JobMode.FLEET),
        [{"i": 1}, {"i": 2}, {"i": 3}],
        [dead],
    )
    assert coordinator.claim_assignment(dead.worker_id) is not None
    _offline(coordinator, dead.worker_id)

    coordinator._requeue_stale_assignments()

    statuses = [
        row[0]
        for row in coordinator._conn.execute(
            "SELECT status FROM distributed_assignments WHERE job_id = ?",
            (job.job_id,),
        )
    ]
    assert statuses == [AssignmentStatus.FAILED.value] * 3
    after = coordinator.get_job(job.job_id)
    assert after is not None
    assert after.status == JobStatus.PARTIAL


def _go_silent(coordinator, worker_id: str, seconds: int = 600) -> None:
    """Backdate last_seen_at so the worker reads as having stopped heartbeating."""
    past = (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat()
    coordinator._conn.execute(
        "UPDATE workers SET last_seen_at = ? WHERE worker_id = ?", (past, worker_id)
    )
    coordinator._conn.commit()


def test_a_worker_that_stops_heartbeating_is_marked_offline(coordinator):
    worker = coordinator.register_worker(WorkerRegisterRequest())
    _go_silent(coordinator, worker.worker_id)

    coordinator._mark_absent_workers()

    assert coordinator.get_worker(worker.worker_id).status == WorkerStatus.OFFLINE


def test_a_recently_seen_worker_stays_online(coordinator):
    """Guard against sweeping up healthy hosts."""
    worker = coordinator.register_worker(WorkerRegisterRequest())
    _go_silent(coordinator, worker.worker_id, seconds=5)

    coordinator._mark_absent_workers()

    assert coordinator.get_worker(worker.worker_id).status == WorkerStatus.ONLINE


def test_the_absence_threshold_is_configurable(coordinator):
    """An operator running slow heartbeats must be able to widen the window.

    `atomics worker --heartbeat-interval` is a flag, but the threshold was fixed
    at four times its *default*, so a worker configured with a 300s interval got
    declared absent and had its pinned fleet work failed while behaving exactly
    as told.
    """
    patient = Coordinator(coordinator._conn, worker_absent_after_seconds=900)
    worker = patient.register_worker(WorkerRegisterRequest())
    _go_silent(patient, worker.worker_id, seconds=600)

    patient._mark_absent_workers()

    assert patient.get_worker(worker.worker_id).status == WorkerStatus.ONLINE
    # Same database, default threshold: the 600s silence is over the line.
    coordinator._mark_absent_workers()
    assert coordinator.get_worker(worker.worker_id).status == WorkerStatus.OFFLINE


def test_a_silent_host_is_not_handed_a_new_fleet_slice(coordinator):
    alive = coordinator.register_worker(WorkerRegisterRequest(labels={"gpu": "4090"}))
    silent = coordinator.register_worker(WorkerRegisterRequest(labels={"gpu": "4090"}))
    _go_silent(coordinator, silent.worker_id)

    matched = coordinator.matching_workers({"gpu": "4090"})

    assert [w.worker_id for w in matched] == [alive.worker_id]


def test_a_silent_hosts_pinned_work_fails_even_if_it_claimed_nothing(coordinator):
    """The realistic dead-host case: killed before it ever asked for work.

    Those assignments stay pending, so the stale-assignment scan never sees them.
    Without the liveness sweep the job waits on a host that cannot claim anything.
    """
    silent = coordinator.register_worker(WorkerRegisterRequest())
    job = coordinator.create_fleet_job(
        DistributedRunRequest(mode=JobMode.FLEET), [{"i": 1}, {"i": 2}], [silent]
    )
    _go_silent(coordinator, silent.worker_id)

    coordinator._requeue_stale_assignments()

    statuses = [
        row[0]
        for row in coordinator._conn.execute(
            "SELECT status FROM distributed_assignments WHERE job_id = ?",
            (job.job_id,),
        )
    ]
    assert statuses == [AssignmentStatus.FAILED.value] * 2
    assert coordinator.get_job(job.job_id).status == JobStatus.PARTIAL


def test_a_silent_worker_does_not_fail_unpinned_work(coordinator):
    """Split mode must still recover rather than lose the task."""
    first = coordinator.register_worker(WorkerRegisterRequest())
    second = coordinator.register_worker(WorkerRegisterRequest())
    coordinator.create_split_job(DistributedRunRequest(mode=JobMode.SPLIT), [{"i": 1}])
    claimed = coordinator.claim_assignment(first.worker_id)
    assert claimed is not None
    _go_silent(coordinator, first.worker_id)

    reclaimed = coordinator.claim_assignment(second.worker_id)

    assert reclaimed is not None
    assert reclaimed.assignment_id == claimed.assignment_id


def test_a_pinned_assignment_fails_once_its_retries_are_exhausted(coordinator):
    """A host that keeps timing out must not hold the job open indefinitely.

    This is the path a killed worker actually takes: nothing marks it offline, so
    its work is only ever judged by the timeout.
    """
    owner = coordinator.register_worker(WorkerRegisterRequest())
    job = coordinator.create_fleet_job(
        DistributedRunRequest(mode=JobMode.FLEET, timeout_seconds=1, max_retries=0),
        [{"i": 1}],
        [owner],
    )
    claimed = coordinator.claim_assignment(owner.worker_id)
    assert claimed is not None
    _backdate(coordinator, claimed.assignment_id)

    coordinator._requeue_stale_assignments()

    after = coordinator.get_assignment(claimed.assignment_id)
    assert after is not None
    assert after.status == AssignmentStatus.FAILED
    job_after = coordinator.get_job(job.job_id)
    assert job_after is not None
    assert job_after.status == JobStatus.PARTIAL


def test_a_pinned_assignment_retries_before_it_fails(coordinator):
    """The retry budget must actually be spent, not skipped."""
    owner = coordinator.register_worker(WorkerRegisterRequest())
    coordinator.create_fleet_job(
        DistributedRunRequest(mode=JobMode.FLEET, timeout_seconds=1, max_retries=1),
        [{"i": 1}],
        [owner],
    )

    first = coordinator.claim_assignment(owner.worker_id)
    assert first is not None
    _backdate(coordinator, first.assignment_id)
    coordinator._requeue_stale_assignments()
    assert coordinator.get_assignment(first.assignment_id).status == (
        AssignmentStatus.PENDING
    )

    second = coordinator.claim_assignment(owner.worker_id)
    assert second is not None
    _backdate(coordinator, second.assignment_id)
    coordinator._requeue_stale_assignments()

    assert coordinator.get_assignment(first.assignment_id).status == (
        AssignmentStatus.FAILED
    )


def test_job_reports_partial_when_one_host_drops_out(coordinator):
    good = coordinator.register_worker(WorkerRegisterRequest())
    bad = coordinator.register_worker(WorkerRegisterRequest())
    job = coordinator.create_fleet_job(
        DistributedRunRequest(mode=JobMode.FLEET), [{"i": 1}], [good, bad]
    )

    finished = coordinator.claim_assignment(good.worker_id)
    assert finished is not None
    coordinator.submit_assignment(
        finished.assignment_id, '{"ok": true}', worker_id=good.worker_id
    )

    assert coordinator.claim_assignment(bad.worker_id) is not None
    _offline(coordinator, bad.worker_id)
    coordinator._requeue_stale_assignments()

    after = coordinator.get_job(job.job_id)
    assert after is not None
    assert after.status == JobStatus.PARTIAL


def test_a_finished_fleet_job_records_a_per_worker_summary(coordinator):
    """summary_json existed since phase 1 with nothing ever writing to it."""
    import json as _json

    first = coordinator.register_worker(WorkerRegisterRequest(labels={"gpu": "4090"}))
    second = coordinator.register_worker(WorkerRegisterRequest(labels={"gpu": "3060"}))
    job = coordinator.create_fleet_job(
        DistributedRunRequest(mode=JobMode.FLEET), [{"i": 1}], [first, second]
    )

    for worker, latency in ((first, 100.0), (second, 300.0)):
        claimed = coordinator.claim_assignment(worker.worker_id)
        assert claimed is not None
        coordinator.submit_assignment(
            claimed.assignment_id,
            _json.dumps(
                {
                    "latency_ms": latency,
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "estimated_cost_usd": 0.01,
                    "tokens_per_second": 25.0,
                    "provider": "ollama",
                    "model": "qwen3:14b",
                }
            ),
            worker_id=worker.worker_id,
        )

    finished = coordinator.get_job(job.job_id)
    assert finished is not None
    assert finished.status == JobStatus.COMPLETED
    assert finished.summary_json is not None

    summary = _json.loads(finished.summary_json)
    by_id = {w["worker_id"]: w for w in summary["workers"]}
    assert set(by_id) == {first.worker_id, second.worker_id}
    assert by_id[first.worker_id]["mean_latency_ms"] == 100.0
    assert by_id[second.worker_id]["mean_latency_ms"] == 300.0
    assert by_id[first.worker_id]["labels"] == {"gpu": "4090"}
    assert summary["total_output_tokens"] == 40


def test_a_summary_records_the_host_that_failed(coordinator):
    good = coordinator.register_worker(WorkerRegisterRequest())
    bad = coordinator.register_worker(WorkerRegisterRequest())
    job = coordinator.create_fleet_job(
        DistributedRunRequest(mode=JobMode.FLEET), [{"i": 1}], [good, bad]
    )
    claimed = coordinator.claim_assignment(good.worker_id)
    assert claimed is not None
    coordinator.submit_assignment(
        claimed.assignment_id, '{"latency_ms": 50}', worker_id=good.worker_id
    )
    _go_silent(coordinator, bad.worker_id)
    coordinator._requeue_stale_assignments()

    import json as _json

    finished = coordinator.get_job(job.job_id)
    assert finished is not None
    summary = _json.loads(finished.summary_json)
    by_id = {w["worker_id"]: w for w in summary["workers"]}
    assert by_id[bad.worker_id]["failed"] == 1
    assert by_id[good.worker_id]["completed"] == 1


def test_submit_assignment_completes_job(coordinator):
    w = coordinator.register_worker(WorkerRegisterRequest())
    job = coordinator.create_split_job(
        DistributedRunRequest(mode=JobMode.SPLIT), [{"i": 1}]
    )
    a = coordinator.claim_assignment(w.worker_id)
    assert a is not None
    coordinator.submit_assignment(a.assignment_id, '{"ok": true}', worker_id=w.worker_id)
    job2 = coordinator.get_job(job.job_id)
    assert job2 is not None
    assert job2.status.value == "completed"


def test_requeue_offline_worker_assignment(coordinator):
    w1 = coordinator.register_worker(WorkerRegisterRequest())
    job = coordinator.create_split_job(
        DistributedRunRequest(mode=JobMode.SPLIT), [{"i": 1}]
    )
    a1 = coordinator.claim_assignment(w1.worker_id)
    assert a1 is not None
    assert a1.worker_id == w1.worker_id

    # Mark worker offline and heartbeat a second worker.
    coordinator._conn.execute(
        "UPDATE workers SET status = ? WHERE worker_id = ?",
        ("offline", w1.worker_id),
    )
    coordinator._conn.commit()

    w2 = coordinator.register_worker(WorkerRegisterRequest())
    a2 = coordinator.claim_assignment(w2.worker_id)
    assert a2 is not None
    assert a2.assignment_id == a1.assignment_id
    assert a2.worker_id == w2.worker_id


def test_requeue_timed_out_assignment(coordinator):
    w1 = coordinator.register_worker(WorkerRegisterRequest())
    job = coordinator.create_split_job(
        DistributedRunRequest(mode=JobMode.SPLIT, timeout_seconds=1),
        [{"i": 1}],
    )
    a1 = coordinator.claim_assignment(w1.worker_id)
    assert a1 is not None

    # Manually age the started_at timestamp so the timeout requeue fires.
    old_started = "2026-01-01T00:00:00+00:00"
    coordinator._conn.execute(
        "UPDATE distributed_assignments SET started_at = ? WHERE assignment_id = ?",
        (old_started, a1.assignment_id),
    )
    coordinator._conn.commit()

    a2 = coordinator.claim_assignment(w1.worker_id)
    assert a2 is not None
    assert a2.assignment_id == a1.assignment_id
    assert a2.retry_count == 2


def test_partial_job_status_on_failure(coordinator):
    w = coordinator.register_worker(WorkerRegisterRequest())
    job = coordinator.create_split_job(
        DistributedRunRequest(mode=JobMode.SPLIT), [{"i": 1}, {"i": 2}]
    )
    a1 = coordinator.claim_assignment(w.worker_id)
    a2 = coordinator.claim_assignment(w.worker_id)
    assert a1 is not None and a2 is not None

    coordinator.submit_assignment(
        a1.assignment_id, '{"ok": true}', worker_id=w.worker_id
    )
    coordinator.submit_assignment(
        a2.assignment_id, None, worker_id=w.worker_id, error="boom"
    )

    job2 = coordinator.get_job(job.job_id)
    assert job2 is not None
    assert job2.status.value == "partial"


def test_recover_jobs_requeues_stale_assigned_work(coordinator):
    w = coordinator.register_worker(WorkerRegisterRequest())
    coordinator.create_split_job(
        DistributedRunRequest(mode=JobMode.SPLIT, timeout_seconds=1),
        [{"i": 1}],
    )
    a1 = coordinator.claim_assignment(w.worker_id)
    assert a1 is not None

    old_started = "2026-01-01T00:00:00+00:00"
    coordinator._conn.execute(
        "UPDATE distributed_assignments SET started_at = ? WHERE assignment_id = ?",
        (old_started, a1.assignment_id),
    )
    coordinator._conn.commit()

    coordinator.recover_jobs()
    a2 = coordinator.claim_assignment(w.worker_id)
    assert a2 is not None
    assert a2.assignment_id == a1.assignment_id


# ── Full-mode tests ──────────────────────────────────────────────────────────


def test_full_job_creates_one_assignment(coordinator):
    req = DistributedRunRequest(mode=JobMode.FULL, run_request={"iterations": 5})
    job = coordinator.create_full_job(req, [])
    assert job.mode == JobMode.FULL
    rows = coordinator._conn.execute(
        "SELECT target_worker_id, task_spec FROM distributed_assignments WHERE job_id = ?",
        (job.job_id,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] is None
    spec = json.loads(rows[0][1])
    assert spec["mode"] == "full"
    assert spec["run_request"]["iterations"] == 5


def test_full_job_with_selector_pins_to_first_matching_worker(coordinator):
    w1 = coordinator.register_worker(WorkerRegisterRequest(labels={"box": "a"}))
    coordinator.register_worker(WorkerRegisterRequest(labels={"box": "b"}))
    req = DistributedRunRequest(mode=JobMode.FULL, run_request={"iterations": 1})
    job = coordinator.create_full_job_from_selector(req, {"box": "a"})
    rows = coordinator._conn.execute(
        "SELECT target_worker_id FROM distributed_assignments WHERE job_id = ?",
        (job.job_id,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == w1.worker_id


def test_full_job_with_zero_matches_raises(coordinator):
    coordinator.register_worker(WorkerRegisterRequest(labels={"box": "a"}))
    req = DistributedRunRequest(mode=JobMode.FULL, run_request={"iterations": 1})
    with pytest.raises(ValueError, match="no online workers match"):
        coordinator.create_full_job_from_selector(req, {"box": "missing"})


def test_full_job_claimable_by_any_worker_when_no_selector(coordinator):
    coordinator.register_worker(WorkerRegisterRequest())
    w2 = coordinator.register_worker(WorkerRegisterRequest())
    job = coordinator.create_full_job(
        DistributedRunRequest(mode=JobMode.FULL, run_request={"iterations": 1}), []
    )
    a = coordinator.claim_assignment(w2.worker_id)
    assert a is not None
    assert a.job_id == job.job_id
    assert a.worker_id == w2.worker_id


def test_full_job_with_selector_only_claimable_by_target_worker(coordinator):
    w1 = coordinator.register_worker(WorkerRegisterRequest(labels={"box": "a"}))
    w2 = coordinator.register_worker(WorkerRegisterRequest(labels={"box": "b"}))
    job = coordinator.create_full_job_from_selector(
        DistributedRunRequest(mode=JobMode.FULL, run_request={"iterations": 1}),
        {"box": "a"},
    )
    assert coordinator.claim_assignment(w2.worker_id) is None
    a = coordinator.claim_assignment(w1.worker_id)
    assert a is not None
    assert a.job_id == job.job_id


def test_full_job_reaches_completed_when_result_submitted(coordinator):
    w = coordinator.register_worker(WorkerRegisterRequest())
    job = coordinator.create_full_job(
        DistributedRunRequest(mode=JobMode.FULL, run_request={"iterations": 1}), []
    )
    a = coordinator.claim_assignment(w.worker_id)
    assert a is not None
    coordinator.submit_assignment(a.assignment_id, '{"ok": true}', worker_id=w.worker_id)
    completed = coordinator.get_job(job.job_id)
    assert completed is not None
    assert completed.status == JobStatus.COMPLETED


# ── Capability-based assignment routing ───────────────────────────────────────


def test_python_worker_claims_python_assignment(coordinator):
    w = coordinator.register_worker(WorkerRegisterRequest(capabilities=["python"]))
    coordinator.create_split_job(
        DistributedRunRequest(mode=JobMode.SPLIT), [{"task_name": "t", "runtime": "python"}]
    )
    a = coordinator.claim_assignment(w.worker_id)
    assert a is not None


def test_node_worker_claims_node_assignment(coordinator):
    node = coordinator.register_worker(WorkerRegisterRequest(capabilities=["node"]))
    py = coordinator.register_worker(WorkerRegisterRequest(capabilities=["python"]))
    coordinator.create_split_job(
        DistributedRunRequest(mode=JobMode.SPLIT), [{"task_name": "t", "runtime": "node"}]
    )
    assert coordinator.claim_assignment(py.worker_id) is None
    a = coordinator.claim_assignment(node.worker_id)
    assert a is not None


def test_worker_with_no_capabilities_defaults_to_python(coordinator):
    w = coordinator.register_worker(WorkerRegisterRequest())
    coordinator.create_split_job(
        DistributedRunRequest(mode=JobMode.SPLIT), [{"task_name": "t"}]
    )
    a = coordinator.claim_assignment(w.worker_id)
    assert a is not None


def test_worker_without_runtime_capability_does_not_claim(coordinator):
    w = coordinator.register_worker(WorkerRegisterRequest(capabilities=["python"]))
    coordinator.create_split_job(
        DistributedRunRequest(mode=JobMode.SPLIT), [{"task_name": "t", "runtime": "wasm"}]
    )
    assert coordinator.claim_assignment(w.worker_id) is None
