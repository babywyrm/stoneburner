import pytest

from atomics.distributed.coordinator import Coordinator
from atomics.distributed.models import DistributedRunRequest, JobMode, WorkerRegisterRequest
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


def test_submit_assignment_completes_job(coordinator):
    w = coordinator.register_worker(WorkerRegisterRequest())
    job = coordinator.create_split_job(
        DistributedRunRequest(mode=JobMode.SPLIT), [{"i": 1}]
    )
    a = coordinator.claim_assignment(w.worker_id)
    assert a is not None
    coordinator.submit_assignment(a.assignment_id, '{"ok": true}')
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

    coordinator.submit_assignment(a1.assignment_id, '{"ok": true}')
    coordinator.submit_assignment(a2.assignment_id, None, error="boom")

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
