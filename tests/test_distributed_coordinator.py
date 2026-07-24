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
