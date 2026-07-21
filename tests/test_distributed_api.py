import json

import pytest
from fastapi.testclient import TestClient

from atomics.api.server import create_app
from atomics.tasks import TASK_CATALOG


@pytest.fixture
def client(tmp_path):
    app = create_app(no_auth=True, db_path=tmp_path / "distributed.db")
    with TestClient(app) as tc:
        yield tc


def test_register_worker(client):
    resp = client.post("/api/v1/workers/register", json={"labels": {"provider": "ollama"}})
    assert resp.status_code == 200
    assert "worker_id" in resp.json()


def test_poll_next_assignment(client):
    reg = client.post("/api/v1/workers/register", json={})
    worker_id = reg.json()["worker_id"]
    # create a job
    client.post(
        "/api/v1/distributed/runs",
        json={"mode": "split", "run_request": {"iterations": 1}},
    )
    resp = client.get(f"/api/v1/workers/{worker_id}/jobs/next")
    assert resp.status_code == 200
    assert resp.json() is not None


def test_get_job(client):
    resp = client.post(
        "/api/v1/distributed/runs",
        json={"mode": "split", "run_request": {"iterations": 2}},
    )
    job_id = resp.json()["job_id"]
    resp = client.get(f"/api/v1/distributed/runs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["job_id"] == job_id


def test_distributed_run_uses_real_task_specs(client):
    resp = client.post(
        "/api/v1/distributed/runs",
        json={"mode": "split", "run_request": {"iterations": 2, "tier": "ez"}},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    conn = client.app.state.coordinator._conn
    rows = conn.execute(
        "SELECT task_spec FROM distributed_assignments WHERE job_id = ?",
        (job_id,),
    ).fetchall()
    assert len(rows) == 2
    catalog_names = {t.name for t in TASK_CATALOG}
    for row in rows:
        spec = json.loads(row[0])
        assert "task_name" in spec
        assert "prompt" in spec
        assert spec["prompt"]
        assert spec["task_name"] in catalog_names
        assert "category" in spec
        assert "max_output_tokens" in spec
