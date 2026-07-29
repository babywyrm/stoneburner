import json

import pytest
from fastapi.testclient import TestClient

from atomics.api.config import ServerSettings
from atomics.api.server import create_app
from atomics.tasks import TASK_CATALOG

API_KEY = "test-coordinator-key"


@pytest.fixture
def client(tmp_path):
    app = create_app(no_auth=True, db_path=tmp_path / "distributed.db")
    with TestClient(app) as tc:
        yield tc


@pytest.fixture
def secured_client(tmp_path):
    """A coordinator with authentication actually switched on."""
    app = create_app(
        ServerSettings(
            no_auth=False,
            api_keys={API_KEY},
            db_path=tmp_path / "secured.db",
        )
    )
    with TestClient(app) as tc:
        yield tc


def _auth() -> dict[str, str]:
    return {"X-API-Key": API_KEY}


# Every endpoint on this router, with a body where one is required. Submitting a
# job spends GPU time and cloud budget, so an unauthenticated caller reaching any
# of these is a real exposure rather than an information leak.
DISTRIBUTED_ENDPOINTS = [
    ("post", "/api/v1/workers/register", {}),
    ("post", "/api/v1/distributed/runs", {"mode": "split", "run_request": {"iterations": 1}}),
    ("get", "/api/v1/distributed/runs/any-job-id", None),
]


@pytest.mark.parametrize(("method", "path", "body"), DISTRIBUTED_ENDPOINTS)
def test_distributed_endpoints_require_a_key(secured_client, method, path, body):
    request = getattr(secured_client, method)
    resp = request(path) if body is None else request(path, json=body)
    assert resp.status_code == 401, f"{method.upper()} {path} answered anonymously"


@pytest.mark.parametrize(("method", "path", "body"), DISTRIBUTED_ENDPOINTS)
def test_distributed_endpoints_reject_an_unknown_key(secured_client, method, path, body):
    request = getattr(secured_client, method)
    headers = {"X-API-Key": "not-the-key"}
    resp = (
        request(path, headers=headers)
        if body is None
        else request(path, json=body, headers=headers)
    )
    assert resp.status_code == 401


def test_registering_a_worker_succeeds_with_a_key(secured_client):
    resp = secured_client.post(
        "/api/v1/workers/register", json={"labels": {"gpu": "1"}}, headers=_auth()
    )
    assert resp.status_code == 200
    assert "worker_id" in resp.json()


def test_submitting_and_reading_a_run_succeeds_with_a_key(secured_client):
    submitted = secured_client.post(
        "/api/v1/distributed/runs",
        json={"mode": "split", "run_request": {"iterations": 2}},
        headers=_auth(),
    )
    assert submitted.status_code == 202
    job_id = submitted.json()["job_id"]

    fetched = secured_client.get(
        f"/api/v1/distributed/runs/{job_id}", headers=_auth()
    )
    assert fetched.status_code == 200
    assert fetched.json()["job_id"] == job_id


def test_auth_is_checked_before_the_job_is_looked_up(secured_client):
    """An anonymous caller must not learn whether a job id exists."""
    resp = secured_client.get("/api/v1/distributed/runs/does-not-exist")
    assert resp.status_code == 401


def test_no_auth_backend_permits_the_distributed_endpoints(client):
    """The --no-auth development path must keep working unchanged."""
    registered = client.post("/api/v1/workers/register", json={})
    assert registered.status_code == 200
    submitted = client.post(
        "/api/v1/distributed/runs",
        json={"mode": "split", "run_request": {"iterations": 1}},
    )
    assert submitted.status_code == 202
    assert client.get(
        f"/api/v1/distributed/runs/{submitted.json()['job_id']}"
    ).status_code == 200


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


def _fleet_assignments(client, job_id: str) -> list[tuple[str, str]]:
    rows = client.app.state.coordinator._conn.execute(
        "SELECT target_worker_id, task_spec FROM distributed_assignments "
        "WHERE job_id = ?",
        (job_id,),
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def _register(client, labels: dict[str, str]) -> str:
    resp = client.post("/api/v1/workers/register", json={"labels": labels})
    assert resp.status_code == 200
    return resp.json()["worker_id"]


def test_fleet_run_broadcasts_to_every_matching_worker(client):
    first = _register(client, {"gpu": "4090"})
    second = _register(client, {"gpu": "4090"})
    _register(client, {"gpu": "3060"})

    resp = client.post(
        "/api/v1/distributed/runs",
        json={
            "mode": "fleet",
            "run_request": {"iterations": 2, "tier": "ez"},
            "worker_selector": {"gpu": "4090"},
        },
    )
    assert resp.status_code == 202
    assert resp.json()["mode"] == "fleet"

    rows = _fleet_assignments(client, resp.json()["job_id"])
    assert len(rows) == 4
    assert {target for target, _ in rows} == {first, second}
    # Both hosts must run the same two prompts for the comparison to mean anything.
    per_worker: dict[str, set[str]] = {}
    for target, spec in rows:
        per_worker.setdefault(target, set()).add(spec)
    assert len({frozenset(specs) for specs in per_worker.values()}) == 1


def test_fleet_run_with_no_matching_workers_is_rejected(client):
    _register(client, {"gpu": "3060"})

    resp = client.post(
        "/api/v1/distributed/runs",
        json={
            "mode": "fleet",
            "run_request": {"iterations": 1},
            "worker_selector": {"gpu": "4090"},
        },
    )

    assert resp.status_code == 400
    assert "no online workers" in resp.json()["detail"].lower()
    # A job that can never progress must not be left behind.
    jobs = client.app.state.coordinator._conn.execute(
        "SELECT COUNT(*) FROM distributed_jobs"
    ).fetchone()[0]
    assert jobs == 0


def test_fleet_run_without_a_selector_uses_every_online_worker(client):
    first = _register(client, {})
    second = _register(client, {"gpu": "4090"})

    resp = client.post(
        "/api/v1/distributed/runs",
        json={"mode": "fleet", "run_request": {"iterations": 1}},
    )

    assert resp.status_code == 202
    rows = _fleet_assignments(client, resp.json()["job_id"])
    assert {target for target, _ in rows} == {first, second}


def test_full_mode_is_accepted(client):
    """Full mode creates a single assignment with the full run request."""
    resp = client.post(
        "/api/v1/distributed/runs",
        json={"mode": "full", "run_request": {"iterations": 3}},
    )
    assert resp.status_code == 202
    job = resp.json()
    rows = client.app.state.coordinator._conn.execute(
        "SELECT task_spec FROM distributed_assignments WHERE job_id = ?",
        (job["job_id"],),
    ).fetchall()
    assert len(rows) == 1
    spec = json.loads(rows[0][0])
    assert spec["mode"] == "full"
    assert spec["run_request"]["iterations"] == 3


def test_full_mode_with_selector_and_no_match_is_rejected(client):
    """Full mode with a selector must reject when no worker matches."""
    resp = client.post(
        "/api/v1/distributed/runs",
        json={
            "mode": "full",
            "run_request": {"iterations": 1},
            "worker_selector": {"box": "missing"},
        },
    )
    assert resp.status_code == 400
    assert "selector" in resp.json()["detail"].lower()


def _task_specs(client, job_id: str) -> list[dict]:
    rows = client.app.state.coordinator._conn.execute(
        "SELECT task_spec FROM distributed_assignments WHERE job_id = ?",
        (job_id,),
    ).fetchall()
    return [json.loads(row[0]) for row in rows]


def test_worker_selector_is_rejected_not_ignored(client):
    """An unhonored selector must 400 rather than produce an untargeted run."""
    resp = client.post(
        "/api/v1/distributed/runs",
        json={
            "mode": "split",
            "run_request": {"iterations": 1, "tier": "ez"},
            "worker_selector": {"gpu": "1"},
        },
    )
    assert resp.status_code == 400
    assert "worker_selector" in resp.json()["detail"]


def test_pinned_provider_travels_with_every_task_spec(client):
    """A provider named on the run request must reach the workers executing it."""
    resp = client.post(
        "/api/v1/distributed/runs",
        json={
            "mode": "split",
            "run_request": {
                "iterations": 3,
                "tier": "ez",
                "provider": "vllm",
                "model": "qwen3:14b",
            },
        },
    )
    assert resp.status_code == 202
    specs = _task_specs(client, resp.json()["job_id"])
    assert len(specs) == 3
    assert all(spec["provider"] == "vllm" for spec in specs)
    assert all(spec["model"] == "qwen3:14b" for spec in specs)


def test_unpinned_run_leaves_provider_to_the_worker(client):
    """Omitting a provider must not silently pin one — workers keep their own."""
    resp = client.post(
        "/api/v1/distributed/runs",
        json={"mode": "split", "run_request": {"iterations": 1, "tier": "ez"}},
    )
    specs = _task_specs(client, resp.json()["job_id"])
    assert specs
    assert all("provider" not in spec for spec in specs)
    assert all("model" not in spec for spec in specs)


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
