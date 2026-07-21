import pytest
from fastapi.testclient import TestClient

from atomics.api.config import ServerSettings
from atomics.api.server import create_app


@pytest.fixture
def client():
    app = create_app(settings=ServerSettings(no_auth=True))
    with TestClient(app) as tc:
        yield tc


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_post_runs_unauth():
    app = create_app(settings=ServerSettings(api_keys={"secret"}))
    with TestClient(app) as tc:
        resp = tc.post("/api/v1/runs", json={"provider": "ollama"})
        assert resp.status_code == 401


def test_post_runs_with_auth():
    app = create_app(settings=ServerSettings(api_keys={"secret"}))
    with TestClient(app) as tc:
        resp = tc.post(
            "/api/v1/runs",
            json={"provider": "ollama"},
            headers={"X-API-Key": "secret"},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["kind"] == "run"
        assert body["status"] == "pending"


def test_get_job_not_found(client):
    resp = client.get("/api/v1/jobs/invalid")
    assert resp.status_code == 404

def test_get_job_returns_running_while_in_progress(client):
    import asyncio
    import threading
    from unittest.mock import AsyncMock, patch

    started = threading.Event()
    release = threading.Event()

    async def slow_run(_payload):
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.01)
        return {"run_id": "slow", "tasks": 1, "success": 1}

    with patch(
        "atomics.api.routes.run_benchmark_from_request",
        new=AsyncMock(side_effect=slow_run),
    ):
        resp = client.post("/api/v1/runs", json={"provider": "ollama"})
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        assert started.wait(timeout=2.0)

        poll = client.get(f"/api/v1/jobs/{job_id}")
        assert poll.status_code == 200
        body = poll.json()
        assert body["status"] == "running"
        assert body["error"] is None

        release.set()

        for _ in range(50):
            poll = client.get(f"/api/v1/jobs/{job_id}")
            if poll.json()["status"] == "completed":
                break
        assert poll.json()["status"] == "completed"
        assert poll.json()["result"]["run_id"] == "slow"

