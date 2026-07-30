"""Tests for distributed job/worker listing endpoints used by the dashboard."""

from __future__ import annotations

from fastapi.testclient import TestClient

from atomics.api.config import ServerSettings
from atomics.api.server import create_app

API_KEY = "test-coordinator-key"


def _auth() -> dict[str, str]:
    return {"X-API-Key": API_KEY}


def test_list_jobs_returns_empty_by_default(tmp_path):
    app = create_app(ServerSettings(no_auth=False, api_keys={API_KEY}, db_path=tmp_path / "db.db"))
    with TestClient(app) as tc:
        res = tc.get("/api/v1/distributed/runs", headers=_auth())
        assert res.status_code == 200
        assert res.json() == {"jobs": []}


def test_list_workers_returns_empty_by_default(tmp_path):
    app = create_app(ServerSettings(no_auth=False, api_keys={API_KEY}, db_path=tmp_path / "db.db"))
    with TestClient(app) as tc:
        res = tc.get("/api/v1/workers", headers=_auth())
        assert res.status_code == 200
        assert res.json() == {"workers": []}


def test_list_jobs_after_creating_run(tmp_path):
    app = create_app(ServerSettings(no_auth=True, db_path=tmp_path / "db.db"))
    with TestClient(app) as tc:
        tc.post("/api/v1/distributed/runs", json={"mode": "split", "run_request": {"iterations": 1}})
        res = tc.get("/api/v1/distributed/runs")
        data = res.json()
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["mode"] == "split"


def test_list_workers_after_registration(tmp_path):
    app = create_app(ServerSettings(no_auth=True, db_path=tmp_path / "db.db"))
    with TestClient(app) as tc:
        tc.post("/api/v1/workers/register", json={})
        res = tc.get("/api/v1/workers")
        data = res.json()
        assert len(data["workers"]) == 1
        assert "worker_id" in data["workers"][0]
