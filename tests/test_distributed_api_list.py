"""Tests for distributed job/worker listing endpoints used by the dashboard."""

from __future__ import annotations

from fastapi.testclient import TestClient

from atomics.api.config import ServerSettings
from atomics.api.server import create_app

API_KEY = "test-coordinator-key"
ALICE = "alice-key-0123456789"
BOB = "bob-key-9876543210"


def _auth() -> dict[str, str]:
    return {"X-API-Key": API_KEY}


def test_list_jobs_returns_empty_by_default(tmp_path):
    app = create_app(ServerSettings(no_auth=False, api_keys={API_KEY}, db_path=tmp_path / "db.db"))
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        res = tc.get("/api/v1/distributed/runs", headers=_auth())
        assert res.status_code == 200
        assert res.json() == {"jobs": []}


def test_list_workers_returns_empty_by_default(tmp_path):
    app = create_app(ServerSettings(no_auth=False, api_keys={API_KEY}, db_path=tmp_path / "db.db"))
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        res = tc.get("/api/v1/workers", headers=_auth())
        assert res.status_code == 200
        assert res.json() == {"workers": []}


def test_list_jobs_after_creating_run(tmp_path):
    app = create_app(ServerSettings(no_auth=True, db_path=tmp_path / "db.db"))
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        tc.post(
            "/api/v1/distributed/runs", json={"mode": "split", "run_request": {"iterations": 1}}
        )
        res = tc.get("/api/v1/distributed/runs")
        data = res.json()
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["mode"] == "split"


def test_list_workers_after_registration(tmp_path):
    app = create_app(ServerSettings(no_auth=True, db_path=tmp_path / "db.db"))
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        tc.post("/api/v1/workers/register", json={})
        res = tc.get("/api/v1/workers")
        data = res.json()
        assert len(data["workers"]) == 1
        assert "worker_id" in data["workers"][0]


def test_distributed_list_limit_is_bounded():
    from fastapi.testclient import TestClient

    from atomics.api.config import ServerSettings
    from atomics.api.server import create_app

    app = create_app(settings=ServerSettings(no_auth=True))
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        assert tc.get("/api/v1/distributed/runs?limit=100000").status_code == 422


def test_a_caller_sees_only_their_own_distributed_runs(tmp_path):
    alice, bob = {"X-API-Key": ALICE}, {"X-API-Key": BOB}
    app = create_app(ServerSettings(api_keys={ALICE, BOB}, db_path=tmp_path / "db.db"))
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        job = tc.post(
            "/api/v1/distributed/runs",
            json={"mode": "split", "run_request": {"iterations": 1}},
            headers=alice,
        ).json()
        assert tc.get(f"/api/v1/distributed/runs/{job['job_id']}", headers=alice).status_code == 200
        assert tc.get(f"/api/v1/distributed/runs/{job['job_id']}", headers=bob).status_code == 404
        assert len(tc.get("/api/v1/distributed/runs", headers=alice).json()["jobs"]) == 1
        assert tc.get("/api/v1/distributed/runs", headers=bob).json()["jobs"] == []
        assert "owner" not in job


def test_an_existing_distributed_jobs_table_gains_owner(tmp_path):
    import sqlite3

    from atomics.storage.schema import init_db

    db = tmp_path / "old.db"
    old = sqlite3.connect(db)
    old.execute(
        "CREATE TABLE distributed_jobs (job_id TEXT PRIMARY KEY, mode TEXT NOT NULL, "
        "parent_run_id TEXT, status TEXT NOT NULL DEFAULT 'pending', request_json TEXT NOT NULL, "
        "summary_json TEXT, created_at TEXT NOT NULL, completed_at TEXT)"
    )
    old.execute(
        "INSERT INTO distributed_jobs VALUES ('j1','split',NULL,'pending','{}',NULL,'t',NULL)"
    )
    old.commit()
    old.close()
    conn = init_db(db)
    assert conn.execute("SELECT owner FROM distributed_jobs WHERE job_id='j1'").fetchone()[0] == (
        "anonymous"
    )
