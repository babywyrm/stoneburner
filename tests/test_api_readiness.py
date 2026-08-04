"""Liveness and readiness are separate questions.

`/health` used to answer `ok` unconditionally, which kept a server in a load
balancer's rotation while every request it received was going to fail on an
unreachable database. The split puts the dependency check where an orchestrator
can act on it without restarting a process that is working fine.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from atomics.api.config import ServerSettings
from atomics.api.server import create_app


@pytest.fixture
def app(tmp_path):
    return create_app(ServerSettings(no_auth=True, db_path=tmp_path / "ready.db"))


class TestLiveness:
    def test_health_reports_ok(self, app):
        with TestClient(app) as client:
            res = client.get("/api/v1/health")
            assert res.status_code == 200
            assert res.json()["status"] == "ok"

    def test_health_stays_ok_when_the_database_is_broken(self, app):
        """Liveness must not depend on the database.

        Restarting the API server does not repair a database outage; it just
        removes the endpoint that could have reported one.
        """
        with TestClient(app) as client:
            app.state.coordinator._conn.close()
            res = client.get("/api/v1/health")
            assert res.status_code == 200
            assert res.json()["status"] == "ok"

    def test_health_needs_no_credentials(self, tmp_path):
        keyed = create_app(ServerSettings(api_keys={"k"}, db_path=tmp_path / "h.db"))
        with TestClient(keyed) as client:
            assert client.get("/api/v1/health").status_code == 200


class TestReadiness:
    def test_a_healthy_server_is_ready(self, app):
        with TestClient(app) as client:
            res = client.get("/api/v1/ready")
            assert res.status_code == 200
            assert res.json()["status"] == "ready"

    def test_the_database_check_is_reported_by_name(self, app):
        with TestClient(app) as client:
            checks = client.get("/api/v1/ready").json()["checks"]
            assert [c["name"] for c in checks] == ["database"]
            assert checks[0]["ok"] is True
            assert checks[0]["detail"] is None

    def test_an_unreachable_database_makes_the_server_unready(self, app):
        with TestClient(app) as client:
            app.state.coordinator._conn.close()
            res = client.get("/api/v1/ready")

            assert res.status_code == 503
            assert res.json()["status"] == "not_ready"

    def test_the_failure_explains_itself(self, app):
        """A 503 with no reason sends someone reading source at 3am."""
        with TestClient(app) as client:
            app.state.coordinator._conn.close()
            check = client.get("/api/v1/ready").json()["checks"][0]

            assert check["ok"] is False
            assert check["detail"]
            assert "ProgrammingError" in check["detail"] or "closed" in check["detail"]

    def test_readiness_recovers_when_the_database_does(self, app, tmp_path):
        """Readiness must be live state, not a flag latched at startup."""
        from atomics.storage.schema import init_db

        with TestClient(app) as client:
            app.state.coordinator._conn.close()
            assert client.get("/api/v1/ready").status_code == 503

            app.state.coordinator._conn = init_db(tmp_path / "ready.db")
            assert client.get("/api/v1/ready").status_code == 200

    def test_readiness_needs_no_credentials(self, tmp_path):
        """A probe should not need a key; it reveals no run data."""
        keyed = create_app(ServerSettings(api_keys={"k"}, db_path=tmp_path / "r.db"))
        with TestClient(keyed) as client:
            assert client.get("/api/v1/ready").status_code == 200

    def test_readiness_before_startup_is_not_ready(self, app):
        """Reached before lifespan runs, there is no coordinator to check."""
        client = TestClient(app)  # no context manager, so no lifespan
        res = client.get("/api/v1/ready")

        assert res.status_code == 503
        assert res.json()["checks"][0]["detail"] == "coordinator not initialized"


class TestCoordinatorCheck:
    def test_a_live_connection_reports_no_error(self, tmp_path):
        from atomics.distributed.coordinator import Coordinator
        from atomics.storage.schema import init_db

        coordinator = Coordinator(init_db(tmp_path / "c.db"))
        try:
            assert coordinator.check_database() is None
        finally:
            coordinator._conn.close()

    def test_a_closed_connection_reports_the_error(self, tmp_path):
        from atomics.distributed.coordinator import Coordinator
        from atomics.storage.schema import init_db

        coordinator = Coordinator(init_db(tmp_path / "c.db"))
        coordinator._conn.close()

        error = coordinator.check_database()
        assert error is not None
        assert "Error" in error
