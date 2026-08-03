"""Regression tests for the v0.15.2 security fixes.

Each test corresponds to a finding from the 2026-08-02 audit. They assert the
boundary, not the implementation, so a future refactor that reopens the hole
still fails here.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from atomics.api.config import ServerSettings, is_loopback_host
from atomics.api.server import create_app
from atomics.distributed.coordinator import AssignmentRejectedError, Coordinator
from atomics.distributed.models import (
    DistributedRunRequest,
    JobMode,
    WorkerRegisterRequest,
)
from atomics.storage.schema import init_db


class TestNoAuthRequiresLoopback:
    """--no-auth on a public interface exposes eval submission unauthenticated."""

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "[::1]"])
    def test_loopback_hosts_permit_no_auth(self, host):
        assert is_loopback_host(host) is True
        ServerSettings(host=host, no_auth=True)

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.239", "::", "example.com"])
    def test_public_hosts_reject_no_auth(self, host):
        assert is_loopback_host(host) is False
        with pytest.raises(ValueError, match="no_auth cannot be combined"):
            ServerSettings(host=host, no_auth=True)

    def test_public_hosts_are_fine_with_a_key(self):
        settings = ServerSettings(host="0.0.0.0", api_keys={"k"})
        assert settings.host == "0.0.0.0"


class TestWorkerKeysAreSeparable:
    """A worker credential must not also authorize run and eval submission."""

    def test_worker_keys_default_to_the_submitter_keys(self):
        settings = ServerSettings(api_keys={"shared"})
        assert settings.effective_worker_keys == {"shared"}

    def test_worker_keys_override_when_supplied(self):
        settings = ServerSettings(api_keys={"submit"}, worker_api_keys={"worker"})
        assert settings.effective_worker_keys == {"worker"}

    def test_a_worker_key_cannot_submit_evals(self, tmp_path):
        app = create_app(
            ServerSettings(
                api_keys={"submitter-key"},
                worker_api_keys={"worker-key"},
                db_path=tmp_path / "sec.db",
            )
        )
        with TestClient(app) as client:
            registered = client.post(
                "/api/v1/workers/register",
                json={},
                headers={"X-API-Key": "worker-key"},
            )
            assert registered.status_code == 200

            escalation = client.post(
                "/api/v1/evals",
                json={"suite": "codegen", "provider": "ollama"},
                headers={"X-API-Key": "worker-key"},
            )
            assert escalation.status_code == 401

    def test_a_submitter_key_cannot_act_as_a_worker(self, tmp_path):
        app = create_app(
            ServerSettings(
                api_keys={"submitter-key"},
                worker_api_keys={"worker-key"},
                db_path=tmp_path / "sec.db",
            )
        )
        with TestClient(app) as client:
            res = client.post(
                "/api/v1/workers/register",
                json={},
                headers={"X-API-Key": "submitter-key"},
            )
            assert res.status_code == 401


class TestAssignmentOwnership:
    """An assignment may only be completed by the worker holding it."""

    @pytest.fixture
    def coordinator(self, tmp_path):
        return Coordinator(init_db(tmp_path / "coord.db"))

    def test_another_worker_cannot_submit_your_result(self, coordinator):
        owner = coordinator.register_worker(WorkerRegisterRequest())
        attacker = coordinator.register_worker(WorkerRegisterRequest())
        coordinator.create_split_job(
            DistributedRunRequest(mode=JobMode.SPLIT), [{"i": 1}]
        )
        assignment = coordinator.claim_assignment(owner.worker_id)
        assert assignment is not None

        with pytest.raises(AssignmentRejectedError):
            coordinator.submit_assignment(
                assignment.assignment_id,
                '{"forged": true}',
                worker_id=attacker.worker_id,
            )

        unchanged = coordinator.get_assignment(assignment.assignment_id)
        assert unchanged is not None
        assert unchanged.result_json is None

    def test_a_completed_assignment_cannot_be_overwritten(self, coordinator):
        owner = coordinator.register_worker(WorkerRegisterRequest())
        coordinator.create_split_job(
            DistributedRunRequest(mode=JobMode.SPLIT), [{"i": 1}]
        )
        assignment = coordinator.claim_assignment(owner.worker_id)
        assert assignment is not None
        coordinator.submit_assignment(
            assignment.assignment_id, '{"ok": true}', worker_id=owner.worker_id
        )

        with pytest.raises(AssignmentRejectedError):
            coordinator.submit_assignment(
                assignment.assignment_id, '{"ok": false}', worker_id=owner.worker_id
            )

        final = coordinator.get_assignment(assignment.assignment_id)
        assert final is not None
        assert final.result_json == '{"ok": true}'

    def test_an_unknown_assignment_still_reports_not_found(self, coordinator):
        worker = coordinator.register_worker(WorkerRegisterRequest())
        result = coordinator.submit_assignment(
            "does-not-exist", "{}", worker_id=worker.worker_id
        )
        assert result is None

    def test_the_route_rejects_a_mismatched_worker_with_409(self, tmp_path):
        app = create_app(ServerSettings(no_auth=True, db_path=tmp_path / "route.db"))
        with TestClient(app) as client:
            owner = client.post("/api/v1/workers/register", json={}).json()
            attacker = client.post("/api/v1/workers/register", json={}).json()
            client.post(
                "/api/v1/distributed/runs",
                json={"mode": "split", "run_request": {"iterations": 1}},
            )
            claimed = client.get(
                f"/api/v1/workers/{owner['worker_id']}/jobs/next"
            ).json()
            assert claimed is not None

            forged = client.post(
                f"/api/v1/workers/{attacker['worker_id']}"
                f"/jobs/{claimed['assignment_id']}/result",
                json={"status": "completed", "result_json": '{"forged": true}'},
            )
            assert forged.status_code == 409

    def test_the_rightful_worker_still_succeeds_over_http(self, tmp_path):
        app = create_app(ServerSettings(no_auth=True, db_path=tmp_path / "route.db"))
        with TestClient(app) as client:
            owner = client.post("/api/v1/workers/register", json={}).json()
            client.post(
                "/api/v1/distributed/runs",
                json={"mode": "split", "run_request": {"iterations": 1}},
            )
            claimed = client.get(
                f"/api/v1/workers/{owner['worker_id']}/jobs/next"
            ).json()

            accepted = client.post(
                f"/api/v1/workers/{owner['worker_id']}"
                f"/jobs/{claimed['assignment_id']}/result",
                json={"status": "completed", "result_json": '{"ok": true}'},
            )
            assert accepted.status_code == 200
            assert accepted.json()["status"] == "completed"


class TestDashboardEscaping:
    """Worker-supplied strings must never reach the DOM as markup."""

    def test_the_dashboard_builds_rows_without_innerhtml(self):
        from atomics.api.dashboard import _DASHBOARD_HTML

        script = _DASHBOARD_HTML.split("<script>", 1)[1]
        assert "createElement" in script
        assert "textContent" in script
        # Assigning to .innerHTML in the render path is how the stored XSS
        # happened. Matched with the dot so prose about it stays allowed.
        assert ".innerHTML" not in script

    def test_the_api_key_is_not_read_from_the_query_string_alone(self):
        from atomics.api.dashboard import _DASHBOARD_HTML

        assert "sessionStorage" in _DASHBOARD_HTML
        assert "history.replaceState" in _DASHBOARD_HTML

    def test_a_scripted_worker_label_is_stored_verbatim_not_rendered(self, tmp_path):
        """The payload round-trips as data; escaping happens at render time."""
        app = create_app(
            ServerSettings(
                no_auth=True, with_dashboard=True, db_path=tmp_path / "xss.db"
            )
        )
        payload = "<script>alert(1)</script>"
        with TestClient(app) as client:
            client.post("/api/v1/workers/register", json={"labels": {"gpu": payload}})
            listed = client.get("/api/v1/workers").json()
            assert listed["workers"][0]["labels"]["gpu"] == payload

            page = client.get("/dashboard")
            assert payload not in page.text


class TestConstantTimeKeyComparison:
    def test_key_matching_accepts_a_configured_key(self):
        from atomics.api.auth import key_matches

        assert key_matches("right", {"right", "other"}) is True

    def test_key_matching_rejects_an_unknown_key(self):
        from atomics.api.auth import key_matches

        assert key_matches("wrong", {"right"}) is False

    def test_key_matching_handles_an_empty_key_set(self):
        from atomics.api.auth import key_matches

        assert key_matches("anything", set()) is False
