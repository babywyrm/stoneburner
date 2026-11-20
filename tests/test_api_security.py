"""Regression tests for the v0.15.2 security fixes.

Each test corresponds to a finding from the 2026-08-02 audit. They assert the
boundary, not the implementation, so a future refactor that reopens the hole
still fails here.
"""

from __future__ import annotations

import re

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

    @pytest.mark.parametrize("host", ["0.0.0.0", "203.0.113.10", "::", "example.com"])
    def test_public_hosts_reject_no_auth(self, host):
        assert is_loopback_host(host) is False
        with pytest.raises(ValueError, match="no_auth cannot be combined"):
            ServerSettings(host=host, no_auth=True)

    def test_public_hosts_are_fine_with_a_key(self):
        settings = ServerSettings(host="0.0.0.0", api_keys={"k"})
        assert settings.host == "0.0.0.0"


class TestNoAuthRejectsForeignHostHeader:
    """A page that DNS-rebinds its own name to 127.0.0.1 is same-origin with a
    --no-auth server, and could submit evals and read the results back."""

    @staticmethod
    def _status(settings: ServerSettings, host: str, **headers: str) -> int:
        with TestClient(create_app(settings)) as client:
            return client.get("/api/v1/jobs", headers={"host": host, **headers}).status_code

    @pytest.mark.parametrize("host", ["127.0.0.1:8000", "localhost", "[::1]:8000"])
    def test_loopback_host_header_is_served(self, host, tmp_path):
        settings = ServerSettings(no_auth=True, db_path=tmp_path / "a.db")
        assert self._status(settings, host) == 200

    def test_rebound_host_header_is_refused(self, tmp_path):
        settings = ServerSettings(no_auth=True, db_path=tmp_path / "a.db")
        assert self._status(settings, "attacker.example:8000") == 421

    def test_keyed_server_does_not_check_host(self, tmp_path):
        settings = ServerSettings(api_keys={"k"}, db_path=tmp_path / "a.db")
        assert self._status(settings, "atomics.lab", **{"X-API-Key": "k"}) == 200


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
        with TestClient(app, base_url="http://127.0.0.1") as client:
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
        with TestClient(app, base_url="http://127.0.0.1") as client:
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
        coordinator.create_split_job(DistributedRunRequest(mode=JobMode.SPLIT), [{"i": 1}])
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
        coordinator.create_split_job(DistributedRunRequest(mode=JobMode.SPLIT), [{"i": 1}])
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
        result = coordinator.submit_assignment("does-not-exist", "{}", worker_id=worker.worker_id)
        assert result is None

    def test_the_route_rejects_a_mismatched_worker_with_409(self, tmp_path):
        app = create_app(ServerSettings(no_auth=True, db_path=tmp_path / "route.db"))
        with TestClient(app, base_url="http://127.0.0.1") as client:
            owner = client.post("/api/v1/workers/register", json={}).json()
            attacker = client.post("/api/v1/workers/register", json={}).json()
            client.post(
                "/api/v1/distributed/runs",
                json={"mode": "split", "run_request": {"iterations": 1}},
            )
            claimed = client.get(f"/api/v1/workers/{owner['worker_id']}/jobs/next").json()
            assert claimed is not None

            forged = client.post(
                f"/api/v1/workers/{attacker['worker_id']}/jobs/{claimed['assignment_id']}/result",
                json={"status": "completed", "result_json": '{"forged": true}'},
            )
            assert forged.status_code == 409

    def test_the_rightful_worker_still_succeeds_over_http(self, tmp_path):
        app = create_app(ServerSettings(no_auth=True, db_path=tmp_path / "route.db"))
        with TestClient(app, base_url="http://127.0.0.1") as client:
            owner = client.post("/api/v1/workers/register", json={}).json()
            client.post(
                "/api/v1/distributed/runs",
                json={"mode": "split", "run_request": {"iterations": 1}},
            )
            claimed = client.get(f"/api/v1/workers/{owner['worker_id']}/jobs/next").json()

            accepted = client.post(
                f"/api/v1/workers/{owner['worker_id']}/jobs/{claimed['assignment_id']}/result",
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
            ServerSettings(no_auth=True, with_dashboard=True, db_path=tmp_path / "xss.db")
        )
        payload = "<script>alert(1)</script>"
        with TestClient(app, base_url="http://127.0.0.1") as client:
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


class TestEveryRouteRequiresAuth:
    """A new route that forgets require_auth is open to anyone who can reach
    the port. Walks the live OpenAPI spec so the check grows with the API."""

    OPEN = frozenset({"/api/v1/health", "/api/v1/ready"})

    @pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong-key-0123456789"}])
    def test_every_api_route_rejects_a_caller_without_a_valid_key(self, tmp_path, headers):
        settings = ServerSettings(api_keys={"k" * 32}, db_path=tmp_path / "a.db")
        app = create_app(settings)
        with TestClient(app, base_url="http://127.0.0.1") as client:
            paths = app.openapi()["paths"]
            served = {
                (method.upper(), path): client.request(
                    method.upper(), re.sub(r"\{[^}]+\}", "x", path), json={}, headers=headers
                ).status_code
                for path, ops in paths.items()
                for method in ops
                if path not in self.OPEN
            }
        assert len(served) > 20
        assert {route: code for route, code in served.items() if code != 401} == {}


class TestApiDocsStayLocal:
    """Swagger, ReDoc, and the schema map every route and parameter for an
    anonymous caller; they are served only by a loopback --no-auth server."""

    DOCS = ("/docs", "/redoc", "/openapi.json")

    def test_a_keyed_server_does_not_publish_its_api_map(self, tmp_path):
        settings = ServerSettings(api_keys={"k" * 32}, db_path=tmp_path / "a.db")
        with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
            assert {path: client.get(path).status_code for path in self.DOCS} == dict.fromkeys(
                self.DOCS, 404
            )

    def test_a_local_no_auth_server_still_serves_docs(self, tmp_path):
        settings = ServerSettings(no_auth=True, db_path=tmp_path / "a.db")
        with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
            assert {path: client.get(path).status_code for path in self.DOCS} == dict.fromkeys(
                self.DOCS, 200
            )


class TestOutboundAllowedHosts:
    """When --allowed-hosts is set, callers cannot steer inference or judge traffic elsewhere."""

    @pytest.mark.parametrize(
        "bad",
        [
            set(),
            {"ftp://127.0.0.1:11434"},
            {"http://user:pass@127.0.0.1:11434"},
            {"http://127.0.0.1:11434?leak=1"},
            {"http://127.0.0.1:11434#frag"},
            {"http://127.0.0.1:99999"},
        ],
    )
    def test_empty_or_malformed_allowed_hosts_rejected_at_startup(self, bad):
        with pytest.raises(ValueError):
            ServerSettings(api_keys={"k"}, allowed_hosts=bad)

    def test_origin_and_path_matching_blocks_port_scheme_and_prefix_bypasses(self):
        from atomics.api.config import DisallowedHostError

        settings = ServerSettings(
            api_keys={"k"},
            allowed_hosts={
                "http://10.0.0.5:11434",
                "https://vllm.example.com/v1",
                "http://[::1]:11434",
            },
        )
        for valid in (
            None,
            "http://10.0.0.5:11434",
            "http://10.0.0.5:11434/",
            "http://[::ffff:10.0.0.5]:11434",
            "https://VLLM.EXAMPLE.COM:443/v1",
            "https://vllm.example.com/v1/chat",
            "http://[0:0:0:0:0:0:0:1]:11434",
        ):
            settings.check_outbound_host(valid, label="host")

        for blocked in (
            "http://10.0.0.5:8000",
            "https://10.0.0.5:11434",
            "http://10.0.0.5.evil.example:11434",
            "https://vllm.example.com/v10",
            "https://vllm.example.com/admin",
            "https://vllm.example.com",
            "http://169.254.169.254/latest/meta-data",
        ):
            with pytest.raises(DisallowedHostError) as excinfo:
                settings.check_outbound_host(blocked, label="host")
            # Never enumerate the server's internal allowlist in error details.
            assert "10.0.0.5" not in str(excinfo.value)
            assert "vllm.example.com" not in str(excinfo.value)

    def test_every_outbound_route_enforces_allowed_hosts(
        self, tmp_path, jobs_never_run, monkeypatch
    ):
        async def fake_list_models(provider: str, host: str | None) -> dict[str, object]:
            return {"provider": provider, "models": []}

        async def fake_pt(payload: object) -> dict[str, object]:
            return {"ok": True, "health": True, "provider": "ollama", "response": "4"}

        monkeypatch.setattr("atomics.api.routes.list_models", fake_list_models)
        monkeypatch.setattr("atomics.api.routes.run_provider_test", fake_pt)

        allowed_url = "http://127.0.0.1:11434"
        forbidden_url = "http://10.99.0.9:11434"
        settings = ServerSettings(
            api_keys={"secret-key"},
            db_path=tmp_path / "allow.db",
            allowed_hosts={allowed_url},
        )
        headers = {"X-API-Key": "secret-key"}

        with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
            # 1. Runs
            assert (
                client.post(
                    "/api/v1/runs",
                    json={"provider": "ollama", "host": forbidden_url},
                    headers=headers,
                ).status_code
                == 403
            )
            assert (
                client.post(
                    "/api/v1/runs",
                    json={"provider": "ollama", "host": allowed_url},
                    headers=headers,
                ).status_code
                == 202
            )

            # 2. Evals (both host and judge_host)
            r_eval = client.post(
                "/api/v1/evals",
                json={"suite": "accuracy", "provider": "ollama", "judge_host": forbidden_url},
                headers=headers,
            )
            assert r_eval.status_code == 403
            assert allowed_url not in r_eval.text

            # 3. Sweeps (host and judge_host)
            assert (
                client.post(
                    "/api/v1/sweeps",
                    json={
                        "provider": "ollama",
                        "models": ["m"],
                        "suites": ["eval"],
                        "budget_usd": 1.0,
                        "judge_host": forbidden_url,
                    },
                    headers=headers,
                ).status_code
                == 403
            )

            # 4. Batteries (host, judge_host, and profile target)
            assert (
                client.post(
                    "/api/v1/batteries",
                    json={
                        "name": "desk-pass",
                        "provider": "ollama",
                        "budget_usd": 1.0,
                        "host": forbidden_url,
                    },
                    headers=headers,
                ).status_code
                == 403
            )
            assert (
                client.post(
                    "/api/v1/batteries",
                    json={
                        "name": "desk-pass",
                        "provider": "ollama",
                        "budget_usd": 1.0,
                        "profile": "profiles/examples/http-flask-endpoint.yaml",
                    },
                    headers=headers,
                ).status_code
                == 403
            )

            # 5. Stress & 6. Soak
            assert (
                client.post(
                    "/api/v1/stress",
                    json={
                        "provider": "ollama",
                        "model": "m",
                        "budget_usd": 1.0,
                        "host": forbidden_url,
                    },
                    headers=headers,
                ).status_code
                == 403
            )
            assert (
                client.post(
                    "/api/v1/soak",
                    json={
                        "provider": "ollama",
                        "model": "m",
                        "budget_usd": 1.0,
                        "host": forbidden_url,
                    },
                    headers=headers,
                ).status_code
                == 403
            )

            # 7. Models & 8. Provider-test
            assert (
                client.get(
                    "/api/v1/models",
                    params={"provider": "ollama", "host": forbidden_url},
                    headers=headers,
                ).status_code
                == 403
            )
            assert (
                client.get(
                    "/api/v1/models",
                    params={"provider": "ollama", "host": allowed_url},
                    headers=headers,
                ).status_code
                == 200
            )
            assert (
                client.post(
                    "/api/v1/provider-test",
                    json={"provider": "ollama", "host": forbidden_url},
                    headers=headers,
                ).status_code
                == 403
            )
            assert (
                client.post(
                    "/api/v1/provider-test",
                    json={"provider": "ollama", "host": allowed_url},
                    headers=headers,
                ).status_code
                == 200
            )

            # 9. Distributed full run host overrides
            assert (
                client.post(
                    "/api/v1/distributed/runs",
                    json={
                        "mode": "full",
                        "run_request": {
                            "provider": "ollama",
                            "iterations": 1,
                            "ollama_host": forbidden_url,
                        },
                    },
                    headers=headers,
                ).status_code
                == 403
            )
            assert (
                client.post(
                    "/api/v1/distributed/runs",
                    json={
                        "mode": "full",
                        "run_request": {
                            "provider": "ollama",
                            "iterations": 1,
                            "ollama_host": "ftp://bad",
                        },
                    },
                    headers=headers,
                ).status_code
                == 400
            )
