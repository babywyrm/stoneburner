"""Correlation IDs and access logging.

The load-bearing case is a job: runs and evals are async, so the submitting
request has returned long before the work finishes. Without a shared
identifier, a failure hours later has nothing tying it back to who asked for
it, which is the entire reason these exist.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from fastapi.testclient import TestClient

from atomics.api.config import ServerSettings
from atomics.api.jobs import JobManager
from atomics.api.request_log import (
    REQUEST_ID_HEADER,
    current_request_id,
    new_request_id,
    sanitize_request_id,
)
from atomics.api.server import create_app

ALICE = "alice-key-0123456789"
ACCESS_LOG = "atomics.api.request_log"


def access_lines(caplog) -> list[str]:
    """Only the access log's own records.

    `caplog.at_level` raises the level for one logger but still captures every
    other logger's output, so indexing records[0] picks up whatever unrelated
    warning happened to fire first.
    """
    return [r.getMessage() for r in caplog.records if r.name == ACCESS_LOG]


@pytest.fixture
def client(tmp_path):
    app = create_app(ServerSettings(api_keys={ALICE}, db_path=tmp_path / "log.db"))
    with TestClient(app) as test_client:
        yield test_client


class TestRequestIdSanitizing:
    """An inbound ID is attacker-controlled and goes straight into a log file."""

    def test_a_reasonable_id_is_accepted(self):
        assert sanitize_request_id("abc-123_XYZ.4") == "abc-123_XYZ.4"

    @pytest.mark.parametrize(
        "hostile",
        [
            "has space",
            "newline\ninjected",
            "carriage\rreturn",
            "semi;colon",
            'quote"mark',
            "null\x00byte",
            "unicode\u2028separator",
        ],
    )
    def test_anything_that_could_forge_a_log_line_is_rejected(self, hostile):
        assert sanitize_request_id(hostile) is None

    def test_an_overlong_id_is_rejected(self):
        assert sanitize_request_id("a" * 65) is None
        assert sanitize_request_id("a" * 64) == "a" * 64

    def test_empty_and_missing_are_rejected(self):
        assert sanitize_request_id("") is None
        assert sanitize_request_id(None) is None

    def test_generated_ids_are_unique_and_safe(self):
        ids = {new_request_id() for _ in range(100)}
        assert len(ids) == 100
        assert all(sanitize_request_id(i) == i for i in ids)


class TestResponseHeader:
    def test_every_response_carries_a_correlation_id(self, client):
        res = client.get("/api/v1/health")
        assert res.headers[REQUEST_ID_HEADER]

    def test_a_caller_supplied_id_is_echoed_back(self, client):
        res = client.get("/api/v1/health", headers={REQUEST_ID_HEADER: "trace-abc-123"})
        assert res.headers[REQUEST_ID_HEADER] == "trace-abc-123"

    def test_a_hostile_id_is_replaced_not_echoed(self, client):
        res = client.get("/api/v1/health", headers={REQUEST_ID_HEADER: "bad id with spaces"})
        assert res.headers[REQUEST_ID_HEADER] != "bad id with spaces"
        assert sanitize_request_id(res.headers[REQUEST_ID_HEADER])

    def test_error_responses_are_tagged_too(self, client):
        """A failing request is the one you most want to correlate."""
        res = client.post("/api/v1/evals", json={"suite": "accuracy", "provider": "x"})
        assert res.status_code == 401
        assert res.headers[REQUEST_ID_HEADER]

    def test_distinct_requests_get_distinct_ids(self, client):
        first = client.get("/api/v1/health").headers[REQUEST_ID_HEADER]
        second = client.get("/api/v1/health").headers[REQUEST_ID_HEADER]
        assert first != second


class TestAccessLog:
    def test_one_line_is_logged_per_request(self, client, caplog):
        with caplog.at_level(logging.INFO, logger="atomics.api.request_log"):
            client.get("/api/v1/health")

        lines = access_lines(caplog)
        assert len(lines) == 1
        assert "method=GET" in lines[0]
        assert "path=/api/v1/health" in lines[0]
        assert "status=200" in lines[0]
        assert "duration_ms=" in lines[0]

    def test_the_log_line_carries_the_correlation_id(self, client, caplog):
        with caplog.at_level(logging.INFO, logger="atomics.api.request_log"):
            res = client.get("/api/v1/health", headers={REQUEST_ID_HEADER: "corr-42"})

        assert res.headers[REQUEST_ID_HEADER] == "corr-42"
        assert "request_id=corr-42" in access_lines(caplog)[0]

    def test_the_log_line_identifies_an_authenticated_caller(self, client, caplog):
        from atomics.api.callers import caller_id_from_key

        with caplog.at_level(logging.INFO, logger="atomics.api.request_log"):
            client.post(
                "/api/v1/evals",
                json={"suite": "accuracy", "provider": "ollama"},
                headers={"X-API-Key": ALICE},
            )

        assert f"caller={caller_id_from_key(ALICE)}" in access_lines(caplog)[0]

    def test_the_api_key_never_appears_in_the_log(self, client, caplog):
        with caplog.at_level(logging.INFO):
            client.post(
                "/api/v1/evals",
                json={"suite": "accuracy", "provider": "ollama"},
                headers={"X-API-Key": ALICE},
            )

        assert ALICE not in "\n".join(r.getMessage() for r in caplog.records)

    def test_the_query_string_is_not_logged(self, client, caplog):
        """Query strings have carried API keys before; an access log is the
        wrong place to discover that."""
        with caplog.at_level(logging.INFO, logger="atomics.api.request_log"):
            client.get("/api/v1/health?api_key=super-secret&debug=1")

        logged = access_lines(caplog)[0]
        assert "super-secret" not in logged
        assert "path=/api/v1/health" in logged
        assert "?" not in logged

    def test_a_failed_request_is_still_logged(self, client, caplog):
        with caplog.at_level(logging.INFO, logger="atomics.api.request_log"):
            client.post("/api/v1/evals", json={"suite": "accuracy", "provider": "x"})

        assert "status=401" in access_lines(caplog)[0]


class TestJobCorrelation:
    """The reason correlation IDs exist here: crossing the async boundary."""

    @pytest.mark.asyncio
    async def test_a_job_inherits_the_submitting_request_id(self, monkeypatch):
        from atomics.api import request_log

        token = request_log._request_id.set("req-inherited")
        try:
            manager = JobManager()

            async def work(_job_id: str) -> str:
                return "done"

            job_id = await manager.submit("run", work, owner="alice")
            await manager.wait_for(job_id)
            assert manager.jobs[job_id].request_id == "req-inherited"
        finally:
            request_log._request_id.reset(token)

    @pytest.mark.asyncio
    async def test_the_id_survives_into_the_running_task(self):
        """create_task copies the context, so the job body sees it too."""
        from atomics.api import request_log

        seen: list[str] = []
        token = request_log._request_id.set("req-in-task")
        try:
            manager = JobManager()

            async def work(_job_id: str) -> str:
                await asyncio.sleep(0)
                seen.append(current_request_id())
                return "done"

            job_id = await manager.submit("run", work)
            await manager.wait_for(job_id)
        finally:
            request_log._request_id.reset(token)

        assert seen == ["req-in-task"]

    @pytest.mark.asyncio
    async def test_submission_and_completion_are_both_logged(self, caplog):
        async def work(_job_id: str) -> str:
            return "done"

        with caplog.at_level(logging.INFO, logger="atomics.api.jobs"):
            manager = JobManager()
            job_id = await manager.submit("eval", work, owner="alice")
            await manager.wait_for(job_id)

        messages = "\n".join(r.getMessage() for r in caplog.records)
        assert f"job_submitted job_id={job_id}" in messages
        assert f"job_finished job_id={job_id}" in messages
        assert "status=completed" in messages
        assert "caller=alice" in messages

    @pytest.mark.asyncio
    async def test_a_failed_job_logs_its_status(self, caplog):
        async def boom(_job_id: str) -> str:
            raise RuntimeError("nope")

        with caplog.at_level(logging.INFO, logger="atomics.api.jobs"):
            manager = JobManager()
            job_id = await manager.submit("run", boom)
            await manager.wait_for(job_id)

        assert "status=failed" in "\n".join(r.getMessage() for r in caplog.records)

    def test_a_job_started_over_http_records_the_request_id(self, client):
        res = client.post(
            "/api/v1/evals",
            json={"suite": "accuracy", "provider": "ollama"},
            headers={"X-API-Key": ALICE, REQUEST_ID_HEADER: "http-corr-1"},
        )
        assert res.status_code == 202

        job = client.app.state.job_manager.jobs[res.json()["job_id"]]
        assert job.request_id == "http-corr-1"


def test_the_context_is_empty_outside_a_request():
    assert current_request_id() == ""
