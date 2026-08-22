"""Bounded job poll: completed, cap, Ctrl-C. Never cancels the job."""

from __future__ import annotations

import httpx

from atomics.mcp.client import AtomicsApiClient
from atomics.repl.dispatch import handle_line
from atomics.repl.session import Session
from atomics.repl.wait import WAIT_INTERVAL_SECONDS, WAIT_MAX_POLLS, wait_for_job


def _client(bodies: list[dict]) -> AtomicsApiClient:
    remaining = list(bodies)

    def handler(request: httpx.Request) -> httpx.Response:
        body = remaining.pop(0) if remaining else bodies[-1]
        return httpx.Response(200, json=body)

    return AtomicsApiClient("http://api.test", "k", transport=httpx.MockTransport(handler))


def test_wait_stops_on_completed() -> None:
    sleeps: list[float] = []
    client = _client(
        [
            {"job_id": "abc", "status": "pending"},
            {"job_id": "abc", "status": "completed", "result": {"ok": True}},
        ]
    )
    body = wait_for_job(client, "abc", sleep=sleeps.append)
    assert body["status"] == "completed"
    assert sleeps == [WAIT_INTERVAL_SECONDS]


def test_wait_stops_at_max_polls() -> None:
    sleeps: list[float] = []
    client = _client([{"job_id": "abc", "status": "pending"}])
    body = wait_for_job(client, "abc", sleep=sleeps.append)
    assert body["status"] == "pending"
    assert len(sleeps) == WAIT_MAX_POLLS - 1


def test_wait_ctrl_c_stops_the_poll() -> None:
    def sleep(_seconds: float) -> None:
        raise KeyboardInterrupt

    client = _client([{"job_id": "abc", "status": "pending"}])
    body = wait_for_job(client, "abc", sleep=sleep)
    assert body["status"] == "pending"


def test_wait_uses_last_job_id() -> None:
    session = Session(last_job_id="abc")
    client = _client([{"job_id": "abc", "status": "completed"}])
    result = handle_line("wait", session=session, client=client)
    assert "completed" in result.stdout


def test_wait_without_id_errors() -> None:
    result = handle_line("wait", session=Session(), client=_client([]))
    assert "submit" in result.stderr.lower() or "job" in result.stderr.lower()
    assert result.exit_loop is False
