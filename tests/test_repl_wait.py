"""Bounded job poll: completed, cap, Ctrl-C. Never cancels the job."""

from __future__ import annotations

import io

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
    client = _client(
        [
            {
                "job_id": "abc",
                "status": "completed",
                "request": {"suite": "accuracy", "model": "m"},
                "result": {
                    "overall_accuracy": 1.0,
                    "fixtures_run": 1,
                    "total_tokens": 10,
                },
                "progress": {"current": 1, "total": 1},
            }
        ]
    )
    buf = io.StringIO()
    handle_line("wait", session=session, client=client, write=buf.write)
    text = buf.getvalue()
    assert "accuracy" in text
    assert "1.000" in text
    assert '"status"' not in text


def test_wait_prints_when_progress_changes() -> None:
    session = Session(last_job_id="abc")
    client = _client(
        [
            {
                "job_id": "abc",
                "status": "running",
                "progress": {
                    "current": 0,
                    "total": 2,
                    "in_flight": {"fixture_id": "ev-01", "phase": "generate", "model": "m"},
                },
            },
            {
                "job_id": "abc",
                "status": "running",
                "progress": {
                    "current": 0,
                    "total": 2,
                    "in_flight": {"fixture_id": "ev-01", "phase": "generate", "model": "m"},
                },
            },
            {
                "job_id": "abc",
                "status": "completed",
                "progress": {"current": 1, "total": 2, "in_flight": None},
                "result": {"fixtures_run": 1},
            },
        ]
    )
    written: list[str] = []
    handle_line(
        "wait",
        session=session,
        client=client,
        write=written.append,
    )
    text = "".join(written)
    assert '"status"' not in text
    assert "generate" in text
    assert "ev-01" in text
    assert text.count("generate") == 1


def test_wait_verbose_prints_replies() -> None:
    session = Session(last_job_id="abc")
    body = {
        "job_id": "abc",
        "status": "completed",
        "request": {"suite": "accuracy", "model": "m"},
        "progress": {"current": 1, "total": 1},
        "result": {
            "overall_accuracy": 0.6,
            "fixtures_run": 1,
            "total_tokens": 170,
            "fixtures": [
                {
                    "id": "ev-01",
                    "score": 0.6,
                    "status": "success",
                    "tokens": 170,
                    "latency_ms": 800,
                    "response": "Paris is the capital.",
                }
            ],
        },
    }
    quiet: list[str] = []
    handle_line("wait", session=session, client=_client([body]), write=quiet.append)
    assert "Paris" not in "".join(quiet)
    verbose: list[str] = []
    handle_line(
        "wait --verbose",
        session=session,
        client=_client([body]),
        write=verbose.append,
    )
    text = "".join(verbose)
    assert "Paris is the capital." in text
    assert "800ms" in text


def test_wait_without_id_errors() -> None:
    result = handle_line("wait", session=Session(), client=_client([]))
    assert "submit" in result.stderr.lower() or "job" in result.stderr.lower()
    assert result.exit_loop is False
