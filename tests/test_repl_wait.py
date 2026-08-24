"""Job poll: completed, failed, optional cap, Ctrl-C. Never cancels the job."""

from __future__ import annotations

import io

import httpx

from atomics.mcp.client import AtomicsApiClient
from atomics.repl.dispatch import handle_line
from atomics.repl.session import Session
from atomics.repl.wait import WAIT_INTERVAL_SECONDS, wait_for_job


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
    body = wait_for_job(client, "abc", sleep=sleeps.append, max_polls=30)
    assert body["status"] == "pending"
    assert len(sleeps) == 29


def test_wait_default_polls_past_thirty_until_completed() -> None:
    sleeps: list[float] = []
    bodies = [{"job_id": "abc", "status": "running"}] * 35
    bodies.append({"job_id": "abc", "status": "completed", "result": {"ok": True}})
    client = _client(bodies)
    body = wait_for_job(client, "abc", sleep=sleeps.append)
    assert body["status"] == "completed"
    assert len(sleeps) == 35


def test_wait_stops_on_failed() -> None:
    sleeps: list[float] = []
    client = _client(
        [{"job_id": "abc", "status": "failed", "error": {"message": "budget"}}]
    )
    body = wait_for_job(client, "abc", sleep=sleeps.append)
    assert body["status"] == "failed"
    assert sleeps == []


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


def test_wait_prints_new_trail_entries_on_same_in_flight() -> None:
    session = Session(last_job_id="abc")
    client = _client(
        [
            {
                "job_id": "abc",
                "status": "running",
                "progress": {
                    "current": 0,
                    "total": 1,
                    "in_flight": {
                        "fixture_id": "ev-01",
                        "phase": "generate",
                        "model": "m",
                    },
                    "trail": [
                        {"fixture_id": "ev-01", "phase": "generate", "model": "m"},
                    ],
                },
            },
            {
                "job_id": "abc",
                "status": "running",
                "progress": {
                    "current": 0,
                    "total": 1,
                    "in_flight": {
                        "fixture_id": "ev-01",
                        "phase": "generate",
                        "model": "m",
                    },
                    "trail": [
                        {"fixture_id": "ev-01", "phase": "generate", "model": "m"},
                        {"fixture_id": "ev-01", "phase": "judge", "model": "m"},
                    ],
                },
            },
            {
                "job_id": "abc",
                "status": "completed",
                "progress": {
                    "current": 1,
                    "total": 1,
                    "in_flight": None,
                    "trail": [
                        {"fixture_id": "ev-01", "phase": "generate", "model": "m"},
                        {"fixture_id": "ev-01", "phase": "judge", "model": "m"},
                    ],
                },
                "result": {
                    "overall_accuracy": 1.0,
                    "fixtures_run": 1,
                    "total_tokens": 10,
                    "fixtures": [
                        {"id": "ev-01", "score": 1.0, "status": "success", "tokens": 10}
                    ],
                },
            },
        ]
    )
    written: list[str] = []
    handle_line("wait", session=session, client=client, write=written.append)
    text = "".join(written)
    assert text.count("generate") == 1
    assert "judge" in text


def test_wait_failed_prints_headline() -> None:
    session = Session(last_job_id="abc")
    written: list[str] = []
    handle_line(
        "wait",
        session=session,
        client=_client(
            [
                {
                    "job_id": "abc",
                    "status": "failed",
                    "kind": "eval",
                    "error": {"message": "budget exceeded"},
                }
            ]
        ),
        write=written.append,
    )
    text = "".join(written)
    assert "eval  failed" in text
    assert "budget exceeded" in text
    assert "still running" not in text


def test_wait_prints_stress_in_flight_when_concurrency_changes() -> None:
    session = Session(last_job_id="abc")
    client = _client(
        [
            {
                "job_id": "abc",
                "status": "running",
                "progress": {
                    "current": 0,
                    "total": 2,
                    "in_flight": {"concurrency": 1, "phase_seconds": 5.0},
                },
            },
            {
                "job_id": "abc",
                "status": "running",
                "progress": {
                    "current": 0,
                    "total": 2,
                    "in_flight": {"concurrency": 2, "phase_seconds": 5.0},
                },
            },
            {
                "job_id": "abc",
                "status": "completed",
                "kind": "stress",
                "request": {"model": "m", "host": "h"},
                "progress": {"current": 2, "total": 2, "in_flight": None},
                "result": {
                    "peak_tps": 10.0,
                    "saturation_concurrency": 2,
                    "phases": [
                        {"concurrency": 1, "requests": 3, "aggregate_tps": 8.0},
                        {"concurrency": 2, "requests": 4, "aggregate_tps": 10.0},
                    ],
                },
            },
        ]
    )
    written: list[str] = []
    handle_line("wait", session=session, client=client, write=written.append)
    text = "".join(written)
    assert "c=1  5s" in text
    assert "c=2  5s" in text
    assert "sat=2" in text


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
