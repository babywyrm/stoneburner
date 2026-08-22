"""REPL loop: health on entry, stay on API errors, exit on EOF."""

from __future__ import annotations

import io

import httpx

from atomics.mcp.client import AtomicsApiClient
from atomics.repl.loop import run_repl


def _client(*, health=None, handler=None) -> AtomicsApiClient:
    def default(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return health or httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"ok": True})

    return AtomicsApiClient(
        "http://api.test",
        "k",
        transport=httpx.MockTransport(handler or default),
    )


def test_startup_health_failure_exits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope", request=request)

    stderr = io.StringIO()
    code = run_repl(
        _client(handler=handler),
        input_fn=lambda _prompt: (_ for _ in ()).throw(AssertionError("must not prompt")),
        stdout=io.StringIO(),
        stderr=stderr,
    )
    assert code == 1
    assert "atomics server" in stderr.getvalue()


def test_eof_exits_zero_after_health() -> None:
    stdout = io.StringIO()

    def input_fn(_prompt: str) -> str:
        raise EOFError

    code = run_repl(_client(), input_fn=input_fn, stdout=stdout, stderr=io.StringIO())
    assert code == 0


def test_mid_session_error_stays() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(401, json={"detail": "invalid API key"})

    lines = iter(["list_jobs", "exit"])

    def input_fn(_prompt: str) -> str:
        return next(lines)

    stderr = io.StringIO()
    code = run_repl(
        _client(handler=handler),
        input_fn=input_fn,
        stdout=io.StringIO(),
        stderr=stderr,
    )
    assert code == 0
    assert "invalid API key" in stderr.getvalue()


def test_ctrl_c_at_prompt_reprints() -> None:
    state = {"n": 0}

    def input_fn(_prompt: str) -> str:
        state["n"] += 1
        if state["n"] == 1:
            raise KeyboardInterrupt
        raise EOFError

    code = run_repl(_client(), input_fn=input_fn, stdout=io.StringIO(), stderr=io.StringIO())
    assert code == 0
    assert state["n"] == 2
