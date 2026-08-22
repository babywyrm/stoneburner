"""Local REPL verbs do not touch the API client."""

from __future__ import annotations

import json

import httpx

from atomics.mcp.client import AtomicsApiClient
from atomics.repl.dispatch import handle_line
from atomics.repl.session import Session


class _Boom:
    def __getattr__(self, name: str):
        raise AssertionError(f"local verb must not call client.{name}")


def test_show_prints_session_json() -> None:
    session = Session(provider="ollama", model="gpt-oss:20b")
    result = handle_line("show", session=session, client=_Boom())  # type: ignore[arg-type]
    assert json.loads(result.stdout) == session.as_dict()
    assert result.exit_loop is False


def test_set_key_value() -> None:
    session = Session()
    handle_line("set effort high", session=session, client=_Boom())  # type: ignore[arg-type]
    assert session.effort == "high"


def test_set_key_clears() -> None:
    session = Session(model="x")
    handle_line("set model", session=session, client=_Boom())  # type: ignore[arg-type]
    assert session.model is None


def test_unknown_verb_stays_in_the_prompt() -> None:
    result = handle_line("frobnicate", session=Session(), client=_Boom())  # type: ignore[arg-type]
    assert result.exit_loop is False
    assert "help" in result.stderr.lower()
    assert "frobnicate" in result.stderr


def test_exit_asks_the_loop_to_stop() -> None:
    result = handle_line("exit", session=Session(), client=_Boom())  # type: ignore[arg-type]
    assert result.exit_loop is True
    assert result.exit_code == 0


def test_help_lists_submit_eval() -> None:
    result = handle_line("help", session=Session(), client=_Boom())  # type: ignore[arg-type]
    assert "submit_eval" in result.stdout
    assert "wait" in result.stdout


def _client(requests: list, *, response=None) -> AtomicsApiClient:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response or httpx.Response(200, json={"job_id": "abc", "status": "pending"})

    return AtomicsApiClient("http://api.test", "k", transport=httpx.MockTransport(handler))


def test_submit_eval_fills_provider_from_session() -> None:
    requests: list[httpx.Request] = []
    session = Session(provider="ollama", model="gpt-oss:20b")
    result = handle_line(
        "submit_eval --suite toolcall",
        session=session,
        client=_client(requests),
    )
    payload = json.loads(requests[0].content)
    assert payload["provider"] == "ollama"
    assert payload["model"] == "gpt-oss:20b"
    assert payload["suite"] == "toolcall"
    assert session.last_job_id == "abc"
    assert '"job_id": "abc"' in result.stdout


def test_session_host_fills_submit_eval() -> None:
    requests: list[httpx.Request] = []
    session = Session(provider="ollama", model="llama3.2:1b", host="http://192.168.1.79:11434")
    handle_line(
        "submit_eval --suite accuracy",
        session=session,
        client=_client(requests),
    )
    payload = json.loads(requests[0].content)
    assert payload["host"] == "http://192.168.1.79:11434"


def test_explicit_flag_wins_over_session() -> None:
    requests: list[httpx.Request] = []
    session = Session(provider="ollama")
    handle_line(
        "submit_eval --suite toolcall --provider claude",
        session=session,
        client=_client(requests),
    )
    assert json.loads(requests[0].content)["provider"] == "claude"


def test_submit_sweep_models_from_session_model() -> None:
    requests: list[httpx.Request] = []
    session = Session(provider="ollama", model="a")
    handle_line(
        "submit_sweep --suites eval --budget_usd 2",
        session=session,
        client=_client(requests),
    )
    assert json.loads(requests[0].content)["models"] == ["a"]


def test_submit_sweep_explicit_models_win() -> None:
    requests: list[httpx.Request] = []
    session = Session(provider="ollama", model="a")
    handle_line(
        "submit_sweep --models b,c --suites eval --budget_usd 2",
        session=session,
        client=_client(requests),
    )
    assert json.loads(requests[0].content)["models"] == ["b", "c"]


def test_get_job_takes_one_positional() -> None:
    requests: list[httpx.Request] = []
    handle_line("get_job abc", session=Session(), client=_client(requests))
    assert requests[0].url.path == "/api/v1/jobs/abc"


def test_api_error_stays_in_the_prompt() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "budget_usd is required"})

    client = AtomicsApiClient("http://api.test", "k", transport=httpx.MockTransport(handler))
    result = handle_line(
        "submit_sweep --models a --suites eval --provider ollama --budget_usd 1",
        session=Session(),
        client=client,
    )
    assert result.exit_loop is False
    assert "budget_usd is required" in result.stderr


def test_unknown_flag_does_not_call_the_client() -> None:
    requests: list[httpx.Request] = []
    result = handle_line(
        "health --nope 1",
        session=Session(),
        client=_client(requests),
    )
    assert requests == []
    assert "nope" in result.stderr
