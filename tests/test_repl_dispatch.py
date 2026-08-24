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


_EVAL_JOB = {
    "job_id": "abc",
    "status": "pending",
    "kind": "eval",
    "request": {"suite": "toolcall", "provider": "ollama", "model": "gpt-oss:20b"},
}


def test_submit_eval_fills_provider_from_session() -> None:
    requests: list[httpx.Request] = []
    session = Session(provider="ollama", model="gpt-oss:20b")
    result = handle_line(
        "submit_eval --suite toolcall",
        session=session,
        client=_client(requests, response=httpx.Response(200, json=_EVAL_JOB)),
    )
    payload = json.loads(requests[0].content)
    assert payload["provider"] == "ollama"
    assert payload["model"] == "gpt-oss:20b"
    assert payload["suite"] == "toolcall"
    assert session.last_job_id == "abc"
    assert "abc" in result.stdout
    assert "toolcall" in result.stdout
    assert '"job_id"' not in result.stdout


def test_submit_eval_verbose_keeps_json() -> None:
    session = Session(provider="ollama", model="gpt-oss:20b")
    result = handle_line(
        "submit_eval --suite toolcall --verbose",
        session=session,
        client=_client([], response=httpx.Response(200, json=_EVAL_JOB)),
    )
    assert '"job_id": "abc"' in result.stdout
    assert session.last_job_id == "abc"


def test_session_host_fills_submit_stress() -> None:
    requests: list[httpx.Request] = []
    session = Session(provider="ollama", model="llama3.2:1b", host="http://192.168.1.79:11434")
    handle_line(
        "submit_stress --budget_usd 1",
        session=session,
        client=_client(requests),
    )
    payload = json.loads(requests[0].content)
    assert payload["host"] == "http://192.168.1.79:11434"


def test_session_host_fills_submit_sweep() -> None:
    requests: list[httpx.Request] = []
    session = Session(provider="ollama", model="llama3.2:1b", host="http://192.168.1.79:11434")
    handle_line(
        "submit_sweep --suites eval --budget_usd 1",
        session=session,
        client=_client(requests),
    )
    payload = json.loads(requests[0].content)
    assert payload["host"] == "http://192.168.1.79:11434"


def test_session_host_fills_submit_run() -> None:
    requests: list[httpx.Request] = []
    session = Session(provider="ollama", model="llama3.2:1b", host="http://192.168.1.79:11434")
    handle_line(
        "submit_run",
        session=session,
        client=_client(requests),
    )
    payload = json.loads(requests[0].content)
    assert payload["host"] == "http://192.168.1.79:11434"


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


def test_explicit_host_wins_on_sweep_and_run() -> None:
    requests: list[httpx.Request] = []
    session = Session(
        provider="ollama",
        model="llama3.2:1b",
        host="http://127.0.0.1:11434",
    )
    handle_line(
        "submit_sweep --suites eval --budget_usd 1 --host http://127.0.0.1:11435",
        session=session,
        client=_client(requests),
    )
    assert json.loads(requests[0].content)["host"] == "http://127.0.0.1:11435"
    handle_line(
        "submit_run --host http://127.0.0.1:11435",
        session=session,
        client=_client(requests),
    )
    assert json.loads(requests[1].content)["host"] == "http://127.0.0.1:11435"


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


def test_list_jobs_quiet() -> None:
    body = {
        "jobs": [
            {
                "job_id": "abcdef12ffff",
                "status": "running",
                "kind": "eval",
                "request": {"suite": "accuracy", "model": "m", "host": "h"},
            }
        ]
    }
    result = handle_line(
        "list_jobs",
        session=Session(),
        client=_client([], response=httpx.Response(200, json=body)),
    )
    assert "eval  running  abcdef12" in result.stdout
    assert "accuracy" in result.stdout
    assert '"jobs"' not in result.stdout


def test_list_jobs_verbose_keeps_json() -> None:
    body = {"jobs": [{"job_id": "abc", "status": "pending", "kind": "eval"}]}
    result = handle_line(
        "list_jobs --verbose",
        session=Session(),
        client=_client([], response=httpx.Response(200, json=body)),
    )
    assert '"jobs"' in result.stdout


def test_list_models_verbose_keeps_json() -> None:
    body = {"provider": "ollama", "models": [{"name": "llama3.2:1b"}]}
    result = handle_line(
        "list_models --verbose",
        session=Session(),
        client=_client([], response=httpx.Response(200, json=body)),
    )
    assert '"models"' in result.stdout


def test_list_models_quiet() -> None:
    body = {
        "provider": "ollama",
        "models": [{"name": "llama3.2:1b"}, {"name": "qwen3:14b"}],
    }
    result = handle_line(
        "list_models",
        session=Session(),
        client=_client([], response=httpx.Response(200, json=body)),
    )
    assert "ollama  2" in result.stdout
    assert "llama3.2:1b" in result.stdout
    assert "qwen3:14b" in result.stdout
    assert '"models"' not in result.stdout


def test_provider_test_verbose_keeps_json() -> None:
    body = {"ok": True, "model": "llama3.2:1b", "latency_ms": 191.2, "response": "4"}
    result = handle_line(
        "provider_test --provider ollama --verbose",
        session=Session(),
        client=_client([], response=httpx.Response(200, json=body)),
    )
    assert '"ok"' in result.stdout


def test_provider_test_quiet_ok() -> None:
    body = {
        "ok": True,
        "model": "llama3.2:1b",
        "latency_ms": 191.2,
        "response": "4",
        "error": None,
    }
    result = handle_line(
        "provider_test --provider ollama",
        session=Session(),
        client=_client([], response=httpx.Response(200, json=body)),
    )
    assert "ok  llama3.2:1b  191ms  4" in result.stdout
    assert '"ok"' not in result.stdout


def test_provider_test_quiet_fail() -> None:
    body = {
        "ok": False,
        "model": "llama3.2:1b",
        "latency_ms": 0.0,
        "response": None,
        "error": "Provider health check failed.",
    }
    result = handle_line(
        "provider_test --provider ollama",
        session=Session(),
        client=_client([], response=httpx.Response(200, json=body)),
    )
    assert "fail  llama3.2:1b" in result.stdout
    assert "Provider health check failed." in result.stdout


def test_recent_runs_quiet() -> None:
    body = {
        "runs": [
            {
                "run_id": "rpt-001abc",
                "provider": "ollama",
                "model": "qwen",
                "tier": "ez",
                "total_tasks": 3,
                "successful_tasks": 3,
                "failed_tasks": 0,
                "total_tokens": 450,
                "total_cost_usd": 0.06,
            }
        ]
    }
    result = handle_line(
        "recent_runs",
        session=Session(),
        client=_client([], response=httpx.Response(200, json=body)),
    )
    assert "rpt-001abc  ollama  qwen  ez  3 ok  0 fail  450 tok  $0.06" in result.stdout
    assert '"runs"' not in result.stdout


def test_recent_runs_verbose_keeps_json() -> None:
    body = {"runs": [{"run_id": "rpt-001abc"}]}
    result = handle_line(
        "recent_runs --verbose",
        session=Session(),
        client=_client([], response=httpx.Response(200, json=body)),
    )
    assert '"runs"' in result.stdout


def test_get_run_quiet() -> None:
    body = {
        "run": {
            "run_id": "run-detail",
            "provider": "ollama",
            "model": "qwen",
            "tier": "refusal",
        },
        "fixtures": [
            {"id": "rf-01", "score": 0.8, "status": "complete"},
        ],
    }
    result = handle_line(
        "get_run run-detail",
        session=Session(),
        client=_client([], response=httpx.Response(200, json=body)),
    )
    assert "run-detail  ollama  qwen  refusal" in result.stdout
    assert "  rf-01  0.80  complete" in result.stdout
    assert '"fixtures"' not in result.stdout


def test_get_run_verbose_keeps_json() -> None:
    body = {"run": {"run_id": "run-detail"}, "fixtures": []}
    result = handle_line(
        "get_run run-detail --verbose",
        session=Session(),
        client=_client([], response=httpx.Response(200, json=body)),
    )
    assert '"run"' in result.stdout


def test_compare_quiet() -> None:
    body = {
        "by": "provider",
        "rows": [
            {
                "group_key": "ollama",
                "avg_accuracy_score": 0.91,
                "task_count": 10,
                "total_cost": 0.12,
            }
        ],
    }
    result = handle_line(
        "compare",
        session=Session(),
        client=_client([], response=httpx.Response(200, json=body)),
    )
    assert "provider  1" in result.stdout
    assert "ollama  0.91  10 tasks  $0.12" in result.stdout
    assert '"rows"' not in result.stdout


def test_compare_verbose_keeps_json() -> None:
    body = {"by": "model", "rows": []}
    result = handle_line(
        "compare --verbose",
        session=Session(),
        client=_client([], response=httpx.Response(200, json=body)),
    )
    assert '"rows"' in result.stdout


def test_trends_quiet() -> None:
    body = {
        "hours": 24,
        "rows": [
            {
                "hour": "2026-08-23 17:00",
                "task_count": 1,
                "total_tokens": 10,
                "cost": 0.03,
            }
        ],
    }
    result = handle_line(
        "trends",
        session=Session(),
        client=_client([], response=httpx.Response(200, json=body)),
    )
    assert "24h  1" in result.stdout
    assert "2026-08-23 17:00  1  10 tok  $0.03" in result.stdout
    assert '"rows"' not in result.stdout


def test_trends_verbose_keeps_json() -> None:
    body = {"hours": 24, "rows": []}
    result = handle_line(
        "trends --verbose",
        session=Session(),
        client=_client([], response=httpx.Response(200, json=body)),
    )
    assert '"rows"' in result.stdout


def test_unknown_flag_does_not_call_the_client() -> None:
    requests: list[httpx.Request] = []
    result = handle_line(
        "health --nope 1",
        session=Session(),
        client=_client(requests),
    )
    assert requests == []
    assert "nope" in result.stderr
