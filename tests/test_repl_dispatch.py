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


def test_submit_verbs_reject_save() -> None:
    for verb in ("submit_run", "submit_eval --suite refusal"):
        result = handle_line(
            f"{verb} --save false",
            session=Session(provider="ollama"),
            client=_Boom(),  # type: ignore[arg-type]
        )
        assert "unknown flag --save" in result.stderr


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


def test_submit_eval_forwards_channel() -> None:
    requests: list[httpx.Request] = []
    session = Session(provider="ollama", model="a")
    handle_line(
        "submit_eval --suite toolcall --channel tools",
        session=session,
        client=_client(requests, response=httpx.Response(200, json=_EVAL_JOB)),
    )
    assert json.loads(requests[0].content)["channel"] == "tools"


def test_submit_eval_and_sweep_forward_temperature() -> None:
    requests: list[httpx.Request] = []
    session = Session(provider="ollama", model="a")
    client = _client(requests, response=httpx.Response(200, json=_EVAL_JOB))
    handle_line("submit_eval --suite toolcall --temperature 0", session=session, client=client)
    handle_line(
        "submit_sweep --models a --suites eval --budget_usd 1 --temperature 0.7",
        session=session,
        client=client,
    )
    assert [json.loads(r.content)["temperature"] for r in requests] == [0.0, 0.7]


def test_session_host_fills_submit_stress() -> None:
    requests: list[httpx.Request] = []
    session = Session(provider="ollama", model="llama3.2:1b", host="http://10.0.0.9:11434")
    handle_line(
        "submit_stress --budget_usd 1",
        session=session,
        client=_client(requests),
    )
    payload = json.loads(requests[0].content)
    assert payload["host"] == "http://10.0.0.9:11434"


def test_session_host_fills_submit_sweep() -> None:
    requests: list[httpx.Request] = []
    session = Session(provider="ollama", model="llama3.2:1b", host="http://10.0.0.9:11434")
    handle_line(
        "submit_sweep --suites eval --budget_usd 1",
        session=session,
        client=_client(requests),
    )
    payload = json.loads(requests[0].content)
    assert payload["host"] == "http://10.0.0.9:11434"


def test_session_host_fills_submit_run() -> None:
    requests: list[httpx.Request] = []
    session = Session(provider="ollama", model="llama3.2:1b", host="http://10.0.0.9:11434")
    handle_line(
        "submit_run",
        session=session,
        client=_client(requests),
    )
    payload = json.loads(requests[0].content)
    assert payload["host"] == "http://10.0.0.9:11434"


def test_session_host_fills_submit_eval() -> None:
    requests: list[httpx.Request] = []
    session = Session(provider="ollama", model="llama3.2:1b", host="http://10.0.0.9:11434")
    handle_line(
        "submit_eval --suite accuracy",
        session=session,
        client=_client(requests),
    )
    payload = json.loads(requests[0].content)
    assert payload["host"] == "http://10.0.0.9:11434"


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


def test_submit_sweep_forwards_judge_host() -> None:
    requests: list[httpx.Request] = []
    session = Session(provider="ollama", model="a")
    handle_line(
        "submit_sweep --suites eval --budget_usd 1 --judge_host http://10.0.0.9:11434",
        session=session,
        client=_client(requests),
    )
    assert json.loads(requests[0].content)["judge_host"] == "http://10.0.0.9:11434"


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


def test_provider_test_quiet_empty_visible_with_thinking_is_think() -> None:
    body = {
        "ok": True,
        "model": "qwen3.5:2b",
        "latency_ms": 400.0,
        "response": "",
        "thinking_tokens": 288,
        "error": None,
    }
    result = handle_line(
        "provider_test --provider ollama",
        session=Session(),
        client=_client([], response=httpx.Response(200, json=body)),
    )
    assert "THINK" in result.stdout
    assert "288" in result.stdout
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


async def test_repl_verbs_match_mcp_tools_and_client_parameters() -> None:
    import inspect

    import pytest

    pytest.importorskip("mcp")
    from atomics.mcp.server import build_server
    from atomics.repl.dispatch import _VERBS

    server = build_server(_client([]))
    mcp_names = {tool.name for tool in await server.list_tools()}
    assert set(_VERBS) == mcp_names

    for verb, (method_name, allowed, positional) in _VERBS.items():
        method = getattr(AtomicsApiClient, method_name)
        expected = set(inspect.signature(method).parameters) - {"self"}
        actual = set(allowed) | ({positional} if positional is not None else set())
        assert actual == expected, f"parameter drift on {verb}"


def test_submit_battery_fills_session_fields_and_formats_quietly() -> None:
    requests: list[httpx.Request] = []
    session = Session(
        provider="ollama",
        model="granite4.2:3b",
        host="http://127.0.0.1:11434",
        effort="low",
    )
    job_body = {
        "job_id": "bat-1",
        "status": "pending",
        "kind": "battery",
        "request": {
            "name": "desk-pass",
            "provider": "ollama",
            "model": "granite4.2:3b",
            "host": "http://127.0.0.1:11434",
        },
    }
    result = handle_line(
        "submit_battery --name desk-pass --budget_usd 5",
        session=session,
        client=_client(requests, response=httpx.Response(202, json=job_body)),
    )
    assert requests[0].url.path == "/api/v1/batteries"
    payload = json.loads(requests[0].content)
    assert payload == {
        "name": "desk-pass",
        "provider": "ollama",
        "model": "granite4.2:3b",
        "host": "http://127.0.0.1:11434",
        "effort": "low",
        "budget_usd": 5.0,
    }
    assert session.last_job_id == "bat-1"
    assert "battery  desk-pass  granite4.2:3b" in result.stdout


def test_distributed_verbs_dispatch_and_format_quietly() -> None:
    requests: list[httpx.Request] = []
    workers_body = {
        "workers": [
            {
                "worker_id": "w-alpha",
                "status": "online",
                "endpoint": "http://127.0.0.1:11434",
                "labels": {"gpu": "m3", "site": "lab"},
            }
        ]
    }
    res_workers = handle_line(
        "list_workers",
        session=Session(),
        client=_client(requests, response=httpx.Response(200, json=workers_body)),
    )
    assert "w-alpha  online  http://127.0.0.1:11434  gpu=m3,site=lab" in res_workers.stdout

    submit_requests: list[httpx.Request] = []
    session = Session()
    dist_job = {
        "job_id": "dj-123456789",
        "mode": "full",
        "status": "pending",
        "request_json": json.dumps(
            {"mode": "full", "eval_request": {"suite": "refusal", "model": "gemma3:4b"}}
        ),
        "summary_json": None,
    }
    res_submit = handle_line(
        "submit_distributed_run --mode full "
        '--eval_request \'{"suite":"refusal","provider":"ollama"}\' '
        "--worker_selector site=lab,gpu=m3 --timeout_seconds 120 --max_retries 1",
        session=session,
        client=_client(submit_requests, response=httpx.Response(202, json=dist_job)),
    )
    assert submit_requests[0].url.path == "/api/v1/distributed/runs"
    assert json.loads(submit_requests[0].content) == {
        "mode": "full",
        "eval_request": {"suite": "refusal", "provider": "ollama"},
        "worker_selector": {"site": "lab", "gpu": "m3"},
        "timeout_seconds": 120,
        "max_retries": 1,
    }
    assert session.last_job_id == "dj-123456789"
    assert "distributed  full  refusal  gemma3:4b" in res_submit.stdout

    list_res = handle_line(
        "list_distributed_runs --limit 5",
        session=Session(),
        client=_client([], response=httpx.Response(200, json={"jobs": [dist_job]})),
    )
    assert "distributed  full  pending  dj-12345" in list_res.stdout

    get_requests: list[httpx.Request] = []
    get_res = handle_line(
        "get_distributed_run dj-123456789",
        session=Session(),
        client=_client(get_requests, response=httpx.Response(200, json=dist_job)),
    )
    assert get_requests[0].url.path == "/api/v1/distributed/runs/dj-123456789"
    assert '"job_id": "dj-123456789"' in get_res.stdout


def test_distributed_verbs_reject_malformed_json_and_selector() -> None:
    requests: list[httpx.Request] = []
    bad_json = handle_line(
        "submit_distributed_run --eval_request '[1,2]'",
        session=Session(),
        client=_client(requests),
    )
    assert requests == []
    assert "JSON object" in bad_json.stderr

    bad_sel = handle_line(
        "submit_distributed_run --mode fleet --worker_selector invalid_pair",
        session=Session(),
        client=_client(requests),
    )
    assert requests == []
    assert "worker_selector" in bad_sel.stderr

    json_sel_requests: list[httpx.Request] = []
    handle_line(
        'submit_distributed_run --mode full --eval_request \'{"suite":"refusal"}\' '
        '--worker_selector \'{"site":"lab"}\'',
        session=Session(),
        client=_client(json_sel_requests),
    )
    assert json.loads(json_sel_requests[0].content)["worker_selector"] == {"site": "lab"}
