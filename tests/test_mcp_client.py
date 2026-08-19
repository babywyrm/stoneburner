"""Tests for the MCP server's HTTP client against the atomics API.

The client is the whole of the MCP server's reach into atomics, so these cover
the parts a proxy can get wrong: the exact path and payload each call produces,
that the API key travels as `X-API-Key`, and that a failure keeps the server's
own explanation instead of collapsing into a transport error.
"""

from __future__ import annotations

import httpx
import pytest

from atomics.mcp.client import (
    API_KEY_ENV,
    API_URL_ENV,
    DEFAULT_API_URL,
    AtomicsApiClient,
    AtomicsApiError,
)


def client_recording(requests, *, response=None, api_key="test-key"):
    """A client whose transport records requests and replays one response."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response or httpx.Response(200, json={"ok": True})

    return AtomicsApiClient(
        "http://api.test", api_key, transport=httpx.MockTransport(handler)
    )


def test_submit_run_posts_expected_payload():
    requests: list[httpx.Request] = []
    with client_recording(requests) as client:
        client.submit_run(provider="claude", model="sonnet", tier="mid", iterations=7)

    (request,) = requests
    assert request.method == "POST"
    assert request.url.path == "/api/v1/runs"
    import json

    assert json.loads(request.content) == {
        "provider": "claude",
        "model": "sonnet",
        "tier": "mid",
        "iterations": 7,
        "interval": 5,
        "save": True,
    }


def test_submit_run_omits_model_when_not_given():
    """An absent model must not become `null`; the server picks the default."""
    requests: list[httpx.Request] = []
    with client_recording(requests) as client:
        client.submit_run(provider="ollama")

    import json

    assert "model" not in json.loads(requests[0].content)


def test_api_key_is_sent_as_x_api_key_header():
    requests: list[httpx.Request] = []
    with client_recording(requests, api_key="s3cret") as client:
        client.recent_runs()

    assert requests[0].headers["x-api-key"] == "s3cret"


def test_no_api_key_sends_no_header():
    """Anonymous is a valid mode against a --no-auth server; sending an empty
    key would instead be a credential the server tries and rejects."""
    requests: list[httpx.Request] = []
    with client_recording(requests, api_key=None) as client:
        client.health()

    assert "x-api-key" not in requests[0].headers


def test_submit_eval_omits_budget_so_server_default_applies():
    """Leaving the budget unset must send nothing, not a client-invented number:
    DEFAULT_EVAL_BUDGET_USD on the server is the single definition."""
    requests: list[httpx.Request] = []
    with client_recording(requests) as client:
        client.submit_eval(suite="adversarial", provider="claude")

    import json

    payload = json.loads(requests[0].content)
    assert "budget_usd" not in payload
    assert payload == {"suite": "adversarial", "provider": "claude", "save": True}


def test_submit_eval_includes_budget_when_given():
    requests: list[httpx.Request] = []
    with client_recording(requests) as client:
        client.submit_eval(
            suite="rag", provider="openai", judge_model="j", fixtures=["a"], budget_usd=2.5
        )

    import json

    payload = json.loads(requests[0].content)
    assert payload["budget_usd"] == 2.5
    assert payload["judge_model"] == "j"
    assert payload["fixtures"] == ["a"]


def test_submit_sweep_posts_required_budget():
    requests: list[httpx.Request] = []
    with client_recording(requests) as client:
        client.submit_sweep(
            provider="ollama",
            models=["a"],
            suites=["eval"],
            budget_usd=4.0,
            runs=2,
        )

    import json

    assert requests[0].method == "POST"
    assert requests[0].url.path == "/api/v1/sweeps"
    payload = json.loads(requests[0].content)
    assert payload["budget_usd"] == 4.0
    assert payload["models"] == ["a"]
    assert payload["runs"] == 2


def test_get_job_uses_job_id_in_path():
    requests: list[httpx.Request] = []
    with client_recording(requests) as client:
        client.get_job("abc-123")

    assert requests[0].url.path == "/api/v1/jobs/abc-123"


def test_list_jobs_hits_the_collection_path():
    requests: list[httpx.Request] = []
    with client_recording(requests) as client:
        client.list_jobs()

    assert requests[0].method == "GET"
    assert requests[0].url.path == "/api/v1/jobs"


def test_get_run_uses_run_id_in_path():
    requests: list[httpx.Request] = []
    with client_recording(requests) as client:
        client.get_run("run-99")

    assert requests[0].url.path == "/api/v1/runs/run-99"


def test_trends_passes_hours():
    requests: list[httpx.Request] = []
    with client_recording(requests) as client:
        client.trends(hours=48)

    assert requests[0].url.path == "/api/v1/reports/trends"
    assert requests[0].url.params["hours"] == "48"


def test_compare_omits_unset_filters():
    requests: list[httpx.Request] = []
    with client_recording(requests) as client:
        client.compare(by="model", tier="ez")

    params = requests[0].url.params
    assert params["by"] == "model"
    assert params["tier"] == "ez"
    assert "since_hours" not in params
    assert "category" not in params


def test_list_models_uses_query_params():
    requests: list[httpx.Request] = []
    with client_recording(requests) as client:
        client.list_models(provider="vllm", host="http://gpu:8000/v1")

    assert requests[0].method == "GET"
    assert requests[0].url.path == "/api/v1/models"
    assert requests[0].url.params["provider"] == "vllm"
    assert requests[0].url.params["host"] == "http://gpu:8000/v1"


def test_list_models_omits_host_when_unset():
    requests: list[httpx.Request] = []
    with client_recording(requests) as client:
        client.list_models(provider="ollama")

    assert "host" not in requests[0].url.params


def test_provider_test_posts_expected_payload():
    requests: list[httpx.Request] = []
    with client_recording(requests) as client:
        client.provider_test(provider="ollama", model="qwen3:14b", thinking=False)

    import json

    assert requests[0].method == "POST"
    assert requests[0].url.path == "/api/v1/provider-test"
    assert json.loads(requests[0].content) == {
        "provider": "ollama",
        "model": "qwen3:14b",
        "thinking": False,
    }


def test_recent_runs_passes_limit():
    requests: list[httpx.Request] = []
    with client_recording(requests) as client:
        client.recent_runs(limit=3)

    assert requests[0].url.path == "/api/v1/reports/recent-runs"
    assert requests[0].url.params["limit"] == "3"


def test_error_detail_from_api_is_preserved():
    """The agent should read the server's reason, not a bare status code."""
    response = httpx.Response(400, json={"detail": "Unsupported eval suite: nope"})
    with client_recording([], response=response) as client:
        with pytest.raises(AtomicsApiError) as excinfo:
            client.submit_eval(suite="nope", provider="claude")

    assert "Unsupported eval suite: nope" in str(excinfo.value)
    assert excinfo.value.status_code == 400


def test_error_without_json_body_falls_back_to_text():
    response = httpx.Response(502, text="upstream boom")
    with client_recording([], response=response) as client:
        with pytest.raises(AtomicsApiError) as excinfo:
            client.health()

    assert "upstream boom" in str(excinfo.value)
    assert excinfo.value.status_code == 502


def test_error_with_structured_detail_is_stringified():
    """FastAPI validation errors put a list here; it must not crash the parse."""
    response = httpx.Response(422, json={"detail": [{"loc": ["body", "suite"]}]})
    with client_recording([], response=response) as client:
        with pytest.raises(AtomicsApiError) as excinfo:
            client.submit_eval(suite="", provider="claude")

    assert "suite" in str(excinfo.value)


def test_empty_success_body_returns_empty_dict():
    with client_recording([], response=httpx.Response(204)) as client:
        assert client.health() == {}


def test_connection_refused_names_the_missing_server():
    """The default httpx message explains nothing actionable to an agent."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("All connection attempts failed")

    client = AtomicsApiClient(
        "http://127.0.0.1:9999", "k", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(AtomicsApiError) as excinfo:
        client.health()

    message = str(excinfo.value)
    assert "http://127.0.0.1:9999" in message
    assert "atomics server" in message


def test_from_env_reads_url_and_key(monkeypatch):
    monkeypatch.setenv(API_URL_ENV, "http://elsewhere:9000/")
    monkeypatch.setenv(API_KEY_ENV, "from-env")
    client = AtomicsApiClient.from_env()
    try:
        assert client.base_url == "http://elsewhere:9000"
    finally:
        client.close()


def test_from_env_defaults_to_loopback(monkeypatch):
    monkeypatch.delenv(API_URL_ENV, raising=False)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    client = AtomicsApiClient.from_env()
    try:
        assert client.base_url == DEFAULT_API_URL
    finally:
        client.close()
