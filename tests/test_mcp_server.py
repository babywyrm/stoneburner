"""Tests for the MCP tool surface built over the atomics API client.

These cover what the tool layer is responsible for: which tools exist, whether
they are honestly labelled as read-only or spending, that arguments reach the
API client unchanged, and that an API failure still tells the agent why.
"""

from __future__ import annotations

import json

import pytest

# The [mcp] extra is optional. Skip rather than error out collection when it is
# missing; CI and CONTRIBUTING install it so these run there.
pytest.importorskip("mcp")

from atomics.mcp.client import AtomicsApiError
from atomics.mcp.server import build_server

READ_ONLY_TOOLS = {
    "health",
    "list_models",
    "list_jobs",
    "get_job",
    "get_run",
    "compare",
    "recent_runs",
    "trends",
}
SPENDING_TOOLS = {
    "submit_run",
    "submit_eval",
    "submit_sweep",
    "submit_stress",
    "submit_soak",
    "provider_test",
}


class FakeApi:
    """Stands in for AtomicsApiClient, recording calls instead of making them."""

    def __init__(self, result=None, error=None):
        self.calls: list[tuple[str, dict]] = []
        self.result = result if result is not None else {"job_id": "j1", "status": "queued"}
        self.error = error

    def _record(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if self.error is not None:
            raise self.error
        return self.result

    def health(self):
        return self._record("health")

    def submit_run(self, **kwargs):
        return self._record("submit_run", **kwargs)

    def submit_eval(self, **kwargs):
        return self._record("submit_eval", **kwargs)

    def get_job(self, job_id):
        return self._record("get_job", job_id=job_id)

    def list_jobs(self):
        return self._record("list_jobs")

    def get_run(self, run_id):
        return self._record("get_run", run_id=run_id)

    def trends(self, **kwargs):
        return self._record("trends", **kwargs)

    def compare(self, **kwargs):
        return self._record("compare", **kwargs)

    def recent_runs(self, **kwargs):
        return self._record("recent_runs", **kwargs)

    def list_models(self, **kwargs):
        return self._record("list_models", **kwargs)

    def provider_test(self, **kwargs):
        return self._record("provider_test", **kwargs)

    def submit_sweep(self, **kwargs):
        return self._record("submit_sweep", **kwargs)

    def submit_stress(self, **kwargs):
        return self._record("submit_stress", **kwargs)

    def submit_soak(self, **kwargs):
        return self._record("submit_soak", **kwargs)


def payload_of(result):
    """The JSON a tool returned, as the MCP client would receive it."""
    return json.loads(result.content[0].text)


async def test_registers_exactly_the_supported_api_surface():
    """The proxy must not advertise tools the API cannot serve."""
    tools = await build_server(FakeApi()).list_tools()
    assert {tool.name for tool in tools} == READ_ONLY_TOOLS | SPENDING_TOOLS


@pytest.mark.parametrize("name", sorted(READ_ONLY_TOOLS))
async def test_read_only_tools_are_labelled_read_only(name):
    """Agents throttle destructive-looking tools; mislabelling these as
    spending would discourage the cheap calls we want them making."""
    tools = {tool.name: tool for tool in await build_server(FakeApi()).list_tools()}
    assert tools[name].annotations.read_only_hint is True


@pytest.mark.parametrize("name", sorted(SPENDING_TOOLS))
async def test_spending_tools_are_not_labelled_read_only(name):
    """These charge a real provider account. Claiming read-only would invite an
    agent to call them freely."""
    tools = {tool.name: tool for tool in await build_server(FakeApi()).list_tools()}
    assert tools[name].annotations.read_only_hint is False


async def test_submit_run_forwards_arguments_to_the_api():
    api = FakeApi()
    await build_server(api).call_tool(
        "submit_run", {"provider": "claude", "model": "sonnet", "iterations": 9}
    )

    name, kwargs = api.calls[0]
    assert name == "submit_run"
    assert kwargs["provider"] == "claude"
    assert kwargs["model"] == "sonnet"
    assert kwargs["iterations"] == 9


async def test_submit_run_forwards_effort_to_the_api():
    api = FakeApi()
    await build_server(api).call_tool(
        "submit_run",
        {
            "provider": "openai",
            "effort": "high",
            "reasoning_mode": "pro",
            "thinking": False,
        },
    )

    name, kwargs = api.calls[0]
    assert name == "submit_run"
    assert kwargs["effort"] == "high"
    assert kwargs["reasoning_mode"] == "pro"
    assert kwargs["thinking"] is False


async def test_list_models_forwards_arguments_to_the_api():
    api = FakeApi(result={"provider": "ollama", "models": []})
    await build_server(api).call_tool(
        "list_models", {"provider": "vllm", "host": "http://gpu:8000/v1"}
    )
    name, kwargs = api.calls[0]
    assert name == "list_models"
    assert kwargs["provider"] == "vllm"
    assert kwargs["host"] == "http://gpu:8000/v1"


async def test_provider_test_forwards_arguments_to_the_api():
    api = FakeApi(result={"ok": True, "health": True})
    await build_server(api).call_tool("provider_test", {"provider": "ollama", "model": "qwen3:14b"})
    name, kwargs = api.calls[0]
    assert name == "provider_test"
    assert kwargs["provider"] == "ollama"
    assert kwargs["model"] == "qwen3:14b"


async def test_submit_sweep_forwards_required_budget():
    api = FakeApi()
    await build_server(api).call_tool(
        "submit_sweep",
        {
            "provider": "ollama",
            "models": ["qwen3:14b"],
            "suites": ["redblue", "refusal"],
            "budget_usd": 7.0,
            "runs": 3,
        },
    )
    _, kwargs = api.calls[0]
    assert kwargs["budget_usd"] == 7.0
    assert kwargs["models"] == ["qwen3:14b"]
    assert kwargs["suites"] == ["redblue", "refusal"]
    assert kwargs["runs"] == 3


async def test_submit_stress_forwards_required_budget():
    api = FakeApi()
    await build_server(api).call_tool(
        "submit_stress",
        {
            "provider": "ollama",
            "model": "qwen3:14b",
            "budget_usd": 3.0,
            "max_concurrency": 8,
        },
    )
    _, kwargs = api.calls[0]
    assert kwargs["budget_usd"] == 3.0
    assert kwargs["model"] == "qwen3:14b"
    assert kwargs["max_concurrency"] == 8


async def test_submit_soak_forwards_duration_seconds():
    api = FakeApi()
    await build_server(api).call_tool(
        "submit_soak",
        {
            "provider": "ollama",
            "model": "qwen3:14b",
            "budget_usd": 2.0,
            "duration_seconds": 120,
            "concurrency": 2,
        },
    )
    _, kwargs = api.calls[0]
    assert kwargs["budget_usd"] == 2.0
    assert kwargs["duration_seconds"] == 120
    assert kwargs["concurrency"] == 2


async def test_submit_eval_forwards_budget():
    api = FakeApi()
    await build_server(api).call_tool(
        "submit_eval", {"suite": "adversarial", "provider": "openai", "budget_usd": 4.0}
    )

    _, kwargs = api.calls[0]
    assert kwargs["suite"] == "adversarial"
    assert kwargs["budget_usd"] == 4.0


async def test_get_job_returns_the_api_payload():
    api = FakeApi(result={"job_id": "abc", "status": "finished", "result": {"score": 1}})
    result = await build_server(api).call_tool("get_job", {"job_id": "abc"})

    assert api.calls == [("get_job", {"job_id": "abc"})]
    assert payload_of(result)["status"] == "finished"


async def test_list_jobs_and_get_run_and_trends_forward():
    api = FakeApi(result={"jobs": []})
    server = build_server(api)
    await server.call_tool("list_jobs", {})
    await server.call_tool("get_run", {"run_id": "r1"})
    await server.call_tool("trends", {"hours": 12})

    assert api.calls[0] == ("list_jobs", {})
    assert api.calls[1] == ("get_run", {"run_id": "r1"})
    assert api.calls[2] == ("trends", {"hours": 12})


async def test_api_error_detail_reaches_the_agent():
    """A failed call must carry the server's reason. The MCP runtime wraps a
    raised exception in its own error type, so this asserts on the message
    surviving rather than on that private class.
    """
    api = FakeApi(error=AtomicsApiError("Budget exceeded: limit 10.0 USD", status_code=402))
    with pytest.raises(Exception) as excinfo:
        await build_server(api).call_tool("submit_eval", {"suite": "rag", "provider": "x"})

    assert "Budget exceeded: limit 10.0 USD" in str(excinfo.value)


async def test_instructions_tell_the_agent_to_poll_for_results():
    """Submissions are async; an agent that does not know to poll sees a job id
    and concludes the eval produced nothing. The API status is `completed`."""
    server = build_server(FakeApi())
    text = server.instructions or ""
    assert "get_job" in text
    assert "completed" in text


async def test_submit_run_tool_advertises_effort():
    tools = {t.name: t for t in await build_server(FakeApi()).list_tools()}
    schema = tools["submit_run"].input_schema
    assert "effort" in schema["properties"]
    assert "reasoning_mode" in schema["properties"]
    assert "thinking" in schema["properties"]


async def test_submit_eval_tool_names_the_security_suites():
    tools = {t.name: t for t in await build_server(FakeApi()).list_tools()}
    description = tools["submit_eval"].description or ""
    for suite in ("refusal", "redblue", "toolcall", "codereview"):
        assert suite in description
