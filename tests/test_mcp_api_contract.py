"""Contract tests between the MCP client and the real API server.

The other MCP tests assert the client builds the request we expect. That proves
the code agrees with its own mock, not that it agrees with the server — a
renamed route, a changed field, or the wrong auth header would pass all of them
and fail against a live `atomics server`.

So each test here captures the exact request a tool produces and replays it
against a real FastAPI app: real routing, real API-key auth, real pydantic
validation of the body. If the API moves, these fail.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from atomics.api.config import ServerSettings
from atomics.api.server import create_app
from atomics.mcp.client import AtomicsApiClient

API_KEY = "contract-test-key"


def captured_request(call) -> httpx.Request:
    """The HTTP request one client call produces, without sending it anywhere."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    with AtomicsApiClient(
        "http://api.test", API_KEY, transport=httpx.MockTransport(handler)
    ) as client:
        call(client)
    return seen[0]


def replay(app_client: TestClient, request: httpx.Request, *, with_key: bool = True):
    """Send a captured request to the real app, keeping only meaningful headers."""
    headers = {}
    if with_key and "x-api-key" in request.headers:
        headers["X-API-Key"] = request.headers["x-api-key"]
    if request.content:
        headers["Content-Type"] = "application/json"
    return app_client.request(
        request.method,
        request.url.path,
        content=request.content or None,
        params=dict(request.url.params) or None,
        headers=headers,
    )


@pytest.fixture
def app_client():
    app = create_app(settings=ServerSettings(api_keys={API_KEY}))
    with TestClient(app) as tc:
        yield tc


def test_submit_run_is_accepted_by_the_real_api(app_client):
    request = captured_request(lambda c: c.submit_run(provider="ollama"))
    response = replay(app_client, request)

    assert response.status_code == 202
    assert response.json()["kind"] == "run"


def test_submit_eval_is_accepted_by_the_real_api(app_client):
    request = captured_request(
        lambda c: c.submit_eval(suite="accuracy", provider="ollama", budget_usd=1.0)
    )
    response = replay(app_client, request)

    assert response.status_code == 202
    assert response.json()["kind"] == "eval"


def test_submit_eval_without_budget_is_accepted(app_client):
    """Omitting the budget must satisfy the server's default, not fail validation."""
    request = captured_request(lambda c: c.submit_eval(suite="accuracy", provider="ollama"))
    assert replay(app_client, request).status_code == 202


def test_health_is_reachable(app_client):
    request = captured_request(lambda c: c.health())
    response = replay(app_client, request)

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_compare_is_reachable(app_client):
    request = captured_request(lambda c: c.compare(by="model"))
    assert replay(app_client, request).status_code == 200


def test_recent_runs_is_reachable(app_client):
    request = captured_request(lambda c: c.recent_runs(limit=5))
    assert replay(app_client, request).status_code == 200


def test_unknown_job_is_a_404_not_a_routing_error(app_client):
    """404 proves the path shape matched a route. A 405 or 422 would mean the
    client is building `/jobs/{id}` in a way the server does not recognize."""
    request = captured_request(lambda c: c.get_job("no-such-job"))
    assert replay(app_client, request).status_code == 404


def test_the_api_key_header_is_what_authenticates(app_client):
    """Negative control. If the client sent the key under any other name, the
    positive tests above would still pass whenever auth happened to be off."""
    request = captured_request(lambda c: c.recent_runs())
    assert replay(app_client, request, with_key=False).status_code == 401
    assert replay(app_client, request, with_key=True).status_code == 200
