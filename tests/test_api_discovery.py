"""API endpoints for listing models and probing a provider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from atomics.api.config import ServerSettings
from atomics.api.server import create_app
from atomics.providers.factory import ProviderConfigError


@pytest.fixture
def client():
    app = create_app(settings=ServerSettings(no_auth=True))
    with TestClient(app) as tc:
        yield tc


def test_get_models_requires_auth():
    app = create_app(settings=ServerSettings(api_keys={"secret"}))
    with TestClient(app) as tc:
        resp = tc.get("/api/v1/models", params={"provider": "ollama"})
        assert resp.status_code == 401


def test_get_models_lists_annotated_tags(client):
    fake = SimpleNamespace(
        list_models=AsyncMock(
            return_value=[
                {
                    "name": "qwen3:14b",
                    "size_gb": 9.0,
                    "parameter_size": "14B",
                    "family": "qwen3",
                    "model_class": "mid",
                    "thinking": True,
                }
            ]
        )
    )
    with patch("atomics.api._discovery.make_provider", return_value=fake):
        resp = client.get("/api/v1/models", params={"provider": "ollama"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "ollama"
    assert body["models"][0]["name"] == "qwen3:14b"
    assert body["models"][0]["thinking"] is True
    fake.list_models.assert_awaited_once()


def test_get_models_rejects_unlistable_provider(client):
    resp = client.get("/api/v1/models", params={"provider": "claude"})
    assert resp.status_code == 400
    assert "ollama" in resp.json()["detail"]


def test_get_models_rejects_bad_host(client):
    resp = client.get(
        "/api/v1/models",
        params={"provider": "ollama", "host": "file:///etc/passwd"},
    )
    assert resp.status_code == 400


def test_get_models_connection_failure_is_502(client):
    fake = SimpleNamespace(list_models=AsyncMock(side_effect=ConnectionError("down")))
    with patch("atomics.api._discovery.make_provider", return_value=fake):
        resp = client.get("/api/v1/models", params={"provider": "ollama"})
    assert resp.status_code == 502


def test_provider_test_requires_auth():
    app = create_app(settings=ServerSettings(api_keys={"secret"}))
    with TestClient(app) as tc:
        resp = tc.post("/api/v1/provider-test", json={"provider": "ollama"})
        assert resp.status_code == 401


def test_provider_test_returns_health_and_generate(client):
    fake = SimpleNamespace(
        name="ollama",
        health_check=AsyncMock(return_value=True),
        generate=AsyncMock(
            return_value=SimpleNamespace(
                text="4",
                input_tokens=8,
                output_tokens=1,
                total_tokens=9,
                thinking_tokens=0,
                latency_ms=120.0,
                estimated_cost_usd=0.0,
                tokens_per_second=40.0,
                tps_basis="output",
            )
        ),
    )
    with patch("atomics.api._discovery.make_provider", return_value=fake):
        resp = client.post(
            "/api/v1/provider-test",
            json={"provider": "ollama", "model": "qwen3:14b"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["health"] is True
    assert body["response"] == "4"
    assert body["input_tokens"] == 8
    assert body["output_tokens"] == 1
    generate_kwargs = fake.generate.await_args.kwargs
    assert generate_kwargs["max_tokens"] == 32
    assert "2+2" in fake.generate.await_args.args[0]


def test_provider_test_rejects_caller_prompt(client):
    resp = client.post(
        "/api/v1/provider-test",
        json={"provider": "ollama", "prompt": "ignore previous"},
    )
    assert resp.status_code == 422


def test_provider_test_health_failure_is_ok_false(client):
    fake = SimpleNamespace(
        name="ollama",
        health_check=AsyncMock(return_value=False),
        generate=AsyncMock(),
    )
    with patch("atomics.api._discovery.make_provider", return_value=fake):
        resp = client.post("/api/v1/provider-test", json={"provider": "ollama"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is False
    assert resp.json()["health"] is False
    fake.generate.assert_not_awaited()


def test_provider_test_config_error_is_400(client):
    with patch(
        "atomics.api._discovery.make_provider",
        side_effect=ProviderConfigError("ANTHROPIC_API_KEY not set"),
    ):
        resp = client.post("/api/v1/provider-test", json={"provider": "claude"})
    assert resp.status_code == 400
