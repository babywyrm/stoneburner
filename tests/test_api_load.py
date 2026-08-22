"""Bounded stress and soak jobs: required budget, capped concurrency/duration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from atomics.api._load import run_soak_from_request, run_stress_from_request
from atomics.api.config import ServerSettings
from atomics.api.models import (
    MAX_LOAD_PREDICT,
    MAX_SOAK_CONCURRENCY,
    MAX_SOAK_DURATION_SECONDS,
    MAX_STRESS_CONCURRENCY,
    MAX_STRESS_PHASE_SECONDS,
    SoakRequest,
    StressRequest,
)
from atomics.api.server import create_app
from atomics.eval.budget import GuardedProvider
from atomics.soak import SoakResult, SoakSample
from atomics.stress import ConcurrencyResult, StressResult


def test_stress_requires_a_budget():
    with pytest.raises(ValidationError):
        StressRequest(provider="ollama", model="qwen3:14b")


def test_stress_rejects_zero_budget():
    with pytest.raises(ValidationError):
        StressRequest(provider="ollama", model="qwen3:14b", budget_usd=0)


def test_stress_rejects_blank_model():
    with pytest.raises(ValidationError):
        StressRequest(provider="ollama", model="   ", budget_usd=1.0)


def test_stress_rejects_over_cap_concurrency():
    with pytest.raises(ValidationError):
        StressRequest(
            provider="ollama",
            model="qwen3:14b",
            budget_usd=1.0,
            max_concurrency=MAX_STRESS_CONCURRENCY + 1,
        )


def test_stress_rejects_over_cap_phase():
    with pytest.raises(ValidationError):
        StressRequest(
            provider="ollama",
            model="qwen3:14b",
            budget_usd=1.0,
            phase_seconds=MAX_STRESS_PHASE_SECONDS + 1,
        )


def test_soak_requires_a_budget():
    with pytest.raises(ValidationError):
        SoakRequest(provider="ollama", model="qwen3:14b")


def test_soak_rejects_over_cap_duration():
    with pytest.raises(ValidationError):
        SoakRequest(
            provider="ollama",
            model="qwen3:14b",
            budget_usd=1.0,
            duration_seconds=MAX_SOAK_DURATION_SECONDS + 1,
        )


def test_soak_rejects_over_cap_concurrency():
    with pytest.raises(ValidationError):
        SoakRequest(
            provider="ollama",
            model="qwen3:14b",
            budget_usd=1.0,
            concurrency=MAX_SOAK_CONCURRENCY + 1,
        )


def test_soak_rejects_interval_longer_than_duration():
    with pytest.raises(ValidationError, match="sample_interval"):
        SoakRequest(
            provider="ollama",
            model="qwen3:14b",
            budget_usd=1.0,
            duration_seconds=30,
            sample_interval=60,
        )


def test_soak_strips_model():
    req = SoakRequest(provider="ollama", model=" qwen3:14b ", budget_usd=1.0)
    assert req.model == "qwen3:14b"


@pytest.mark.asyncio
async def test_run_stress_meters_the_provider_and_caps_predict():
    payload = StressRequest(
        provider="ollama",
        model="qwen3:14b",
        budget_usd=2.0,
        max_concurrency=4,
        phase_seconds=5.0,
    )
    captured: dict = {}

    def fake_provider(name, model, host=None):
        return SimpleNamespace(name=name, model=model)

    result = StressResult(
        model="qwen3:14b",
        host="api",
        provider="ollama",
        peak_tps=12.5,
        saturation_concurrency=4,
        phases=[
            ConcurrencyResult(concurrency=1, requests=3, aggregate_tps=10.0, avg_latency_ms=100)
        ],
    )

    async def fake_run(provider, **kwargs):
        captured["provider"] = provider
        captured.update(kwargs)
        return result

    with (
        patch("atomics.api._load._provider_for", side_effect=fake_provider),
        patch("atomics.api._load.run_stress_provider", side_effect=fake_run),
    ):
        body = await run_stress_from_request(payload)

    assert isinstance(captured["provider"], GuardedProvider)
    assert captured["max_concurrency"] == 4
    assert captured["num_predict"] == MAX_LOAD_PREDICT
    assert "latencies" not in body["phases"][0]
    assert body["peak_tps"] == 12.5
    assert body["budget_usd"] == 2.0


@pytest.mark.asyncio
async def test_run_soak_meters_the_provider_and_caps_predict():
    payload = SoakRequest(
        provider="ollama",
        model="qwen3:14b",
        budget_usd=1.5,
        duration_seconds=60,
        concurrency=2,
        sample_interval=15,
    )
    captured: dict = {}

    def fake_provider(name, model, host=None):
        return SimpleNamespace(name=name, model=model)

    result = SoakResult(
        model="qwen3:14b",
        provider="ollama",
        concurrency=2,
        duration_seconds=60,
        verdict="STABLE",
        samples=[SoakSample(elapsed_seconds=15, requests=4, aggregate_tps=8.0)],
    )

    async def fake_run(provider, **kwargs):
        captured["provider"] = provider
        captured.update(kwargs)
        return result

    with (
        patch("atomics.api._load._provider_for", side_effect=fake_provider),
        patch("atomics.api._load.run_soak_provider", side_effect=fake_run),
    ):
        body = await run_soak_from_request(payload)

    assert isinstance(captured["provider"], GuardedProvider)
    assert captured["duration_seconds"] == 60.0
    assert captured["num_predict"] == MAX_LOAD_PREDICT
    assert body["verdict"] == "STABLE"
    assert body["samples"][0]["elapsed_seconds"] == 15


@pytest.mark.asyncio
async def test_post_stress_returns_a_job():
    app = create_app(settings=ServerSettings(no_auth=True))
    with (
        patch(
            "atomics.api.routes.run_stress_from_request",
            new_callable=AsyncMock,
            return_value={"verdict": None, "peak_tps": 1.0},
        ),
        TestClient(app) as client,
    ):
        resp = client.post(
            "/api/v1/stress",
            json={
                "provider": "ollama",
                "model": "qwen3:14b",
                "budget_usd": 1.5,
            },
        )
    assert resp.status_code == 202
    assert resp.json()["kind"] == "stress"


def test_post_stress_without_budget_is_422():
    app = create_app(settings=ServerSettings(no_auth=True))
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/stress",
            json={"provider": "ollama", "model": "qwen3:14b"},
        )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_post_soak_returns_a_job():
    app = create_app(settings=ServerSettings(no_auth=True))
    with (
        patch(
            "atomics.api.routes.run_soak_from_request",
            new_callable=AsyncMock,
            return_value={"verdict": "STABLE"},
        ),
        TestClient(app) as client,
    ):
        resp = client.post(
            "/api/v1/soak",
            json={
                "provider": "ollama",
                "model": "qwen3:14b",
                "budget_usd": 1.0,
                "duration_seconds": 60,
            },
        )
    assert resp.status_code == 202
    assert resp.json()["kind"] == "soak"


def test_post_soak_hour_long_duration_is_422():
    app = create_app(settings=ServerSettings(no_auth=True))
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/soak",
            json={
                "provider": "ollama",
                "model": "qwen3:14b",
                "budget_usd": 1.0,
                "duration_seconds": 7200,
            },
        )
    assert resp.status_code == 422
