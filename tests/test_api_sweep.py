"""Bounded sweep jobs: required budget, named models, no discover-everything."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from atomics.api._sweep import run_sweep_from_request
from atomics.api.config import ServerSettings
from atomics.api.models import MAX_SWEEP_MODELS, MAX_SWEEP_RUNS, SweepRequest
from atomics.api.server import create_app
from atomics.eval.budget import GuardedProvider
from atomics.eval.gauntlet import SuiteJobResult


def test_sweep_requires_a_budget():
    with pytest.raises(ValidationError):
        SweepRequest(provider="ollama", models=["a"], suites=["eval"])


def test_sweep_rejects_zero_budget():
    with pytest.raises(ValidationError):
        SweepRequest(provider="ollama", models=["a"], suites=["eval"], budget_usd=0)


def test_sweep_rejects_too_many_models():
    with pytest.raises(ValidationError):
        SweepRequest(
            provider="ollama",
            models=[f"m{i}" for i in range(MAX_SWEEP_MODELS + 1)],
            suites=["eval"],
            budget_usd=1.0,
        )


def test_sweep_rejects_too_many_runs():
    with pytest.raises(ValidationError):
        SweepRequest(
            provider="ollama",
            models=["a"],
            suites=["eval"],
            runs=MAX_SWEEP_RUNS + 1,
            budget_usd=1.0,
        )


def test_sweep_rejects_unknown_suite():
    with pytest.raises(ValidationError, match="unknown suites"):
        SweepRequest(
            provider="ollama",
            models=["a"],
            suites=["soak"],
            budget_usd=1.0,
        )


def test_sweep_strips_blank_models():
    req = SweepRequest(
        provider="ollama", models=[" a ", "", "b"], suites=["eval"], budget_usd=2.0
    )
    assert req.models == ["a", "b"]


def test_sweep_normalizes_suite_names():
    req = SweepRequest(
        provider="ollama",
        models=["qwen3:14b"],
        suites=["RedBlue", "refusal"],
        budget_usd=5.0,
    )
    assert req.suites == ["redblue", "refusal"]


@pytest.mark.asyncio
async def test_run_sweep_meters_every_model_against_one_budget():
    payload = SweepRequest(
        provider="ollama",
        models=["a", "b"],
        suites=["eval"],
        budget_usd=3.0,
    )
    captured: dict = {}

    def fake_provider(name, model):
        return SimpleNamespace(name=name, model=model)

    def store_factory(**kwargs):
        captured.update(kwargs)
        return AsyncMock()

    rows = [
        SuiteJobResult(model="a", suite="eval", ok=True, headline=0.9),
        SuiteJobResult(model="b", suite="eval", ok=False, error="nope"),
    ]

    with (
        patch("atomics.api._sweep._provider_for", side_effect=fake_provider),
        patch("atomics.api._sweep.make_suite_runner", side_effect=store_factory),
        patch(
            "atomics.api._sweep.run_gauntlet",
            new_callable=AsyncMock,
            return_value=rows,
        ) as gauntlet,
    ):
        result = await run_sweep_from_request(payload)

    wrapped = captured["provider_factory"]("a")
    judge = captured["judge_provider"]
    assert isinstance(wrapped, GuardedProvider)
    assert isinstance(judge, GuardedProvider)
    assert wrapped.guard is judge.guard
    assert gauntlet.await_args.kwargs["skip_incapable"] is False
    assert gauntlet.await_args.kwargs["models"] == ["a", "b"]
    assert result["ok"] == 1
    assert result["fail"] == 1
    assert result["jobs"][0]["headline"] == 0.9
    assert result["budget_usd"] == 3.0


@pytest.mark.asyncio
async def test_post_sweeps_returns_a_job():
    app = create_app(settings=ServerSettings(no_auth=True))
    with (
        patch(
            "atomics.api.routes.run_sweep_from_request",
            new_callable=AsyncMock,
            return_value={"ok": 1, "fail": 0, "jobs": []},
        ),
        TestClient(app) as client,
    ):
        resp = client.post(
            "/api/v1/sweeps",
            json={
                "provider": "ollama",
                "models": ["qwen3:14b"],
                "suites": ["eval"],
                "budget_usd": 1.5,
            },
        )
    assert resp.status_code == 202
    assert resp.json()["kind"] == "sweep"


def test_post_sweeps_without_budget_is_422():
    app = create_app(settings=ServerSettings(no_auth=True))
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/sweeps",
            json={"provider": "ollama", "models": ["a"], "suites": ["eval"]},
        )
    assert resp.status_code == 422
