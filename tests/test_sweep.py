"""Tests for the multi-model sweep orchestrator."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from atomics.sweep import ModelSweepResult, run_model_sweep


def _make_mock_provider(name: str = "ollama") -> AsyncMock:
    provider = AsyncMock()
    provider.name = name
    provider.generate = AsyncMock(
        return_value=SimpleNamespace(
            text="test response",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            model="test-model",
            latency_ms=100.0,
            estimated_cost_usd=0.0,
            tokens_per_second=50.0,
            thinking_tokens=0,
            thinking_text="",
            raw={},
        )
    )
    return provider


def _make_judge_provider() -> AsyncMock:
    provider = AsyncMock()
    provider.name = "ollama"
    provider.generate = AsyncMock(
        return_value=SimpleNamespace(
            text='{"accuracy": 3, "completeness": 2, "format": 2, "rationale": "Good."}',
            input_tokens=50,
            output_tokens=30,
            total_tokens=80,
            model="judge-model",
            latency_ms=50.0,
            estimated_cost_usd=0.0,
            tokens_per_second=100.0,
            thinking_tokens=0,
            thinking_text="",
            raw={},
        )
    )
    return provider


@pytest.mark.asyncio
async def test_sweep_runs_multiple_models():
    provider = _make_mock_provider()
    judge = _make_judge_provider()

    models = ["qwen2.5:1.5b", "qwen2.5:3b", "mistral:7b"]
    results = await run_model_sweep(
        provider_factory=lambda m: provider,
        judge_provider=judge,
        models=models,
        fixture_ids=["ev-01"],
    )

    assert len(results) == 3
    assert all(isinstance(r, ModelSweepResult) for r in results)
    assert [r.model for r in results] == models


@pytest.mark.asyncio
async def test_sweep_result_has_required_fields():
    provider = _make_mock_provider()
    judge = _make_judge_provider()

    results = await run_model_sweep(
        provider_factory=lambda m: provider,
        judge_provider=judge,
        models=["qwen2.5:7b"],
        fixture_ids=["ev-01"],
    )

    r = results[0]
    assert r.model == "qwen2.5:7b"
    assert r.fixtures_run >= 1
    assert isinstance(r.overall_quality, float) or r.overall_quality is None
    assert isinstance(r.avg_latency_ms, float)
    assert isinstance(r.total_cost_usd, float)
    assert isinstance(r.total_tokens, int)


@pytest.mark.asyncio
async def test_sweep_filters_fixtures_by_id():
    provider = _make_mock_provider()
    judge = _make_judge_provider()

    results = await run_model_sweep(
        provider_factory=lambda m: provider,
        judge_provider=judge,
        models=["qwen2.5:7b"],
        fixture_ids=["ev-01", "ev-02"],
    )

    r = results[0]
    assert r.fixtures_run == 2


@pytest.mark.asyncio
async def test_sweep_handles_provider_error():
    """A model that fails should produce a result with None quality, not crash the sweep."""
    fail_provider = AsyncMock()
    fail_provider.name = "ollama"
    fail_provider.generate = AsyncMock(side_effect=ConnectionError("refused"))

    judge = _make_judge_provider()

    results = await run_model_sweep(
        provider_factory=lambda m: fail_provider,
        judge_provider=judge,
        models=["broken-model:1b"],
        fixture_ids=["ev-01"],
    )

    assert len(results) == 1
    assert results[0].model == "broken-model:1b"
    assert results[0].overall_quality is None


@pytest.mark.asyncio
async def test_sweep_on_model_done_callback():
    provider = _make_mock_provider()
    judge = _make_judge_provider()
    completed: list[str] = []

    def on_done(r: ModelSweepResult) -> None:
        completed.append(r.model)

    await run_model_sweep(
        provider_factory=lambda m: provider,
        judge_provider=judge,
        models=["a:1b", "b:2b"],
        fixture_ids=["ev-01"],
        on_model_done=on_done,
    )

    assert completed == ["a:1b", "b:2b"]
