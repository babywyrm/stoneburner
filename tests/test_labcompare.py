"""Tests for labcompare orchestration helpers."""
from __future__ import annotations

import pytest

from atomics.labcompare import HostSpec, parse_host_specs


def test_parse_single_host():
    specs = parse_host_specs(["laptop=http://192.168.1.205:11434"])
    assert specs == [HostSpec(name="laptop", url="http://192.168.1.205:11434")]


def test_parse_multiple_hosts():
    specs = parse_host_specs([
        "laptop=http://192.168.1.205:11434",
        "brainbox=http://192.168.1.239:11434",
    ])
    assert len(specs) == 2
    assert specs[1].name == "brainbox"


def test_parse_host_missing_equals_raises():
    with pytest.raises(ValueError, match="expected NAME=URL"):
        parse_host_specs(["http://192.168.1.205:11434"])


def test_parse_host_empty_name_raises():
    with pytest.raises(ValueError, match="empty host name"):
        parse_host_specs(["=http://x:11434"])


# ── VRAM fit ──────────────────────────────────────────────

from atomics.labcompare import vram_fit_from_ps


def test_vram_fit_full_gpu():
    ps = {"models": [{"name": "qwen3.6:27b", "size": 100, "size_vram": 100,
                      "details": {"family": "qwen"}}]}
    fit, gpu = vram_fit_from_ps(ps, "qwen3.6:27b")
    assert fit == 1.0


def test_vram_fit_partial_offload():
    ps = {"models": [{"name": "qwen3.6:27b", "size": 100, "size_vram": 65}]}
    fit, _ = vram_fit_from_ps(ps, "qwen3.6:27b")
    assert abs(fit - 0.65) < 1e-6


def test_vram_fit_model_not_loaded_returns_none():
    ps = {"models": []}
    fit, gpu = vram_fit_from_ps(ps, "qwen3.6:27b")
    assert fit is None
    assert gpu is None


def test_vram_fit_zero_size_returns_none():
    ps = {"models": [{"name": "m", "size": 0, "size_vram": 0}]}
    fit, _ = vram_fit_from_ps(ps, "m")
    assert fit is None


# ── Verdict math ──────────────────────────────────────────

from atomics.labcompare import parity_verdict, speedup_ratio


def test_speedup_ratio_basic():
    assert speedup_ratio(48.0, 7.0) == 6.9


def test_speedup_ratio_zero_baseline_returns_none():
    assert speedup_ratio(48.0, 0.0) is None


def test_speedup_ratio_none_input_returns_none():
    assert speedup_ratio(None, 7.0) is None


def test_parity_verdict_within_tolerance():
    ok, delta = parity_verdict(0.94, 0.94, tolerance=0.05)
    assert ok is True
    assert delta == 0.0


def test_parity_verdict_outside_tolerance():
    ok, delta = parity_verdict(0.94, 0.80, tolerance=0.05)
    assert ok is False
    assert abs(delta - 0.14) < 1e-6


def test_parity_verdict_missing_score_returns_none():
    ok, delta = parity_verdict(0.94, None)
    assert ok is None
    assert delta is None


# ── Throughput probe ──────────────────────────────────────

from atomics.labcompare import ThroughputResult, probe_throughput
from atomics.providers.base import ProviderResponse


class _FakeProvider:
    def __init__(self):
        self.calls = 0

    async def generate(self, prompt, *, system=None, model=None, max_tokens=256,
                        thinking=None, thinking_budget=None, temperature=None):
        self.calls += 1
        return ProviderResponse(
            text="hello world",
            input_tokens=10, output_tokens=20, total_tokens=30,
            model=model or "m", latency_ms=250.0, estimated_cost_usd=0.0,
            tokens_per_second=80.0, tps_basis="generation",
            raw={"prompt_eval_count": 10, "prompt_eval_duration": 10_000_000},
        )


@pytest.mark.asyncio
async def test_probe_throughput_averages_tps():
    prov = _FakeProvider()

    async def ps():
        return {"models": [{"name": "m", "size": 100, "size_vram": 100}]}

    res = await probe_throughput(prov, "m", ps_fetcher=ps, prompts=["a", "b", "c"])
    assert isinstance(res, ThroughputResult)
    assert prov.calls == 3
    assert res.tokens_per_second == 80.0
    assert res.latency_ms == 250.0
    assert res.vram_fit_pct == 1.0


@pytest.mark.asyncio
async def test_probe_throughput_handles_ps_failure():
    prov = _FakeProvider()

    async def ps():
        raise RuntimeError("ps unavailable")

    res = await probe_throughput(prov, "m", ps_fetcher=ps, prompts=["a"])
    assert res.tokens_per_second == 80.0
    assert res.vram_fit_pct is None


# ── Orchestrator ──────────────────────────────────────────

from atomics.labcompare import CellResult, run_labcompare


@pytest.mark.asyncio
async def test_run_labcompare_two_hosts_throughput_and_quality():
    hosts = [HostSpec("laptop", "http://a:11434"), HostSpec("brainbox", "http://b:11434")]

    def provider_factory(url):
        return _FakeProvider()

    def ps_fetcher_factory(url):
        async def _ps():
            return {"models": [{"name": "m", "size": 100, "size_vram": 100}]}
        return _ps

    async def quality_fn(provider, judge_provider, judge_model, model):
        return 0.9

    cells = await run_labcompare(
        hosts=hosts, models=["m"], dimensions=["throughput", "quality"],
        quality_suite="eval", judge_host=None, judge_model="judge",
        prompts=2, provider_factory=provider_factory, quality_fn=quality_fn,
        ps_fetcher_factory=ps_fetcher_factory,
    )
    assert len(cells) == 2
    assert {c.host_name for c in cells} == {"laptop", "brainbox"}
    assert all(c.tokens_per_second == 80.0 for c in cells)
    assert all(c.quality_score == 0.9 for c in cells)


@pytest.mark.asyncio
async def test_run_labcompare_unreachable_host_skipped():
    hosts = [HostSpec("up", "http://a:11434"), HostSpec("down", "http://b:11434")]

    class _DeadProvider:
        async def generate(self, *a, **k):
            raise ConnectionError("host down")

    def provider_factory(url):
        return _DeadProvider() if "b:" in url else _FakeProvider()

    def ps_fetcher_factory(url):
        async def _ps():
            return {"models": []}
        return _ps

    async def quality_fn(provider, judge_provider, judge_model, model):
        return 0.9

    cells = await run_labcompare(
        hosts=hosts, models=["m"], dimensions=["throughput"],
        quality_suite="eval", judge_host=None, judge_model=None,
        prompts=1, provider_factory=provider_factory, quality_fn=quality_fn,
        ps_fetcher_factory=ps_fetcher_factory,
    )
    up = [c for c in cells if c.host_name == "up"][0]
    down = [c for c in cells if c.host_name == "down"][0]
    assert up.tokens_per_second == 80.0
    assert down.tokens_per_second is None
    assert down.error is not None
