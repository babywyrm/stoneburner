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
