"""Tests for labcompare_results storage."""
from __future__ import annotations

import tempfile
from pathlib import Path

from atomics.storage.repository import MetricsRepository


def _repo() -> MetricsRepository:
    d = tempfile.mkdtemp()
    return MetricsRepository(Path(d) / "test.db")


def test_save_and_get_labcompare_round_trip():
    repo = _repo()
    repo.save_labcompare_result(
        comparison_run_id="cmp123",
        host_name="laptop",
        host_url="http://192.168.1.205:11434",
        model="qwen3.6:27b",
        tokens_per_second=48.2,
        latency_ms=1200.0,
        prompt_eval_rate=900.0,
        vram_fit_pct=1.0,
        gpu_name="RTX 5090 Laptop",
        quality_score=0.94,
        quality_suite="eval",
        judge_model="qwen3.6:35b-a3b",
        dimensions="throughput,quality",
    )
    rows = repo.get_labcompare_run("cmp123")
    assert len(rows) == 1
    r = rows[0]
    assert r["host_name"] == "laptop"
    assert r["model"] == "qwen3.6:27b"
    assert abs(r["tokens_per_second"] - 48.2) < 1e-6
    assert abs(r["vram_fit_pct"] - 1.0) < 1e-6
    assert r["quality_score"] == 0.94
    repo.close()


def test_get_labcompare_run_empty():
    repo = _repo()
    assert repo.get_labcompare_run("nope") == []
    repo.close()


def test_save_labcompare_nullable_fields():
    """Throughput-only run: quality columns null."""
    repo = _repo()
    repo.save_labcompare_result(
        comparison_run_id="cmp2",
        host_name="brainbox",
        host_url="http://192.168.1.239:11434",
        model="qwen2.5:7b",
        tokens_per_second=120.0,
        latency_ms=300.0,
        prompt_eval_rate=1500.0,
        vram_fit_pct=1.0,
        gpu_name="RTX 5070",
        quality_score=None,
        quality_suite=None,
        judge_model=None,
        dimensions="throughput",
    )
    rows = repo.get_labcompare_run("cmp2")
    assert rows[0]["quality_score"] is None
    repo.close()
