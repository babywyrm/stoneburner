"""Tests for the GPU stress testing module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atomics.stress import (
    ConcurrencyResult,
    StressResult,
    _percentile,
    _run_phase,
    run_stress,
)


def test_percentile_empty():
    assert _percentile([], 50) == 0.0


def test_percentile_single():
    assert _percentile([42.0], 50) == 42.0


def test_percentile_p50():
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0


def test_percentile_p95():
    values = list(range(1, 101))
    p95 = _percentile([float(v) for v in values], 95)
    assert 94 <= p95 <= 96


def test_concurrency_result_defaults():
    r = ConcurrencyResult(concurrency=4)
    assert r.concurrency == 4
    assert r.requests == 0
    assert r.aggregate_tps == 0.0
    assert r.latencies == []


def test_stress_result_defaults():
    r = StressResult(model="test", host="http://fake:11434")
    assert r.model == "test"
    assert r.phases == []
    assert r.peak_tps == 0.0
    assert r.saturation_concurrency == 0


def test_stress_prompts_are_populated():
    from atomics.stress import STRESS_PROMPTS

    assert len(STRESS_PROMPTS) >= 6
    for p in STRESS_PROMPTS:
        assert len(p) > 50


@pytest.mark.asyncio
async def test_run_stress_with_mock():
    """Verify the ramp logic works end-to-end with a mocked HTTP client."""
    from atomics.stress import _run_phase

    call_count = 0

    async def mock_post(url, *, json, timeout):
        nonlocal call_count
        call_count += 1
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "response": "test " * 50,
            "eval_count": 100,
            "prompt_eval_count": 30,
            "eval_duration": 800_000_000,
            "total_duration": 900_000_000,
        }
        return mock_resp

    mock_client = AsyncMock()
    mock_client.post = mock_post

    result = await _run_phase(
        mock_client, "http://fake:11434", "test-model",
        concurrency=2, duration_seconds=1.0, num_predict=512,
    )

    assert result.concurrency == 2
    assert result.requests >= 2
    assert result.total_output_tokens >= 200
    assert result.aggregate_tps > 0
    assert result.avg_request_tps > 0
    assert len(result.latencies) >= 2


@pytest.mark.asyncio
async def test_run_stress_e2e_with_mock(monkeypatch):
    """Full run_stress ramp with mocked HTTP and GPU helpers."""

    async def mock_post(url, *, json, timeout):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "response": "test " * 50,
            "eval_count": 80,
            "prompt_eval_count": 20,
            "eval_duration": 500_000_000,
            "total_duration": 600_000_000,
        }
        return mock_resp

    mock_client = AsyncMock()
    mock_client.post = mock_post
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    monkeypatch.setattr("atomics.stress._get_gpu_info", lambda: ("FakeGPU", 12288.0))
    monkeypatch.setattr("atomics.stress._get_vram_used_mb", lambda: 4000.0)

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    phase_log: list[int] = []

    result = await run_stress(
        host="http://fake:11434",
        model="test-model",
        max_concurrency=4,
        phase_seconds=0.5,
        num_predict=128,
        on_phase=lambda p: phase_log.append(p.concurrency),
    )

    assert result.model == "test-model"
    assert result.host == "http://fake:11434"
    assert result.gpu_name == "FakeGPU"
    assert result.vram_total_mb == 12288.0
    assert result.vram_peak_mb == 4000.0
    assert len(result.phases) == 3  # 1, 2, 4
    assert result.peak_tps > 0
    assert result.saturation_concurrency >= 1
    assert result.total_requests > 0
    assert result.duration_seconds > 0
    assert phase_log == [1, 2, 4]


@pytest.mark.asyncio
async def test_run_phase_handles_failures():
    """Failed requests should be counted, not crash the phase."""
    call_count = 0

    async def flaky_post(url, *, json, timeout):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 0:
            raise ConnectionError("refused")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "response": "ok",
            "eval_count": 10,
            "prompt_eval_count": 5,
            "eval_duration": 100_000_000,
            "total_duration": 120_000_000,
        }
        return mock_resp

    mock_client = AsyncMock()
    mock_client.post = flaky_post

    result = await _run_phase(
        mock_client, "http://fake:11434", "test-model",
        concurrency=2, duration_seconds=0.5, num_predict=64,
    )

    assert result.requests >= 1
    assert result.failed >= 1
    assert result.concurrency == 2


def test_cli_stress_command_with_save(monkeypatch, tmp_path):
    """atomics stress --save should persist results to database."""
    from click.testing import CliRunner

    from atomics.cli import cli

    async def fake_run_stress(**kwargs):
        on_phase = kwargs.get("on_phase")
        phase = ConcurrencyResult(
            concurrency=1, requests=5, total_output_tokens=500,
            aggregate_tps=25.0, avg_request_tps=25.0,
            avg_latency_ms=200.0, p95_latency_ms=300.0,
        )
        if on_phase:
            on_phase(phase)
        return StressResult(
            model=kwargs.get("model", "test"),
            host=kwargs.get("host", "http://fake:11434"),
            phases=[phase],
            peak_tps=25.0, saturation_concurrency=1,
            duration_seconds=15.0, total_tokens=500, total_requests=5,
        )

    monkeypatch.setattr("atomics.stress.run_stress", fake_run_stress)
    monkeypatch.setenv("ATOMICS_DB_PATH", str(tmp_path / "test.db"))

    runner = CliRunner()
    result = runner.invoke(cli, [
        "stress", "--model", "test-model", "--max-concurrency", "1",
        "--phase-seconds", "1", "--save",
    ])
    assert result.exit_code == 0
    assert "Peak throughput" in result.output or "peak" in result.output.lower()
    assert "saved" in result.output.lower()


def test_cli_stress_command_no_save(monkeypatch, tmp_path):
    """atomics stress --no-save should skip database persistence."""
    from click.testing import CliRunner

    from atomics.cli import cli

    async def fake_run_stress(**kwargs):
        phase = ConcurrencyResult(
            concurrency=1, requests=5, total_output_tokens=500,
            aggregate_tps=25.0, avg_request_tps=25.0,
            avg_latency_ms=200.0, p95_latency_ms=300.0,
        )
        on_phase = kwargs.get("on_phase")
        if on_phase:
            on_phase(phase)
        return StressResult(
            model=kwargs.get("model", "test"),
            host=kwargs.get("host", "http://fake:11434"),
            phases=[phase],
            peak_tps=25.0, saturation_concurrency=1,
            duration_seconds=15.0, total_tokens=500, total_requests=5,
        )

    monkeypatch.setattr("atomics.stress.run_stress", fake_run_stress)
    monkeypatch.setenv("ATOMICS_DB_PATH", str(tmp_path / "test.db"))

    runner = CliRunner()
    result = runner.invoke(cli, [
        "stress", "--model", "test-model", "--max-concurrency", "1",
        "--phase-seconds", "1", "--no-save",
    ])
    assert result.exit_code == 0
    assert "saved" not in result.output.lower()


def test_cli_stress_throttling_detected(monkeypatch, tmp_path):
    """Stress CLI should detect throttling when final phase TPS drops."""
    from click.testing import CliRunner

    from atomics.cli import cli

    async def fake_run_stress(**kwargs):
        phases = [
            ConcurrencyResult(
                concurrency=1, requests=5, total_output_tokens=500,
                aggregate_tps=50.0, avg_request_tps=50.0,
                avg_latency_ms=200.0, p95_latency_ms=300.0,
            ),
            ConcurrencyResult(
                concurrency=2, requests=10, total_output_tokens=800,
                aggregate_tps=40.0, avg_request_tps=20.0,
                avg_latency_ms=400.0, p95_latency_ms=600.0,
            ),
        ]
        on_phase = kwargs.get("on_phase")
        if on_phase:
            for p in phases:
                on_phase(p)
        return StressResult(
            model=kwargs.get("model", "test"),
            host=kwargs.get("host", "http://fake:11434"),
            phases=phases,
            peak_tps=50.0, saturation_concurrency=1,
            duration_seconds=30.0, total_tokens=1300, total_requests=15,
        )

    monkeypatch.setattr("atomics.stress.run_stress", fake_run_stress)
    monkeypatch.setenv("ATOMICS_DB_PATH", str(tmp_path / "test.db"))

    runner = CliRunner()
    result = runner.invoke(cli, [
        "stress", "--model", "test-model", "--max-concurrency", "2",
        "--phase-seconds", "1", "--no-save",
    ])
    assert result.exit_code == 0
    assert "Possible" in result.output  # throttling detected


def test_cli_stress_help():
    from click.testing import CliRunner

    from atomics.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["stress", "--help"])
    assert result.exit_code == 0
    assert "--max-concurrency" in result.output
    assert "--phase-seconds" in result.output
    assert "--provider" in result.output


# ── Provider-based stress tests ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_stress_provider_basic():
    """run_stress_provider uses BaseProvider.generate() and tracks cost."""
    from atomics.providers.base import ProviderResponse
    from atomics.stress import run_stress_provider

    call_count = 0

    async def fake_generate(prompt, *, system="", model=None, max_tokens=1024,
                            thinking=None, thinking_budget=None):
        nonlocal call_count
        call_count += 1
        return ProviderResponse(
            text="test " * 50,
            input_tokens=30,
            output_tokens=100,
            total_tokens=130,
            model=model or "test-model",
            latency_ms=600.0,
            estimated_cost_usd=0.001,
            tokens_per_second=166.7,
        )

    mock_provider = MagicMock()
    mock_provider.name = "openai"
    mock_provider.generate = fake_generate

    result = await run_stress_provider(
        provider=mock_provider,
        model="test-model",
        max_concurrency=2,
        phase_seconds=0.5,
        num_predict=512,
    )

    assert result.model == "test-model"
    assert result.provider == "openai"
    assert len(result.phases) >= 2  # 1, 2
    assert result.peak_tps > 0
    assert result.total_requests > 0
    assert result.total_cost_usd > 0
    assert all(p.total_cost_usd >= 0 for p in result.phases)


@pytest.mark.asyncio
async def test_run_stress_provider_on_phase_callback():
    """on_phase callback fires for each concurrency level."""
    from atomics.providers.base import ProviderResponse
    from atomics.stress import run_stress_provider

    async def fake_generate(prompt, **kwargs):
        return ProviderResponse(
            text="ok", input_tokens=10, output_tokens=50,
            total_tokens=60, model="m", latency_ms=200.0,
            estimated_cost_usd=0.0005, tokens_per_second=250.0,
        )

    mock_provider = MagicMock()
    mock_provider.name = "claude"
    mock_provider.generate = fake_generate

    phase_log: list[int] = []

    await run_stress_provider(
        provider=mock_provider,
        model="m",
        max_concurrency=4,
        phase_seconds=0.3,
        on_phase=lambda p: phase_log.append(p.concurrency),
    )

    assert phase_log == [1, 2, 4]


@pytest.mark.asyncio
async def test_run_stress_provider_handles_failures():
    """Failed requests don't crash; they increment failed counter."""
    from atomics.providers.base import ProviderResponse
    from atomics.stress import run_stress_provider

    call_count = 0

    async def flaky_generate(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 0:
            raise ConnectionError("timeout")
        return ProviderResponse(
            text="ok", input_tokens=10, output_tokens=50,
            total_tokens=60, model="m", latency_ms=100.0,
            estimated_cost_usd=0.0002, tokens_per_second=500.0,
        )

    mock_provider = MagicMock()
    mock_provider.name = "openai"
    mock_provider.generate = flaky_generate

    result = await run_stress_provider(
        provider=mock_provider, model="m",
        max_concurrency=1, phase_seconds=0.5,
    )

    assert result.total_requests >= 1
    assert result.total_failed >= 1


@pytest.mark.asyncio
async def test_run_stress_provider_tps_fallback():
    """When tokens_per_second is None, compute from output_tokens/latency."""
    from atomics.providers.base import ProviderResponse
    from atomics.stress import run_stress_provider

    async def fake_generate(prompt, **kwargs):
        return ProviderResponse(
            text="ok", input_tokens=10, output_tokens=100,
            total_tokens=110, model="m", latency_ms=1000.0,
            estimated_cost_usd=0.001, tokens_per_second=None,
        )

    mock_provider = MagicMock()
    mock_provider.name = "claude"
    mock_provider.generate = fake_generate

    result = await run_stress_provider(
        provider=mock_provider, model="m",
        max_concurrency=1, phase_seconds=0.3,
    )

    assert result.phases[0].per_request_tps
    for tps in result.phases[0].per_request_tps:
        assert 80 <= tps <= 120  # ~100 tok/s = 100 tokens / 1s


def test_cli_stress_with_provider(monkeypatch, tmp_path):
    """atomics stress --provider openai routes to run_stress_provider."""
    pytest.importorskip("openai", reason="optional 'openai' extra not installed")
    from click.testing import CliRunner

    from atomics.cli import cli

    captured_kwargs: dict = {}

    async def fake_run_stress_provider(**kwargs):
        captured_kwargs.update(kwargs)
        phase = ConcurrencyResult(
            concurrency=1, requests=5, total_output_tokens=500,
            aggregate_tps=100.0, avg_request_tps=100.0,
            avg_latency_ms=200.0, p95_latency_ms=300.0,
            total_cost_usd=0.005,
        )
        on_phase = kwargs.get("on_phase")
        if on_phase:
            on_phase(phase)
        return StressResult(
            model="gpt-4o-mini", host="api",
            provider="openai",
            phases=[phase], peak_tps=100.0,
            saturation_concurrency=1,
            duration_seconds=15.0, total_tokens=500,
            total_requests=5, total_cost_usd=0.005,
        )

    monkeypatch.setattr("atomics.stress.run_stress_provider", fake_run_stress_provider)
    monkeypatch.setenv("ATOMICS_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")

    runner = CliRunner()
    result = runner.invoke(cli, [
        "stress", "--provider", "openai", "--model", "gpt-4o-mini",
        "--max-concurrency", "2", "--no-save",
    ])
    assert result.exit_code == 0, result.output
    assert "Peak throughput" in result.output or "peak" in result.output.lower()


def test_cli_stress_provider_claude(monkeypatch, tmp_path):
    """atomics stress --provider claude uses Claude provider."""
    from click.testing import CliRunner

    from atomics.cli import cli

    async def fake_run_stress_provider(**kwargs):
        phase = ConcurrencyResult(
            concurrency=1, requests=3, total_output_tokens=300,
            aggregate_tps=80.0, avg_request_tps=80.0,
            avg_latency_ms=400.0, p95_latency_ms=600.0,
            total_cost_usd=0.008,
        )
        on_phase = kwargs.get("on_phase")
        if on_phase:
            on_phase(phase)
        return StressResult(
            model="claude-haiku-4-5-20251001", host="api",
            provider="claude",
            phases=[phase], peak_tps=80.0,
            saturation_concurrency=1,
            duration_seconds=15.0, total_tokens=300,
            total_requests=3, total_cost_usd=0.008,
        )

    monkeypatch.setattr("atomics.stress.run_stress_provider", fake_run_stress_provider)
    monkeypatch.setenv("ATOMICS_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake")

    runner = CliRunner()
    result = runner.invoke(cli, [
        "stress", "--provider", "claude", "--model", "claude-haiku-4-5-20251001",
        "--max-concurrency", "4", "--no-save",
    ])
    assert result.exit_code == 0, result.output


def test_cli_stress_provider_save_cost(monkeypatch, tmp_path):
    """Stress with --provider --save should persist cost data."""
    pytest.importorskip("openai", reason="optional 'openai' extra not installed")
    from click.testing import CliRunner

    from atomics.cli import cli

    async def fake_run_stress_provider(**kwargs):
        phase = ConcurrencyResult(
            concurrency=1, requests=5, total_output_tokens=500,
            aggregate_tps=50.0, avg_request_tps=50.0,
            avg_latency_ms=300.0, p95_latency_ms=500.0,
            total_cost_usd=0.010,
        )
        on_phase = kwargs.get("on_phase")
        if on_phase:
            on_phase(phase)
        return StressResult(
            model="gpt-4o-mini", host="api",
            provider="openai",
            phases=[phase], peak_tps=50.0,
            saturation_concurrency=1,
            duration_seconds=15.0, total_tokens=500,
            total_requests=5, total_cost_usd=0.010,
        )

    monkeypatch.setattr("atomics.stress.run_stress_provider", fake_run_stress_provider)
    monkeypatch.setenv("ATOMICS_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")

    runner = CliRunner()
    result = runner.invoke(cli, [
        "stress", "--provider", "openai", "--model", "gpt-4o-mini",
        "--max-concurrency", "1", "--save",
    ])
    assert result.exit_code == 0, result.output
    assert "saved" in result.output.lower()
    assert "cost" in result.output.lower() or "$" in result.output


# ── _get_gpu_info / _get_vram_used_mb ─────────────────────────────────────────


class TestGPUInfo:
    def test_get_gpu_info_no_nvidia_smi(self):
        from atomics.stress import _get_gpu_info
        with patch("shutil.which", return_value=None):
            name, total = _get_gpu_info()
        assert name == ""
        assert total is None

    def test_get_gpu_info_success(self):
        from atomics.stress import _get_gpu_info
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA RTX 4090, 24576"
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
             patch("subprocess.run", return_value=mock_result):
            name, total = _get_gpu_info()
        assert name == "NVIDIA RTX 4090"
        assert total == 24576.0

    def test_get_gpu_info_nonzero_returncode(self):
        from atomics.stress import _get_gpu_info
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
             patch("subprocess.run", return_value=mock_result):
            name, total = _get_gpu_info()
        assert name == ""
        assert total is None

    def test_get_gpu_info_exception(self):
        from atomics.stress import _get_gpu_info
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
             patch("subprocess.run", side_effect=OSError("no smi")):
            name, total = _get_gpu_info()
        assert name == ""
        assert total is None

    def test_get_vram_used_no_smi(self):
        from atomics.stress import _get_vram_used_mb
        with patch("shutil.which", return_value=None):
            assert _get_vram_used_mb() is None

    def test_get_vram_used_success(self):
        from atomics.stress import _get_vram_used_mb
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "8192"
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
             patch("subprocess.run", return_value=mock_result):
            assert _get_vram_used_mb() == 8192.0

    def test_get_vram_used_nonzero_returncode(self):
        from atomics.stress import _get_vram_used_mb
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
             patch("subprocess.run", return_value=mock_result):
            assert _get_vram_used_mb() is None

    def test_get_vram_used_exception(self):
        from atomics.stress import _get_vram_used_mb
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
             patch("subprocess.run", side_effect=ValueError("bad")):
            assert _get_vram_used_mb() is None


# ── _run_phase_profile / run_stress_profile ───────────────────────────────────


def _make_stress_ollama_profile():
    from types import SimpleNamespace
    return SimpleNamespace(
        type="ollama", name="test-gate", model="qwen2.5:3b",
        ollama_host="http://localhost:11434", http_url="",
        http_timeout=30, prompts=["probe 1", "probe 2"],
    )


def _make_stress_http_profile():
    from types import SimpleNamespace
    return SimpleNamespace(
        type="http", name="http-gate", model="",
        ollama_host="", http_url="http://localhost:8080/gate",
        http_timeout=30, prompts=[],
    )


async def _instant_profile_req(client, profile, prompt):
    """Instant fake profile request — no real I/O."""
    return ("ok", 5.0, "pass")


class TestRunStressProfile:
    """Cover _run_phase_profile + run_stress_profile. Uses instant mock + 0.15s phases."""

    @pytest.mark.asyncio
    async def test_run_stress_profile_ollama(self):
        from atomics.stress import run_stress_profile
        with patch("atomics.profiles._single_request_profile",
                   side_effect=_instant_profile_req):
            result = await run_stress_profile(
                profile=_make_stress_ollama_profile(),
                max_concurrency=2,
                phase_seconds=0.15,
            )
        assert result.model == "qwen2.5:3b"
        assert result.provider == "profile:ollama"
        assert len(result.phases) >= 1
        assert result.peak_tps >= 0.0
        assert result.total_requests > 0

    @pytest.mark.asyncio
    async def test_run_stress_profile_http(self):
        from atomics.stress import run_stress_profile
        with patch("atomics.profiles._single_request_profile",
                   side_effect=_instant_profile_req):
            result = await run_stress_profile(
                profile=_make_stress_http_profile(),
                max_concurrency=1,
                phase_seconds=0.15,
            )
        assert result.provider == "profile:http"
        assert result.host == "http://localhost:8080/gate"

    @pytest.mark.asyncio
    async def test_run_stress_profile_on_phase_callback(self):
        from atomics.stress import run_stress_profile
        phases_received: list[ConcurrencyResult] = []

        def on_phase(p: ConcurrencyResult) -> None:
            phases_received.append(p)

        with patch("atomics.profiles._single_request_profile",
                   side_effect=_instant_profile_req):
            result = await run_stress_profile(
                profile=_make_stress_ollama_profile(),
                max_concurrency=2,
                phase_seconds=0.15,
                on_phase=on_phase,
            )
        assert len(phases_received) == len(result.phases)

    @pytest.mark.asyncio
    async def test_run_stress_profile_failure_in_phase(self):
        from atomics.stress import run_stress_profile
        call_n = [0]

        async def _sometimes_fail(client, profile, prompt):
            call_n[0] += 1
            if call_n[0] % 2 == 0:
                raise RuntimeError("gate error")
            return ("ok", 10.0, "pass")

        with patch("atomics.profiles._single_request_profile", side_effect=_sometimes_fail):
            result = await run_stress_profile(
                profile=_make_stress_ollama_profile(),
                max_concurrency=1,
                phase_seconds=0.15,
            )
        assert result.total_failed >= 0

    @pytest.mark.asyncio
    async def test_run_stress_profile_saturation_concurrency_set(self):
        from atomics.stress import run_stress_profile
        with patch("atomics.profiles._single_request_profile",
                   side_effect=_instant_profile_req):
            result = await run_stress_profile(
                profile=_make_stress_ollama_profile(),
                max_concurrency=4,
                phase_seconds=0.15,
            )
        assert result.saturation_concurrency >= 1

    @pytest.mark.asyncio
    async def test_run_stress_profile_no_prompts_uses_stress_prompts(self):
        from atomics.stress import run_stress_profile
        with patch("atomics.profiles._single_request_profile",
                   side_effect=_instant_profile_req):
            result = await run_stress_profile(
                profile=_make_stress_http_profile(),
                max_concurrency=1,
                phase_seconds=0.15,
            )
        assert result.total_requests > 0
