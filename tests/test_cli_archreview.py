from unittest.mock import patch

from click.testing import CliRunner

from atomics.archreview.models import ArchReviewResult
from atomics.cli import cli


def _fake_results():
    return [
        ArchReviewResult(run_id="", repo="juice-shop", tier="floor",
                         model="qwen2.5:14b", provider="ollama", round=1,
                         findings=[], objective_recall=0.71, objective_precision=0.74,
                         objective_f=0.72, judge_score=0.77, matched_categories=["injection"]),
    ]


def _verbose_results():
    from atomics.archreview.models import Finding
    return [
        ArchReviewResult(run_id="", repo="juice-shop", tier="floor",
                         model="qwen2.5:14b", provider="ollama", round=1,
                         findings=[Finding("injection", "routes/search.ts", "high", "raw sql")],
                         objective_recall=0.6, objective_precision=1.0, objective_f=0.75,
                         judge_score=0.7, matched_categories=["injection"]),
    ]


def test_archreview_cli_runs_and_prints_table(tmp_path, monkeypatch):
    monkeypatch.setenv("JUICE_SHOP_PATH", str(tmp_path))
    (tmp_path / "server.ts").write_text("// app\n")

    runner = CliRunner()
    with patch("atomics.archreview.runner.run_archreview") as m:
        async def _shim(**kwargs):
            return _fake_results()
        m.side_effect = _shim
        result = runner.invoke(cli, [
            "archreview", "--repo", "juice-shop",
            "--models", "qwen2.5:14b", "--provider", "ollama",
            "--judge-provider", "ollama", "--judge-model", "deepseek-r1:14b",
            "--tier", "floor", "--no-save",
        ])
    assert result.exit_code == 0, result.output
    assert "juice-shop" in result.output
    assert "qwen2.5:14b" in result.output
    assert "deepseek-r1:14b" in result.output
    assert "Judge Model" in result.output


def test_archreview_cli_verbose_streams_findings(tmp_path, monkeypatch):
    monkeypatch.setenv("JUICE_SHOP_PATH", str(tmp_path))
    (tmp_path / "server.ts").write_text("// app\n")

    runner = CliRunner()
    with patch("atomics.archreview.runner.run_archreview") as m:
        async def _shim(**kwargs):
            return _verbose_results()
        m.side_effect = _shim
        result = runner.invoke(cli, [
            "archreview", "--repo", "juice-shop",
            "--models", "qwen2.5:14b", "--provider", "ollama",
            "--judge-provider", "ollama", "--judge-model", "deepseek-r1:7b",
            "--tier", "floor", "--verbose", "--no-save",
        ])
    assert result.exit_code == 0, result.output
    assert "analyzing with" in result.output
    assert "round 1" in result.output
    assert "routes/search.ts" in result.output  # per-finding detail printed


def test_archreview_cli_passes_larger_ollama_context(tmp_path, monkeypatch):
    monkeypatch.setenv("JUICE_SHOP_PATH", str(tmp_path))
    (tmp_path / "server.ts").write_text("// app\n")

    built = []

    class _FakeOllamaProvider:
        name = "ollama"

        def __init__(self, *, host, default_model, timeout, context_tokens=None):
            self._host = host
            self._default_model = default_model
            self._context_tokens = context_tokens
            built.append(self)

        @property
        def default_model(self):
            return self._default_model

    runner = CliRunner()
    with patch("atomics.providers.ollama.OllamaProvider", _FakeOllamaProvider):
        with patch("atomics.archreview.runner.run_archreview") as m:
            async def _shim(**kwargs):
                return _fake_results()
            m.side_effect = _shim
            result = runner.invoke(cli, [
                "archreview", "--repo", "juice-shop",
                "--models", "qwen2.5:14b", "--provider", "ollama",
                "--judge-provider", "ollama", "--judge-model", "deepseek-r1:7b",
                "--tier", "floor", "--no-save",
            ])
    assert result.exit_code == 0, result.output
    assert [p._context_tokens for p in built] == [8192, 22144]
    assert "context=22144" in result.output
    assert "reserve=2048" in result.output


def test_archreview_cli_expanded_context_reserves_output_room(tmp_path, monkeypatch):
    monkeypatch.setenv("JUICE_SHOP_PATH", str(tmp_path))
    (tmp_path / "server.ts").write_text("// app\n")

    built = []

    class _FakeOllamaProvider:
        name = "ollama"

        def __init__(self, *, host, default_model, timeout, context_tokens=None):
            self._default_model = default_model
            self._context_tokens = context_tokens
            built.append(self)

        @property
        def default_model(self):
            return self._default_model

    runner = CliRunner()
    with patch("atomics.providers.ollama.OllamaProvider", _FakeOllamaProvider):
        with patch("atomics.archreview.runner.run_archreview") as m:
            async def _shim(**kwargs):
                return _fake_results()
            m.side_effect = _shim
            result = runner.invoke(cli, [
                "archreview", "--repo", "juice-shop",
                "--models", "qwen3.5:4b", "--provider", "ollama",
                "--judge-provider", "ollama", "--judge-model", "deepseek-r1:7b",
                "--tier", "expanded", "--no-save",
            ])
    assert result.exit_code == 0, result.output
    assert built[-1]._context_tokens == 134144
    assert "context=134144" in result.output
    assert "reserve=2048" in result.output


def test_archreview_cli_accepts_wide_tier_for_local_models(tmp_path, monkeypatch):
    monkeypatch.setenv("JUICE_SHOP_PATH", str(tmp_path))
    (tmp_path / "server.ts").write_text("// app\n")

    built = []

    class _FakeOllamaProvider:
        name = "ollama"

        def __init__(self, *, host, default_model, timeout, context_tokens=None):
            self._default_model = default_model
            self._context_tokens = context_tokens
            built.append(self)

        @property
        def default_model(self):
            return self._default_model

    runner = CliRunner()
    with patch("atomics.providers.ollama.OllamaProvider", _FakeOllamaProvider):
        with patch("atomics.archreview.runner.run_archreview") as m:
            async def _shim(**kwargs):
                return _fake_results()
            m.side_effect = _shim
            result = runner.invoke(cli, [
                "archreview", "--repo", "juice-shop",
                "--models", "qwen3.5:4b", "--provider", "ollama",
                "--judge-provider", "ollama", "--judge-model", "deepseek-r1:7b",
                "--tier", "wide", "--no-save",
            ])
    assert result.exit_code == 0, result.output
    assert built[-1]._context_tokens == 54144
    assert "tier=wide" in result.output
    assert "context=54144" in result.output


def test_archreview_cli_accepts_local_tier_for_brainbox_models(tmp_path, monkeypatch):
    monkeypatch.setenv("JUICE_SHOP_PATH", str(tmp_path))
    (tmp_path / "server.ts").write_text("// app\n")

    built = []

    class _FakeOllamaProvider:
        name = "ollama"

        def __init__(self, *, host, default_model, timeout, context_tokens=None):
            self._default_model = default_model
            self._context_tokens = context_tokens
            built.append(self)

        @property
        def default_model(self):
            return self._default_model

    runner = CliRunner()
    with patch("atomics.providers.ollama.OllamaProvider", _FakeOllamaProvider):
        with patch("atomics.archreview.runner.run_archreview") as m:
            async def _shim(**kwargs):
                return _fake_results()
            m.side_effect = _shim
            result = runner.invoke(cli, [
                "archreview", "--repo", "juice-shop",
                "--models", "qwen3.5:4b", "--provider", "ollama",
                "--judge-provider", "ollama", "--judge-model", "deepseek-r1:7b",
                "--tier", "local", "--no-save",
            ])
    assert result.exit_code == 0, result.output
    assert built[-1]._context_tokens == 38144
    assert "tier=local" in result.output
    assert "context=38144" in result.output


def test_archreview_cli_max_output_tokens_adjusts_reserve_and_runner_arg(tmp_path, monkeypatch):
    monkeypatch.setenv("JUICE_SHOP_PATH", str(tmp_path))
    (tmp_path / "server.ts").write_text("// app\n")

    built = []
    calls = []

    class _FakeOllamaProvider:
        name = "ollama"

        def __init__(self, *, host, default_model, timeout, context_tokens=None):
            self._default_model = default_model
            self._context_tokens = context_tokens
            built.append(self)

        @property
        def default_model(self):
            return self._default_model

    runner = CliRunner()
    with patch("atomics.providers.ollama.OllamaProvider", _FakeOllamaProvider):
        with patch("atomics.archreview.runner.run_archreview") as m:
            async def _shim(**kwargs):
                calls.append(kwargs)
                return _fake_results()
            m.side_effect = _shim
            result = runner.invoke(cli, [
                "archreview", "--repo", "juice-shop",
                "--models", "mistral-small3.2:24b", "--provider", "ollama",
                "--judge-provider", "ollama", "--judge-model", "deepseek-r1:7b",
                "--tier", "wide", "--max-output-tokens", "512", "--no-save",
            ])
    assert result.exit_code == 0, result.output
    assert built[-1]._context_tokens == 52608
    assert calls[0]["max_output_tokens"] == 512
    assert "context=52608" in result.output
    assert "reserve=512" in result.output


def test_archreview_cli_inference_timeout_overrides_ollama_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("JUICE_SHOP_PATH", str(tmp_path))
    (tmp_path / "server.ts").write_text("// app\n")

    built = []

    class _FakeOllamaProvider:
        name = "ollama"

        def __init__(self, *, host, default_model, timeout, context_tokens=None):
            self._default_model = default_model
            self._timeout = timeout
            built.append(self)

        @property
        def default_model(self):
            return self._default_model

    runner = CliRunner()
    with patch("atomics.providers.ollama.OllamaProvider", _FakeOllamaProvider):
        with patch("atomics.archreview.runner.run_archreview") as m:
            async def _shim(**kwargs):
                return _fake_results()
            m.side_effect = _shim
            result = runner.invoke(cli, [
                "archreview", "--repo", "juice-shop",
                "--models", "mistral-small3.2:24b", "--provider", "ollama",
                "--judge-provider", "ollama", "--judge-model", "deepseek-r1:7b",
                "--tier", "wide", "--inference-timeout", "900", "--no-save",
            ])
    assert result.exit_code == 0, result.output
    assert [p._timeout for p in built] == [900.0, 900.0]


def test_archreview_cli_unknown_repo_errors():
    runner = CliRunner()
    result = runner.invoke(cli, ["archreview", "--repo", "does-not-exist",
                                 "--models", "m", "--no-save"])
    assert result.exit_code != 0
    assert "does-not-exist" in result.output
