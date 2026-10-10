from unittest.mock import patch

import pytest
from click.testing import CliRunner

from atomics.archreview.models import ArchReviewResult
from atomics.cli import cli


def _fake_results():
    return [
        ArchReviewResult(
            run_id="",
            repo="juice-shop",
            tier="floor",
            model="qwen2.5:14b",
            provider="ollama",
            round=1,
            findings=[],
            objective_recall=0.71,
            objective_precision=0.74,
            objective_f=0.72,
            judge_score=0.77,
            matched_categories=["injection"],
        ),
    ]


def _recording_ollama(built: list):
    """An OllamaProvider stand-in that records each instance it builds."""

    class _FakeOllamaProvider:
        name = "ollama"

        def __init__(self, *, host, default_model, timeout, context_tokens=None):
            self._default_model = default_model
            self._context_tokens = context_tokens
            self._timeout = timeout
            built.append(self)

        @property
        def default_model(self):
            return self._default_model

    return _FakeOllamaProvider


def _verbose_results():
    from atomics.archreview.models import Finding

    return [
        ArchReviewResult(
            run_id="",
            repo="juice-shop",
            tier="floor",
            model="qwen2.5:14b",
            provider="ollama",
            round=1,
            findings=[Finding("injection", "routes/search.ts", "high", "raw sql")],
            objective_recall=0.6,
            objective_precision=1.0,
            objective_f=0.75,
            judge_score=0.7,
            matched_categories=["injection"],
        ),
    ]


def test_archreview_cli_runs_and_prints_table(tmp_path, monkeypatch):
    monkeypatch.setenv("JUICE_SHOP_PATH", str(tmp_path))
    (tmp_path / "server.ts").write_text("// app\n")

    runner = CliRunner()
    with patch("atomics.archreview.runner.run_archreview") as m:

        async def _shim(**kwargs):
            return _fake_results()

        m.side_effect = _shim
        result = runner.invoke(
            cli,
            [
                "archreview",
                "--repo",
                "juice-shop",
                "--models",
                "qwen2.5:14b",
                "--provider",
                "ollama",
                "--judge-provider",
                "ollama",
                "--judge-model",
                "deepseek-r1:14b",
                "--tier",
                "floor",
                "--no-save",
            ],
        )
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
        result = runner.invoke(
            cli,
            [
                "archreview",
                "--repo",
                "juice-shop",
                "--models",
                "qwen2.5:14b",
                "--provider",
                "ollama",
                "--judge-provider",
                "ollama",
                "--judge-model",
                "deepseek-r1:7b",
                "--tier",
                "floor",
                "--verbose",
                "--no-save",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "analyzing with" in result.output
    assert "round 1" in result.output
    assert "routes/search.ts" in result.output  # per-finding detail printed


@pytest.mark.parametrize(
    ("model", "tier", "contexts", "shown"),
    [
        ("qwen2.5:14b", "floor", [8192, 22144], ["context=22144", "reserve=2048"]),
        ("qwen3.5:4b", "expanded", [134144], ["context=134144", "reserve=2048"]),
        ("qwen3.5:4b", "wide", [54144], ["tier=wide", "context=54144"]),
        ("qwen3.5:4b", "local", [38144], ["tier=local", "context=38144"]),
    ],
)
def test_archreview_cli_sizes_ollama_context_per_tier(
    tmp_path, monkeypatch, model, tier, contexts, shown
):
    monkeypatch.setenv("JUICE_SHOP_PATH", str(tmp_path))
    (tmp_path / "server.ts").write_text("// app\n")

    built = []

    _FakeOllamaProvider = _recording_ollama(built)

    async def _shim(**kwargs):
        return _fake_results()

    with (
        patch("atomics.providers.ollama.OllamaProvider", _FakeOllamaProvider),
        patch("atomics.archreview.runner.run_archreview", side_effect=_shim),
    ):
        result = CliRunner().invoke(
            cli,
            [
                "archreview",
                "--repo",
                "juice-shop",
                "--models",
                model,
                "--provider",
                "ollama",
                "--judge-provider",
                "ollama",
                "--judge-model",
                "deepseek-r1:7b",
                "--tier",
                tier,
                "--no-save",
            ],
        )
    assert result.exit_code == 0, result.output
    assert [p._context_tokens for p in built][-len(contexts) :] == contexts
    for text in shown:
        assert text in result.output


def test_archreview_cli_max_output_tokens_adjusts_reserve_and_runner_arg(tmp_path, monkeypatch):
    monkeypatch.setenv("JUICE_SHOP_PATH", str(tmp_path))
    (tmp_path / "server.ts").write_text("// app\n")

    built = []
    calls = []

    _FakeOllamaProvider = _recording_ollama(built)

    runner = CliRunner()
    with patch("atomics.providers.ollama.OllamaProvider", _FakeOllamaProvider):
        with patch("atomics.archreview.runner.run_archreview") as m:

            async def _shim(**kwargs):
                calls.append(kwargs)
                return _fake_results()

            m.side_effect = _shim
            result = runner.invoke(
                cli,
                [
                    "archreview",
                    "--repo",
                    "juice-shop",
                    "--models",
                    "mistral-small3.2:24b",
                    "--provider",
                    "ollama",
                    "--judge-provider",
                    "ollama",
                    "--judge-model",
                    "deepseek-r1:7b",
                    "--tier",
                    "wide",
                    "--max-output-tokens",
                    "512",
                    "--no-save",
                ],
            )
    assert result.exit_code == 0, result.output
    assert built[-1]._context_tokens == 52608
    assert calls[0]["max_output_tokens"] == 512
    assert "context=52608" in result.output
    assert "reserve=512" in result.output


def test_archreview_cli_inference_timeout_overrides_ollama_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("JUICE_SHOP_PATH", str(tmp_path))
    (tmp_path / "server.ts").write_text("// app\n")

    built = []

    _FakeOllamaProvider = _recording_ollama(built)

    runner = CliRunner()
    with patch("atomics.providers.ollama.OllamaProvider", _FakeOllamaProvider):
        with patch("atomics.archreview.runner.run_archreview") as m:

            async def _shim(**kwargs):
                return _fake_results()

            m.side_effect = _shim
            result = runner.invoke(
                cli,
                [
                    "archreview",
                    "--repo",
                    "juice-shop",
                    "--models",
                    "mistral-small3.2:24b",
                    "--provider",
                    "ollama",
                    "--judge-provider",
                    "ollama",
                    "--judge-model",
                    "deepseek-r1:7b",
                    "--tier",
                    "wide",
                    "--inference-timeout",
                    "900",
                    "--no-save",
                ],
            )
    assert result.exit_code == 0, result.output
    assert [p._timeout for p in built] == [900.0, 900.0]


def test_archreview_cli_unknown_repo_errors():
    runner = CliRunner()
    result = runner.invoke(
        cli, ["archreview", "--repo", "does-not-exist", "--models", "m", "--no-save"]
    )
    assert result.exit_code != 0
    assert "does-not-exist" in result.output


def test_archreview_cli_rejects_path_traversal():
    """Repo names with path separators or '..' are rejected."""
    runner = CliRunner()
    for bad_name in ["../../etc/passwd", "../cli", "foo/bar", "a\\b"]:
        result = runner.invoke(
            cli, ["archreview", "--repo", bad_name, "--models", "m", "--no-save"]
        )
        assert result.exit_code != 0, f"expected failure for {bad_name!r}"
        assert "Invalid repo name" in result.output, f"bad message for {bad_name!r}"


def test_archreview_runs_is_alias_for_rounds():
    """--runs is accepted as an alias for --rounds (cross-suite consistency)."""
    from unittest.mock import patch

    from atomics.archreview.models import ArchReviewResult

    captured = {}

    async def _shim(**kwargs):
        captured["rounds"] = kwargs.get("rounds")
        return [
            ArchReviewResult(
                run_id="r",
                repo="juice-shop",
                tier="floor",
                model="m",
                provider="ollama",
                round=0,
                findings=[],
            )
        ]

    runner = CliRunner()
    with patch("atomics.archreview.runner.run_archreview", side_effect=_shim):
        result = runner.invoke(
            cli,
            [
                "archreview",
                "--repo",
                "juice-shop",
                "--models",
                "m",
                "--provider",
                "ollama",
                "--runs",
                "2",
                "--judge-only",
                "--no-save",
            ],
        )
    # It should parse --runs into rounds=2 (or fail later for env reasons, not on the flag).
    assert "no such option" not in result.output.lower()
    if captured:
        assert captured["rounds"] == 2


def test_cli_archreview_extra_judges_option():
    result = CliRunner().invoke(cli, ["archreview", "--help"])
    assert result.exit_code == 0
    assert "--extra-judges" in result.output
