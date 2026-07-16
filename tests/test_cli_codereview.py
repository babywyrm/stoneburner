"""CLI contract tests for secure code review."""

from __future__ import annotations

import json
from types import SimpleNamespace

from click.testing import CliRunner

from atomics.cli import cli


class _Summary:
    detection_rate = 1.0
    false_positive_rate = 0.0
    review_score = 1.0

    def __init__(self) -> None:
        self.results = [
            SimpleNamespace(
                fixture=SimpleNamespace(
                    id="cr-characterization",
                    is_vulnerable=True,
                    cwe="CWE-79",
                    mode="snippet",
                ),
                verdict="detected",
            )
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "detection_rate": self.detection_rate,
            "false_positive_rate": self.false_positive_rate,
            "review_score": self.review_score,
            "results": [{"id": "cr-characterization"}],
        }


def _patch_codereview(monkeypatch) -> None:
    provider = SimpleNamespace(name="mock", default_model="mock-model")

    async def fake_run_codereview(*_args, **kwargs):
        summary = _Summary()
        callback = kwargs.get("on_fixture_done")
        if callback is not None:
            callback(summary.results[0])
        return summary

    monkeypatch.setattr(
        "atomics.cli._make_provider",
        lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr("atomics.eval.codereview.run_codereview", fake_run_codereview)


def test_codereview_help_preserves_public_options() -> None:
    result = CliRunner().invoke(cli, ["codereview", "--help"])

    assert result.exit_code == 0
    for option in (
        "--provider",
        "--model",
        "--ollama-host",
        "--vllm-host",
        "--judge-provider",
        "--judge-model",
        "--judge-host",
        "--json-out",
    ):
        assert option in result.output


def test_codereview_invocation_preserves_output(monkeypatch) -> None:
    _patch_codereview(monkeypatch)

    result = CliRunner().invoke(cli, ["codereview"])

    assert result.exit_code == 0
    assert "Secure Code Review Summary" in result.output
    assert "cr-characterization" in result.output
    assert "100.0%" in result.output


def test_codereview_json_preserves_results_key(monkeypatch, tmp_path) -> None:
    _patch_codereview(monkeypatch)
    output = tmp_path / "codereview.json"

    result = CliRunner().invoke(cli, ["codereview", "--json-out", str(output)])

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["review_score"] == 1.0
    assert payload["results"] == [{"id": "cr-characterization"}]
