"""CLI contract tests for refusal calibration."""

from __future__ import annotations

import json
from types import SimpleNamespace

from click.testing import CliRunner

from atomics.cli import cli


class _Summary:
    calibration_score = 1.0
    over_refusal_rate = 0.0
    under_refusal_rate = 0.0

    def __init__(self) -> None:
        self.results = [
            SimpleNamespace(
                fixture=SimpleNamespace(id="ref-characterization", expected="comply"),
                classification="comply",
                correct=True,
                over_refusal=False,
                under_refusal=False,
            )
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "calibration_score": self.calibration_score,
            "over_refusal_rate": self.over_refusal_rate,
            "under_refusal_rate": self.under_refusal_rate,
            "results": [{"id": "ref-characterization"}],
        }


def _patch_refusal(monkeypatch) -> None:
    provider = SimpleNamespace(name="mock", default_model="mock-model")

    async def fake_run_refusal(*_args, **kwargs):
        summary = _Summary()
        callback = kwargs.get("on_fixture_done")
        if callback is not None:
            callback(summary.results[0])
        return summary

    monkeypatch.setattr(
        "atomics.cli._make_provider",
        lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr("atomics.eval.refusal.run_refusal", fake_run_refusal)


def test_refusal_help_preserves_public_options() -> None:
    result = CliRunner().invoke(cli, ["refusal", "--help"])

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


def test_refusal_invocation_preserves_output(monkeypatch) -> None:
    _patch_refusal(monkeypatch)

    result = CliRunner().invoke(cli, ["refusal"])

    assert result.exit_code == 0
    assert "Refusal Calibration Summary" in result.output
    assert "ref-characterization" in result.output
    assert "100.0%" in result.output


def test_refusal_json_preserves_results_key(monkeypatch, tmp_path) -> None:
    _patch_refusal(monkeypatch)
    output = tmp_path / "refusal.json"

    result = CliRunner().invoke(cli, ["refusal", "--json-out", str(output)])

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["calibration_score"] == 1.0
    assert payload["results"] == [{"id": "ref-characterization"}]
