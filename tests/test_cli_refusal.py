"""CLI contract tests for refusal calibration."""

from __future__ import annotations

import json
from types import SimpleNamespace

from click.testing import CliRunner

from atomics.cli import cli
from atomics.eval.outcomes import (
    AttemptResult,
    JudgeOutcome,
    JudgeOutcomeStatus,
    ProviderOutcome,
    ProviderOutcomeKind,
    RunIntegrity,
)


class _Summary:
    calibration_score = 1.0
    over_refusal_rate = 0.0
    under_refusal_rate = 0.0

    def __init__(self, *, partial: bool = False) -> None:
        self.results = [
            SimpleNamespace(
                fixture=SimpleNamespace(
                    id="ref-characterization",
                    category="benign",
                    expected="comply",
                ),
                classification="comply",
                correct=True,
                over_refusal=False,
                under_refusal=False,
            )
        ]
        judge = (
            None
            if partial
            else JudgeOutcome(
                status=JudgeOutcomeStatus.SCORED,
                score=1.0,
                judges_expected=1,
                judges_scored=1,
            )
        )
        attempt = AttemptResult(
            attempt_index=0,
            provider=ProviderOutcome(ProviderOutcomeKind.COMPLETED),
            response_text="response",
            latency_ms=1.0,
            estimated_cost_usd=0.0,
            input_tokens=1,
            output_tokens=1,
            thinking_tokens=0,
            judge=judge,
        )
        self.integrity = RunIntegrity.from_fixture_attempts([[attempt]])

    def to_dict(self) -> dict[str, object]:
        return {
            "calibration_score": self.calibration_score,
            "over_refusal_rate": self.over_refusal_rate,
            "under_refusal_rate": self.under_refusal_rate,
            "integrity": {"status": self.integrity.status.value},
            "results": [{"id": "ref-characterization"}],
        }


def _patch_refusal(monkeypatch, *, partial: bool = False) -> None:
    provider = SimpleNamespace(name="mock", default_model="mock-model")

    async def fake_run_refusal(*_args, **kwargs):
        summary = _Summary(partial=partial)
        start_callback = kwargs.get("on_fixture_start")
        if start_callback is not None:
            start_callback(summary.results[0].fixture)
        callback = kwargs.get("on_fixture_done")
        if callback is not None:
            callback(summary.results[0])
        return summary

    monkeypatch.setattr(
        "atomics.commands.refusal._make_provider",
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
        "--save",
        "--allow-partial",
    ):
        assert option in result.output


def test_refusal_command_is_registered_and_reexported() -> None:
    from atomics import cli as cli_module
    from atomics.commands.refusal import refusal

    assert cli_module.cli.commands["refusal"] is refusal
    assert cli_module.refusal is refusal


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


def test_refusal_partial_run_exits_nonzero(monkeypatch) -> None:
    _patch_refusal(monkeypatch, partial=True)

    result = CliRunner().invoke(
        cli,
        ["--no-progress", "refusal", "--no-save"],
    )

    assert result.exit_code == 1
    assert "infrastructure_invalid" in result.output


def test_refusal_allow_partial_writes_json(monkeypatch, tmp_path) -> None:
    _patch_refusal(monkeypatch, partial=True)
    output = tmp_path / "partial.json"

    result = CliRunner().invoke(
        cli,
        [
            "--no-progress",
            "refusal",
            "--no-save",
            "--allow-partial",
            "--json-out",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["integrity"]["status"] == (
        "infrastructure_invalid"
    )
