"""CLI contract tests for secure code review."""

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
    detection_rate = 1.0
    false_positive_rate = 0.0
    review_score = 1.0

    def __init__(self, *, partial: bool = False) -> None:
        self.results = [
            SimpleNamespace(
                fixture=SimpleNamespace(
                    id="cr-characterization",
                    is_vulnerable=True,
                    cwe="CWE-79",
                    mode="snippet",
                ),
                verdict="detected",
                passed=True,
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
            response_text="review",
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
            "detection_rate": self.detection_rate,
            "false_positive_rate": self.false_positive_rate,
            "review_score": self.review_score,
            "integrity": {"status": self.integrity.status.value},
            "results": [{"id": "cr-characterization"}],
        }


def _patch_codereview(monkeypatch, *, partial: bool = False) -> None:
    provider = SimpleNamespace(name="mock", default_model="mock-model")

    async def fake_run_codereview(*_args, **kwargs):
        summary = _Summary(partial=partial)
        start_callback = kwargs.get("on_fixture_start")
        if start_callback is not None:
            start_callback(summary.results[0].fixture)
        callback = kwargs.get("on_fixture_done")
        if callback is not None:
            callback(summary.results[0])
        return summary

    monkeypatch.setattr(
        "atomics.commands.codereview._make_provider",
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
        "--save",
        "--allow-partial",
    ):
        assert option in result.output


def test_codereview_command_is_registered_and_reexported() -> None:
    from atomics import cli as cli_module
    from atomics.commands.codereview import codereview

    assert cli_module.cli.commands["codereview"] is codereview
    assert cli_module.codereview is codereview


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


def test_codereview_partial_json_is_written_before_nonzero_exit(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_codereview(monkeypatch, partial=True)
    output = tmp_path / "codereview.json"

    result = CliRunner().invoke(
        cli,
        [
            "--no-progress",
            "codereview",
            "--no-save",
            "--json-out",
            str(output),
        ],
    )

    assert result.exit_code == 1
    assert json.loads(output.read_text(encoding="utf-8"))["integrity"]["status"] == (
        "infrastructure_invalid"
    )


def test_codereview_allow_partial_exits_zero(monkeypatch) -> None:
    _patch_codereview(monkeypatch, partial=True)

    result = CliRunner().invoke(
        cli,
        ["--no-progress", "codereview", "--no-save", "--allow-partial"],
    )

    assert result.exit_code == 0
