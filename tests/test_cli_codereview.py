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

    def __init__(
        self,
        *,
        partial: bool = False,
        run_id: str = "codereview-run",
    ) -> None:
        self.run_id = run_id
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
        payload: dict[str, object] = {
            "id": "cr-characterization",
            "status": self.integrity.status.value,
            "score": None if partial else 1.0,
            "generation_status": "completed",
            "judge_status": "skipped" if partial else "scored",
            "latency_ms": 1.0,
            "estimated_cost_usd": 0.0,
            "attempt_count": 1,
            "generation_failures": 0,
            "infrastructure_failures": 0,
            "judge_failures": 1 if partial else 0,
            "parse_failed": False,
            "error_class": "",
            "error_message": "",
            "attempts": [
                {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "thinking_tokens": 0,
                }
            ],
        }
        fixture_result = SimpleNamespace(
            fixture=SimpleNamespace(
                id="cr-characterization",
                is_vulnerable=True,
                cwe="CWE-79",
                mode="snippet",
            ),
            verdict="detected",
            passed=True,
            to_dict=lambda: payload,
        )
        self.results = [fixture_result]

    def to_dict(self) -> dict[str, object]:
        return {
            "detection_rate": self.detection_rate,
            "false_positive_rate": self.false_positive_rate,
            "review_score": self.review_score,
            "integrity": {"status": self.integrity.status.value},
            "results": [{"id": "cr-characterization"}],
        }


def _patch_codereview(
    monkeypatch,
    *,
    partial: bool = False,
    db_path=None,
    fail_after_callback: bool = False,
) -> None:
    provider = SimpleNamespace(name="mock", default_model="mock-model")

    async def fake_run_codereview(*_args, **kwargs):
        summary = _Summary(
            partial=partial,
            run_id=kwargs.get("run_id") or "codereview-run",
        )
        start_callback = kwargs.get("on_fixture_start")
        if start_callback is not None:
            start_callback(summary.results[0].fixture)
        callback = kwargs.get("on_fixture_done")
        if callback is not None:
            callback(summary.results[0])
        if fail_after_callback:
            raise RuntimeError("fixture batch failed")
        return summary

    monkeypatch.setattr(
        "atomics.commands.codereview._make_provider",
        lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr("atomics.eval.codereview.run_codereview", fake_run_codereview)
    if db_path is not None:
        monkeypatch.setattr(
            "atomics.commands.codereview.load_settings",
            lambda: SimpleNamespace(db_path=db_path),
        )


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

    result = CliRunner().invoke(cli, ["codereview", "--no-save"])

    assert result.exit_code == 0
    assert "Secure Code Review Summary" in result.output
    assert "cr-characterization" in result.output
    assert "100.0%" in result.output


def test_codereview_json_preserves_results_key(monkeypatch, tmp_path) -> None:
    _patch_codereview(monkeypatch)
    output = tmp_path / "codereview.json"

    result = CliRunner().invoke(
        cli,
        ["codereview", "--no-save", "--json-out", str(output)],
    )

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


def test_codereview_save_persists_and_finalizes_parent(
    monkeypatch,
    tmp_path,
) -> None:
    from atomics.storage import MetricsRepository

    db_path = tmp_path / "metrics.db"
    _patch_codereview(monkeypatch, db_path=db_path)

    result = CliRunner().invoke(cli, ["--no-progress", "codereview"])

    assert result.exit_code == 0
    repo = MetricsRepository(db_path)
    rows = repo.get_evaluation_results(suite="codereview")
    assert len(rows) == 1
    parent = next(
        run
        for run in repo.get_recent_runs()
        if run["run_id"] == rows[0]["run_id"]
    )
    assert parent["total_tasks"] == 1
    assert parent["successful_tasks"] == 1
    assert parent["completed_at"] is not None
    repo.close()


def test_codereview_no_save_skips_database(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "metrics.db"
    _patch_codereview(monkeypatch, db_path=db_path)

    result = CliRunner().invoke(
        cli,
        ["--no-progress", "codereview", "--no-save"],
    )

    assert result.exit_code == 0
    assert not db_path.exists()


def test_codereview_failure_preserves_fixture_and_finalizes_parent(
    monkeypatch,
    tmp_path,
) -> None:
    from atomics.storage import MetricsRepository

    db_path = tmp_path / "metrics.db"
    _patch_codereview(
        monkeypatch,
        db_path=db_path,
        fail_after_callback=True,
    )

    result = CliRunner().invoke(cli, ["--no-progress", "codereview"])

    assert result.exit_code == 1
    assert "Code review evaluation failed" in result.output
    repo = MetricsRepository(db_path)
    rows = repo.get_evaluation_results(suite="codereview")
    parent = next(
        run
        for run in repo.get_recent_runs()
        if run["run_id"] == rows[0]["run_id"]
    )
    assert len(rows) == 1
    assert parent["total_tasks"] == 1
    assert parent["completed_at"] is not None
    repo.close()
