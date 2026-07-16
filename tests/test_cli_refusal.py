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

    def __init__(
        self,
        *,
        partial: bool = False,
        run_id: str = "refusal-run",
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
            response_text="response",
            latency_ms=1.0,
            estimated_cost_usd=0.0,
            input_tokens=1,
            output_tokens=1,
            thinking_tokens=0,
            judge=judge,
        )
        self.integrity = RunIntegrity.from_fixture_attempts([[attempt]])
        payload: dict[str, object] = {
            "id": "ref-characterization",
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
                id="ref-characterization",
                category="benign",
                expected="comply",
            ),
            classification="unknown" if partial else "comply",
            correct=not partial,
            over_refusal=False,
            under_refusal=False,
            to_dict=lambda: payload,
        )
        self.results = [fixture_result]

    def to_dict(self) -> dict[str, object]:
        return {
            "calibration_score": self.calibration_score,
            "over_refusal_rate": self.over_refusal_rate,
            "under_refusal_rate": self.under_refusal_rate,
            "integrity": {"status": self.integrity.status.value},
            "results": [{"id": "ref-characterization"}],
        }


def _patch_refusal(
    monkeypatch,
    *,
    partial: bool = False,
    db_path=None,
    fail_after_callback: bool = False,
) -> None:
    provider = SimpleNamespace(name="mock", default_model="mock-model")

    async def fake_run_refusal(*_args, **kwargs):
        summary = _Summary(
            partial=partial,
            run_id=kwargs.get("run_id") or "refusal-run",
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
        "atomics.commands.refusal._make_provider",
        lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr("atomics.eval.refusal.run_refusal", fake_run_refusal)
    if db_path is not None:
        monkeypatch.setattr(
            "atomics.commands.refusal.load_settings",
            lambda: SimpleNamespace(db_path=db_path),
        )


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

    result = CliRunner().invoke(cli, ["refusal", "--no-save"])

    assert result.exit_code == 0
    assert "Refusal Calibration Summary" in result.output
    assert "ref-characterization" in result.output
    assert "100.0%" in result.output
    assert "Run ID" in result.output


def test_refusal_json_preserves_results_key(monkeypatch, tmp_path) -> None:
    _patch_refusal(monkeypatch)
    output = tmp_path / "refusal.json"

    result = CliRunner().invoke(
        cli,
        ["refusal", "--no-save", "--json-out", str(output)],
    )

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
    assert "ERROR" in result.output
    assert "MISS" not in result.output
    assert "Judge failures" in result.output


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


def test_refusal_save_persists_and_finalizes_parent(monkeypatch, tmp_path) -> None:
    from atomics.storage import MetricsRepository

    db_path = tmp_path / "metrics.db"
    _patch_refusal(monkeypatch, db_path=db_path)

    result = CliRunner().invoke(cli, ["--no-progress", "refusal"])

    assert result.exit_code == 0
    repo = MetricsRepository(db_path)
    rows = repo.get_evaluation_results(suite="refusal")
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


def test_refusal_no_save_skips_database(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "metrics.db"
    _patch_refusal(monkeypatch, db_path=db_path)

    result = CliRunner().invoke(
        cli,
        ["--no-progress", "refusal", "--no-save"],
    )

    assert result.exit_code == 0
    assert not db_path.exists()


def test_refusal_failure_preserves_fixture_and_finalizes_parent(
    monkeypatch,
    tmp_path,
) -> None:
    from atomics.storage import MetricsRepository

    db_path = tmp_path / "metrics.db"
    _patch_refusal(
        monkeypatch,
        db_path=db_path,
        fail_after_callback=True,
    )

    result = CliRunner().invoke(cli, ["--no-progress", "refusal"])

    assert result.exit_code == 1
    assert "Refusal evaluation failed" in result.output
    repo = MetricsRepository(db_path)
    rows = repo.get_evaluation_results(suite="refusal")
    parent = next(
        run
        for run in repo.get_recent_runs()
        if run["run_id"] == rows[0]["run_id"]
    )
    assert len(rows) == 1
    assert parent["total_tasks"] == 1
    assert parent["completed_at"] is not None
    repo.close()


def test_refusal_finalizer_failure_is_sanitized(monkeypatch, tmp_path) -> None:
    from atomics.storage import MetricsRepository

    db_path = tmp_path / "metrics.db"
    _patch_refusal(monkeypatch, db_path=db_path)

    def fail_finalizer(*_args, **_kwargs) -> None:
        raise RuntimeError("Bearer secret-finalizer-token")

    monkeypatch.setattr(
        MetricsRepository,
        "complete_evaluation_run",
        fail_finalizer,
    )

    result = CliRunner().invoke(cli, ["--no-progress", "refusal"])

    assert result.exit_code == 1
    assert "Refusal evaluation failed" in result.output
    assert "secret-finalizer-token" not in result.output


def test_refusal_json_failure_keeps_persisted_fixture(monkeypatch, tmp_path) -> None:
    from atomics.storage import MetricsRepository

    db_path = tmp_path / "metrics.db"
    _patch_refusal(monkeypatch, db_path=db_path)

    def fail_json(*_args, **_kwargs) -> None:
        raise OSError("Bearer secret-json-token")

    monkeypatch.setattr(
        "atomics.commands.refusal.write_summary_json",
        fail_json,
    )

    result = CliRunner().invoke(
        cli,
        ["--no-progress", "refusal", "--json-out", str(tmp_path / "out.json")],
    )

    assert result.exit_code == 1
    assert "secret-json-token" not in result.output
    repo = MetricsRepository(db_path)
    rows = repo.get_evaluation_results(suite="refusal")
    assert len(rows) == 1
    parent = next(
        run
        for run in repo.get_recent_runs()
        if run["run_id"] == rows[0]["run_id"]
    )
    assert parent["completed_at"] is not None
    repo.close()


def test_refusal_primary_failure_survives_close_failure(monkeypatch, tmp_path) -> None:
    from atomics.storage import MetricsRepository

    db_path = tmp_path / "metrics.db"
    _patch_refusal(
        monkeypatch,
        db_path=db_path,
        fail_after_callback=True,
    )
    original_close = MetricsRepository.close

    def fail_close(repository, *_args, **_kwargs) -> None:
        original_close(repository)
        raise RuntimeError("close failure")

    monkeypatch.setattr(MetricsRepository, "close", fail_close)

    result = CliRunner().invoke(cli, ["--no-progress", "refusal"])

    assert result.exit_code == 1
    assert "fixture batch failed" in result.output
    assert "close failure" not in result.output


def test_refusal_progress_respects_root_toggle(monkeypatch) -> None:
    events: list[str] = []

    class _Progress:
        def __init__(self, *_args, **_kwargs) -> None:
            events.append("init")

        def on_start(self, *_args, **_kwargs) -> None:
            events.append("start")

        def on_done(self, *_args, **_kwargs) -> None:
            events.append("done")

    _patch_refusal(monkeypatch)
    monkeypatch.setattr("atomics.commands.refusal.FixtureProgress", _Progress)

    enabled = CliRunner().invoke(cli, ["refusal", "--no-save"])
    assert enabled.exit_code == 0
    assert events == ["init", "start", "done"]

    events.clear()
    disabled = CliRunner().invoke(
        cli,
        ["--no-progress", "refusal", "--no-save"],
    )
    assert disabled.exit_code == 0
    assert events == []
