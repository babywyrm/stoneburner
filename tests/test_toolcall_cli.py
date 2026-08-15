"""CLI-level behaviour for `atomics toolcall`."""

from __future__ import annotations

from click.testing import CliRunner

from atomics.cli import cli


def test_cli_toolcall_extra_judges_option():
    result = CliRunner().invoke(cli, ["toolcall", "--help"])
    assert result.exit_code == 0
    assert "--extra-judges" in result.output


def test_toolcall_is_registered():
    result = CliRunner().invoke(cli, ["toolcall", "--help"])
    assert result.exit_code == 0
    assert "divergence" in result.output.lower()


def test_the_help_states_that_calls_are_never_executed():
    """Anyone pointing this at real infrastructure needs to know that up front."""
    result = CliRunner().invoke(cli, ["toolcall", "--help"])
    assert "never executed" in result.output.lower()


def test_unknown_provider_is_rejected_before_any_request():
    result = CliRunner().invoke(cli, ["toolcall", "-p", "nonexistent", "-m", "x"])
    assert result.exit_code != 0
    assert "nonexistent" in result.output


def test_unknown_category_is_rejected_with_the_valid_list():
    """fixtures_for_category returns () for an unknown name, so validating here
    is what prevents a run over zero fixtures exiting successfully."""
    result = CliRunner().invoke(
        cli, ["toolcall", "-p", "ollama", "-m", "x", "--category", "nope", "--no-save"]
    )
    assert result.exit_code != 0
    assert "nope" in result.output
    # The message must name the alternatives, not just reject.
    assert "injection" in result.output


def test_channel_choice_is_validated():
    result = CliRunner().invoke(
        cli, ["toolcall", "-p", "ollama", "-m", "x", "--channel", "sideways"]
    )
    assert result.exit_code != 0


def test_runs_must_be_positive():
    result = CliRunner().invoke(
        cli, ["toolcall", "-p", "ollama", "-m", "x", "--runs", "0"]
    )
    assert result.exit_code != 0


def test_every_group_alias_is_documented_in_the_help():
    from atomics.eval.toolcall.fixtures import GROUP_ALIASES

    result = CliRunner().invoke(cli, ["toolcall", "--help"])
    for alias in GROUP_ALIASES:
        assert alias in result.output, f"--category alias {alias!r} is undocumented"


def test_export_offers_the_toolcall_suite():
    """A suite that cannot be exported is evidence nobody can hand over."""
    result = CliRunner().invoke(cli, ["export", "--help"])
    assert "toolcall" in result.output


def test_toolcall_appears_in_the_top_level_command_list():
    result = CliRunner().invoke(cli, ["--help"])
    assert "toolcall" in result.output


def test_export_toolcall_returns_saved_rows(tmp_path):
    """Asserting only on the exit code would pass with no dispatch branch at all:
    the chain falls through and still exits zero. So save a row and require it
    back out."""
    from atomics.storage.records import EvaluationResultRecord
    from atomics.storage.repository import MetricsRepository

    db = tmp_path / "toolcall.db"
    repo = MetricsRepository(db)
    repo.create_run("r-toolcall", tier="toolcall", provider="ollama", model="test-model")
    repo.save_evaluation_result(
        EvaluationResultRecord(
            run_id="r-toolcall",
            suite="toolcall",
            fixture_id="tc-01",
            status="complete",
            generation_status="ok",
            judge_status="unscored",
            latency_ms=12.0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            score=1.0,
            provider="ollama",
            model="test-model",
            result_json={"id": "tc-01", "tool_outcome": "dangerous_call"},
        )
    )
    repo.close()

    runner = CliRunner(env={"ATOMICS_DB_PATH": str(db)})
    result = runner.invoke(cli, ["export", "--suite", "toolcall", "--format", "jsonl"])
    assert result.exit_code == 0, result.output
    assert "tc-01" in result.output
    assert "dangerous_call" in result.output


def _one_fixture_summary():
    from atomics.eval.toolcall.runner import ToolCallSummary

    return ToolCallSummary(
        run_id="r-save",
        provider="ollama",
        model="test-model",
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
        tool_capable=True,
        fixtures=[{
            "id": "tc-01",
            "category": "direct_dangerous",
            "severity": "CRITICAL",
            "tool_outcome": "dangerous_call",
            "prose_label": "resisted",
            "error": None,
            "latency_ms": 10.0,
            "cost_usd": 0.0,
            "runs": [{}],
        }],
    )


def test_saving_creates_the_parent_run_row(tmp_path):
    """evaluation_results has a foreign key to runs, so without the parent row
    every --save run raises IntegrityError on its first fixture."""
    from atomics.commands.toolcall import _save
    from atomics.storage.repository import MetricsRepository

    db = tmp_path / "save.db"
    _save(_one_fixture_summary(), db_path=db)

    repo = MetricsRepository(db)
    rows = repo.get_evaluation_results(suite="toolcall")
    repo.close()
    assert [r["fixture_id"] for r in rows] == ["tc-01"]
    # Compliance is recorded as 1.0: the tool channel is deterministic, so this
    # is the outcome rather than a judged score.
    assert rows[0]["score"] == 1.0


def test_saving_finalizes_the_parent_run_row(tmp_path):
    """Writing the fixture rows is only half the job.

    A parent left without `completed_at` reads as a run still in progress for as
    long as the database exists, and its rolled-up counts stay empty.
    """
    from atomics.commands.toolcall import _save
    from atomics.storage.repository import MetricsRepository

    db = tmp_path / "finalize.db"
    _save(_one_fixture_summary(), db_path=db)

    repo = MetricsRepository(db)
    try:
        parent = next(
            run for run in repo.get_recent_runs() if run["run_id"] == "r-save"
        )
    finally:
        repo.close()
    assert parent["completed_at"] is not None
    assert parent["total_tasks"] == 1
