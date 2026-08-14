"""Persistence lifecycle contract shared by the suite evaluation commands.

Each suite command opens a repository only under `--save`, writes a parent `runs`
row, finalizes that row once the fixture rows are in, and closes the connection.
The last two have to happen even when the run raises. A missed finalize leaves a
parent row without `completed_at` in the database forever; a missed close leaks
the SQLite connection. Neither is visible in a passing run, which is why these
are asserted directly rather than inferred from a successful invocation.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from atomics.cli import cli
from atomics.config import AtomicsSettings
from atomics.storage import MetricsRepository


def _settings(db_path) -> AtomicsSettings:  # type: ignore[no-untyped-def]
    """Real settings with a temporary database, so no attribute is missing."""
    return AtomicsSettings(anthropic_api_key="fake-key-for-tests", db_path=db_path)


def _tracking_repository(opened: list[MetricsRepository]) -> type[MetricsRepository]:
    """A repository that records every instance so a test can assert it closed."""

    class TrackingRepository(MetricsRepository):
        def __init__(self, db_path):  # type: ignore[no-untyped-def]
            super().__init__(db_path)
            opened.append(self)

    return TrackingRepository


def _assert_closed(repo: MetricsRepository) -> None:
    with pytest.raises(sqlite3.ProgrammingError):
        repo._conn.execute("SELECT 1")


def _parent_rows(db_path) -> list[sqlite3.Row]:  # type: ignore[no-untyped-def]
    """Read the parent `runs` rows directly, without reopening the CLI's repo."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute("SELECT run_id, tier, completed_at FROM runs"))
    finally:
        conn.close()


# ── redblue ───────────────────────────────────────────────────────────────────


def _patch_redblue(monkeypatch, db_path, *, fail: bool = False) -> list[MetricsRepository]:
    from atomics.eval.redblue import runner as redblue_runner
    from atomics.eval.redblue.runner import RedBlueSummary
    from atomics.storage import repository as repository_module

    opened: list[MetricsRepository] = []
    provider = SimpleNamespace(name="mock", default_model="mock-model")

    async def fake_run_redblue(*_args, **kwargs):
        if fail:
            raise RuntimeError("api_key=redblue-secret")
        now = datetime.now(UTC)
        return RedBlueSummary(
            run_id=kwargs.get("run_id") or "redblue-run",
            provider="mock",
            model="mock-model",
            mode=kwargs.get("mode") or "all",
            started_at=now,
            completed_at=now,
        )

    monkeypatch.setattr(
        "atomics.commands.security.cmd_redblue._make_provider",
        lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr(
        "atomics.commands.security.cmd_redblue.load_settings",
        lambda: _settings(db_path),
    )
    monkeypatch.setattr(redblue_runner, "run_redblue", fake_run_redblue)
    monkeypatch.setattr(
        repository_module, "MetricsRepository", _tracking_repository(opened)
    )
    return opened


def test_redblue_save_finalizes_parent_and_closes(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "redblue.db"
    opened = _patch_redblue(monkeypatch, db_path)

    result = CliRunner().invoke(cli, ["--no-progress", "redblue", "--save"])

    assert result.exit_code == 0
    rows = _parent_rows(db_path)
    assert len(rows) == 1
    assert rows[0]["tier"] == "redblue-all"
    assert rows[0]["completed_at"] is not None
    assert len(opened) == 1
    _assert_closed(opened[0])


def test_redblue_runner_failure_still_finalizes_parent(monkeypatch, tmp_path) -> None:
    """A failed run must not leave its parent row unfinalized forever."""
    db_path = tmp_path / "redblue-fail.db"
    _patch_redblue(monkeypatch, db_path, fail=True)

    result = CliRunner().invoke(cli, ["--no-progress", "redblue", "--save"])

    assert result.exit_code != 0
    rows = _parent_rows(db_path)
    assert len(rows) == 1
    assert rows[0]["completed_at"] is not None


def test_redblue_runner_failure_closes_connection(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "redblue-leak.db"
    opened = _patch_redblue(monkeypatch, db_path, fail=True)

    result = CliRunner().invoke(cli, ["--no-progress", "redblue", "--save"])

    assert result.exit_code != 0
    assert len(opened) == 1
    _assert_closed(opened[0])


def test_redblue_runner_failure_is_reported_and_sanitized(monkeypatch, tmp_path) -> None:
    """The failure must reach the user as a message, with the secret stripped.

    An unhandled exception satisfies "secret not in output" trivially, because
    nothing is printed at all — so this also asserts the report is present.
    """
    db_path = tmp_path / "redblue-sanitize.db"
    _patch_redblue(monkeypatch, db_path, fail=True)

    result = CliRunner().invoke(cli, ["--no-progress", "redblue", "--save"])

    assert result.exit_code != 0
    assert "Red/blue evaluation failed" in result.output
    assert "redblue-secret" not in result.output


def test_redblue_no_save_writes_nothing(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "redblue-nosave.db"
    opened = _patch_redblue(monkeypatch, db_path)

    result = CliRunner().invoke(cli, ["--no-progress", "redblue", "--no-save"])

    assert result.exit_code == 0
    assert opened == []
    assert not db_path.exists()


# ── multiturn ─────────────────────────────────────────────────────────────────


def _patch_multiturn(
    monkeypatch, db_path, *, fail: bool = False
) -> list[MetricsRepository]:
    from atomics.eval.multiturn import runner as multiturn_runner
    from atomics.eval.multiturn.runner import MultiturnRunSummary
    from atomics.storage import repository as repository_module

    opened: list[MetricsRepository] = []
    provider = SimpleNamespace(name="mock", default_model="mock-model")

    async def fake_run_multiturn(*_args, **kwargs):
        if fail:
            raise RuntimeError("api_key=multiturn-secret")
        now = datetime.now(UTC)
        return MultiturnRunSummary(
            run_id=kwargs.get("run_id") or "multiturn-run",
            provider="mock",
            model="mock-model",
            judge_provider="mock",
            judge_model="mock-model",
            started_at=now,
            completed_at=now,
        )

    monkeypatch.setattr(
        "atomics.commands.security.cmd_multiturn._make_provider",
        lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr(
        "atomics.commands.security.cmd_multiturn.load_settings",
        lambda: _settings(db_path),
    )
    monkeypatch.setattr(multiturn_runner, "run_multiturn", fake_run_multiturn)
    monkeypatch.setattr(
        repository_module, "MetricsRepository", _tracking_repository(opened)
    )
    return opened


def test_multiturn_save_finalizes_parent_and_closes(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "multiturn.db"
    opened = _patch_multiturn(monkeypatch, db_path)

    result = CliRunner().invoke(cli, ["--no-progress", "multiturn", "--save"])

    assert result.exit_code == 0
    rows = _parent_rows(db_path)
    assert len(rows) == 1
    assert rows[0]["tier"] == "multiturn"
    assert rows[0]["completed_at"] is not None
    assert len(opened) == 1
    _assert_closed(opened[0])


def test_multiturn_runner_failure_still_finalizes_parent(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "multiturn-fail.db"
    _patch_multiturn(monkeypatch, db_path, fail=True)

    result = CliRunner().invoke(cli, ["--no-progress", "multiturn", "--save"])

    assert result.exit_code != 0
    rows = _parent_rows(db_path)
    assert len(rows) == 1
    assert rows[0]["completed_at"] is not None


def test_multiturn_runner_failure_closes_connection(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "multiturn-leak.db"
    opened = _patch_multiturn(monkeypatch, db_path, fail=True)

    result = CliRunner().invoke(cli, ["--no-progress", "multiturn", "--save"])

    assert result.exit_code != 0
    assert len(opened) == 1
    _assert_closed(opened[0])


def test_multiturn_runner_failure_is_reported_and_sanitized(
    monkeypatch, tmp_path
) -> None:
    db_path = tmp_path / "multiturn-sanitize.db"
    _patch_multiturn(monkeypatch, db_path, fail=True)

    result = CliRunner().invoke(cli, ["--no-progress", "multiturn", "--save"])

    assert result.exit_code != 0
    assert "Multi-turn evaluation failed" in result.output
    assert "multiturn-secret" not in result.output


# ── eval ──────────────────────────────────────────────────────────────────────


def _patch_eval(monkeypatch, db_path, *, fail: bool = False) -> list[MetricsRepository]:
    from atomics.eval import runner as eval_runner
    from atomics.eval.runner import EvalRunSummary
    from atomics.storage import repository as repository_module

    opened: list[MetricsRepository] = []
    provider = SimpleNamespace(name="mock", default_model="mock-model")

    async def fake_run_eval(*_args, **kwargs):
        if fail:
            raise RuntimeError("api_key=eval-secret")
        now = datetime.now(UTC)
        return EvalRunSummary(
            run_id=kwargs.get("run_id") or "eval-run",
            provider="mock",
            model="mock-model",
            judge_provider="mock",
            judge_model="mock-model",
            started_at=now,
            completed_at=now,
            fixture_results=[],
        )

    monkeypatch.setattr(
        "atomics.commands.eval._make_provider",
        lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr(
        "atomics.commands.eval.load_settings",
        lambda: _settings(db_path),
    )
    monkeypatch.setattr(eval_runner, "run_eval", fake_run_eval)
    monkeypatch.setattr(
        repository_module, "MetricsRepository", _tracking_repository(opened)
    )
    return opened


def test_eval_save_finalizes_parent_and_closes(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "eval.db"
    opened = _patch_eval(monkeypatch, db_path)

    result = CliRunner().invoke(cli, ["--no-progress", "eval", "--save"])

    assert result.exit_code == 0
    rows = _parent_rows(db_path)
    assert len(rows) == 1
    assert rows[0]["tier"] == "eval"
    assert rows[0]["completed_at"] is not None
    assert len(opened) == 1
    _assert_closed(opened[0])


def test_eval_runner_failure_still_finalizes_parent(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "eval-fail.db"
    _patch_eval(monkeypatch, db_path, fail=True)

    result = CliRunner().invoke(cli, ["--no-progress", "eval", "--save"])

    assert result.exit_code != 0
    rows = _parent_rows(db_path)
    assert len(rows) == 1
    assert rows[0]["completed_at"] is not None


def test_eval_runner_failure_closes_connection(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "eval-leak.db"
    opened = _patch_eval(monkeypatch, db_path, fail=True)

    result = CliRunner().invoke(cli, ["--no-progress", "eval", "--save"])

    assert result.exit_code != 0
    assert len(opened) == 1
    _assert_closed(opened[0])


def test_eval_runner_failure_is_reported_and_sanitized(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "eval-sanitize.db"
    _patch_eval(monkeypatch, db_path, fail=True)

    result = CliRunner().invoke(cli, ["--no-progress", "eval", "--save"])

    assert result.exit_code != 0
    assert "Eval run failed" in result.output
    assert "eval-secret" not in result.output


# ── rag ───────────────────────────────────────────────────────────────────────


def _patch_rag(monkeypatch, db_path, *, fail: bool = False) -> list[MetricsRepository]:
    from atomics.eval.rag import runner as rag_runner
    from atomics.eval.rag.runner import RAGRunSummary
    from atomics.storage import repository as repository_module

    opened: list[MetricsRepository] = []
    provider = SimpleNamespace(name="mock", default_model="mock-model")

    async def fake_run_rag(*_args, **kwargs):
        if fail:
            raise RuntimeError("api_key=rag-secret")
        now = datetime.now(UTC)
        return RAGRunSummary(
            run_id=kwargs.get("run_id") or "rag-run",
            provider="mock",
            model="mock-model",
            judge_provider="mock",
            judge_model="mock-model",
            started_at=now,
            completed_at=now,
            fixture_results=[],
        )

    monkeypatch.setattr(
        "atomics.commands.rag._make_provider",
        lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr(
        "atomics.commands.rag.load_settings",
        lambda: _settings(db_path),
    )
    monkeypatch.setattr(rag_runner, "run_rag", fake_run_rag)
    monkeypatch.setattr(
        repository_module, "MetricsRepository", _tracking_repository(opened)
    )
    return opened


def test_rag_save_finalizes_parent_and_closes(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "rag.db"
    opened = _patch_rag(monkeypatch, db_path)

    result = CliRunner().invoke(cli, ["--no-progress", "rag", "--save"])

    assert result.exit_code == 0, result.output
    rows = _parent_rows(db_path)
    assert len(rows) == 1
    assert rows[0]["tier"] == "rag"
    assert rows[0]["completed_at"] is not None
    _assert_closed(opened[0])


def test_rag_runner_failure_finalizes_and_closes(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "rag-fail.db"
    opened = _patch_rag(monkeypatch, db_path, fail=True)

    result = CliRunner().invoke(cli, ["--no-progress", "rag", "--save"])

    assert result.exit_code != 0
    assert _parent_rows(db_path)[0]["completed_at"] is not None
    _assert_closed(opened[0])
    assert "RAG eval failed" in result.output
    assert "rag-secret" not in result.output


# ── codegen ───────────────────────────────────────────────────────────────────


def _patch_codegen(
    monkeypatch, db_path, *, fail: bool = False
) -> list[MetricsRepository]:
    from atomics.eval.codegen import runner as codegen_runner
    from atomics.eval.codegen.runner import CodegenRunSummary
    from atomics.storage import repository as repository_module

    opened: list[MetricsRepository] = []
    provider = SimpleNamespace(name="mock", default_model="mock-model")

    async def fake_run_codegen(*_args, **kwargs):
        if fail:
            raise RuntimeError("api_key=codegen-secret")
        now = datetime.now(UTC)
        return CodegenRunSummary(
            run_id=kwargs.get("run_id") or "codegen-run",
            provider="mock",
            model="mock-model",
            started_at=now,
            completed_at=now,
            fixture_results=[],
        )

    monkeypatch.setattr(
        "atomics.commands.rag._make_provider",
        lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr(
        "atomics.commands.rag.load_settings",
        lambda: _settings(db_path),
    )
    monkeypatch.setattr(codegen_runner, "run_codegen", fake_run_codegen)
    monkeypatch.setattr(
        repository_module, "MetricsRepository", _tracking_repository(opened)
    )
    return opened


def test_codegen_save_finalizes_parent_and_closes(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "codegen.db"
    opened = _patch_codegen(monkeypatch, db_path)

    result = CliRunner().invoke(cli, ["--no-progress", "codegen", "--save"])

    assert result.exit_code == 0, result.output
    rows = _parent_rows(db_path)
    assert len(rows) == 1
    assert rows[0]["tier"] == "codegen"
    assert rows[0]["completed_at"] is not None
    _assert_closed(opened[0])


def test_codegen_runner_failure_finalizes_and_closes(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "codegen-fail.db"
    opened = _patch_codegen(monkeypatch, db_path, fail=True)

    result = CliRunner().invoke(cli, ["--no-progress", "codegen", "--save"])

    assert result.exit_code != 0
    assert _parent_rows(db_path)[0]["completed_at"] is not None
    _assert_closed(opened[0])
    assert "Code generation eval failed" in result.output
    assert "codegen-secret" not in result.output


# ── probe ─────────────────────────────────────────────────────────────────────


def _patch_probe(monkeypatch, db_path, *, fail: bool = False) -> list[MetricsRepository]:
    from atomics.probe import runner as probe_runner
    from atomics.probe.runner import ProbeSummary
    from atomics.storage import repository as repository_module

    opened: list[MetricsRepository] = []
    provider = SimpleNamespace(name="mock", default_model="mock-model")

    async def fake_run_probe(*_args, **_kwargs):
        if fail:
            raise RuntimeError("api_key=probe-secret")
        return ProbeSummary()

    monkeypatch.setattr(
        "atomics.commands.rag._make_provider",
        lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr(
        "atomics.commands.rag.load_settings",
        lambda: _settings(db_path),
    )
    monkeypatch.setattr(probe_runner, "run_probe", fake_run_probe)
    monkeypatch.setattr(
        repository_module, "MetricsRepository", _tracking_repository(opened)
    )
    return opened


def _probe_args(tmp_path) -> list[str]:  # type: ignore[no-untyped-def]
    artifact = tmp_path / "finding.json"
    artifact.write_text("{}", encoding="utf-8")
    return [
        "--no-progress",
        "probe",
        "--artifact",
        "config-file",
        "--file",
        str(artifact),
        "--save",
    ]


def test_probe_save_finalizes_parent_and_closes(monkeypatch, tmp_path) -> None:
    """Also covers `finalize_probe_run`: probe rows live in their own table."""
    db_path = tmp_path / "probe.db"
    opened = _patch_probe(monkeypatch, db_path)

    result = CliRunner().invoke(cli, _probe_args(tmp_path))

    assert result.exit_code == 0, result.output
    rows = _parent_rows(db_path)
    assert len(rows) == 1
    assert rows[0]["tier"] == "probe"
    assert rows[0]["completed_at"] is not None
    _assert_closed(opened[0])


def test_probe_runner_failure_finalizes_and_closes(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "probe-fail.db"
    opened = _patch_probe(monkeypatch, db_path, fail=True)

    result = CliRunner().invoke(cli, _probe_args(tmp_path))

    assert result.exit_code != 0
    assert _parent_rows(db_path)[0]["completed_at"] is not None
    _assert_closed(opened[0])
    assert "Probe run failed" in result.output
    assert "probe-secret" not in result.output


# ── archreview ────────────────────────────────────────────────────────────────


def _archreview_args() -> list[str]:
    return [
        "archreview",
        "--repo",
        "juice-shop",
        "--models",
        "mock-model",
        "--provider",
        "ollama",
        "--tier",
        "floor",
        "--judge-only",
        "--save",
    ]


def _patch_archreview(
    monkeypatch, tmp_path, db_path, *, fail: bool = False
) -> list[MetricsRepository]:
    from atomics.archreview import runner as archreview_runner
    from atomics.archreview.models import ArchReviewResult
    from atomics.storage import repository as repository_module

    monkeypatch.setenv("JUICE_SHOP_PATH", str(tmp_path))
    (tmp_path / "server.ts").write_text("// app\n", encoding="utf-8")

    opened: list[MetricsRepository] = []

    async def fake_run_archreview(**kwargs):
        if fail:
            raise RuntimeError("api_key=archreview-secret")
        # One round rather than none: the table row averages over the results, so
        # an empty list divides by zero before persistence is ever reached.
        return [
            ArchReviewResult(
                run_id=kwargs.get("run_id") or "",
                repo="juice-shop",
                tier="floor",
                model="mock-model",
                provider="ollama",
                round=1,
                findings=[],
                objective_recall=0.5,
                objective_precision=0.5,
                objective_f=0.5,
                judge_score=0.5,
                matched_categories=[],
            )
        ]

    monkeypatch.setattr(
        "atomics.commands.rag.load_settings",
        lambda: _settings(db_path),
    )
    monkeypatch.setattr(archreview_runner, "run_archreview", fake_run_archreview)
    monkeypatch.setattr(
        repository_module, "MetricsRepository", _tracking_repository(opened)
    )
    return opened


def test_archreview_save_finalizes_parent_and_closes(monkeypatch, tmp_path) -> None:
    """Also covers `finalize_archreview_run`, whose rows live in their own table."""
    db_path = tmp_path / "archreview.db"
    opened = _patch_archreview(monkeypatch, tmp_path, db_path)

    result = CliRunner().invoke(cli, _archreview_args())

    assert result.exit_code == 0, result.output
    rows = _parent_rows(db_path)
    assert len(rows) == 1
    assert rows[0]["tier"] == "archreview"
    assert rows[0]["completed_at"] is not None
    _assert_closed(opened[0])


# ── JSON output ───────────────────────────────────────────────────────────────
#
# These commands used to write `--json-out` with a hand-rolled `json.dump` after
# the connection was already closed. They now go through `write_summary_json`
# inside the managed lifetime, which reports a failed write as a sanitized CLI
# error instead of a raw traceback, and still finalizes the run.


def test_redblue_json_out_writes_the_summary(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "redblue-json.db"
    _patch_redblue(monkeypatch, db_path)
    out = tmp_path / "redblue.json"

    result = CliRunner().invoke(
        cli,
        ["--no-progress", "redblue", "--no-save", "--json-out", str(out)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text(encoding="utf-8"))["mode"] == "all"


def test_multiturn_json_out_writes_the_summary(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "multiturn-json.db"
    _patch_multiturn(monkeypatch, db_path)
    out = tmp_path / "multiturn.json"

    result = CliRunner().invoke(
        cli,
        ["--no-progress", "multiturn", "--no-save", "--json-out", str(out)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["provider"] == "mock"
    assert payload["run_id"]


def test_redblue_unwritable_json_is_reported_and_still_finalizes(
    monkeypatch, tmp_path
) -> None:
    db_path = tmp_path / "redblue-json-fail.db"
    opened = _patch_redblue(monkeypatch, db_path)
    unwritable = tmp_path / "no-such-directory" / "out.json"

    result = CliRunner().invoke(
        cli,
        ["--no-progress", "redblue", "--save", "--json-out", str(unwritable)],
    )

    assert result.exit_code != 0
    assert "Unable to write JSON output" in result.output
    assert _parent_rows(db_path)[0]["completed_at"] is not None
    _assert_closed(opened[0])


def test_archreview_runner_failure_finalizes_and_closes(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "archreview-fail.db"
    opened = _patch_archreview(monkeypatch, tmp_path, db_path, fail=True)

    result = CliRunner().invoke(cli, _archreview_args())

    assert result.exit_code != 0
    assert _parent_rows(db_path)[0]["completed_at"] is not None
    _assert_closed(opened[0])
    assert "Architecture review failed" in result.output
    assert "archreview-secret" not in result.output
