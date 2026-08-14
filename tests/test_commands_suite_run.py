"""Unit tests for the shared suite persistence lifetime."""

from __future__ import annotations

import sqlite3

import click
import pytest

from atomics.commands.suite_run import suite_run
from atomics.storage import MetricsRepository


def _finalize(repo: MetricsRepository, run_id: str) -> object:
    return repo.complete_run(run_id)


def test_no_save_opens_nothing(tmp_path) -> None:
    db_path = tmp_path / "unused.db"
    with suite_run(
        suite="demo",
        db_path=db_path,
        save=False,
        finalize=_finalize,
        failure_prefix="Demo failed",
    ) as run:
        run.begin("r1", provider="p", model="m")
        assert run.repository is None

    assert not db_path.exists()


def test_parent_row_is_finalized_and_connection_closed(tmp_path) -> None:
    db_path = tmp_path / "ok.db"
    with suite_run(
        suite="demo",
        db_path=db_path,
        save=True,
        finalize=_finalize,
        failure_prefix="Demo failed",
    ) as run:
        run.begin("r1", provider="p", model="m", tier="demo-tier")
        repository = run.repository

    assert repository is not None
    with pytest.raises(sqlite3.ProgrammingError):
        repository._conn.execute("SELECT 1")
    row = _read_run(db_path, "r1")
    assert row["tier"] == "demo-tier"
    assert row["trigger"] == "manual"
    assert row["completed_at"] is not None


def test_body_failure_still_finalizes_and_closes(tmp_path) -> None:
    db_path = tmp_path / "boom.db"
    captured: list[MetricsRepository] = []

    with pytest.raises(click.ClickException) as excinfo:
        with suite_run(
            suite="demo",
            db_path=db_path,
            save=True,
            finalize=_finalize,
            failure_prefix="Demo failed",
        ) as run:
            assert run.repository is not None
            captured.append(run.repository)
            run.begin("r1", provider="p", model="m")
            raise RuntimeError("api_key=body-secret")

    assert "Demo failed" in str(excinfo.value)
    assert "body-secret" not in str(excinfo.value)
    assert _read_run(db_path, "r1")["completed_at"] is not None
    with pytest.raises(sqlite3.ProgrammingError):
        captured[0]._conn.execute("SELECT 1")


def test_click_exception_passes_through_unwrapped(tmp_path) -> None:
    with pytest.raises(click.ClickException) as excinfo:
        with suite_run(
            suite="demo",
            db_path=tmp_path / "click.db",
            save=True,
            finalize=_finalize,
            failure_prefix="Demo failed",
        ) as run:
            run.begin("r1", provider="p", model="m")
            raise click.ClickException("already phrased")

    assert str(excinfo.value) == "already phrased"


def test_exit_passes_through_so_exit_codes_survive(tmp_path) -> None:
    with pytest.raises(click.exceptions.Exit) as excinfo:
        with suite_run(
            suite="demo",
            db_path=tmp_path / "exit.db",
            save=True,
            finalize=_finalize,
            failure_prefix="Demo failed",
        ) as run:
            run.begin("r1", provider="p", model="m")
            raise click.exceptions.Exit(1)

    assert excinfo.value.exit_code == 1


def test_keyboard_interrupt_is_not_wrapped(tmp_path) -> None:
    with pytest.raises(KeyboardInterrupt):
        with suite_run(
            suite="demo",
            db_path=tmp_path / "sigint.db",
            save=True,
            finalize=_finalize,
            failure_prefix="Demo failed",
        ) as run:
            run.begin("r1", provider="p", model="m")
            raise KeyboardInterrupt

    assert _read_run(tmp_path / "sigint.db", "r1")["completed_at"] is not None


def test_finalize_failure_is_named_and_sanitized(tmp_path) -> None:
    def failing_finalize(_repo: MetricsRepository, _run_id: str) -> object:
        raise RuntimeError("api_key=finalize-secret")

    with pytest.raises(click.ClickException) as excinfo:
        with suite_run(
            suite="demo",
            db_path=tmp_path / "finalize.db",
            save=True,
            finalize=failing_finalize,
            failure_prefix="Demo failed",
        ) as run:
            run.begin("r1", provider="p", model="m")

    message = str(excinfo.value)
    assert "Failed to finalize demo run r1" in message
    assert "finalize-secret" not in message


def test_every_parent_is_finalized_even_when_one_fails(tmp_path) -> None:
    attempted: list[str] = []

    def flaky_finalize(repo: MetricsRepository, run_id: str) -> object:
        attempted.append(run_id)
        if run_id == "r1":
            raise RuntimeError("first parent is broken")
        return repo.complete_run(run_id)

    with pytest.raises(click.ClickException):
        with suite_run(
            suite="demo",
            db_path=tmp_path / "multi.db",
            save=True,
            finalize=flaky_finalize,
            failure_prefix="Demo failed",
        ) as run:
            run.begin("r1", provider="p", model="m")
            run.begin("r2", provider="p", model="m2")

    assert attempted == ["r1", "r2"]
    assert _read_run(tmp_path / "multi.db", "r2")["completed_at"] is not None


def test_cleanup_failure_does_not_mask_the_body_failure(tmp_path, caplog) -> None:
    def failing_finalize(_repo: MetricsRepository, _run_id: str) -> object:
        raise RuntimeError("cleanup went wrong too")

    with pytest.raises(click.ClickException) as excinfo:
        with suite_run(
            suite="demo",
            db_path=tmp_path / "both.db",
            save=True,
            finalize=failing_finalize,
            failure_prefix="Demo failed",
        ) as run:
            run.begin("r1", provider="p", model="m")
            raise RuntimeError("the real problem")

    assert "the real problem" in str(excinfo.value)
    assert "cleanup went wrong too" in caplog.text


def test_cleanup_is_idempotent(tmp_path) -> None:
    calls: list[str] = []

    def counting_finalize(repo: MetricsRepository, run_id: str) -> object:
        calls.append(run_id)
        return repo.complete_run(run_id)

    with suite_run(
        suite="demo",
        db_path=tmp_path / "twice.db",
        save=True,
        finalize=counting_finalize,
        failure_prefix="Demo failed",
    ) as run:
        run.begin("r1", provider="p", model="m")
        assert run.cleanup() is None

    assert calls == ["r1"]


def test_named_finalizers_dispatch_on_the_instance(tmp_path) -> None:
    """A substituted repository must be honored, not the base implementation.

    An unbound method reference (`MetricsRepository.complete_run`) would call the
    base class no matter what instance was opened, silently bypassing any subclass
    or wrapper — including the ones tests use to simulate a failing database.
    """
    from atomics.commands.suite_run import (
        finalize_adversarial_run,
        finalize_evaluation_run,
        finalize_task_run,
    )

    called: list[str] = []

    class OverridingRepository(MetricsRepository):
        def complete_run(self, run_id: str):  # type: ignore[override]
            called.append("task")
            return super().complete_run(run_id)

        def complete_evaluation_run(self, run_id: str):  # type: ignore[override]
            called.append("evaluation")
            return super().complete_evaluation_run(run_id)

        def complete_adversarial_run(self, run_id: str) -> None:
            called.append("adversarial")

    repository = OverridingRepository(tmp_path / "dispatch.db")
    try:
        repository.create_run("r1")
        finalize_task_run(repository, "r1")
        finalize_evaluation_run(repository, "r1")
        finalize_adversarial_run(repository, "r1")
    finally:
        repository.close()

    assert called == ["task", "evaluation", "adversarial"]


def _read_run(db_path, run_id: str) -> sqlite3.Row:  # type: ignore[no-untyped-def]
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT tier, trigger, completed_at FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, f"no parent row for {run_id}"
    return row
