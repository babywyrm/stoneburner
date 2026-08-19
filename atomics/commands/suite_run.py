"""Shared database lifetime for every command that records a run.

They all need the same four things, in the same order: a repository opened only
under `--save`, a parent `runs` row per model under test, that row finalized once
the fixture rows are in, and the connection closed. The last two have to happen
even when the run raises, and getting them wrong is invisible in a passing run —
the symptoms are a parent row that never receives `completed_at`, which reads as
a run still in progress, and a leaked SQLite connection.

Commands differ in only three details, so those are the parameters: which table
the parent aggregates from (`finalize`), what the run is called in an error
message (`suite`), and how a failure is phrased (`failure_prefix`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NoReturn

import click

from atomics.storage import MetricsRepository
from atomics.validation import sanitize_error

#: Rolls a parent `runs` row up from whichever table holds the suite's results.
#: Use one of the `finalize_*` functions below rather than an unbound method:
#: those dispatch on the instance, so a subclass or wrapper is honored.
Finalizer = Callable[[MetricsRepository, str], None]

_LOGGER = logging.getLogger("atomics.cli")


def finalize_task_run(repository: MetricsRepository, run_id: str) -> None:
    """For suites whose fixture rows land in `task_results`."""
    repository.complete_run(run_id)


def finalize_evaluation_run(repository: MetricsRepository, run_id: str) -> None:
    """For suites whose fixture rows land in the generic `evaluation_results`."""
    repository.complete_evaluation_run(run_id)


def finalize_adversarial_run(repository: MetricsRepository, run_id: str) -> None:
    """For `adversarial`, whose fixture rows land in `adversarial_results`."""
    repository.complete_adversarial_run(run_id)


def finalize_probe_run(repository: MetricsRepository, run_id: str) -> None:
    """For `probe`, whose fixture rows land in `probe_results`."""
    repository.complete_probe_run(run_id)


def finalize_archreview_run(repository: MetricsRepository, run_id: str) -> None:
    """For `archreview`, whose fixture rows land in `archreview_results`."""
    repository.complete_archreview_run(run_id)


class SuiteRun:
    """The repository and parent run rows for one suite command invocation.

    `repository` is None when the command was invoked with `--no-save`, which is
    the signal every callback already checks before persisting a fixture.
    """

    def __init__(
        self,
        *,
        suite: str,
        db_path: Path,
        save: bool,
        finalize: Finalizer,
    ) -> None:
        self._suite = suite
        self._db_path = db_path
        self._save = save
        self._finalize = finalize
        self._run_ids: list[str] = []
        self._cleaned = False
        self.repository: MetricsRepository | None = None

    def open(self) -> None:
        if not self._save:
            return
        # Resolved here rather than at import so substituting the class on the
        # storage module takes effect. Tests rely on that to inject a repository
        # that fails on demand, which is the only way to exercise cleanup.
        from atomics.storage.repository import MetricsRepository as Repository

        try:
            self.repository = Repository(self._db_path)
        except Exception as exc:
            raise RuntimeError(
                f"unable to open database {self._db_path}: {sanitize_error(exc)}"
            ) from exc

    def require_repository(self) -> MetricsRepository:
        """The repository, for callers reached only when saving is on.

        Saves those callers a `None` check that can never fire, without letting
        them reach past the attribute and lose the type.
        """
        if self.repository is None:
            raise RuntimeError("suite_run was entered with save=False")
        return self.repository

    def begin(
        self,
        run_id: str,
        *,
        provider: str,
        model: str,
        tier: str | None = None,
        trigger: str = "manual",
        pass_count: int = 1,
    ) -> None:
        """Write the parent row a suite's fixture rows will reference.

        Called per model under test, so a command comparing two models finalizes
        both. A no-op without `--save`, so callers need no second condition.
        """
        if self.repository is None:
            return
        self.repository.create_run(
            run_id,
            tier=tier or self._suite,
            provider=provider,
            model=model,
            trigger=trigger,
            pass_count=pass_count,
        )
        self._run_ids.append(run_id)

    def cleanup(self) -> click.ClickException | None:
        """Finalize every parent row and close the connection, exactly once.

        Returns the problems rather than raising them: the caller decides whether
        a cleanup failure is the headline or whether it is being reported behind a
        failure that already happened. Each parent is attempted even if an earlier
        one fails, and the connection is closed regardless.
        """
        if self.repository is None or self._cleaned:
            return None
        self._cleaned = True
        failures: list[str] = []
        for run_id in self._run_ids:
            try:
                self._finalize(self.repository, run_id)
            except Exception as exc:
                failures.append(
                    f"Failed to finalize {self._suite} run {run_id}: {sanitize_error(exc)}"
                )
        try:
            self.repository.close()
        except Exception as exc:
            failures.append(f"Failed to close {self._suite} repository: {sanitize_error(exc)}")
        if failures:
            return click.ClickException("; ".join(failures))
        return None


@contextmanager
def suite_run(
    *,
    suite: str,
    db_path: Path,
    save: bool,
    finalize: Finalizer,
    failure_prefix: str,
) -> Iterator[SuiteRun]:
    """Run a suite command body with its persistence lifetime managed.

    Whatever happens in the body, the parent rows are finalized and the
    connection is closed. A failure inside the body is what the user hears about;
    a cleanup problem that surfaces behind it is logged rather than raised, since
    replacing the original failure would hide the thing that actually went wrong.
    `ClickException` and Click's control-flow exceptions pass through untouched so
    exit codes and already-phrased messages survive.
    """
    run = SuiteRun(suite=suite, db_path=db_path, save=save, finalize=finalize)
    failure: BaseException | None = None
    try:
        run.open()
        yield run
    except BaseException as exc:  # noqa: BLE001 — re-raised below, possibly wrapped
        failure = exc
    finally:
        cleanup_failure = run.cleanup()
        if cleanup_failure is not None:
            if failure is None:
                failure = cleanup_failure
            else:
                _LOGGER.error(str(cleanup_failure))

    if failure is not None:
        _reraise(failure, failure_prefix)


def _reraise(failure: BaseException, failure_prefix: str) -> NoReturn:
    """Surface a failure as a CLI error, without leaking its detail."""
    if isinstance(failure, (click.ClickException, click.exceptions.Exit, click.Abort)):
        raise failure
    if not isinstance(failure, Exception):
        # KeyboardInterrupt and SystemExit are the operator's intent, not a bug.
        raise failure
    raise click.ClickException(f"{failure_prefix}: {sanitize_error(failure)}") from failure
