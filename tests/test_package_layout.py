"""Moved modules keep their original import paths."""

from __future__ import annotations


def test_load_modules_resolve_through_legacy_paths() -> None:
    from atomics.load.stress import StressResult as Moved
    from atomics.stress import StressResult

    assert StressResult is Moved


def test_soak_and_sweep_shims() -> None:
    from atomics.benchmark.sweep import ModelSweepResult as MovedSweep
    from atomics.load.soak import SoakResult as MovedSoak
    from atomics.soak import SoakResult
    from atomics.sweep import ModelSweepResult

    assert SoakResult is MovedSoak
    assert ModelSweepResult is MovedSweep


def test_reporting_package_still_exports_the_old_module() -> None:
    import atomics.reporting as reporting_pkg
    from atomics.reporting.reporting import print_recent_runs

    assert reporting_pkg.print_recent_runs is print_recent_runs
