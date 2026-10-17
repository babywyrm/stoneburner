"""Moved modules keep their original import paths."""

from __future__ import annotations


def test_reporting_package_still_exports_the_old_module() -> None:
    import atomics.reporting as reporting_pkg
    from atomics.reporting.reporting import print_recent_runs

    assert reporting_pkg.print_recent_runs is print_recent_runs
