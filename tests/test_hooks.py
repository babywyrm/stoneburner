"""Tests for post-run hooks and notifications."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from atomics.hooks import hook_env, notify_run_complete, run_post_hook
from atomics.models import RunSummary


def _sample_summary() -> RunSummary:
    return RunSummary(
        run_id="abc123",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        total_tasks=2,
        successful_tasks=2,
        failed_tasks=0,
        total_tokens=100,
        total_cost_usd=0.01,
    )


def test_hook_env_contains_expected_keys():
    env = hook_env(_sample_summary(), tier="ez", provider="claude")
    assert env["ATOMICS_RUN_ID"] == "abc123"
    assert env["ATOMICS_TIER"] == "ez"
    assert env["ATOMICS_PROVIDER"] == "claude"
    assert "ATOMICS_TOTAL_COST_USD" in env


def test_run_post_hook_merges_env(monkeypatch):
    captured: dict = {}

    def fake_run(cmd, shell=True, env=None, check=False):
        captured["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("atomics.hooks.subprocess.run", fake_run)
    rc = run_post_hook("true", {"ATOMICS_RUN_ID": "x"})
    assert rc == 0
    assert captured["env"]["ATOMICS_RUN_ID"] == "x"


@pytest.mark.parametrize("system", ["Darwin", "Linux"])
def test_notify_skips_when_no_binary(monkeypatch, system):
    monkeypatch.setattr("atomics.hooks.platform.system", lambda: system)
    monkeypatch.setattr("atomics.hooks.shutil.which", lambda _n: None)
    notify_run_complete(_sample_summary())  # should not raise


def test_notify_macos_runs_osascript(monkeypatch):
    monkeypatch.setattr("atomics.hooks.platform.system", lambda: "Darwin")
    def _which(name: str) -> str | None:
        return "/usr/bin/osascript" if name == "osascript" else None

    monkeypatch.setattr("atomics.hooks.shutil.which", _which)
    mock_run = MagicMock(return_value=MagicMock(returncode=0))
    monkeypatch.setattr("atomics.hooks.subprocess.run", mock_run)
    notify_run_complete(_sample_summary())
    assert mock_run.called
