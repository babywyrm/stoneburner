"""Post-run shell hooks and optional desktop notifications (macOS / Linux)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from collections.abc import Mapping

from atomics.models import RunSummary


def hook_env(
    summary: RunSummary,
    *,
    tier: str,
    provider: str,
) -> dict[str, str]:
    return {
        "ATOMICS_RUN_ID": summary.run_id,
        "ATOMICS_TIER": tier,
        "ATOMICS_PROVIDER": provider,
        "ATOMICS_TOTAL_TASKS": str(summary.total_tasks),
        "ATOMICS_SUCCESSFUL_TASKS": str(summary.successful_tasks),
        "ATOMICS_FAILED_TASKS": str(summary.failed_tasks),
        "ATOMICS_TOTAL_TOKENS": str(summary.total_tokens),
        "ATOMICS_TOTAL_COST_USD": f"{summary.total_cost_usd:.6f}",
    }


def run_post_hook(cmd: str, extra_env: Mapping[str, str]) -> int:
    """Run a shell command with extra env vars merged into the current environment."""
    env = {**os.environ, **extra_env}
    return subprocess.run(cmd, shell=True, env=env, check=False).returncode


def _escape_applescript_string(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def notify_run_complete(summary: RunSummary, *, title: str = "Atomics") -> None:
    """Best-effort desktop notification (osascript on macOS, notify-send on Linux)."""
    body = (
        f"Run {summary.run_id}: {summary.total_tasks} tasks, "
        f"${summary.total_cost_usd:.4f}, {summary.total_tokens} tokens"
    )
    system = platform.system()
    if system == "Darwin" and shutil.which("osascript"):
        script = (
            f'display notification "{_escape_applescript_string(body)}" '
            f'with title "{_escape_applescript_string(title)}"'
        )
        subprocess.run(["osascript", "-e", script], check=False, capture_output=True)
    elif system == "Linux" and shutil.which("notify-send"):
        subprocess.run(["notify-send", title, body], check=False, capture_output=True)
