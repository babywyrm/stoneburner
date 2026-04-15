"""Environment diagnostics for macOS and Linux."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import sqlite3
import sys

from rich.console import Console

from atomics.config import AtomicsSettings
from atomics.paths import default_data_dir, default_db_path
from atomics.scheduler.cron import detect_best_scheduler


def run_doctor(settings: AtomicsSettings | None = None) -> int:
    """Print diagnostics. Returns 0 if healthy, 1 if blocking issues."""
    console = Console()
    settings = settings or AtomicsSettings()
    errors = 0

    v = sys.version_info
    major, minor, micro = v[0], v[1], v[2]
    if (major, minor) < (3, 11):
        console.print(f"[red]Python 3.11+ required; found {major}.{minor}[/red]")
        errors += 1
    else:
        console.print(f"[green]Python {major}.{minor}.{micro}[/green] OK")

    console.print(f"[dim]Platform:[/dim] {platform.system()} ({platform.machine()})")

    db_path = settings.db_path
    console.print(f"[dim]Database path:[/dim] {db_path.resolve()}")
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("SELECT 1")
        conn.close()
        console.print("[green]SQLite database[/green] OK (readable / creatable)")
    except OSError as exc:
        console.print(f"[red]Database path not usable:[/red] {exc}")
        errors += 1

    if settings.anthropic_api_key:
        console.print("[green]ANTHROPIC_API_KEY[/green] set")
    else:
        console.print(
            "[yellow]ANTHROPIC_API_KEY[/yellow] not set (required for Claude / provider-test)"
        )

    if settings.openai_api_key:
        console.print("[green]OPENAI_API_KEY[/green] set")
    else:
        console.print("[yellow]OPENAI_API_KEY[/yellow] not set (optional; needed for OpenAI)")

    if importlib.util.find_spec("openai") is not None:
        console.print("[green]openai[/green] SDK available (OpenAI / Codex)")
    else:
        console.print(
            "[yellow]openai[/yellow] SDK not installed (optional; uv sync --extra openai)"
        )

    if importlib.util.find_spec("boto3") is not None:
        console.print("[green]boto3[/green] available (Bedrock)")
        try:
            import boto3

            sts = boto3.client("sts")
            identity = sts.get_caller_identity()
            console.print(
                f"[green]AWS credentials[/green] valid (account {identity.get('Account', '?')})"
            )
        except Exception:
            console.print(
                "[yellow]AWS credentials[/yellow] not configured or invalid "
                "(check AWS_ACCESS_KEY_ID / AWS_PROFILE)"
            )
    else:
        console.print("[yellow]boto3[/yellow] not installed (optional; needed for Bedrock)")

    sched = detect_best_scheduler()
    console.print(f"[dim]Preferred scheduler:[/dim] [cyan]{sched}[/cyan]")
    if sched == "crontab" and not shutil.which("crontab"):
        console.print("[yellow]crontab[/yellow] binary not found on PATH")
    if sched == "systemd" and not shutil.which("systemctl"):
        console.print("[yellow]systemctl[/yellow] not found on PATH")
    if sched == "launchd" and platform.system() != "Darwin":
        console.print("[dim]launchd is normal on macOS only[/dim]")

    if platform.system() == "Linux":
        data_dir = default_data_dir()
        console.print(f"[dim]Default data dir (Linux):[/dim] {data_dir}")
        console.print(f"[dim]Default DB without ATOMICS_DB_PATH:[/dim] {default_db_path()}")

    notify_bins = (
        ("osascript", "macOS notifications"),
        ("notify-send", "Linux notifications"),
    )
    for binary, desc in notify_bins:
        if shutil.which(binary):
            console.print(f"[green]{binary}[/green] found ({desc})")

    return 1 if errors else 0
