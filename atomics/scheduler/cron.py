"""Helpers to generate crontab, systemd timer, and launchd plist entries."""

from __future__ import annotations

import platform
import plistlib
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from atomics.models import BurnTier
from atomics.providers.factory import PROVIDER_NAMES

_VALID_TIERS: frozenset[str] = frozenset(t.value for t in BurnTier)
_VALID_PROVIDERS: frozenset[str] = frozenset(PROVIDER_NAMES)


def _has_control_chars(value: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in value)


def _validate_interval_minutes(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > 1440:
        raise ValueError(f"interval_minutes must be an integer in 1..1440, got {value!r}")
    return value


def _validate_max_iterations(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > 100_000:
        raise ValueError(f"max_iterations must be an integer in 1..100000, got {value!r}")
    return value


def _validate_tier(tier: str) -> str:
    if tier not in _VALID_TIERS:
        raise ValueError(f"Invalid tier {tier!r}; expected one of {sorted(_VALID_TIERS)}")
    return tier


def _validate_provider(provider: str) -> str:
    if provider not in _VALID_PROVIDERS:
        raise ValueError(
            f"Invalid provider {provider!r}; expected one of {sorted(_VALID_PROVIDERS)}"
        )
    return provider


def _validate_label(label: str) -> str:
    if (
        not label
        or _has_control_chars(label)
        or "/" in label
        or "\\" in label
        or ".." in label
        or label.startswith(".")
    ):
        raise ValueError(f"Invalid schedule label {label!r}")
    return label


def _validate_single_line_path(value: str, *, field: str) -> str:
    if not value or _has_control_chars(value):
        raise ValueError(f"{field} must be non-empty and free of control characters")
    return value


class _CmdResult(Protocol):
    @property
    def returncode(self) -> int: ...

    @property
    def stdout(self) -> str: ...

    @property
    def stderr(self) -> str: ...


_RunCmd = Callable[..., _CmdResult]


def generate_crontab_entry(
    interval_minutes: int = 30,
    max_iterations: int = 10,
    python_path: str | None = None,
    tier: str = "baseline",
    provider: str = "claude",
) -> str:
    _validate_interval_minutes(interval_minutes)
    _validate_max_iterations(max_iterations)
    _validate_tier(tier)
    _validate_provider(provider)
    py = _validate_single_line_path(
        python_path or shutil.which("python3") or sys.executable, field="python_path"
    )
    cwd = _validate_single_line_path(str(Path.cwd()), field="working_dir")
    return (
        f"*/{interval_minutes} * * * * "
        f"cd {shlex.quote(cwd)} && {shlex.quote(py)} -m atomics run "
        f"--tier {tier} --provider {provider} "
        f"--trigger scheduled "
        f"--max-iterations {max_iterations} >> logs/atomics-cron.log 2>&1"
    )


def generate_systemd_timer(
    interval_minutes: int = 30,
    max_iterations: int = 10,
    working_dir: str | None = None,
    tier: str = "baseline",
    provider: str = "claude",
) -> tuple[str, str]:
    _validate_interval_minutes(interval_minutes)
    _validate_max_iterations(max_iterations)
    _validate_tier(tier)
    _validate_provider(provider)
    wd = _validate_single_line_path(working_dir or str(Path.cwd()), field="working_dir")
    py = _validate_single_line_path(shutil.which("python3") or sys.executable, field="python_path")
    exec_start = (
        f"{shlex.quote(py)} -m atomics run --tier {tier} --provider {provider} "
        f"--trigger scheduled --max-iterations {max_iterations}"
    )

    service = f"""[Unit]
Description=Atomics token usage benchmark run ({tier})
After=network.target

[Service]
Type=oneshot
WorkingDirectory={wd}
ExecStart={exec_start}
StandardOutput=append:{wd}/logs/atomics.log
StandardError=append:{wd}/logs/atomics.log
"""

    timer = f"""[Unit]
Description=Atomics benchmark timer ({tier})

[Timer]
OnCalendar=*:0/{interval_minutes}
Persistent=true
RandomizedDelaySec=60

[Install]
WantedBy=timers.target
"""
    return service, timer


def generate_launchd_plist(
    interval_minutes: int = 30,
    max_iterations: int = 10,
    working_dir: str | None = None,
    label: str = "com.babywyrm.atomics",
    tier: str = "baseline",
    provider: str = "claude",
) -> str:
    _validate_interval_minutes(interval_minutes)
    _validate_max_iterations(max_iterations)
    _validate_label(label)
    _validate_tier(tier)
    _validate_provider(provider)
    wd = _validate_single_line_path(working_dir or str(Path.cwd()), field="working_dir")
    py = _validate_single_line_path(shutil.which("python3") or sys.executable, field="python_path")
    interval_seconds = interval_minutes * 60
    log_path = f"{wd}/logs/atomics.log"
    payload = {
        "Label": f"{label}.{tier}",
        "WorkingDirectory": wd,
        "ProgramArguments": [
            py,
            "-m",
            "atomics",
            "run",
            "--tier",
            tier,
            "--provider",
            provider,
            "--trigger",
            "scheduled",
            "--max-iterations",
            str(max_iterations),
        ],
        "StartInterval": interval_seconds,
        "StandardOutPath": log_path,
        "StandardErrorPath": log_path,
        "RunAtLoad": True,
    }
    return plistlib.dumps(payload, sort_keys=False).decode("utf-8")


# ── Installation helpers ──────────────────────────────────


def install_crontab(entry: str, *, run_cmd: _RunCmd = subprocess.run) -> str:
    """Append an atomics crontab entry, avoiding duplicates."""
    _validate_single_line_path(entry, field="crontab entry")
    result = run_cmd(["crontab", "-l"], capture_output=True, text=True)
    existing = result.stdout if result.returncode == 0 else ""

    marker = "# atomics-managed"
    clean_lines = [line for line in existing.splitlines() if marker not in line]
    clean_lines.append(f"{entry}  {marker}")
    new_crontab = "\n".join(clean_lines) + "\n"

    proc = run_cmd(["crontab", "-"], input=new_crontab, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to install crontab: {proc.stderr}")
    return "Crontab entry installed"


def uninstall_crontab(*, run_cmd: _RunCmd = subprocess.run) -> str:
    """Remove all atomics-managed crontab entries."""
    result = run_cmd(["crontab", "-l"], capture_output=True, text=True)
    if result.returncode != 0:
        return "No crontab found"

    marker = "# atomics-managed"
    clean_lines = [line for line in result.stdout.splitlines() if marker not in line]
    new_crontab = "\n".join(clean_lines) + "\n"

    run_cmd(["crontab", "-"], input=new_crontab, capture_output=True, text=True)
    return "Atomics crontab entries removed"


def install_launchd(
    plist_content: str,
    label: str = "com.babywyrm.atomics",
    tier: str = "baseline",
    *,
    run_cmd: _RunCmd = subprocess.run,
    home_dir: Path | None = None,
) -> str:
    """Write plist and load it via launchctl."""
    _validate_label(label)
    _validate_tier(tier)
    home = home_dir or Path.home()
    plist_dir = home / "Library" / "LaunchAgents"
    plist_dir.mkdir(parents=True, exist_ok=True)
    plist_path = plist_dir / f"{label}.{tier}.plist"

    run_cmd(
        ["launchctl", "unload", str(plist_path)],
        capture_output=True,
    )

    plist_path.write_text(plist_content)
    proc = run_cmd(
        ["launchctl", "load", str(plist_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to load launchd plist: {proc.stderr}")
    return f"LaunchAgent installed at {plist_path}"


def uninstall_launchd(
    label: str = "com.babywyrm.atomics",
    tier: str = "baseline",
    *,
    run_cmd: _RunCmd = subprocess.run,
    home_dir: Path | None = None,
) -> str:
    """Unload and remove launchd plist."""
    _validate_label(label)
    _validate_tier(tier)
    home = home_dir or Path.home()
    plist_path = home / "Library" / "LaunchAgents" / f"{label}.{tier}.plist"
    if not plist_path.exists():
        return f"No plist found at {plist_path}"

    run_cmd(["launchctl", "unload", str(plist_path)], capture_output=True)
    plist_path.unlink()
    return f"LaunchAgent removed: {plist_path}"


def install_systemd(
    service_content: str,
    timer_content: str,
    tier: str = "baseline",
    *,
    run_cmd: _RunCmd = subprocess.run,
    home_dir: Path | None = None,
) -> str:
    """Write systemd user units and enable the timer."""
    _validate_tier(tier)
    home = home_dir or Path.home()
    unit_dir = home / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)

    service_path = unit_dir / f"atomics-{tier}.service"
    timer_path = unit_dir / f"atomics-{tier}.timer"
    service_path.write_text(service_content)
    timer_path.write_text(timer_content)

    run_cmd(["systemctl", "--user", "daemon-reload"], capture_output=True)
    proc = run_cmd(
        ["systemctl", "--user", "enable", "--now", f"atomics-{tier}.timer"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to enable systemd timer: {proc.stderr}")
    return f"Systemd timer enabled: atomics-{tier}.timer"


def uninstall_systemd(
    tier: str = "baseline",
    *,
    run_cmd: _RunCmd = subprocess.run,
    home_dir: Path | None = None,
) -> str:
    """Disable and remove systemd user units."""
    _validate_tier(tier)
    run_cmd(
        ["systemctl", "--user", "disable", "--now", f"atomics-{tier}.timer"],
        capture_output=True,
    )
    home = home_dir or Path.home()
    unit_dir = home / ".config" / "systemd" / "user"
    for suffix in (".service", ".timer"):
        path = unit_dir / f"atomics-{tier}{suffix}"
        if path.exists():
            path.unlink()
    run_cmd(["systemctl", "--user", "daemon-reload"], capture_output=True)
    return f"Systemd units removed: atomics-{tier}"


def detect_best_scheduler(
    *,
    system_name: Callable[[], str] = platform.system,
    which: Callable[[str], str | None] = shutil.which,
) -> str:
    """Auto-detect the best scheduler for this platform."""
    if system_name() == "Darwin":
        return "launchd"
    if which("systemctl"):
        return "systemd"
    return "crontab"


def check_schedule_health(
    fmt: str,
    tier: str,
    *,
    label: str = "com.babywyrm.atomics",
    run_cmd: _RunCmd = subprocess.run,
    home_dir: Path | None = None,
) -> bool:
    """Check whether the OS-level schedule is actually active."""
    if fmt not in {"launchd", "systemd", "crontab"}:
        return False
    _validate_tier(tier)
    if fmt == "launchd":
        _validate_label(label)
        result = run_cmd(
            ["launchctl", "list", f"{label}.{tier}"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    elif fmt == "systemd":
        result = run_cmd(
            ["systemctl", "--user", "is-active", f"atomics-{tier}.timer"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and "active" in result.stdout
    elif fmt == "crontab":
        result = run_cmd(
            ["crontab", "-l"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False
        return "atomics-managed" in result.stdout and f"--tier {tier}" in result.stdout
    return False
