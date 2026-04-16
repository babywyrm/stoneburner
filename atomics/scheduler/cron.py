"""Helpers to generate crontab, systemd timer, and launchd plist entries."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path


def generate_crontab_entry(
    interval_minutes: int = 30,
    max_iterations: int = 10,
    python_path: str | None = None,
    tier: str = "baseline",
    provider: str = "claude",
) -> str:
    py = python_path or shutil.which("python3") or sys.executable
    return (
        f"*/{interval_minutes} * * * * "
        f"cd {Path.cwd()} && {py} -m atomics run --tier {tier} --provider {provider} "
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
    wd = working_dir or str(Path.cwd())
    py = shutil.which("python3") or sys.executable

    service = f"""[Unit]
Description=Atomics token usage benchmark run ({tier})
After=network.target

[Service]
Type=oneshot
WorkingDirectory={wd}
ExecStart={py} -m atomics run --tier {tier} --provider {provider} --trigger scheduled --max-iterations {max_iterations}
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
    wd = working_dir or str(Path.cwd())
    py = shutil.which("python3") or sys.executable
    interval_seconds = interval_minutes * 60

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}.{tier}</string>
    <key>WorkingDirectory</key>
    <string>{wd}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{py}</string>
        <string>-m</string>
        <string>atomics</string>
        <string>run</string>
        <string>--tier</string>
        <string>{tier}</string>
        <string>--provider</string>
        <string>{provider}</string>
        <string>--trigger</string>
        <string>scheduled</string>
        <string>--max-iterations</string>
        <string>{max_iterations}</string>
    </array>
    <key>StartInterval</key>
    <integer>{interval_seconds}</integer>
    <key>StandardOutPath</key>
    <string>{wd}/logs/atomics.log</string>
    <key>StandardErrorPath</key>
    <string>{wd}/logs/atomics.log</string>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
"""


# ── Installation helpers ──────────────────────────────────


def install_crontab(entry: str, *, run_cmd=subprocess.run) -> str:
    """Append an atomics crontab entry, avoiding duplicates."""
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


def uninstall_crontab(*, run_cmd=subprocess.run) -> str:
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
    run_cmd=subprocess.run,
    home_dir: Path | None = None,
) -> str:
    """Write plist and load it via launchctl."""
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
    run_cmd=subprocess.run,
    home_dir: Path | None = None,
) -> str:
    """Unload and remove launchd plist."""
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
    run_cmd=subprocess.run,
    home_dir: Path | None = None,
) -> str:
    """Write systemd user units and enable the timer."""
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
    run_cmd=subprocess.run,
    home_dir: Path | None = None,
) -> str:
    """Disable and remove systemd user units."""
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


def detect_best_scheduler(*, system_name=platform.system, which=shutil.which) -> str:
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
    run_cmd=subprocess.run,
    home_dir: Path | None = None,
) -> bool:
    """Check whether the OS-level schedule is actually active."""
    if fmt == "launchd":
        result = run_cmd(
            ["launchctl", "list", f"{label}.{tier}"],
            capture_output=True, text=True,
        )
        return result.returncode == 0
    elif fmt == "systemd":
        result = run_cmd(
            ["systemctl", "--user", "is-active", f"atomics-{tier}.timer"],
            capture_output=True, text=True,
        )
        return result.returncode == 0 and "active" in result.stdout
    elif fmt == "crontab":
        result = run_cmd(
            ["crontab", "-l"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return False
        return "atomics-managed" in result.stdout and f"--tier {tier}" in result.stdout
    return False
