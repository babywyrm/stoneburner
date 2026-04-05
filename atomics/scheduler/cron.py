"""Cron integration helpers — generate crontab entries, systemd timers, launchd plists."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def generate_crontab_entry(
    interval_minutes: int = 30,
    max_iterations: int = 10,
    python_path: str | None = None,
    tier: str = "baseline",
) -> str:
    py = python_path or shutil.which("python3") or sys.executable
    return (
        f"*/{interval_minutes} * * * * "
        f"cd {Path.cwd()} && {py} -m atomics run --tier {tier} "
        f"--max-iterations {max_iterations} >> logs/atomics-cron.log 2>&1"
    )


def generate_systemd_timer(
    interval_minutes: int = 30,
    max_iterations: int = 10,
    working_dir: str | None = None,
    tier: str = "baseline",
) -> tuple[str, str]:
    wd = working_dir or str(Path.cwd())
    py = shutil.which("python3") or sys.executable

    service = f"""[Unit]
Description=Atomics token usage benchmark run ({tier})
After=network.target

[Service]
Type=oneshot
WorkingDirectory={wd}
ExecStart={py} -m atomics run --tier {tier} --max-iterations {max_iterations}
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
