"""REPL modules must not import providers or storage."""

from __future__ import annotations

import subprocess
import sys


def test_importing_repl_does_not_load_providers_or_storage() -> None:
    code = """
import atomics.repl.loop
import atomics.repl.dispatch
import sys
banned = [
    n
    for n in sys.modules
    if n == "atomics.providers"
    or n.startswith("atomics.providers.")
    or n == "atomics.storage"
    or n.startswith("atomics.storage.")
]
assert banned == [], banned
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
