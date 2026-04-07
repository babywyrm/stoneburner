"""Cross-platform default paths (XDG on Linux)."""

from __future__ import annotations

import os
import platform
from pathlib import Path


def default_data_dir() -> Path:
    """User data directory for Atomics state (database).

    Linux: ``$XDG_DATA_HOME/atomics`` or ``~/.local/share/atomics``.
    Other platforms: ``data`` under the current working directory (unchanged).
    """
    if platform.system() == "Linux":
        xdg = os.environ.get("XDG_DATA_HOME", "").strip()
        base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
        return base / "atomics"
    return Path("data")


def default_db_path() -> Path:
    return default_data_dir() / "atomics.db"
