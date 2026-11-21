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


def write_private_text(path: Path, data: str, *, mkdir_parents: bool = False) -> None:
    """Write UTF-8 text to ``path`` with ``0o600`` owner-only permissions."""
    if mkdir_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    try:
        path.chmod(0o600)
    except OSError:
        pass
