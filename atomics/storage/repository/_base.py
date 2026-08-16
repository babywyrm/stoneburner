"""Shared connection for MetricsRepository mixins."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from atomics.storage.schema import init_db


class RepositoryBase:
    _conn: sqlite3.Connection

    def __init__(self, db_path: Path) -> None:
        self._conn = init_db(db_path)

    def close(self) -> None:
        self._conn.close()
