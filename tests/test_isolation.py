"""Tests must never touch the operator's real database or keychain.

On macOS the default is ``data/atomics.db`` in the checkout, so a test that
built settings without a ``db_path`` used to open (and could migrate) it, and
``load_settings`` filled API keys from the login keychain.
"""

from __future__ import annotations

from pathlib import Path

import keyring

from atomics.api.config import ServerSettings
from atomics.config import load_settings


def test_default_database_is_a_temp_file(tmp_path_factory) -> None:
    temp_root = Path(tmp_path_factory.getbasetemp()).resolve()
    for db_path in (ServerSettings().db_path, load_settings().db_path):
        assert temp_root in Path(db_path).resolve().parents


def test_keychain_is_the_null_backend() -> None:
    assert type(keyring.get_keyring()).__module__ == "keyring.backends.null"


def test_a_hung_test_times_out(request) -> None:
    assert request.config.pluginmanager.hasplugin("timeout")
    assert float(request.config.getini("timeout")) > 0
