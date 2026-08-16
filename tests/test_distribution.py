"""The PyPI name is stoneburner; the import and CLI stay atomics."""

from importlib.metadata import entry_points, metadata, version

import atomics
from atomics.api.config import ServerSettings
from atomics.api.server import create_app


def test_installed_distribution_is_stoneburner():
    assert metadata("stoneburner")["Name"].lower() == "stoneburner"


def test_import_version_reads_the_stoneburner_distribution():
    assert atomics.__version__ == version("stoneburner")
    assert atomics.__version__ != "0.0.0+unknown"


def test_console_script_is_still_atomics():
    names = {ep.name for ep in entry_points(group="console_scripts")}
    assert "atomics" in names


def test_the_api_advertises_the_stoneburner_version():
    app = create_app(settings=ServerSettings(no_auth=True))
    assert app.version == version("stoneburner")
