"""The PyPI name is stoneburner-atomics; the import and CLI stay atomics.

`stoneburner` ultranormalizes to the same string as the existing
`stone-burner` Terraform helper, so PyPI rejects it as too similar.
"""

from importlib.metadata import entry_points, metadata, version

import atomics
from atomics.api.config import ServerSettings
from atomics.api.server import create_app


def test_installed_distribution_is_stoneburner_atomics():
    assert atomics.DIST_NAME == "stoneburner-atomics"
    assert metadata(atomics.DIST_NAME)["Name"].lower() == "stoneburner-atomics"


def test_import_version_reads_the_distribution():
    assert atomics.__version__ == version(atomics.DIST_NAME)
    assert atomics.__version__ != "0.0.0+unknown"


def test_console_script_is_still_atomics():
    names = {ep.name for ep in entry_points(group="console_scripts")}
    assert "atomics" in names


def test_the_api_advertises_the_distribution_version():
    app = create_app(settings=ServerSettings(no_auth=True))
    assert app.version == version(atomics.DIST_NAME)
