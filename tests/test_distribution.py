"""The PyPI name is stoneburner-atomics; the import and CLI stay atomics.

`stoneburner` ultranormalizes to the same string as the existing
`stone-burner` Terraform helper, so PyPI rejects it as too similar.
"""

from importlib.metadata import entry_points, metadata, version
from pathlib import Path

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


def test_pypi_listing_has_discoverability_metadata():
    meta = metadata(atomics.DIST_NAME)
    keywords = {part.strip().lower() for part in meta.get("Keywords", "").split(",") if part.strip()}
    assert {"llm", "eval", "ollama", "prompt-injection", "mcp"} <= keywords
    classifiers = meta.get_all("Classifier") or []
    assert "License :: OSI Approved :: MIT License" in classifiers
    assert "Topic :: Security" in classifiers
    summary = meta["Summary"]
    assert "atomics" in summary.lower()
    assert "ollama" in summary.lower() or "local" in summary.lower()
    urls = " ".join(meta.get_all("Project-URL") or [])
    assert "Documentation" in urls
    assert "Changelog" in urls
    assert "Issues" in urls


def test_readme_leads_with_pypi_install():
    text = Path("README.md").read_text()
    assert "uv tool install stoneburner-atomics" in text
    assert "once a release has been uploaded" not in text
    assert "http://localhost:11434" in text


def test_readme_landing_is_a_storefront_not_only_a_manual():
    """A stranger on GitHub or PyPI should see what it is in one scroll."""
    text = Path("README.md").read_text()
    assert "https://img.shields.io/pypi/v/stoneburner-atomics" in text
    assert "https://pypi.org/project/stoneburner-atomics/" in text
    assert "n/a (scored/total scored)" in text
    assert "atomics doctor" in text
    assert "atomics toolcall" in text
    lowered = text.lower()
    assert "brainbox" not in lowered
    assert "garak" not in lowered
    assert "promptfoo" not in lowered
