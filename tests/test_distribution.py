"""The PyPI name is stoneburner-atomics; the import and CLI stay atomics.

`stoneburner` ultranormalizes to the same string as the existing
`stone-burner` Terraform helper, so PyPI rejects it as too similar.
"""

import hashlib
import re
import subprocess
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
    keywords = {
        part.strip().lower() for part in meta.get("Keywords", "").split(",") if part.strip()
    }
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
    assert "atomics provider-test --provider ollama --no-thinking" in text
    lowered = text.lower()
    assert "garak" not in lowered
    assert "promptfoo" not in lowered


# ponytail: sha256 hides the names from grep and readers, not from a
# dictionary guess. Add a name by hashing its lowercase form.
_LAB_HOST_HASHES = frozenset(
    {
        "6267fcbbe76a73a3",
        "7a70093427b414ca",
        "9535deb12f7d4b83",
        "a856911711504fe4",
        "bd68bebf91242ada",
        "ea28aab4ade0f034",
    }
)
_FICTIONAL_LAN_FILES = frozenset(
    {"atomics/eval/fixtures.py", "atomics/eval/redblue/fixtures.py", "tests/test_distribution.py"}
)


def _lab_leaks(name: str, text: str) -> bool:
    words = {
        hashlib.sha256(w.encode()).hexdigest()[:16] for w in re.findall(r"[a-z0-9]+", text.lower())
    }
    lan = name not in _FICTIONAL_LAN_FILES and re.search(r"\b192\.168\.\d+\.\d+\b", text)
    return bool(words & _LAB_HOST_HASHES or lan)


def test_no_tracked_file_names_a_lab_host():
    """Docs and fixtures use neutral hosts (localhost, gpu-host-a, <host>)."""
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    hits = [
        name
        for name in tracked
        if Path(name).is_file() and _lab_leaks(name, Path(name).read_text(errors="ignore"))
    ]
    assert hits == []


def test_lab_leak_check_catches_names_and_lan_addresses():
    assert not _lab_leaks("README.md", "gpu-host-a at http://<host>:11434")
    assert _lab_leaks("README.md", "http://192.168.7.7:11434")
    assert not _lab_leaks("atomics/eval/fixtures.py", "10.0.0.1 and 192.168.7.7")


def test_readme_stays_a_storefront():
    """Landing page, not a second CLI manual.

    ponytail: 160-line ceiling. Raise it only if a stranger still cannot
    tell what the tool is; recipes belong in QUICKSTART / CLI_REFERENCE.
    """
    lines = Path("README.md").read_text().splitlines()
    assert len(lines) <= 160, f"README is {len(lines)} lines; keep storefront-only"


def test_readme_links_the_operator_manuals():
    """PyPI renders README, so these have to be absolute GitHub URLs."""
    text = Path("README.md").read_text()
    root = "https://github.com/babywyrm/stoneburner/blob/main/"
    for path in (
        "QUICKSTART.md",
        "CONTRIBUTING.md",
        "ARCHITECTURE.md",
        "docs/CLI_REFERENCE.md",
        "docs/REPL.md",
        "docs/API_SERVER.md",
        "docs/MCP_SERVER.md",
        "docs/SECURITY_SUITES.md",
        "docs/LOAD_TESTING.md",
        "docs/THINKING.md",
        "docs/INFERENCE_ENV.md",
        "docs/COMPARING.md",
        "docs/LEADERBOARD.md",
    ):
        assert f"{root}{path}" in text, path


def test_mypy_strictness_flags_stay_enabled():
    """Guard against relaxing [tool.mypy] once strict is green across atomics/ and scripts/."""
    import tomllib

    import yaml

    cfg = tomllib.loads(Path("pyproject.toml").read_text())["tool"]["mypy"]
    assert cfg.get("strict") is True, "[tool.mypy] strict must remain true"
    assert cfg.get("extra_checks") is True, "[tool.mypy] extra_checks must remain true"

    for entry in cfg.get("overrides", []):
        mods = (
            entry.get("module", [])
            if isinstance(entry.get("module"), list)
            else [entry.get("module")]
        )
        assert not any(
            isinstance(m, str) and (m.startswith("atomics") or m.startswith("scripts"))
            for m in mods
        ), f"Do not add per-module mypy overrides for atomics/scripts: {mods}"

    ci_wf = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())
    ci_steps = "\n".join(step.get("run", "") for step in ci_wf["jobs"]["test"]["steps"])
    assert "uv run mypy atomics/ scripts/" in ci_steps


def test_uv_lock_records_sha256_hashes_for_all_packages():
    """Every resolved external distribution in uv.lock must carry sha256 digests."""
    import tomllib

    lock = tomllib.loads(Path("uv.lock").read_text())
    packages = lock.get("package", [])
    assert packages, "uv.lock must contain resolved packages"
    missing: list[str] = []
    for pkg in packages:
        source = pkg.get("source", {})
        if source.get("editable") or source.get("virtual"):
            continue
        sdist_hash = (pkg.get("sdist") or {}).get("hash", "")
        wheel_hashes = [w.get("hash", "") for w in pkg.get("wheels", [])]
        hashes = [h for h in [sdist_hash, *wheel_hashes] if h]
        if not hashes or not all(h.startswith("sha256:") for h in hashes):
            missing.append(str(pkg.get("name")))
    assert missing == [], f"uv.lock packages missing sha256 digests: {missing}"


def test_security_workflow_enforces_hash_verified_audit_and_all_gates():
    """Guard CI security gates: hash-verified pip-audit, zizmor, bandit, gitleaks, trufflehog."""
    import yaml

    wf = yaml.safe_load(Path(".github/workflows/security.yml").read_text())
    jobs = wf["jobs"]

    dep_steps = "\n".join(step.get("run", "") for step in jobs["dependency-audit"]["steps"])
    assert "uv lock --check" in dep_steps
    assert "uv export --frozen" in dep_steps
    assert "--no-hashes" not in dep_steps
    assert "--require-hashes" in dep_steps
    assert "--strict" in dep_steps
    assert "--disable-pip" in dep_steps
    assert "pypi" in dep_steps and "osv" in dep_steps

    static_steps = "\n".join(step.get("run", "") for step in jobs["static-analysis"]["steps"])
    assert "bandit" in static_steps
    assert "zizmor" in static_steps

    secret_steps = "\n".join(step.get("run", "") for step in jobs["secret-scan"]["steps"])
    assert "gitleaks git" in secret_steps
    assert "gitleaks dir" in secret_steps
    assert "trufflehog" in secret_steps
    assert "trufflehog --no-update git file://." in secret_steps
    assert "trufflehog --no-update filesystem ." in secret_steps
    assert "sha256sum --check --strict" in secret_steps
