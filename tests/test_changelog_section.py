"""Tests for the release-notes extractor.

Release notes are only trustworthy if they cannot drift from the changelog, so
these cover both the parsing and the invariant that every released tag has a
section to publish.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from changelog_section import (  # noqa: E402
    CHANGELOG as CHANGELOG_PATH,
)
from changelog_section import (
    VersionNotFoundError,
    list_versions,
    main,
    section,
)

SAMPLE = """# Changelog

## 1.2.0 (2026-01-02) — Second thing

### Added
- A feature.

## 1.1.0 (2026-01-01) — First thing

### Fixed
- A bug.

## 1.0.0

Initial release.
"""


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    path = tmp_path / "CHANGELOG.md"
    path.write_text(SAMPLE, encoding="utf-8")
    return path


def test_body_stops_at_the_next_version(sample):
    _, body = section("1.2.0", sample)
    assert body == "### Added\n- A feature."
    # The neighbouring entry must not bleed in, which is the whole risk here.
    assert "First thing" not in body
    assert "A bug." not in body


def test_the_last_section_runs_to_the_end_of_the_file(sample):
    _, body = section("1.0.0", sample)
    assert body == "Initial release."


def test_the_title_carries_the_summary(sample):
    title, _ = section("1.2.0", sample)
    assert title == "v1.2.0 — Second thing"


def test_a_heading_without_a_summary_still_yields_a_title(sample):
    title, _ = section("1.0.0", sample)
    assert title == "v1.0.0"


def test_a_leading_v_is_accepted(sample):
    """Tags are `v1.2.0`; changelog headings are `1.2.0`."""
    assert section("v1.2.0", sample) == section("1.2.0", sample)


def test_an_unknown_version_names_the_ones_that_exist(sample):
    with pytest.raises(VersionNotFoundError) as exc:
        section("9.9.9", sample)
    assert "1.2.0" in str(exc.value)


def test_versions_are_listed_newest_first(sample):
    assert list_versions(sample) == ["1.2.0", "1.1.0", "1.0.0"]


def test_a_missing_version_exits_nonzero():
    """The release workflow relies on this to fail rather than publish nothing."""
    assert main(["9.9.9"]) == 1


def test_the_real_changelog_parses():
    versions = list_versions()
    assert "0.13.1" in versions
    title, body = section("0.13.1")
    assert title.startswith("v0.13.1 — ")
    assert body


def _released_tags() -> list[str]:
    out = subprocess.run(
        ["git", "tag", "--list", "v[0-9]*.[0-9]*.[0-9]*"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    return sorted(t for t in out.stdout.split() if t)


def _require_full_history() -> None:
    """Skip when the checkout cannot answer the question.

    These invariants are about tags and the version history of pyproject.toml,
    and `actions/checkout` clones shallow and tagless by default. That made this
    module fail CI while passing locally, so the workflows now fetch history and
    a checkout that still lacks it skips loudly instead of failing or, worse,
    passing vacuously against an empty tag list.
    """
    if not _released_tags():
        pytest.skip("no tags in this checkout; needs fetch-depth: 0 and fetch-tags")
    shallow = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    if shallow.stdout.strip() == "true":
        pytest.skip("shallow checkout; needs fetch-depth: 0")


def test_every_released_tag_has_release_notes():
    """A tag with no changelog section publishes an empty GitHub release.

    That is exactly how v0.8.0, v0.12.0, v0.13.0 and v0.13.1 ended up with a bare
    compare link for a body.
    """
    _require_full_history()
    missing = []
    for tag in _released_tags():
        try:
            _, body = section(tag)
        except VersionNotFoundError:
            missing.append(tag)
            continue
        if not body.strip():
            missing.append(f"{tag} (empty)")
    assert not missing, f"tags without usable release notes: {missing}"


class TestPlaceholderDetection:
    """`sync_releases` decides from this whether to overwrite a release body.

    Getting it wrong in one direction leaves an empty release published; in the
    other it destroys hand-written notes, so both are covered.
    """

    @staticmethod
    def _is_placeholder(body: str) -> bool:
        from sync_releases import is_placeholder

        return is_placeholder(body)

    def test_an_empty_body_is_a_placeholder(self):
        assert self._is_placeholder("")
        assert self._is_placeholder("   \n\n  ")

    def test_a_bare_compare_link_is_a_placeholder(self):
        body = (
            "**Full Changelog**: "
            "https://github.com/babywyrm/stoneburner/compare/v0...v0.13.1"
        )
        assert self._is_placeholder(body)
        assert self._is_placeholder(f"\n{body}\n\n")

    def test_hand_written_notes_are_not_a_placeholder(self):
        assert not self._is_placeholder("## Highlights\n\n- Something real.")

    def test_notes_that_also_carry_a_compare_link_are_kept(self):
        """The link is boilerplate; the prose above it is not."""
        body = (
            "## Highlights\n\n- Something real.\n\n"
            "**Full Changelog**: https://example.com/compare/v1...v2"
        )
        assert not self._is_placeholder(body)


def _versions_ever_declared() -> set[str]:
    """Every version that has appeared in pyproject.toml, from git history."""
    out = subprocess.run(
        ["git", "log", "-p", "--", "pyproject.toml"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    return {
        line.split("=", 1)[1].strip().strip('"')
        for line in out.stdout.splitlines()
        if line.startswith("+version = ")
    }


def test_no_version_ever_shipped_undocumented():
    """A version that existed in pyproject must be accounted for somewhere.

    0.4.0 and 0.5.0 were bumped and never written up, so the changelog jumped
    from 0.3.0 to 0.6.0 with nothing explaining it. Either a section or an
    explicit note is enough — the point is that no version silently vanishes.
    """
    _require_full_history()
    changelog_text = CHANGELOG_PATH.read_text(encoding="utf-8")
    documented = set(list_versions())
    unaccounted = [
        v
        for v in sorted(_versions_ever_declared())
        if v not in documented and v not in changelog_text
    ]
    assert not unaccounted, f"versions with no changelog entry or note: {unaccounted}"


def test_every_documented_version_is_tagged():
    """The reverse drift: a changelog entry nobody ever tagged.

    0.11.0 was documented, bumped in pyproject, and never released.
    """
    _require_full_history()
    tagged = {t.lstrip("v") for t in _released_tags()}
    documented = set(list_versions())
    # Pre-0.6.0 predates tagging in this repo.
    expected = {v for v in documented if tuple(map(int, v.split("."))) >= (0, 6, 0)}
    assert expected - tagged == set(), f"documented but never tagged: {sorted(expected - tagged)}"
