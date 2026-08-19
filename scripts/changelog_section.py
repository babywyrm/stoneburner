"""Extract one version's section from CHANGELOG.md.

The release workflow used `generate_release_notes`, which asks GitHub to
summarize commits since the previous tag. That produced empty release bodies:
the workflow also force-updates a floating `v0` tag to each release, so GitHub
picked `v0` as the previous tag and found nothing between it and the new one.
Meanwhile the hand-written changelog entry — the actual release notes — went
unpublished.

This reads the section straight out of CHANGELOG.md so the release notes and the
changelog cannot drift, and exits non-zero when a tag has no entry, which turns
"someone forgot to write the changelog" into a failed release rather than a
silently empty one.

    python scripts/changelog_section.py 0.13.1           # body
    python scripts/changelog_section.py --title 0.13.1   # release title
    python scripts/changelog_section.py --list           # every documented version
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"

# "## 0.13.1 (2026-07-25) — Fixes found by running it for real"
# The date and the em-dash summary are both optional: older entries have neither.
HEADING = re.compile(
    r"^##\s+v?(?P<version>\d+\.\d+\.\d+)"
    r"(?:\s+\((?P<date>[^)]*)\))?"
    r"(?:\s*[—-]\s*(?P<summary>.*))?$"
)


class VersionNotFoundError(LookupError):
    """No section in the changelog matches the requested version."""


def _lines(changelog: Path) -> list[str]:
    return changelog.read_text(encoding="utf-8").splitlines()


def list_versions(changelog: Path = CHANGELOG) -> list[str]:
    return [match.group("version") for line in _lines(changelog) if (match := HEADING.match(line))]


def section(version: str, changelog: Path = CHANGELOG) -> tuple[str, str]:
    """Return `(title, body)` for a version.

    The title is `vX.Y.Z — summary` when the heading carries a summary, and
    plain `vX.Y.Z` when it does not. The body is everything up to the next
    version heading, with surrounding blank lines removed.
    """
    wanted = version.lstrip("v")
    lines = _lines(changelog)
    start: int | None = None
    summary = ""

    for index, line in enumerate(lines):
        match = HEADING.match(line)
        if match is None:
            continue
        if start is not None:
            # The next version heading closes the section.
            return _build(wanted, summary, lines[start:index])
        if match.group("version") == wanted:
            start = index + 1
            summary = (match.group("summary") or "").strip()

    if start is None:
        available = ", ".join(list_versions(changelog)) or "none"
        raise VersionNotFoundError(
            f"No '## {wanted}' section in {changelog.name}. Documented: {available}"
        )
    return _build(wanted, summary, lines[start:])


def _build(version: str, summary: str, body_lines: list[str]) -> tuple[str, str]:
    title = f"v{version} — {summary}" if summary else f"v{version}"
    return title, "\n".join(body_lines).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", nargs="?", help="e.g. 0.13.1 or v0.13.1")
    parser.add_argument(
        "--title", action="store_true", help="print the release title instead of the body"
    )
    parser.add_argument(
        "--list", action="store_true", help="print every version documented in the changelog"
    )
    args = parser.parse_args(argv)

    if args.list:
        print("\n".join(list_versions()))
        return 0
    if not args.version:
        parser.error("a version is required unless --list is given")

    try:
        title, body = section(args.version)
    except VersionNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(title if args.title else body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
