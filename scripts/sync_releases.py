"""Reconcile GitHub releases with CHANGELOG.md.

Releases cut before the workflow published real notes have a bare compare link
for a body and a bare `vX.Y.Z` for a title. This rewrites them from the
changelog so the published history matches what was actually written, and
reports any tag that has no release at all.

Dry run by default — it prints what would change and touches nothing. Pass
`--apply` to write, which requires `gh` to be authenticated.

    python scripts/sync_releases.py                 # show the diff
    python scripts/sync_releases.py --apply         # write it
    python scripts/sync_releases.py --only v0.12.0  # one release
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from changelog_section import VersionNotFoundError, section  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=REPO, capture_output=True, text=True)


def local_tags() -> list[str]:
    out = _run(["git", "tag", "--list", "v[0-9]*.[0-9]*.[0-9]*", "--sort=creatordate"])
    out.check_returncode()
    return [t for t in out.stdout.split() if t]


def published_releases() -> dict[str, dict]:
    """Map tag -> {name, body}. Empty when gh is unavailable or unauthenticated."""
    out = _run(["gh", "release", "list", "--limit", "100", "--json", "tagName"])
    if out.returncode != 0:
        print(f"warning: could not list releases: {out.stderr.strip()}", file=sys.stderr)
        return {}
    releases = {}
    for entry in json.loads(out.stdout or "[]"):
        tag = entry["tagName"]
        detail = _run(["gh", "release", "view", tag, "--json", "name,body"])
        if detail.returncode == 0:
            releases[tag] = json.loads(detail.stdout)
    return releases


def _summarize(text: str, limit: int = 88) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def is_placeholder(body: str) -> bool:
    """True when a release body carries no writing of its own.

    `generate_release_notes` leaves exactly one line — a compare link — when it
    finds no commits between the tag and its predecessor. Those bodies are the
    ones worth replacing. Several older releases have curated notes that read
    better than the changelog section and are deliberately left alone.
    """
    meaningful = [
        line for line in body.strip().splitlines()
        if line.strip() and not line.strip().startswith("**Full Changelog**:")
    ]
    return not meaningful


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write the changes")
    parser.add_argument("--only", metavar="TAG", help="restrict to one tag")
    parser.add_argument(
        "--retitle",
        action="store_true",
        help="also normalize titles on releases whose notes are kept",
    )
    parser.add_argument(
        "--overwrite-curated",
        action="store_true",
        help="replace hand-written release notes with the changelog section too",
    )
    args = parser.parse_args(argv)

    releases = published_releases()
    tags = [t for t in local_tags() if not args.only or t == args.only]
    if not tags:
        print("no matching tags")
        return 1

    changed = 0
    for tag in tags:
        try:
            title, body = section(tag)
        except VersionNotFoundError:
            print(f"{tag}: SKIP — no changelog section")
            continue

        if tag not in releases:
            print(f"{tag}: NO RELEASE — would create, titled {title!r}")
            changed += 1
            if args.apply:
                result = _run([
                    "gh", "release", "create", tag,
                    "--title", title, "--notes", body,
                ])
                print(f"       {'created' if result.returncode == 0 else result.stderr.strip()}")
            continue

        current = releases[tag]
        current_body = current.get("body") or ""
        current_title = current.get("name", "")
        curated = not is_placeholder(current_body)

        if curated and not args.overwrite_curated:
            note = "" if current_title == title else f" (title differs: {current_title!r})"
            print(f"{tag}: keep — hand-written notes{note}")
            if current_title != title and args.retitle:
                changed += 1
                print(f"       retitle -> {title!r}")
                if args.apply:
                    result = _run(["gh", "release", "edit", tag, "--title", title])
                    print(f"       {'updated' if result.returncode == 0 else result.stderr.strip()}")
            continue

        if current_title == title and current_body.strip() == body.strip():
            print(f"{tag}: ok")
            continue

        changed += 1
        print(f"{tag}: {'OVERWRITE' if curated else 'FILL'}")
        if current_title != title:
            print(f"       title: {current_title!r} -> {title!r}")
        print(f"       body:  {_summarize(current_body or '(empty)')}")
        print(f"          ->  {_summarize(body)}")
        if args.apply:
            result = _run([
                "gh", "release", "edit", tag, "--title", title, "--notes", body,
            ])
            print(f"       {'updated' if result.returncode == 0 else result.stderr.strip()}")

    if not changed:
        print("\nEvery release matches the changelog.")
    elif not args.apply:
        print(f"\n{changed} release(s) would change. Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
