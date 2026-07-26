# Releasing

How a version of Atomics gets cut, and the conventions the tooling enforces.

## Conventions

| Thing | Convention |
|-------|------------|
| Version source | `pyproject.toml` only. `atomics.__version__` and the FastAPI app both read installed metadata, so there is one number to change. |
| Scheme | Semantic versioning, pre-1.0. Breaking changes bump the minor while the major is `0`. |
| Tag name | `vX.Y.Z`, annotated, never lightweight. |
| Tag message | `Release vX.Y.Z — <summary>` plus the notable changes. |
| Changelog heading | `## X.Y.Z (YYYY-MM-DD) — <summary>` — the summary becomes the GitHub release title. |
| Release title | `vX.Y.Z — <summary>`, derived from the changelog. Never a bare `vX.Y.Z`. |
| Release body | The changelog section for that version, verbatim. |

Two tests enforce the parts that can silently rot, both in
`tests/test_changelog_section.py`:

- `test_every_released_tag_has_release_notes` — a tag with no changelog section
  would publish an empty release.
- `test_every_documented_version_is_tagged` — a changelog entry nobody tagged.
  This is how `0.11.0` was documented, bumped, and never released.

## Cutting a release

1. **Write the changelog first.** Promote `## Unreleased` to
   `## X.Y.Z (YYYY-MM-DD) — <summary>`. This text is the release notes, so write
   it for someone deciding whether to upgrade: what changed, why, and what they
   have to do about it. Put anything breaking or migration-related under an
   `### Upgrade notes` heading at the top.

2. **Bump the version** in `pyproject.toml`, then `uv sync --extra dev --extra api`
   so the lockfile and installed metadata follow.

3. **Verify.** All of these, not a subset:

   ```bash
   uv run pytest -q --cov-fail-under=85
   uv run mypy atomics/
   uv run ruff check atomics/ tests/ --ignore E501
   gitleaks detect --config .gitleaks.toml --no-banner --redact
   uv run python scripts/smoke_fleet.py       # real processes, not TestClient
   ```

4. **Preview the release notes** exactly as the workflow will render them:

   ```bash
   python3 scripts/changelog_section.py --title X.Y.Z
   python3 scripts/changelog_section.py X.Y.Z
   ```

5. **Commit, tag, push.**

   ```bash
   git commit -m "release: X.Y.Z — <summary>"
   git tag -a vX.Y.Z -m "Release vX.Y.Z — <summary>"
   git push origin main --follow-tags
   ```

6. **Check the workflow.** `gh run list --limit 3`. The `publish` workflow builds
   the distribution, creates the GitHub release from the changelog, and moves the
   floating `v0` tag.

## What the tag triggers

`.github/workflows/publish.yml` runs on `v[0-9]*`:

- **Build distribution** — `uv build`, uploaded as an artifact.
- **GitHub Release** — title and body extracted from `CHANGELOG.md` by
  `scripts/changelog_section.py`, with the built artifacts attached. A tag with
  no changelog section fails the job rather than publishing an empty release.
- **Update floating major tag** — force-moves `v0` to the new release, for
  anything that wants to follow the major line. It is deliberately mutable.
- **Publish to PyPI** — currently fails, see below.

## Known issues

**PyPI publishing cannot succeed.** The `publish-pypi` job uses trusted
publishing for a project named `atomics`, but that name on PyPI belongs to an
unrelated package (`doodspav/atomics`, "Atomic lock-free primitives"). Every tag
therefore shows one failed job. Resolving it means either renaming the
distribution and configuring a trusted publisher, or dropping the job. Nothing
else in the release is affected — the build and the GitHub release both succeed.

**Release notes did not always come from the changelog.** Releases through
v0.13.1 were created with `generate_release_notes`, which diffs against the
previous tag. Because the workflow also moves the floating `v0` tag to each new
release, GitHub compared `v0` against the tag being released, found nothing in
between, and published a body containing only a compare link. `v0.8.0`,
`v0.12.0`, `v0.13.0` and `v0.13.1` are affected.

To reconcile already-published releases with the changelog:

```bash
python3 scripts/sync_releases.py            # dry run, prints what would change
python3 scripts/sync_releases.py --apply    # write it
```

It fills placeholder bodies and creates missing releases. Releases with
hand-written notes — `v0.6.0`, `v0.7.0`, `v0.9.0`, `v0.10.0` — are left alone
unless you pass `--overwrite-curated`, since in several cases those read better
than the changelog section. `--retitle` normalizes their titles without touching
the notes.

**Two legacy lightweight tags.** `v0.9.0` and `v0.10.0` are lightweight rather
than annotated. Converting them means deleting and recreating the tags, which
requires a force push, so they are left as they are. Everything from `v0.11.0`
onward is annotated.
