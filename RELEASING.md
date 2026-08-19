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

Three tests enforce the parts that can silently rot, all in
`tests/test_changelog_section.py`:

- `test_every_released_tag_has_release_notes` — a tag with no changelog section
  would publish an empty release.
- `test_every_documented_version_is_tagged` — a changelog entry nobody tagged.
  This is how `0.11.0` was documented, bumped, and never released. It exempts the
  version being cut right now — the newest entry, when it is also the version
  `pyproject.toml` declares and no tag exists yet — because step 3 below runs
  before step 5, so every correct release passes through that state. The
  exemption lapses as soon as a later version is documented on top of an
  untagged one, which is the shape `0.11.0` actually had.
- `test_no_version_ever_shipped_undocumented` — a version that appeared in
  `pyproject.toml` and never got written up at all. `0.4.0` and `0.5.0` went
  undocumented this way, leaving an unexplained jump from 0.3.0 to 0.6.0.

If a version was bumped but never actually released, say so in the changelog
under a heading that does not parse as a version — see the 0.4.0 / 0.5.0 note.
That satisfies the guard without inventing a release that never happened.

## Cutting a release

1. **Write the changelog first.** Promote `## Unreleased` to
   `## X.Y.Z (YYYY-MM-DD) — <summary>`. This text is the release notes, so write
   it for someone deciding whether to upgrade: what changed, why, and what they
   have to do about it. Put anything breaking or migration-related under an
   `### Upgrade notes` heading at the top.

2. **Bump the version** in `pyproject.toml`, then `uv sync --all-extras` so the
   lockfile and installed metadata follow. Use `--all-extras` rather than
   naming a subset: syncing only `dev` and `api` uninstalls the RAG
   dependencies, and the suite then *skips* those tests rather than failing, so
   verification passes while covering twenty fewer tests than it appears to.

3. **Verify.** All of these, not a subset:

   ```bash
   uv run pytest -q --cov-fail-under=85
   uv run mypy atomics/
   uv run ruff check atomics/ tests/ scripts/
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
   the distribution, creates the GitHub release from the changelog, moves the
   floating `v0` tag, and uploads `stoneburner-atomics` to PyPI if the
   trusted publisher is registered.

## What the tag triggers

`.github/workflows/publish.yml` runs on `v[0-9]*`:

- **Build distribution** — `uv build`, uploaded as an artifact.
- **GitHub Release** — title and body extracted from `CHANGELOG.md` by
  `scripts/changelog_section.py`, with the built artifacts attached. A tag with
  no changelog section fails the job rather than publishing an empty release.
- **Update floating major tag** — force-moves `v0` to the new release, for
  anything that wants to follow the major line. It is deliberately mutable.

- **PyPI** — `uv publish` via trusted publishing, as a job that does not
  block the GitHub release. The listing name is `stoneburner-atomics`
  (`stoneburner` collides with `stone-burner`). See Distribution below.

## Distribution

The PyPI / `uv add` name is `stoneburner-atomics`. The importable package
and the CLI stay `atomics`:

```bash
uv add stoneburner-atomics
uv tool install stoneburner-atomics
uv add 'stoneburner-atomics[api,mcp]'
```

From a clone, `uv sync` is unchanged. `import atomics` and `atomics --version`
are unchanged. `atomics.__version__` reads metadata for the
`stoneburner-atomics` distribution.

Trusted publishing is registered. v0.18.0 claimed `stoneburner-atomics`.
Later tags reuse the same GitHub Environment `pypi` and the same
publisher row:

| Field | Value |
|-------|--------|
| PyPI project name | `stoneburner-atomics` |
| Owner | `babywyrm` |
| Repository | `stoneburner` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

The GitHub release and the PyPI job are independent on purpose. The old
`atomics` trusted-publish job failed on every tag because that name
belongs to an unrelated C++ package. `stoneburner` is too similar to
`stone-burner`.

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
the notes. It is idempotent: a second run reports no changes.

Two hazards it accounts for, both hit while backfilling:

- **Backfilling an old version steals the "Latest" badge.** GitHub flags
  whichever release was published most recently, so creating `v0.11.0` today
  marked a July version as current. New releases are created with
  `--latest=false`; if it happens anyway, `gh release edit vX.Y.Z --latest`
  puts it back.
- **Pushing an old tag runs the workflow as it existed at that commit.** For
  `v0.11.0` that meant the pre-fix `publish.yml`, which would have published an
  empty release, force-moved the floating `v0` tag backwards, and added another
  failed PyPI job. Disable the workflow around the push:

  ```bash
  gh workflow disable publish
  git push origin refs/tags/vX.Y.Z
  gh workflow enable publish
  ```

**Two legacy lightweight tags.** `v0.9.0` and `v0.10.0` are lightweight rather
than annotated. Converting them means deleting and recreating the tags, which
requires a force push, so they are left as they are. Everything from `v0.11.0`
onward is annotated.
