# Contributing

Thanks for looking. This is the entry point — the deeper material lives in
[`ARCHITECTURE.md`](ARCHITECTURE.md) (how the layers fit together),
[`RELEASING.md`](RELEASING.md) (how versions get cut), and
[`SECURITY.md`](SECURITY.md) (including how to report a vulnerability, which is
**not** through a public issue).

## Setup

Requires **Python 3.11+** and [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/babywyrm/stoneburner.git
cd stoneburner
uv sync --extra dev --extra api --extra mcp
uv run atomics doctor      # checks providers, credentials, and state paths
```

No provider credentials are needed to run the test suite; every test either
stubs the provider or skips.

## The checks

Run all four before opening a pull request. These are exactly what CI runs, so
a local pass means a green build:

```bash
uv run pytest -q --cov-fail-under=85
uv run mypy atomics/
uv run ruff check atomics/ tests/ scripts/ --ignore E501
gitleaks detect --config .gitleaks.toml --no-banner --redact
```

`--ignore E501` is not optional — there are several hundred pre-existing
line-length reports, and CI ignores them too. Coverage is enforced at 85%; the
project currently sits near 88%, so new code is expected to arrive tested.

## Conventions worth knowing before you write code

**Layering is enforced, not just documented.** `tests/test_layering.py` parses
the package and fails the build on an import that points the wrong way.
`ARCHITECTURE.md` has the dependency direction.

**Every eval suite ships the same seven things**: fixtures, a judge rubric, a
runner, a CLI command, `--json-out`, `--save/--no-save`, and tests. A suite
missing one of those is incomplete rather than minimal.

**Persistence is additive.** New columns are nullable and reconciled in place.
Anything that would rewrite or drop existing data needs to be raised first.

**No breaking changes to existing CLI commands.** Add flags, keep defaults.

**Untrusted input stays untrusted.** Model output is escaped before terminal
rendering, errors are scrubbed before they reach the database, URLs are
validated, and model-generated code runs in the subprocess sandbox rather than
in-process. If you are adding a path that handles model output, follow the
existing pattern rather than inventing one.

## Commits and pull requests

Commit messages use a `type(scope): summary` subject — `feat`, `fix`, `docs`,
`test`, `refactor`, `chore` — with a body explaining *why* when the reason is
not obvious from the diff. Look at `git log` for the house style.

Group related work into logical commits rather than one commit per file edit.

For a pull request, describe what changed and how you verified it. If it
touches behavior, say which of the checks above you ran.

## Adding a provider

Providers implement the contract in `atomics/providers/base.py` and are
constructed through `atomics/providers/factory.py`. Add the provider module,
register it in the factory, add it to `PROVIDER_CHOICES`, give it pricing
metadata if the API is paid, and add tests that stub the HTTP layer. Document
it in `README.md` and `docs/CLI_REFERENCE.md`.

## Questions

Open an issue. For anything security-related, use the private reporting path in
[`SECURITY.md`](SECURITY.md) instead.
