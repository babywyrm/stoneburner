# Roadmap

Where Stoneburner / Atomics is going next. Shipped work lives in
[CHANGELOG.md](CHANGELOG.md); this file is only about what is not done yet.

The 2026-08-02 project audit found no critical architectural problems and one
critical security finding, since fixed in v0.15.2. What it did surface was
concentration: a few modules doing too much, and a server whose authorization
model was built for a single trusted operator and then exposed over HTTP. The
next two milestones work through that.

## v0.16.0 — Bounded and observable

Making the server safe to leave running unattended.

- [x] Cap concurrent and retained jobs so the in-memory job dict cannot grow
      without limit
- [x] Upper bounds on `iterations`, `interval`, and eval fixture lists
- [x] Security headers on every response; nonce-based CSP for the dashboard
- [x] Dependency auditing and secret scanning in CI, on a schedule
- [ ] Apply the existing budget/rate `Guard` to API-triggered runs — the CLI
      enforces spend ceilings that the API currently bypasses entirely
- [ ] Per-caller request accounting, so one key cannot monopolize capacity
- [ ] Structured request logging with correlation IDs
- [ ] Split readiness from liveness so `/health` stops answering `ok` while the
      coordinator's database is unreachable

## v0.17.0 — Structural consolidation

Paying down concentration before it compounds. No user-visible behavior change;
every item here is a refactor with the test suite as the contract.

- [ ] Split `MetricsRepository` (1132 lines, ~35 methods spanning runs, eight
      suite tables, analytics, and schedules) along domain seams
- [ ] Route `commands/benchmark.py` and `commands/admin.py` through
      `providers/factory.py` instead of hand-rolling the ten-provider branch
      that the factory exists to own
- [ ] Extract a shared eval runner base — the eight suite runners reimplement
      the same attempt/integrity serialization, and CLI persistence boilerplate
      repeats about eight times
- [ ] Group the 30 flat top-level modules into `load/`, `benchmark/`, and
      `reporting/` packages
- [ ] Enforce the configured line length instead of `--ignore E501`, which
      currently suppresses ~386 violations and makes the setting inert
- [ ] Decide the fate of `inference.py` and `workers/bridge.py` — both are
      documented but reachable only from tests

## Beyond

Not scheduled, roughly in order of how often they come up.

- **Non-destructive migrations.** Only nullable column adds are safe today; a
  type or constraint change resets the table. Fine pre-1.0, a blocker for
  anyone treating the database as durable.
- **Dashboard depth.** Currently four read-only cards. Historical trends, drill
  into a single run, and live job progress are the obvious next steps.
- **PyPI distribution.** Needs a rename — `atomics` is taken by an unrelated
  package — which changes `pip install` for every consumer. See
  [RELEASING.md](RELEASING.md).
- **Judge quality.** Multi-judge consensus exists for accuracy; extending it to
  the security suites would reduce single-judge variance.

## Design Principles

- **No breaking changes** to existing CLI commands or persistence
- **Additive schema migrations** with fresh-start policy pre-1.0
- **Every eval suite** gets: fixtures, judge rubric, runner, CLI command,
  `--json-out`, `--save/--no-save`, tests
- **Security by default** — sanitize errors, validate URLs, no self-judging,
  secrets in keychain, untrusted code in a sandbox
