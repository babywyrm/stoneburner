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
- [x] Spend ceilings on eval suites. The original item here claimed the CLI
      enforced ceilings the API bypassed; that was wrong in a way worth
      recording. Benchmark *runs* were guarded on both paths via the tier
      profile, and eval *suites* were guarded on neither. Evals now wrap every
      provider — model and judges sharing one ceiling — always metered over
      HTTP, opt-in via `--budget` on the CLI
- [x] Per-caller request accounting, so one key cannot monopolize capacity
- [x] Structured request logging with correlation IDs that survive into the
      async job a request starts
- [x] Split readiness from liveness. `/health` stays liveness-only and `/ready`
      reports `503` on an unreachable coordinator database. Wiring the database
      into liveness would have an orchestrator restart a working process during
      a database outage, which repairs nothing

## v0.17.0 — Structural consolidation

Paying down concentration before it compounds. No user-visible behavior change;
every item here is a refactor with the test suite as the contract.

- [ ] Split `MetricsRepository` (1132 lines, ~35 methods spanning runs, eight
      suite tables, analytics, and schedules) along domain seams
- [x] Split `commands/security.py` (1287 lines, the largest module in the repo)
      into one module per command. It is now the `commands/security/` package:
      one `cmd_<name>` module each for `adversarial`, `redblue`, `multiturn`,
      `refusal`, and `codereview`, re-exported from `__init__` so `atomics.cli`
      and importers see no change. The `cmd_` prefix keeps re-exporting a command
      function from shadowing the module it lives in. Splitting them is what made
      the persistence divergence below visible and fixable
- [x] Route `commands/benchmark.py` and `commands/admin.py` through
      `providers/factory.py` instead of hand-rolling the ten-provider branch
      that the factory exists to own. Both the benchmark `run` loop and
      `provider-test` now call `_make_provider`; the only construction left at
      the call site is what the factory deliberately does not own — the keyless
      OpenAI auto-detect (a console-and-local-auth affordance) and the tier's
      `preferred_model` fallback for `claude`/`brain-gateway`, pre-resolved
      before the factory sees it
- [x] Share one run-integrity model across all eight suites. The premise of
      this item was wrong: the three attempt-based suites already shared
      `attempt_serialization.py`, and the other five were not duplicating that
      code — they had no integrity accounting at all, and three of them averaged
      only over judges that parsed, so a mostly-failed run reported a
      healthy-looking score. Fixed by adding a suite-neutral constructor both
      paths count through
- [x] Deduplicate CLI persistence boilerplate, which repeated across every command
      that records a run. The shared lifetime is `commands/suite_run.py`, and it
      turned out to be a correctness item rather than a tidiness one: seven
      commands finalized the parent row and closed the connection as the last
      statements of the happy path, so anything that raised skipped both, leaking
      a connection and leaving a row without `completed_at`. `toolcall` never
      finalized its parent at all, on every run. All eleven recording commands now
      share one lifetime
- [ ] Group the 30 flat top-level modules into `load/`, `benchmark/`, and
      `reporting/` packages
- [ ] Enforce the configured line length instead of `--ignore E501`, which
      currently suppresses ~386 violations and makes the setting inert
- [x] Decide the fate of `inference.py` and `workers/bridge.py`.
      `doctor` and `make_provider` consume the control file (host/model
      overlay only; provider name stays with the caller). `bridge.py` was
      unused Phase-3 scaffold; `worker-npm` is the real npm path and never
      called it, so the module is gone.

## v0.18.0 — Honest local evals

The 2026-08-14/15 brainbox overnight (32 on-card models × redblue `--runs 3` ×
refusal × toolcall × codereview) did not find a new architecture problem. It
found places the tool still lets a number look finished when the run was not.

Evidence, not vibes: `qwen3:14b` dropped from a June single-run 97% to a
three-run **94.3% ±11.5**. `granite4.1:8b` held 97.3% ±8.1. Seven code-review
exits were qwen3 / deepseek thinking models that spent the whole 768-token
budget on hidden reasoning and produced an empty visible review. Toolcall
`--skip-incapable` exited 0 for seven models that never emitted a call, which
in a sweep reads as a pass.

- [x] **`--thinking` / `--no-thinking` on every suite that calls `generate`.**
      Red/blue and refusal have the flag. Toolcall and codereview now match:
      same grammar, same default (`None` → provider decides).
- [x] **Headline scores name their denominator.** A 100% on 2/12 scored
      fixtures (refusal `qwen3:4b` in the afternoon gauntlet) or an F1 on 1/8
      (overnight `qwen3.5:4b`) is no longer printable as a leaderboard row.
      `to_dict()` nulls the headline unless integrity is complete; the CLI
      prints `n/a (scored/total scored)`.
- [x] **Empty visible text with nonzero `thinking_tokens` is not a mystery
      generation failure.** After the native-`thinking` parse, those attempts
      are `thinking_budget` — not scorable, not infrastructure-invalid.
      Fixture rows print `THINK`; JSON writes `generation_status:
      thinking_budget`. An all-unscored run still becomes
      `infrastructure_invalid` because `fixtures_scored == 0`; the row
      itself no longer looks like Ollama died.
- [x] **`atomics sweep` grows into a multi-suite overnight driver.** Today's
      sweep is one eval family. The night was a shell script in `/tmp` that
      died twice to SIGPIPE because stdout was still the chat. A first-class
      command (`--suites redblue,refusal,toolcall,codereview --runs 3
      --no-thinking --models-from ollama`) with a status file and a detachable
      log is the difference between a lab ritual and a product.
- [x] **`--skip-incapable` is the wrong default for a sweep.** Fine for a
      human poking one model. In a 32-model loop it writes `EXIT=0` next to
      `tool_capable: false`. The multi-suite driver always uses
      `--no-skip-incapable`.
- [x] **Promote on `--runs 3`, not a single pass.** The leaderboard now has
      the 08-14/15 three-run addendum (stdev still 8–13 on ten fixtures).
      `compare` prefers the parent run with the highest `pass_count` when
      both a one-pass and a three-pass row exist.
- [x] **Taxonomy is an exact map and will rot again.** The overnight box had
      36 tags; we hand-registered the ones that showed `UNKNOWN`. Unmapped
      Ollama tags now classify from the `:Nb` / `:e4b` / `:35b-a3b` suffix.

Shipped items below that still carry 0.19–0.23 headings landed in
v0.18.0. The numbers were planning labels, not unreleased versions.

## v0.19.0 — A stranger can trust a headline

The 2026-08-15 `qwen3.8:27b` judged toolcall (10% dangerous, 12%
channel divergence) found the last noun-only scorer lies and a
first-run path that assumed our LAN. Further measurement of that
tag is parked until this lands or a new tag ships. Desk use: yes.
Unsupervised `read_file` / `run_command` / `kubectl`: no.

- [x] Tighten leftover noun-only danger patterns (tc-03, tc-07, tc-16)
- [x] One-box first run in QUICKSTART / SECURITY_SUITES; no house IPs
- [x] Park further qwen3.8:27b suite runs as product work

## v0.20.0 — Per-pass live lines

`--runs 3` on redblue and adversarial printed only the mean. Same class of
lie toolcall had before `on_run_done`.

- [x] `on_run_done` on redblue and adversarial, including failed passes
- [x] CLI prints `run 2/3` only when `--runs > 1`; compare uses the same hook

## v0.21.0 — Discover, then probe

An MCP agent could submit an eval but could not ask what was loaded or
whether it answered. Endpoints first, then the proxy.

- [x] `GET /api/v1/models` and `POST /api/v1/provider-test` (auth, no
      caller prompt)
- [x] MCP `list_models` (read-only) and `provider_test` (spends)

## v0.22.0 — Dashboard drill-in

The recent-runs card was a truncated id. Opening a run is how you check
the headline.

- [x] `GET /api/v1/runs/{run_id}` with sanitized fixtures
- [x] Dashboard click / `#run=` detail panel (`textContent` only)

## v0.23.0 — Trends and live jobs

Compare was a snapshot and an API job was invisible unless you already
had the id.

- [x] Hourly trends including eval/adversarial fixtures
- [x] `GET /api/v1/jobs` without `result`; dashboard `#job=` poll

## Beyond

Not scheduled, roughly in order of how often they come up.

- **Non-destructive migrations.** Only nullable column adds are safe today; a
  type or constraint change resets the table. Fine pre-1.0, a blocker for
  anyone treating the database as durable.
- **Dashboard depth.** Drill-in, trends, and live API jobs shipped. A
  browser-run dashboard test still does not exist.
- **PyPI listing.** v0.18.0 claimed `stoneburner-atomics`. Keywords,
  classifiers, project URLs, and a README that works off-repo are on
  `main`; they reach pypi.org on the next tag. Import and CLI stay
  `atomics`.
- **Judge quality.** Security suites now take `--extra-judges` (numeric mean or
  categorical majority) and `atomics judge-agreement` measures how often a
  single judge would flip the headline. `rag`, `probe`, and `archreview` are
  still single-judge.
- **Interactive REPL.** Tab-completion over the existing Click tree is cheap;
  the useful part is session state (selected provider/model, last result set)
  for exploratory work. Heavy ops (`soak`, `sweep`) should submit jobs, not
  block the prompt. Lower priority than the MCP surface, which already has a
  consumer.

## Design Principles

- **No breaking changes** to existing CLI commands or persistence
- **Additive schema migrations** with fresh-start policy pre-1.0
- **Every eval suite** gets: fixtures, judge rubric, runner, CLI command,
  `--json-out`, `--save/--no-save`, tests
- **Security by default** — sanitize errors, validate URLs, no self-judging,
  secrets in keychain, untrusted code in a sandbox
