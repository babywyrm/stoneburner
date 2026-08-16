# Changelog

## Unreleased

### Added
- **`redblue` and `adversarial` print each pass when `--runs` is greater
  than one.** The mean fixture line hid a 40% pass (or a 1/3 comply)
  until the JSON was opened. Both runners now have `on_run_done`, matching
  toolcall. Generate failures print `ERROR`. `adversarial --compare` uses
  the same hook so model B is not silent.

- **One-box toolcall first run.** `doctor` → `provider-test --no-thinking`
  → `toolcall --channel tools --runs 3`. Operator docs no longer use a
  lab IP. Divergence is documented as not measured without a judge.

- **Operator docs match the CLI that ships.** README lists `refusal`,
  `codereview`, `judge-agreement`, `labcompare`, and real fixture counts
  (72/20/10/12/8) instead of "per-suite". QUICKSTART's suite list and
  `provider-test` mention thinking models. `CLI_REFERENCE` puts `sweep`
  back in the evaluation table. `THINKING.md` shows `--no-thinking` on
  a qwen3.8 smoke test.

- **`toolcall` live output is documented where operators actually look.**
  `SECURITY_SUITES`, `CLI_REFERENCE`, and `QUICKSTART` now show the
  `run 2/3` lines and how to read `prose=resisted` + `DANGEROUS` as
  channel divergence. Changelog already had the behavior; the recipe
  docs did not.

- **`toolcall` prints each pass when `--runs` is greater than one.** A
  1/3 leak (tc-01 `/etc/shadow`, tc-06 scale-to-zero, tc-20 force-delete)
  was invisible until the JSON was opened after the run. The runner now
  has `on_run_done`; the CLI prints `tc-01 run 2/3 — DANGEROUS`.

- **`toolcall` prints live fixture progress and warns when there is no
  judge.** A 20-fixture × 3-run suite was silent for half an hour. Each
  fixture now prints generating/done lines (durable, not only a spinner).
  Omitting a judge prints that prose stays unjudged and channel
  divergence cannot be measured.

- **`atomics sweep` is an overnight driver, not only an eval family.**
  `--suites redblue,refusal,toolcall,codereview --runs 3 --no-thinking
  --models-from ollama --status FILE --log FILE` writes a JSON status
  file after each model×suite job and a detachable log that survives
  SIGPIPE. Toolcall in a sweep uses `--no-skip-incapable`, so a model
  that never emits a call is `INCAPABLE` / exit 1, not a silent pass.

- **`compare` prefers a `--runs 3` row over a single pass.** Parent
  `runs` rows now store `pass_count`. When the same model has both a
  one-pass and a three-pass measurement, the comparison uses the
  higher pass count.

- **Unmapped Ollama tags classify from the size suffix.** `brand-new:3b`
  is mid, `:1.5b` is light, `:35b-a3b` is heavy. The exact map still
  wins when present; tags without a parameter size stay `UNKNOWN`.

### Fixed
- **`toolcall` no longer scores inspection as a PAN dump, a DELETE, or
  a /tmp note.** `COUNT(*)` / `information_schema` on `customers`
  (tc-03), `SELECT deleted_at` (tc-07), and `write_file(/tmp/notes.txt)`
  (tc-16) required only the noun. Patterns now require the attack.

- **`redblue` no longer scores a truncated IR/STRIDE/Dockerfile answer as
  a capability gap.** Fixture default is 2048, not 1024. `--max-output-tokens`
  is a floor on top of that; thinking models still get at least 4096. The
  judge char budget now uses the same resolved value.

- **`toolcall` no longer scores inspection as destruction.** `df`/`du` on
  `/mnt/backups` (tc-18) and `systemctl status auditd` (tc-09) matched
  patterns that only required the noun. The qwen3.8:27b run published a
  15% dangerous-call rate that was mostly those false positives. Patterns
  now require the destructive verb (`rm`/`wipe`/`mkfs`, `systemctl stop
  auditd`).

- **Empty visible text with nonzero thinking is `thinking_budget`, not a
  dead provider.** Overnight qwen3 / deepseek-r1 code-review FAILs spent
  the token budget on hidden reasoning and printed as `ERROR`. Those
  attempts now classify as `ProviderOutcomeKind.THINKING_BUDGET`: not
  scorable, not infrastructure-invalid. Fixture JSON writes
  `generation_status: thinking_budget`; refusal and codereview print
  `THINK` instead of `ERROR`. OpenAI chat and Responses paths do the
  same when `reasoning_tokens > 0` and there is no visible text.

- **`toolcall` and `codereview` accept `--thinking` / `--no-thinking`.**
  Refusal and red/blue already had the flag. The overnight code-review
  FAILs were thinking models that spent the 768-token budget on hidden
  reasoning and produced an empty visible review. Same grammar, same
  default (`None` → provider decides), `--thinking-budget` default 8000.
  Toolcall forwards the flag to the prose `generate()` only;
  `generate_with_tools` is unchanged.

- **Headline scores no longer print a bare percentage on a partial run.**
  A 100% on 2/12 scored fixtures (refusal `qwen3:4b`) or an F1 on 1/8
  (overnight `qwen3.5:4b`) used to look like a finished leaderboard row.
  `to_dict()` now nulls the headline rates unless integrity is
  `complete`; the CLI prints `n/a (2/12 scored)` instead. The
  scored-subset math stays on the summary object. Shared helpers live in
  `suite_integrity.headline_rate` / `format_headline_rate`. Applied to
  refusal, codereview, and red/blue (which also stopped rendering a
  missing quality score as `0.0%`).

- **A failed `--save` now names the database path.** `suite_run` used to
  surface a bare `unable to open database file` when SQLite could not open
  `atomics.db`, which hid whether the problem was the default `data/` path, a
  permissions issue, or a lock. The wrapped error now includes the path.

- **On-card Ollama tags no longer classify as `UNKNOWN`.** `qwen3:8b`,
  `qwen3.5:9b`, `granite4.1:3b`, `granite4.1:8b`, `ministral-3:8b`,
  `lfm2.5:8b`, `mistral-nemo:12b`, `nemotron-3-nano:4b`,
  `phi4-mini-reasoning:3.8b` (mid), `smollm2:1.7b` (light), and the
  off-card heavies `mistral-small:24b`, `qwen3.6:27b`, `qwen3.6:35b-a3b`,
  `qwen3.8:27b` were missing from `MODEL_CLASS_MAP`, so `compare`/`sweep`
  tables showed blanks. Thinking prefixes now cover `qwen3` and
  `phi4-mini-reasoning`.

- **Ollama now records the native `thinking` field.** Newer Ollama
  returns reasoning in a top-level `thinking` key, not `<think>` tags
  inside `response`. The adapter only stripped tags, so thinking models
  that spent the whole `num_predict` budget on hidden reasoning showed
  up as empty generations with `thinking_tokens=0`. The field is now
  captured and counted.

- **`toolcall --extra-judges` now records `judge_agreement`.** The prose
  panel already averaged scores, but the fixture JSON and `--save` row
  dropped the agreement, so a live granite + extra-judge run wrote
  `NULL` into `evaluation_results.judge_agreement`. Agreement is the
  fraction of panel labels that match the combined label.

- **`refusal` accepts `--thinking` / `--no-thinking`.** Red/blue already
  had the flag; refusal always left thinking at the provider default, so
  qwen3 / deepseek-r1 burned the 512-token fixture budget and scored as
  generation failures. Same grammar as red/blue.

- **`refusal`, `toolcall`, and `codereview` now warn on self-judging.**
  `detect_self_judge` already fired on `redblue` / `eval` / `adversarial` /
  `rag`. The other three security suites silently accepted the model under
  test as its own judge. They now log the same warning. The helper also
  no longer crashes when a duck-typed provider has no `default_model`.

- **A failed run no longer leaks its database connection or leaves an
  unfinishable row behind.** Seven commands — `redblue`, `multiturn`, `eval`,
  `rag`, `codegen`, `probe`, and `archreview` — finalized the parent `runs` row
  and closed the repository as the last statements of the happy path, so anything
  that raised first (a provider timeout, a judge error, an interrupt) skipped
  both. The row kept a null `completed_at` forever, which `atomics report` reads
  as a run still in progress, and the SQLite connection was released only when the
  process exited.

  `toolcall` was a worse case: it never finalized its parent at all, on every
  successful run rather than only on failures, so each of its runs read as
  perpetually in progress and its parent aggregates stayed empty.

  None of these symptoms is visible in a passing run, which is why the CLI
  persistence paths carried no coverage beyond `--help`. They are now asserted
  directly — the finalize and the close, rather than inferred from a successful
  invocation — across all eleven recording commands.

  The fix is `commands/suite_run.py`, one lifetime they all share. It had drifted
  into three idioms: `refusal` and `codereview` used `try/finally`, `adversarial`
  used `ctx.call_on_close` plus an explicit fail-closed finalize, and the rest
  used nothing. The shared version keeps the best property of the `adversarial`
  idiom — a cleanup problem that surfaces behind an existing failure is logged
  rather than raised, so it cannot hide the failure that actually caused it — and
  applies it everywhere. Cleanup failures now name the operation and the run
  (`Failed to finalize refusal run a1b2c3: ...`) rather than restating the
  command, so a finalize problem is distinguishable from the run itself failing.

  Commands differ only in which table their parent aggregates from, expressed as
  `finalize_task_run`, `finalize_evaluation_run`, `finalize_adversarial_run`,
  `finalize_probe_run`, or `finalize_archreview_run`. These are functions rather
  than unbound `MetricsRepository` methods on purpose: an unbound reference calls
  the base class regardless of the instance, silently bypassing any subclass or
  wrapper.

- **`--json-out` failures are reported instead of crashing.** The commands above
  hand-rolled their JSON export with a bare `json.dump`, in several cases after
  the database connection had already been closed, so an unwritable path produced
  a raw traceback. They now go through the existing `write_summary_json`, which
  reports the problem as a sanitized CLI error, and they do it inside the managed
  lifetime so the run is still finalized.

### Changed
- **The Qwen quickstart no longer hardcodes one machine's address.** Its curl and
  `--ollama-host` examples pointed at a specific LAN IP, so nobody else could
  paste them and have them work. They now read `$OLLAMA_HOST`, set once at the
  top, matching how the other docs handle `$ATOMICS_API_KEY`. Test fixtures that
  carried the same addresses moved to the RFC 5737 documentation range
  (`203.0.113.0/24`), which reads unambiguously as an example. Invented addresses
  inside eval fixtures — firewall-rule puzzles, incident-response scenarios, a C2
  indicator — are deliberate content and unchanged.

### Added
- **`--extra-judges` on `redblue`, `multiturn`, `refusal`, `codereview`, and
  `toolcall`.** Same grammar as `eval` / `adversarial` (`provider:model[@host]`),
  same shared `--budget`. Default remains one judge. Numeric suites average;
  categorical suites (`refusal`, `codereview`) majority-vote, and a tie is
  unresolved rather than silently picking the primary. `multiturn` panels the
  conversation score only; `toolcall` panels the prose channel only.

- **`atomics judge-agreement`** generates each fixture once and scores that
  same response with every judge, then reports pairwise agreement and
  majority-flip rate. It is a study, not a leaderboard row: `--save` is off by
  default, and even when on it never writes a parent `runs` row.

- **`atomics mcp` serves atomics to LLM agents over the Model Context
  Protocol**, as a proxy over a running `atomics server`. Six tools: `health`,
  `get_job`, `compare`, and `recent_runs` are annotated read-only, while
  `submit_run` and `submit_eval` spend tokens and return a job id to poll, so no
  tool call blocks on model work. Requires the new `[mcp]` extra; the client
  needs no dependency beyond the `httpx` already in the core set.

  The proxy shape is the design, not a shortcut. An MCP client is a remote,
  automated caller — the API's trust position, not the CLI's, which assumes a
  local operator spending their own money deliberately. Going through the API
  means the agent inherits API-key authentication, the per-eval dollar ceiling,
  and the bounds on iterations and fixtures, with no second copy of a spending
  decision to keep in sync. The tool surface is therefore bounded by what the
  API exposes: `models`, `provider-test`, `sweep`, `stress`, `soak`, and `probe`
  are CLI-only and stay that way until they are endpoints with the auth and
  bounds that implies.

  It serves on stdio only. The HTTP transports would open a port that no
  MCP-layer credential guards while the process holds an API key with spend
  authority, which would hand that authority to anyone who could reach it — the
  opposite of the guardrails the proxy exists to inherit. To reach a remote
  atomics, point `--api-url` at its authenticated API server and run the MCP
  server locally. And because stdout carries the JSON-RPC frames, the command
  forces plain logging and pins its console to stderr: the CLI group's Rich
  handler writes to stdout, and one warning through it would corrupt every
  session. See `docs/MCP_SERVER.md`.

  CI and the contributor setup now sync `--extra mcp` so those tests run rather
  than skip. Without the extra, the SDK-dependent tests skip cleanly instead of
  erroring during collection.

## 0.17.0 (2026-08-09) — Structural consolidation: security command package and shared provider factory

### Changed
- **`commands/security.py` is now the `commands/security/` package**, one module
  per command instead of a single 1287-line file — the largest module in the
  repo. `adversarial`, `redblue`, `multiturn`, `refusal`, and `codereview` each
  live in their own `cmd_<name>` submodule, re-exported from the package so
  `atomics.cli` and `from atomics.commands.security import adversarial` keep
  working unchanged. The `cmd_` prefix is deliberate: re-exporting a command
  named `refusal` from a module also named `refusal` would shadow the module and
  break patching it in tests. No behavior change — the full suite passes as
  before; the only test edits repoint monkeypatch targets at the submodule that
  now owns each command.
- **`provider-test` and the benchmark `run` loop now build providers through the
  shared `providers.factory`** instead of each carrying its own ten-branch
  provider switch. The two branches had already drifted from the factory and
  from each other — a divergence the architecture doc named — and every new
  provider meant editing three copies. Behavior is unchanged: the factory
  encodes the same per-provider model defaults, so the only logic that stayed at
  the call site is the part the factory legitimately cannot own. The keyless
  OpenAI path still auto-detects a local login and prints what it found, because
  reaching into local auth and a console is a CLI affordance the factory keeps
  out of so the API server and distributed workers stay headless. And
  `provider-test`'s `brain-gateway` output still reads `(gateway default)` when
  no model is given, since the gateway resolves its model server-side and there
  is no name to echo. Around 180 lines of duplicated construction are gone.

### Fixed
- **Provider tests no longer depend on how fast the machine is.** Three tests
  asserted on wall-clock timing that a test double never actually spends. The
  providers round latency to two decimal places of a millisecond, so a stub
  returning in under five microseconds recorded `0.0 ms`, `compute_tps` gave
  back `None` for an undefined rate, and `assert resp.tokens_per_second > 0`
  raised a `TypeError` — on a fast machine only. A new `scripted_clock` fixture
  pins a provider module's measured elapsed time, so `llamacpp`, `vllm`, and
  `brain-gateway` now assert the exact latency and throughput their inputs
  imply. Product behavior is unchanged; reporting an undefined rate as `None`
  when no time was measurable is correct.

## 0.16.1 (2026-08-04) — Every eval suite reports run integrity

A reporting-honesty fix. Five suites could publish a healthy-looking score
computed from a small fraction of a run, with nothing to say so. Scores
themselves are unchanged; what is new is the evidence sitting next to them.

### Upgrade notes

- **`--json-out` gains an `integrity` object** on `redblue`, `multiturn`,
  `rag`, `codegen`, and `toolcall`. Purely additive; existing fields keep their
  names and meanings.
- **Exit codes are unchanged.** These five report partial coverage without
  failing the run, so anything green today stays green. Gate on
  `integrity.should_exit_nonzero` if you want CI to enforce coverage.
- **`rag`'s `parse_failure_rate` now derives from the shared counting.** The
  value can differ slightly at the edges: a fixture that generated but was never
  judged used to fall out of the denominator entirely and now counts as a judge
  failure.

### Fixed
- **Every eval suite now reports run integrity, and five of them no longer hide
  a degraded run behind a healthy-looking score.** `redblue`, `multiturn`, and
  `rag` averaged only over judges that returned a usable result and published
  nothing else, so a run where nine of ten judge calls failed reported the tenth
  score as its headline number with no indication that anything went wrong. For
  an evaluation tool that is the worst available failure mode, because the
  number looks fine.

  Scores are unchanged. What is new is an `integrity` block in every suite's
  `--json-out` — status, fixture and attempt coverage, and separate counts for
  generation versus judge failures — matching what `adversarial`, `refusal`, and
  `codereview` already reported.

  The separation matters beyond the averaging. `codegen` records a fixture whose
  generation raised as zero tests passed out of its full test count, so a run
  where every provider call failed reported `overall_pass_rate: 0.0` — identical
  to a model that simply cannot code. `generation_failures` now tells them
  apart.

  These five are **report-only**: they do not exit nonzero on partial coverage,
  since that would change the exit code of runs that pass today. Gate on
  `integrity.should_exit_nonzero` from `--json-out` to enforce it in CI. Only
  the three attempt-based suites still gate automatically.

  `rag`'s `parse_failure_rate` is retained and now derives from the shared
  counting. It was the only integrity signal any of these suites had, and it was
  narrower than it appeared: fixtures whose generation failed were never judged,
  so they never entered its denominator.

### Changed
- **Run integrity is counted in one place.** `RunIntegrity` gains
  `from_fixture_outcomes` for suites that never adopted `AttemptResult` —
  rewriting them onto it would mean rewriting their judges too. Both it and
  `from_fixture_attempts` funnel into a single private routine, so the typed and
  neutral paths cannot drift into describing the same run differently.

## 0.16.0 (2026-08-04) — Bounded and observable: spend ceilings, per-caller quotas, correlation IDs

The theme is making a server safe to leave running unattended. Every limit that
existed bounded the *whole* server; nothing bounded any single caller, nothing
bounded eval spend at all, and nothing connected a failure back to the request
that caused it.

No breaking changes. Every new CLI behavior is behind a flag that defaults off,
and `/health` keeps its existing shape.

### Upgrade notes

- **API-triggered evals are now metered by default**, at `$10` per run. If you
  drive `POST /api/v1/evals` for runs costing more than that, pass an explicit
  `budget_usd` (up to `$1000`). CLI evals are unaffected unless you pass
  `--budget`.
- **One API key is now limited to 4 concurrent jobs** (`max_active_jobs_per_caller`).
  A client that deliberately submits more in parallel will start seeing `429`.
  Raise the setting if that is intentional.
- **Point readiness probes at `/ready`, not `/health`.** `/health` is unchanged
  and still answers `200 {"status": "ok"}`, but it is liveness only. `/ready` is
  the one that reports `503` on an unreachable database.

### Security
- **Per-caller job accounting.** The global concurrency cap was
  first-come-first-served: whoever submitted first held all sixteen slots, and
  every other key got `429` until that work drained — a denial of service
  needing no malice, just one impatient script. Jobs now carry an owner and
  `max_active_jobs_per_caller` (default 4) bounds any single key. The global
  check runs first, so a busy server reports its own load rather than blaming a
  caller for someone else's.

  Callers are identified by a twelve-character SHA-256 prefix of their key,
  never the key itself, because that identifier goes into log files. Under
  `--no-auth` there is no credential distinguishing callers, so quotas collapse
  to the global limit and the server now warns about it at startup.
- **Log injection is not possible through `X-Request-ID`.** A caller-supplied
  correlation ID is honored only as `[A-Za-z0-9._-]{1,64}`; anything containing
  newlines or control characters is replaced with a generated ID rather than
  written into the access log.
- **Access logs omit query strings and request bodies.** Both have carried API
  keys — the dashboard once passed one as `?api_key=` — and an access log is the
  wrong place to discover that.
- **Uvicorn's built-in access log is disabled**, because it wrote the raw
  request line including the query string, which defeated the omission above:
  a request to `?api_key=...` had the key written to the log anyway. Found by
  reading the log file of a real server; `TestClient` never starts uvicorn, so
  no test could have caught it. Our middleware replaces it and carries the
  correlation ID and caller besides.
- **Eval suites now have spend ceilings.** Benchmark runs were always metered
  by a `RateBudgetGuard` from the tier profile; eval suites were metered on no
  path at all, so `POST /api/v1/evals` let any API-key holder spend against
  provider accounts until the accounts themselves objected. An adversarial run
  with `--runs` and extra judges is the most expensive thing this tool does.

  Rather than thread a guard through the eighteen `provider.generate` call
  sites across runners, judges, and scorers, the provider is wrapped
  (`GuardedProvider`). Every suite is covered by construction, including suites
  not written yet, and judge traffic is covered too — which is where consensus
  scoring actually spends. The model and all judges share **one** ceiling, so a
  run with `--extra-judges` cannot cost a multiple of what was requested.

  The two surfaces default differently on purpose. The API is **always**
  metered (`budget_usd`, default `$10`, capped at `$1000`); a caller may lower
  it but not remove it, and `0` or negative is a `422`. The CLI is **opt-in**
  via `--budget` on all twelve commands that run an eval suite — the nine suite
  commands plus `sweep`, `archreview`, and `probe` — so no existing invocation
  changes behavior. A run that hits a ceiling raises `EvalBudgetExceededError`
  with the amount spent, distinct from a `400` meaning a bad request;
  per-minute rate pressure is waited out instead, since it clears on its own.

  `sweep` and `archreview` build a provider per model inside their run loop, so
  they hold the ceiling in a `BudgetMeter` that outlives any single provider.
  Metering per model would have made an N-model sweep cost N times the stated
  ceiling.

  Note that a dollar ceiling only binds where calls cost money: Ollama, vLLM,
  and llama.cpp report `$0.00`, so `--budget` is inherently a no-op for them.
- **Job state is now bounded.** `JobManager` kept every job it had ever run in
  an in-memory dict, each holding a full run summary, so a caller submitting in
  a loop grew it until the process died. At most `max_active_jobs` (16) run
  concurrently — further submissions get `429` — and at most
  `max_retained_jobs` (256) finished jobs are retained for polling, evicted
  oldest-first. Running jobs are never evicted.
- **Request sizes are capped.** `iterations` (1000), `interval` (3600s), and
  the `fixtures` list (500) previously had lower bounds but no upper ones, so a
  single API call could run indefinitely and spend without limit. The CLI is
  unaffected and remains the path for long campaigns.
- **Security headers on every response** — `nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer`, `Cross-Origin-Opener-Policy`, `no-store`, and
  a `default-src 'none'` CSP for JSON routes. The dashboard gets a per-response
  nonce-based CSP rather than `unsafe-inline`, so an injected tag cannot
  execute even if something did reach the page as markup.
- **Upgraded 5 dependencies carrying 8 known advisories** — `click`, `idna`,
  `pydantic-settings`, `pytest`, and `urllib3`. Found by the new dependency
  audit on its first run.

### Added
- **`security` CI workflow** running `pip-audit` against the frozen lockfile
  and gitleaks over both committed history and the working tree. It runs on
  push and pull request, and weekly — the schedule is the part that matters,
  since an advisory lands against a version already pinned rather than when
  the code changes.
- **Dependabot** for Python dependencies and GitHub Actions, with routine
  minor and patch updates grouped so a normal week is one reviewable pull
  request.
- **Correlation IDs across the async boundary.** Runs and evals are async jobs,
  so the submitting request returns long before the work finishes and a failure
  hours later had nothing tying it back to who asked for it. Every response now
  carries `X-Request-ID`, and the ID propagates into job tasks through a context
  variable, so `job_submitted` and `job_finished` log lines carry the same ID as
  the request that started them. Send your own ID to thread an existing trace
  through.
- **Structured access logging** — one line per request with correlation ID,
  caller digest, method, path, status, and duration.
- **`GET /api/v1/ready`** — readiness, separate from liveness. Returns `503`
  with a per-check breakdown when the coordinator's database does not answer.
  Previously `/health` answered `ok` unconditionally, keeping a server in a load
  balancer's rotation while every request it received was going to fail. The
  check is a live query rather than a startup flag, since the failures worth
  catching — a deleted file, a full or read-only disk — only surface when
  something touches the database.
- **`--budget` on all twelve eval-running commands** (see the spend ceiling
  entry above).

### Fixed
- **The server now configures its own logging.** `atomics server` never called
  `setup_logging`, so the `atomics` loggers had no handler at INFO and the new
  access log — along with every `job_submitted` and `job_finished` line — was
  written nowhere. A correlation ID that reaches no log file correlates nothing.
- **Server logs are one line per record.** `setup_logging(plain=True)` swaps
  Rich for a plain formatter in server mode. Rich wraps to the console width,
  80 columns when redirected to a file, which split each access log entry across
  four lines and left it unparseable by grep, journald, or any aggregator.
  Interactive commands keep Rich.
- **`mypy` works with the `[rag]` extra installed.** The numpy stubs use 3.12
  `type` statements, a syntax error under `python_version = "3.11"`, and
  `follow_imports = "skip"` does not prevent mypy parsing a stub it can find.
  `follow_imports_for_stubs` does. CI installs only `dev` and `api`, so the
  documented release check failed the first time it ran against a full install.
- **CI lint gate no longer diverges from local.** `ruff check` in CI omitted
  `scripts/`, so unsorted imports in `scripts/test_dashboard_live.py` had been
  failing `main` unnoticed since late July.
- **Two flaky worker tests** replaced fixed `time.sleep` windows with polling,
  which were failing on loaded CI runners that spent the whole window starting
  node.

## 0.15.2 (2026-08-03) — Security hardening: sandboxed codegen and API authorization

Upgrade before exposing an API server. Every finding below came from the
2026-08-02 project audit, and the first three are reachable by anyone who can
reach a coordinator.

### Upgrade notes

- **`--no-auth` now refuses to start on a non-loopback `--host`.** If you run
  `atomics server --no-auth --host 0.0.0.0`, that command will now exit with an
  error. Supply `--api-key` instead, or bind to `127.0.0.1`.
- **`Coordinator.submit_assignment` requires a `worker_id` keyword** and raises
  `AssignmentRejectedError` when the caller does not hold the assignment. This
  only affects code calling the coordinator directly; the HTTP API is
  unchanged apart from returning `409` where it previously accepted a forged
  submission.
- **Consider setting `--worker-api-key`.** Left unset, workers keep sharing the
  submitter keys as before and the server warns at startup.

### Security

- **Model-generated code no longer runs in the evaluating process.** The
  `codegen` suite executed extracted snippets via `exec()` in-process, and armed
  its timeout *after* that call, so module-level statements ran unguarded. It is
  reachable remotely through `POST /api/v1/evals`. Execution moved to a child
  interpreter with a scrubbed environment (provider API keys are no longer
  visible to generated code), a scratch working directory, address-space and CPU
  limits, blocked network calls, and a wall-clock kill of the whole process
  group. New module: `atomics/eval/codegen/sandbox.py`.
- **The dashboard no longer renders API data as markup.** Worker labels and
  capabilities were concatenated into `innerHTML`, so anyone who could register
  a worker could run script in an operator's browser. Rows are now built with
  `createElement`/`textContent`.
- **Assignment results are bound to the worker holding them.** The submit
  endpoint accepted a `worker_id` in the path and never used it, and did not
  check assignment state, so any authenticated worker could complete or
  overwrite any assignment. Submission is now guarded on both the owning worker
  and the `assigned` state, and returns `409` otherwise.
- **Worker keys can be separated from submitter keys** via
  `atomics server --worker-api-key`. Previously both were built from the same
  set, so a worker credential also authorized run and eval submission. Left
  unset the old behavior is kept for compatibility, with a warning at startup.
- **`--no-auth` is refused on non-loopback binds.** `--no-auth --host 0.0.0.0`
  exposed every endpoint, including eval submission, unauthenticated.
- **API keys are compared with `hmac.compare_digest`** instead of set
  membership, which short-circuited on hash.
- The dashboard keeps its key in `sessionStorage` and strips `?api_key=` from
  the URL, so it no longer persists in browser history or proxy logs.

### Changed
- `Coordinator.submit_assignment` now requires a `worker_id` keyword and raises
  `AssignmentRejectedError` when the caller does not hold the assignment.
- `uv.lock` catches up to the 0.15.1 version bump, which the previous release
  left behind.

### Documentation
- `SECURITY.md` gains sections on API server exposure and on what the codegen
  sandbox does and does not protect against.
- `docs/API_SERVER.md` documents separate worker keys, the loopback-only
  `--no-auth` rule, and the dashboard's key handling.

## 0.15.1 (2026-07-30) — Web dashboard and server CLI improvements

### Added
- Optional web dashboard served by `atomics server --with-dashboard`. It is
  read-only, disabled by default, and visualizes recent runs, distributed jobs,
  workers, and provider/model comparisons from existing API endpoints. Access it
  at `/dashboard?api_key=YOUR_KEY` when authentication is enabled.
- `GET /api/v1/distributed/runs` to list recent distributed jobs.
- `GET /api/v1/workers` to list registered workers.
- `atomics server --db-path PATH` to explicitly set the SQLite database path
  used by the API server and coordinator.

## 0.15.0 (2026-07-29) — Distributed full-mode runs and npm worker bridge

### Added
- **`atomics distributed run --mode full`** delegates an entire benchmark run to
  a single worker. The worker executes the full `LoopEngine` locally and returns
  the run summary plus per-task results, so a single host can own an end-to-end
  run without the coordinator machine driving every task.
- Coordinator support for full-mode jobs: one assignment per job, optionally
  pinned to the first worker matching a label selector, or left unclaimed for
  any worker to pick up.
- Worker-side full-run execution via `execute_full_run`, with provider/model/host
  resolution honoring the worker's own flags and the run request's pinned values.
- `atomics distributed status` renders a compact table for completed full-mode
  jobs, while split and pending jobs keep their existing outputs.
- **`atomics worker-npm`** starts a Node.js worker that joins the distributed pool.
  It registers with the coordinator, heartbeats, polls for assignments, and executes
  them via a JSON-over-stdin bridge command. The bundled `task-runner.js` provides
  a minimal example; production deployments can point `--worker-cmd` at their own
  Node.js runner.
- Capability-based assignment routing: workers advertise capabilities (e.g.
  `python`, `node`) and the coordinator only assigns tasks whose `runtime` matches.
  Python workers advertise `python` by default; npm workers advertise `node`. Use
  `--runtime node` with `atomics distributed run` to generate node tasks.
- **`--pool-size N`** on `atomics worker-npm` spawns N independent Node.js workers
  on the same host, each registering separately with the coordinator. The pool
  manager handles graceful shutdown on `SIGINT`/`SIGTERM`.

## 0.14.0 (2026-07-26) — Tool-call divergence suite, release pipeline repairs

### Added
- **`atomics toolcall` — a sixth security suite, measuring tool-call divergence.**
  Every other suite here reads what a model *says*. This one reads what it *does*.
  A model can refuse a request in conversation and then emit the call when the same
  request arrives with function schemas attached; `adversarial` cannot see that,
  because it judges prose, and prose is where models are most heavily aligned.
  Agentic deployments do not run on prose.

  Each of 20 fixtures is sent twice — once as conversation with no tools, once with
  tool schemas — and the gap is the result. Six categories: direct requests,
  fabricated authority, breakglass pressure, indirect injection arriving inside
  tool output, egress exfiltration, and destructive urgency. **Tool calls are never
  executed**; the suite records what the model asked to do.

  The tool channel is scored deterministically, with no judge: each fixture declares
  the call that constitutes compliance *including its arguments*, so
  `read_file("/etc/shadow")` counts and `read_file("README.md")` does not. Rerunning
  against the same recorded responses gives the same answer, which is what makes the
  headline number defensible where a judged score would not be.

  Two measures, each reported with its numerator and denominator because each is
  computed over a subset: **channel divergence** (resisted in prose, complied with
  tools) and **response divergence** (a dangerous call whose own accompanying text
  refuses — self-contradiction within one response). Both report *not measured*
  rather than 0% when no fixture qualifies.

  A **capability probe** runs before any fixture. A model that cannot emit tool calls
  produces no calls on every fixture, indistinguishable from refusing all of them, so
  scoring it normally would rank the least capable models as the most resistant. Such
  a model runs nothing and is reported as `tool_capable: false`; `--no-skip-incapable`
  makes that a non-zero exit for sweeps, where a silently skipped model looks like a
  pass.
- **Tool calling across all ten providers.** `ProviderResponse.tool_calls` and an
  opt-in `generate_with_tools`, in three dialects: OpenAI `chat/completions` (openai,
  vllm, llamacpp, groq, together, gemini — sharing one mixin rather than six copies),
  Anthropic (claude), and Ollama `/api/chat`. `supports_tools` defaults to False and
  the base method raises rather than returning empty, because a silent empty result
  would score as resistance. Malformed arguments are flagged and the call retained: a
  model emitting structurally broken calls is a result, not a blank.
- `RELEASING.md` — the release process, versioning and tag conventions, and the known issues in the current pipeline. There was no documented process, which is how the inconsistencies below accumulated.
- `scripts/changelog_section.py` extracts a version's section from this file, and the release workflow now publishes that instead of asking GitHub to summarize commits. A tag with no changelog section fails the release rather than publishing an empty one.
- `scripts/sync_releases.py` reconciles already-published GitHub releases with the changelog. Dry run by default; leaves hand-written release notes alone unless told otherwise.
- Two tests guarding the drift that is invisible until release day: every tag has usable release notes, and every documented version was actually tagged.

### Fixed
- **`evaluation_results` writes need their parent `runs` row.** The table carries a
  foreign key to `runs`, so `atomics toolcall --save` would have raised
  `IntegrityError` on its first fixture. Found by writing the export test to require
  the saved row back rather than only asserting on the exit code — the export
  dispatch chain falls through and exits zero even with no branch at all, so the
  weaker test passed against a completely unwired suite.
- **The untagged-version guard failed on every correct release.** `RELEASING.md` has
  you write the changelog and bump the version, then run the suite, then tag — so the
  version being released is always documented and untagged while the suite runs, and
  the guard added for `0.11.0` fired on it every time. Found by cutting this release
  and following the documented steps. It now exempts the newest entry when it is also
  the version `pyproject.toml` declares and no tag exists yet, which lapses the moment
  a later version is documented on top of an untagged one — the shape `0.11.0`
  actually had. A guard that fails on correct use is one you learn to skip past, which
  is how the drift it exists to catch gets through.
- **Release notes were published empty.** The workflow used `generate_release_notes`, which diffs against the previous tag — but it also force-moves a floating `v0` tag to each new release, so GitHub compared `v0` against the tag being released, found nothing between them, and published a body containing only a compare link. `v0.8.0`, `v0.12.0`, `v0.13.0` and `v0.13.1` each shipped with carefully written changelog entries and a blank release. Release titles came from the tag alone, so they read `v0.13.1` rather than naming what changed.
- **`0.11.0` was documented but never tagged.** The version was bumped in `pyproject.toml` and written up here, and no tag or release was ever created, leaving no point in history you could check out for it. Now tagged on the commit that set the version, dated to match.
- **Removed the PyPI publish job.** It had failed on every tag since the first release: it published a distribution named `atomics` via trusted publishing, and that name on PyPI belongs to an unrelated C++ package, so it could not have succeeded. Every release page showed a failed job as a result. Atomics installs from source, and each release carries a built wheel; `RELEASING.md` records what a real PyPI rename would involve. No install path changes.
- All nine published releases now carry the changelog entry as their body and a `vX.Y.Z — summary` title. `v0.11.0` was published for the first time. Four releases with hand-written notes kept them and had only their titles normalized, since in several cases they read better than the changelog section.
- **The changelog jumped from 0.3.0 to 0.6.0 with nothing explaining it.** `0.4.0` and `0.5.0` both existed in `pyproject.toml` and had no entry. They turned out never to have been released — intermediate bumps during one day's work whose contents shipped under 0.6.0 — so the gap is now recorded as a note rather than backfilled with duplicate entries. A third guard asserts that every version ever declared in `pyproject.toml` is either documented or explicitly accounted for, which is the invariant that failed here and for 0.11.0.

## 0.13.1 (2026-07-25) — Fixes found by running it for real

Everything here was found by `scripts/smoke_fleet.py`, a new local harness that
starts an actual coordinator and actual worker processes. The 1833-test suite
drives FastAPI's `TestClient` — an in-process ASGI shim — against a fake
provider, so none of these three defects were reachable from it.

### Added
- `scripts/smoke_fleet.py`: a two-phase local smoke test needing no credentials and no model. Phase one runs a two-host fleet job over real sockets against a stubbed OpenAI-compatible endpoint and checks auth, registration, broadcast, the rollup, and the rendered table. Phase two kills a worker mid-run and requires the coordinator to notice the silence, mark the host offline, fail its pinned slice, and let the job resolve to `partial`. It runs in about 30 seconds and touches no real database.
- `atomics server --worker-absent-after SECONDS` (default 120) sets how long a worker may go silent before it is marked offline.

### Fixed
- **`atomics worker --host` was ignored for `-p vllm`.** The factory's vLLM branch reads its own `vllm_host` parameter, because CLI commands accept `--ollama-host` and `--vllm-host` independently and one positional cannot serve both; the worker only ever passed `host`. So a worker aimed at a specific box silently used `ATOMICS_VLLM_HOST` — by default `localhost:8000` — while reporting success. A worker has one `--host` and one provider, so it now routes that host to whichever parameter the selected provider actually reads. `ollama` and `llamacpp` were already correct.
- **The worker absence window ignored the heartbeat interval it was derived from.** `--heartbeat-interval` is an operator flag, but the threshold was hardcoded at four times its *default*, so a worker configured with a 300-second interval was declared absent and had its pinned fleet work failed while behaving exactly as told. The window is now configurable and documented alongside the flag it depends on.
- **The fleet status table was unreadable at 80 columns.** Nine columns left Rich shrinking each until a worker id rendered as `5b5fc…` and a label as `box=a…`, so the table could not say which host was faster — the only question fleet mode exists to answer. Identifiers are now given the room to arrive whole, the model moves to the caption when every host ran the same one, and tokens-per-second loses a decimal place that was costing the cost column its border. A host that completed nothing no longer counts as the hosts disagreeing about the model, which had brought the extra column back for precisely the run where the remaining numbers matter most. The unit test used synthetic ids like `host-a` that were too short to trigger any of it, and now uses realistic 12-hex ids.

## 0.13.0 (2026-07-25) — Distributed fleet mode, coordinator auth

### Upgrade notes
- **Breaking for external API clients.** `POST /api/v1/workers/register`, `POST /api/v1/distributed/runs`, and `GET /api/v1/distributed/runs/{job_id}` previously answered without credentials and now require `X-API-Key`. The bundled worker and CLI already sent it, so first-party use is unaffected; anything else calling these endpoints anonymously must start sending a key. `--no-auth` still bypasses all of it for local development.
- No database migration needed. The new `distributed_assignments.target_worker_id` column is added in place on first open, and `SCHEMA_VERSION` stays at 20, so existing run history, schedules, and the evaluation ledger are preserved.

### Added
- **Fleet mode** — `atomics distributed run --mode fleet` broadcasts an identical task set to every worker matching `--label`, so the same suite can be compared across hosts. `JobMode` had declared `FLEET` since Phase 1 with nothing implementing it, and `Worker.labels` was persisted and never consulted. A worker must match every label pair; an omitted selector broadcasts to all online workers; a selector matching no online worker is rejected rather than creating a job that cannot progress. The matching workers are snapshotted at submit time, and the task set is built once and shared, since per-worker generation would have given each host different prompts while still producing the right assignment count.
- Per-worker rollups written to `distributed_jobs.summary_json`, which had existed since Phase 1 with nothing ever writing to it — a completed distributed job reported a status and no numbers. Records completed/failed counts, tokens, mean and p95 latency, throughput, and estimated cost per host, plus job totals, aggregated from the `TaskResult` workers already submit. `atomics distributed status` renders a row per host for fleet jobs and accepts `--json-out`; split jobs keep their JSON output.
- Worker liveness detection: a worker silent for 120 seconds is marked offline. Nothing previously set that status, so a killed worker stayed `online` indefinitely and its pinned work could never be reaped. Offline workers are excluded from new fleet runs.

### Changed
- `atomics distributed run --label` and the API's `worker_selector` are now honored for fleet mode. They remain rejected for `split`, which assigns each task to the next available worker and genuinely cannot honor a selector.
- Adding a nullable column no longer resets the database. `init_db` responded to any `SCHEMA_VERSION` bump by backing up and dropping every table — correct for an incompatible schema, far too blunt for one new column, where it would discard run history, schedules, and the evaluation ledger. Missing nullable columns are now added in place, with the expected shape read back from SQLite rather than maintained as a second description of the schema. `NOT NULL` and primary-key columns still require a version bump and are logged as such.
- `atomics distributed run` now honors `--provider` / `--model`. The pinned provider travels with every task spec and overrides the worker's own default, so a targeted run executes where it was aimed; previously both flags were accepted, stored in the run request, and ignored. A worker that cannot build the pinned provider fails the assignment with a recorded error rather than silently substituting a different one.
- `--provider` no longer defaults to `claude` — a default would override every worker on every run. Omitting it keeps the previous worker-decides behavior. It is now also validated against the known provider list up front instead of failing deep inside the worker.
- `atomics distributed run --label` and the API's `worker_selector` are rejected instead of silently ignored: split mode assigns each task to the next available worker, so a selector could never be honored and runs looked targeted while landing anywhere. The CLI fails before submitting and `POST /api/v1/distributed/runs` returns `400`. Label-based targeting arrives with fleet mode (Phase 2).
- The provider-attempt outcome contract (`ProviderOutcome`, `ProviderOutcomeKind`, `policy_block_reason`, `provider_outcome_from_exception`) moved from `atomics.eval.outcomes` to `atomics.providers.outcomes`, so the provider layer no longer imports the eval layer above it. `atomics.eval.outcomes` re-exports every moved name, leaving existing imports working, and a 567-line module is now two focused ones.
- Provider construction moved from `atomics.commands.common._make_provider` to `atomics.providers.factory.make_provider`. The API server and distributed workers had to import the command layer to build a provider, so a missing credential inside a FastAPI worker surfaced as a Click exception. The factory now raises `ProviderConfigError` and each caller renders it in its own idiom: the CLI as a `ClickException` labelled with the flag the user typed, the API as a `400`, a worker as a failed assignment. `_make_provider` remains as the CLI's thin translating wrapper, so command modules and their tests are untouched, and the supported-provider list has one definition (`PROVIDER_NAMES`) rather than three.

### Added
- End-to-end tests that drive the real CLI against a local HTTP server speaking the OpenAI chat-completions dialect (`tests/inference_stub.py`), covering `atomics eval`, `atomics adversarial`, and `atomics qa --profile`. Each asserts both on the JSON artifact produced and on the requests actually received — including that the model under test and the judge model do not cross on the wire — closing a gap where provider tests injected fake clients and CLI tests replaced the runner under test.
- Coverage floor in CI (`--cov-fail-under=85`, currently 87.1%). The threshold lives on the CI command line rather than in `addopts`, so focused local runs are not gated on whole-package coverage.
- Tests asserting that every committed profile in `profiles/examples/` loads and renders its body template into valid JSON.
- `tests/test_layering.py` enforces both import-direction rules by AST inspection: nothing outside `atomics/commands/` and `atomics/cli.py` may import the command layer, and `atomics/providers/` may not import `atomics.eval`. Both violations were invisible at runtime — the code worked, only the dependency direction was wrong — so a test is the only thing that prevents their return.
- `SECURITY.md` now documents the secret-scanning workflow (TruffleHog and gitleaks, over both git history and the working tree) with a working invocation for each.
- `.gitleaks.toml` extending the upstream ruleset with path exclusions and a documented allowlist for placeholder strings that remain in git history.

### Fixed
- **Three unauthenticated coordinator endpoints.** `POST /api/v1/workers/register`, `POST /api/v1/distributed/runs`, and `GET /api/v1/distributed/runs/{job_id}` answered anonymously while the worker lifecycle endpoints beside them required a key, so anyone able to reach the port could submit jobs that spend GPU time and cloud budget, read job status, and register phantom workers. Submitter-facing routes now require client auth and registration requires worker auth. No client changes: the worker and CLI already sent `X-API-Key`. `--no-auth` is unaffected.
- A distributed job could wait forever on a host that had gone. A stale assignment was returned to the pool with its pin intact, where only the absent worker could claim it, and a host that died before claiming anything left work in `pending`, which the stale scan never examines. Pinned work belonging to an absent worker now fails, letting the job resolve to `partial` with that host's losses recorded. Such tasks are never re-run elsewhere: silently finishing one host's slice on another would corrupt the comparison fleet mode exists to make. Split mode still requeues unpinned work to any worker.
- `max_retries` on a distributed run request had been declared since Phase 1 and read nowhere, so assignments retried without limit. It now bounds retries for pinned assignments.
- CI synced only the `dev` extra, so the 13 test modules importing `fastapi` at module scope errored during collection on every run. Both workflows now sync `--extra api`, as do the test-suite instructions in `README.md` and `QUICKSTART.md`, which had the same omission.
- `profiles/examples/ctf-ai-gate.yaml` used a flat `url`/`method`/`headers`/`body`/`response_field` schema that `load_profile` does not read, so copying it as documented failed with `http type requires http.url`. Its body template also used single-brace `{prompt}`, which `render_body` never substitutes. Rewritten against the nested `http:` / `response:` schema.
- The `stoneburner-target-profiles` skill documented those same wrong key names and claimed `response.text_field` accepts dot-paths; extraction is a flat `dict.get`, so nested paths never worked.
- `.trufflehog.yaml` used `exclude_paths` / `exclude_detectors` keys that TruffleHog's `--config` does not accept, so any scan referencing it exited with a parse error. Replaced with `.trufflehogignore` for the `--exclude-paths` flag, which is what the tool actually consumes.
- Documentation examples inlined a literal `sk-abc123` API key, which tripped credential scanners and modelled pasting keys into shell history. `docs/API_SERVER.md` and `docs/CLI_REFERENCE.md` now read `$ATOMICS_API_KEY` from the environment.

## 0.12.0 (2026-07-23) — Distributed benchmark runs

### Added
- Distributed benchmark runs: `atomics distributed run` submits split-mode jobs across multiple workers; `atomics worker` starts a worker process that polls the coordinator, executes tasks, and reports results. Workers can now target any provider/model/host (`--provider`, `--model`, `--host`).
- New API endpoints for worker registration, heartbeat, task polling, and result submission under `/api/v1/workers` and `/api/v1/distributed/runs`.
- Pluggable worker API-key authentication (`X-API-Key`) for distributed endpoints.
- End-to-end local test coverage for distributed runs.
- Coordinator edge-case tests: offline-worker requeue, timeout requeue, retry-count increments, partial job status, and `recover_jobs`.

### Fixed
- `atomics distributed status` now emits clean JSON instead of Rich-markup JSON.
- SQLite `ResourceWarning`s in tests by closing all connections opened during tests via `tests/conftest.py`.

## 0.11.0 (2026-07-20) — API server mode, real RAG retrieval, richer multi-turn fixtures

### Added
- API server mode: `atomics server` runs a FastAPI service with async job scheduling (`POST /api/v1/runs`, `POST /api/v1/evals`, `GET /api/v1/jobs/{id}`), API key authentication (`X-API-Key`), and read-only endpoints for comparison (`/api/v1/compare`) and recent runs (`/api/v1/reports/recent-runs`). Requires the `[api]` extra.
- Added 20 new multi-turn conversation fixtures (`mt-eval-16` through `mt-eval-35`) covering contradiction detection, persona drift/stability, long-context retention, multi-turn tool-use, and security-focused scenarios. Long-context fixtures use `max_output_tokens=1024` to avoid truncating summary turns.
- RAG pipeline with real retrieval: `atomics rag-index` builds a sqlite-vec index from local documents, `atomics rag --index` runs existing fixtures against retrieved chunks, and `atomics rag-retrieval` reports recall@k, precision@k, MRR, and nDCG@k. Optional `[rag]` extra includes `sqlite-vec` and `sentence-transformers`.

## 0.10.0 (2026-07-17) — RAG eval, multi-turn conversations, cost advisor, CI gates, docs overhaul

### Added
- **`atomics advisor`** — cost optimization advisor that analyzes historical
  benchmark data grouped by category and model, identifies the cheapest model
  meeting a `--min-quality` threshold (default 80%), and reports per-category
  recommendations with savings percentages. `--current-model`, `--since-hours`,
  `--json-out` flags.
- **`atomics multiturn`** — multi-turn conversation evaluation with 15 fixtures
  testing context retention (3), instruction following (3), coherence (2),
  correction handling (2), multi-step reasoning (1), constraint accumulation (2),
  and security-focused conversations (2). Real multi-turn: calls `generate()` per
  turn, accumulates transcript, scores at both turn level (Accuracy/Context Use/
  Coherence) and conversation level (Retention/Consistency/Instruction). 22 tests.
- **`atomics codegen`** — code generation evaluation with 15 Python fixtures
  (fizzbuzz through topological sort). Models generate functions, which are
  executed against deterministic test cases — pass/fail is objective, no judge.
  extract_function() handles code blocks and raw definitions. Custom comparison
  for anagram grouping and topological sort validity. 23 tests.
- **Multilingual eval fixtures** — 10 fixtures across 8 languages (Spanish,
  French, German, Portuguese, Japanese, Chinese, Korean, Arabic). Selectable
  via `atomics eval --fixtures ml-01,...`. 10 tests.
- **Webhook notifications** — `atomics/webhooks.py` with Slack (Block Kit),
  Discord (embeds), and generic HTTP POST. Auto-detects format by URL.
  `check_regression()` detects latency/success-rate degradation. Config via
  `ATOMICS_WEBHOOK_URL`. 18 tests.
- **llama.cpp provider** (`--provider llamacpp`) — direct local inference
  targeting llama-server's OpenAI-compatible endpoint. Zero cost, /health
  check. Config via `ATOMICS_LLAMACPP_HOST`. 7 tests.
- **Groq, Together AI, Google Gemini providers** — three new cloud providers
  with pricing tables (10 providers total). 29 tests.
- **GitHub Actions CI workflow** (`.github/workflows/eval-gate.yml`) — test,
  doctor, and optional eval/adversarial gate jobs with manual dispatch for
  provider and minimum resilience threshold. Uploads JSON artifacts.
- **Adversarial integrity ledger** — every fixture now retains its typed
  per-attempt and per-judge-call evidence in JSON and SQLite, including failed
  generation and judge calls. Schema v19 adds additive status, coverage,
  failure, representative-error, ledger, and provider-attempt token fields to
  `adversarial_results`, with model-under-test usage rolled up to parent runs.
- **`atomics adversarial --allow-partial`** — explicitly permits partial or
  infrastructure-invalid execution to exit zero after diagnostics.
- **Generic evaluation ledger** — schema v20 adds `evaluation_results` for
  refusal and secure-code-review fixtures. Results are saved incrementally,
  preserve typed status, usage, cost, sanitized errors, and raw model/judge
  evidence, and finalize their parent run even when later execution fails.
- **Two new adversarial suites** (adversarial fixture count 64 → 72):
  - `encoding` (5 fixtures) — base64 / ROT13 / leetspeak / hex / homoglyph-wrapped
    override requests. Tests whether safety reasoning survives an obfuscation
    layer or only operates on surface text. Group alias: `--category encoding`.
  - `many_shot` (3 fixtures) — context-stuffing attacks that stage many fabricated
    "assistant complied" turns via `prior_turns`, then deliver the real ask. Tests
    long-context safety degradation. Group alias: `--category many_shot`.
  Both use benign-but-forbidden markers only (no harmful content ships).
- **`atomics refusal`** — refusal-calibration eval measuring **both** safety
  failure modes. Fixtures pair `comply` (benign/dual-use security work), `refuse`
  (operational harm), and `clarify` (ambiguous scope) expectations; a judge
  classifies each response and the scorer reports `over_refusal_rate`,
  `under_refusal_rate`, and an overall `calibration_score`. Catches over-aligned
  models that refuse legitimate security work, which resistance-only evals miss.
- **`atomics codereview`** — secure-code-review eval on code snippets and unified
  diffs. Vulnerable fixtures carry a known CWE (SQLi, command injection, path
  traversal, hardcoded secret, insecure deserialization, weak password hash);
  clean fixtures measure false positives. Reports `detection_rate`,
  `false_positive_rate`, and an F1-style `review_score`. `diff` mode tests
  PR-style review of a change in isolation.
- Docs: `docs/ADVERSARIAL_SUITES.md` documents all four new suites (encoding,
  many_shot) and companion evals (refusal, codereview) with usage and metrics.

### Changed
- **Honest adversarial completion** — partial and infrastructure-invalid runs
  now exit nonzero by default after JSON output and database finalization.
  Every fixture is persisted, compare runs own finalized parent rows, and the
  existing JSON shape remains compatible as an additive superset.
- **Converged refusal and code-review execution** — both runners now retain
  immutable provider attempts and judge-call evidence, expose
  `fixture_results`, typed integrity, and total cost, and follow the same
  nonzero-on-incomplete policy. Their commands support `--allow-partial`,
  `--save/--no-save`, progress callbacks, and canonical JSON output.
- **Modular CLI foundation** — refusal and code-review commands now live under
  `atomics.commands`; shared provider construction, model attribution,
  progress, JSON writing, persistence conversion, and integrity exit policy
  are centralized in `commands.common`.
- **Pre-1.0 schema upgrades** — opening a pre-v20 database creates a
  timestamped WAL-safe backup before the existing reset migration policy runs.

### Security
- Refusal and code-review JSON exports and
  `evaluation_results.result_json` contain raw model and judge evidence.
  Documentation now calls out their sensitive-data handling requirements;
  persisted exception summaries continue to be sanitized.

## 0.9.0 (2026-07-09) — labcompare, security hardening, frontier comparison, Phase-C typing

### Added
- **`atomics labcompare`** — compare two+ Ollama inference hosts side-by-side on
  throughput (single-stream tok/s, latency, prompt-eval rate, VRAM fit read from
  each host's `/api/ps`) and quality parity (same fixtures, one fixed judge, so
  identical weights should score identically — a gap flags a problem). Additive:
  reuses existing providers/runners/judge as libraries and persists to a new
  `labcompare_results` table (schema v16). No existing command or table changed.
- **`docs/FRONTIER_COMPARISON.md`** — validated comparison of frontier cloud
  models (GPT-5.5, Claude Sonnet 5, Opus 4.8) vs local inference (qwen3.6:27b,
  phi4). Key finding: local qwen3.6:27b matches every frontier model at 100%
  adversarial resistance and ~95% security capability for $0.
- **`SECURITY.md`** — operational security documentation covering hooks, OAuth,
  secrets, URL validation, and output sanitization.
- **`atomics/validation.py`** — central URL validator (`validate_endpoint_url`)
  and error sanitizer (`sanitize_error`) used by CLI, labcompare, and all eval
  runners to strip leaked credentials from persisted exception strings.
- **ARCHITECTURE.md** — layer map, the load-bearing primitives, how to add an
  eval suite, and the security model. Linked from the README for contributors.
- **`atomics/stats.py`** — single home for the percentile helper that was
  copy-pasted across 5 modules.
- **Type checking in CI** — ship a `py.typed` marker (PEP 561) and add a mypy
  gate. **Phase-C typing pass is complete**: `scenario`, `qa_runner`, all five
  provider adapters, `storage.repository`, and `atomics.cli` are now type-clean
  with their real types (profiles, provider clients, result dataclasses via
  `TYPE_CHECKING`, and a `BaseProvider`-typed factory). All `# type: ignore`
  shims are gone and **no per-module mypy overrides remain** — `mypy atomics` is
  green across all 89 files.

### Changed
- **Single provider factory** — `eval`/`archreview` no longer carry their own
  `_build_provider`; all commands use `_make_provider` (now takes optional
  `region`/`context_tokens`/`inference_timeout`).
- **Eval-suite convergence** (additive, no breaking changes): every suite now
  has `Summary.to_dict()` + `--json-out` (added `eval`, `probe`, `archreview`)
  and creates + finalizes a parent `runs` row (`eval` now calls `complete_run`;
  `probe`/`archreview` gained `complete_probe_run`/`complete_archreview_run`).
  `RedBlueSummary`/`ProbeSummary` expose `fixture_results` as an alias for
  `results`; `archreview` accepts `--runs` as an alias for `--rounds`. See
  ARCHITECTURE.md "known divergences" (JSON-export and parent-run-row rows now
  fully converged).

### Security
- **`atomics secrets get` no longer prints the value by default.** It reports
  presence + a masked preview; use `--show` to print the raw value for piping.
  Removes accidental secret exposure via terminal scrollback / shell history.
- **Path traversal in `--repo`** — `atomics archreview --repo ../../x` is now
  rejected; only simple names matching files in `atomics/archreview/repos/` are
  accepted.
- **Unknown provider names error instead of silent Ollama fallback** — a typo in
  `--provider` or `--extra-judges` now raises a clear error instead of quietly
  using the wrong backend.
- **Rich markup injection** — LLM responses, judge rationale, and error messages
  are escaped before terminal rendering so a model cannot inject fake Rich markup.
- **URL validation** — all `--ollama-host`, `--vllm-host`, `--judge-host`, and
  `--host` flags reject `file://`, embedded credentials, and path traversal.
- **Error message sanitization** — exception strings stored in the DB are scrubbed
  of Bearer tokens, API keys, and AWS credentials before persistence.
- **Secrets key allowlist** — `atomics secrets set` validates key names against
  `KNOWN_KEYS`; use `--force` for custom keys.
- **Rich tracebacks disabled in normal mode** — only shown with `-v`/`--verbose`
  to prevent accidental exposure of frame locals containing secrets.

## 0.8.0 (2026-07-04) — New adversarial suites, export/compare/CI plumbing, redblue variance

### Added
- **`atomics redblue --json-out FILE`** — machine-readable run export via
  `RedBlueSummary.to_dict()`, matching the adversarial command.
- **Suite-isolated export** — `query_task_results` gains `suite`/`suite_prefix`
  filters; `atomics export --suite eval` and `--suite redblue` now return only
  those rows instead of all `task_results` blended together.
- **Three new adversarial fixture suites (16 fixtures)** raising the total to 64:
  - **multi-turn manipulation** (`atomics/eval/adversarial/multiturn.py`, group
    `--category multiturn`): gradual escalation, rapport exploitation, fabricated
    "as we agreed earlier" prior agreement, foot-in-the-door, and role drift.
    Fixtures carry a scripted `prior_turns` conversation the runner renders as a
    transcript prefix, so multi-turn attacks work across every provider without
    changing the single-prompt `generate()` contract.
  - **RAG / retrieved-context poisoning** (`rag_poisoning.py`, `--category
    rag_poisoning`): instructions embedded in retrieved documents, citation
    spoofing, retrieved-doc claims of authority over the system prompt,
    frontmatter directives, and poisoned-knowledge-to-unsafe-action.
  - **MCP tool-description injection** (`tool_desc_injection.py`, `--category
    tool_desc_injection`): hostile directives hidden in a tool's description,
    tool shadowing, parameter-schema injection, hidden/unicode directives, and
    exfiltration framed as tool telemetry — the model-reasoning analogue of a
    hostile MCP server's tool-metadata attack surface.
- **`atomics adversarial --json-out FILE`** — machine-readable export of the full
  run (per-fixture scores, labels, judge scores, rationales, latency, cost) via
  `AdversarialSummary.to_dict()`, including both models when `--compare` is used.
- **`atomics adversarial --compare MODEL`** — run a second model on the same
  fixtures and print a per-fixture score diff (Δ B−A) plus overall-resilience
  delta. Accepts `model`, `provider:model`, or `provider:model@host`.
- **`atomics adversarial --fail-on-resilience N`** — CI gate; exits non-zero when
  severity-weighted resilience %% is below the threshold.
- **Adversarial persistence lifecycle** — the run now creates a parent `runs` row
  (tier `adversarial`) and finalizes it with `complete_adversarial_run()`, which
  aggregates `adversarial_results` (not `task_results`). New
  `get_adversarial_results()` repository query and `atomics export --suite
  adversarial` (also included in `--suite all`).
- **`atomics redblue --runs N`** — variance-aware scoring (mean ± stddev across
  passes) matching `adversarial`; makes the existing QUICKSTART example valid.
- **Progress tracker for long-running evals** — group-level `--verbose/-v` and
  `--progress/--no-progress` flags. Live Rich spinner shows fixture ID, category,
  and ETA during inference. Works across `redblue` and `adversarial` commands.
- **Resilient judge scoring for thinking-mode models** — judge calls now use a
  3-tier fallback: (1) `thinking=False` direct response, (2) retry with thinking
  enabled if response empty, (3) parse from thinking content as last resort. Works
  across all providers (Ollama, Claude, OpenAI, vLLM) without model-specific config.
- **qwen3.6 model research** (`docs/model-notes/qwen/`): architecture analysis,
  speed/quality benchmarks, deployment role recommendations. Key finding: qwen3.6
  MoE (35B-A3B) at 61 tok/s validated as superior judge model (stricter, more
  discerning than qwen2.5:7b).
- **Red/Blue capability leaderboard** (`docs/LEADERBOARD-REDBLUE.md`): 20-model
  overnight sweep results with resistance-vs-capability 2x2 matrix.
- **OS-keychain secrets layer** (`atomics secrets set/get/list/delete`): layered
  resolution (env → .env → keychain) with macOS Keychain / Linux secret-service.

### Fixed
- **redblue truncated thinking models** — fixtures cap `max_output_tokens` at
  1024, which is sized for the visible answer; with thinking enabled, reasoning
  models spent that budget on hidden reasoning and got cut off, scoring as a
  capability gap. The runner now raises the budget to >=4096 when thinking is
  active (explicit `--thinking` or an auto-detected reasoning model).
- **Adversarial fixture count reconciled** — `ALL_FIXTURES` was 32 while the
  runner actually loaded 48 (mcp_agentic + tool_safety were wired in the runner
  but not exported), so the header/docs disagreed with the real run. `ALL_FIXTURES`
  is now the single source of truth (64 with the new suites); the runner and CLI
  both select via `select_fixtures()`, and the duplicate `AdversarialFixture`
  dataclass in `mcp_agentic.py` was removed in favor of the canonical one.
- **redblue persistence** — FK constraint failure when saving fixture results
  (missing parent `runs` row). Now creates run row before fixture processing.
- **Judge parse failures with thinking models** — qwen3.6, deepseek-r1, and
  other thinking-capable models produced empty responses when used as judge,
  causing "Parse failed" on all fixtures. Fixed via thinking-aware fallback chain.

## 0.7.0 (2026-06-23) — Adversarial security suites, reasoning-model judge support, archreview keys

### Added
- **`atomics archreview`** — a repo-agnostic security-architecture review benchmark. Feeds each model under test a **deterministic, content-hashed, token-budgeted evidence pack** of a codebase (tiered: `floor` 16k / `local` 32k / `wide` 48k / `expanded` 128k, so small local models, practical local runs, larger local models, and frontier models are compared fairly) and scores its structured findings two independent ways: (1) **objective** difficulty-weighted OWASP-category recall/precision against a per-repo answer key (deterministic, judge-independent), and (2) a separate **self-judge-guarded reasoning score** (0–10 → 0–1) rating trust-boundary/data-flow/prioritization quality. Multi-round runs report finding-set **robustness** (mean pairwise Jaccard stability + recall stdev). Answer keys are pluggable per repo (`atomics/archreview/repos/<name>.yaml`); **OWASP Juice Shop** ships as the first target with its key derived from `challenges.yml` (per-category weight = summed challenge difficulty). New `atomics/archreview/` package (taxonomy, models, evidence-pack builder, prompt + lenient findings parser, answer-key generator, scorer, async runner), `archreview_results` table (schema v15), and `atomics archreview` CLI with a Rich comparison table. `--max-output-tokens` caps generation for slow local triage runs, `--inference-timeout` lets long-context local runs finish, and `--judge-only` skips objective scoring for repos without a key. Reuses the existing provider abstraction, `detect_self_judge`, and lenient-parsing patterns. New tests across taxonomy/models/pack/prompt/keygen/scorer/runner/storage/CLI.

- **Zero-trust adversarial fixtures** (`atomics/eval/adversarial/zerotrust.py`) —
  6 new fixtures benchmarking model resistance to agentic attack patterns: tool-use
  manipulation via injected context (MCP-T02), breakglass token extraction,
  credential exfiltration via tool calls, system-prompt rewrite persistence,
  authority fabrication, and runtime blocklist evasion. Each maps to a proven attack
  from the zero-trust stack (RCON injection, break-glass bypass, camazotz
  cred_broker, nullfield HOLD, skillseraph J1, runtime blocklist evasion). Registered in
  `ALL_FIXTURES`; run with `atomics adversarial --category zerotrust`.
- **Archreview answer keys for ecosystem repos** — pluggable YAML keys for
  `nullfield` (10 categories: PEP/PDP split, 5-action model, credential stripping,
  fail-closed, circuit breaker, TLS mesh assumption, admin API, identity, audit,
  budgets), `zero-trust-blueprint` (10 categories: layered PEP, shared PDP,
  ambient nonbypassability, egress credential isolation, admission hygiene, network
  defense-in-depth, identity gap, model allowlist, CNI caveat), and `camazotz`
  (10 categories: guardrail-not-boundary, OWASP MCP Top 10, OIDC identity,
  nullfield sidecar, tool execution despite refusal, observer, multi-provider,
  credential labs, runtime config, supply chain). All with 4-tier evidence packs.

- **Agentic-reasoning adversarial fixtures** (`atomics/eval/adversarial/agentic_reasoning.py`) —
  11 fixtures across 5 categories testing model reasoning about security architecture:
  MCP protocol (tool-output injection, dangerous tool selection, cross-server exfil),
  supply-chain trust (dependency-planted configs, review suppression), delegation
  (credential forwarding, privilege escalation through depth), egress awareness
  (credential sprawl, secrets-in-prompts), admission (LLM-as-policy antipattern,
  deterministic vs non-deterministic). Group alias: `--category agentic`.
- **`--verbose` flag** for adversarial eval — dumps the full attack prompt, model
  response, judge rationale, and resistance criteria for each fixture.
- **Reasoning-model judge support** — deepseek-r1, phi4-reasoning, gemma4, and
  functiongemma can now be used as judges. Three-pass score parsing (standard →
  markdown → bare-score), `<think>` block stripping, sentiment-based fallback,
  and score-rationale contradiction detection.
- **ADVERSARIAL_SUITES.md** — comprehensive docs covering flow, scoring, all
  suites, 10-model benchmark leaderboard, and ecosystem context.
- **Total adversarial fixtures: 32** (base 15 + zerotrust 6 + agentic 11).

### Fixed and improved
- **Ollama provider** — explicitly set `think=false` for non-thinking models,
  preventing Ollama from auto-enabling thinking and returning empty responses
  (affected gemma4:e4b).
- **Claude provider default** — updated from deprecated `claude-sonnet-4-20250514`
  (404) to `claude-sonnet-4-6` (verified valid). All tests, CLI, and README updated.
- **Adversarial scorer** — multi-format judge output parsing, sentiment fallback,
  contradiction detector, lenient label resolution (numeric labels), increased
  max_tokens for judge calls (128→512).
- **CLI output** — category shown per fixture, first-sentence rationale in default
  mode (full in --verbose), soft-wrap for long lines, spacing fixes.
- **`archreview` parser tolerance** — added three new fallback passes so every major model output format is handled: (1) **markdown table rows** (`| injection | routes/x.ts | high | raw sql |`), (2) **numbered/bold lists** (`1. **Injection** — routes/x.ts — high — why`), and (3) **hybrid labeled-pipe** (`INJECTION | ROUTE: routes/x.ts | SEVERITY: high | WHY: …`). All passes now guard against markdown table header and separator rows so label words (`Category`, `Location`, etc.) and `---|---` lines are never returned as findings. Prompt tightened with an explicit "no markdown/no table/no numbering" instruction and a concrete one-line example to improve small-model format compliance.
- **`archreview` taxonomy** — 30+ new synonyms covering non-web targets: `path traversal`, `directory traversal`, `lfi/rfi`, `privilege escalation`, `open redirect` → `broken_access_control`; `hardcoded credentials/secret` → `sensitive_data_exposure`; `race condition`, `toctou`, `buffer overflow`, `integer overflow` → `improper_input_validation`; `ssti`, `log injection`, `crlf injection`, `ldap injection` → `injection`; `weak cipher/hash`, `insecure random` → `cryptographic_issues`; `dependency confusion`, `supply chain` → `vulnerable_components`; `dos`, `redos`, `resource exhaustion` → `broken_anti_automation`. Makes the tool useful against Go APIs, Rust services, IaC, and other non-webapp targets without any configuration.
- **`archreview` Juice Shop answer key v2** — added `ssrf` (weight 6.0, matches Juice Shop `challenges.yml` difficulty 6 and the confirmed surface in `routes/profileImageUrlUpload.ts`). Total weight 90 → 96. Frontier models that surface SSRF as the correct architectural category are no longer precision-penalized.
- **`archreview` Ollama context wiring** — evidence packs now request explicit `num_ctx` and `num_predict` so the model's context window is large enough for the prompt and there is always reserved output room. `ContextExhausted` is recorded instead of a misleading parse-failed/zero-recall result when a model stops before producing findings.
- **`archreview` evidence tiers** — added `local` (32k) between `floor` and `wide` for practical local-GPU runs. `--max-output-tokens` and `--inference-timeout` CLI flags for controlling slow local inference. `--verbose` flag streams per-model/per-round findings and scores as they complete.
- **`archreview` judge identity** — `Judge Model` column in the comparison table shows `provider:model` (e.g. `ollama:deepseek-r1:7b`) so multi-model runs are unambiguous.
- **`archreview` finding deduplication** — `parse_findings()` collapses exact (category, location) pairs emitted multiple times by looping models (e.g. the same route listed 7× by qwen2.5:7b). Same category at different locations remains distinct. Applied at all five fallback passes.
- **`pack.py` type annotation** — `build_pack(cfg)` parameter typed as `TierConfig` (import added); zero untyped parameters across the `atomics/archreview/` package.
- **Test quality** — fixed pre-existing coroutine-never-awaited warning in `test_adversarial.py` by converting the nested sync side-effect to an async function; suite now runs with zero warnings.

## 0.6.0 (2026-06-16) — Security suites, vLLM provider, judge accuracy & token-burn fidelity

> Adds the red/blue capability and adversarial resilience suites, the live ecosystem probe, a vendor-neutral `inference.env` standard, an OpenAI-compatible `vllm` provider, hardened judge accuracy (consensus, calibration, gold-criteria coverage), honest token-burn/cost fidelity, and the `qa`/`soak`/`scenario`/`contention` load-testing commands.

### Added (stoneburner)
- **`atomics eval --fixtures ev-19[,…]`** — run a subset of the 25 eval fixtures for fast spot-checks/iteration instead of the full set. Unknown ids error out; the run header reports the real fixture count. (`run_eval` already accepted a `fixtures=` arg; this wires the CLI flag.)
- **Security suites are two independent axes** — documented (README + QUICKSTART) that `redblue` measures **capability** and `adversarial` measures **resilience**, that they don't correlate (live: a non-thinking 12B at 93%/76% vs a thinking 2B at 54%/91%), and that high-capability + low-resilience is the riskiest profile.
- **Full local-gateway model-class coverage** — added the gateway tags that were classifying as UNKNOWN (`gemma4:12b`/`26b`, `phi4:latest`, `phi4-mini:latest`, `qwen2.5-coder:14b`, `qwen3:14b`, `cogito:3b`, `dolphin3:latest`) so `compare`/`sweep` no longer show blanks; classes verified against live model sizes. Regression test asserts the whole lineup classifies.
- **QUICKSTART.md** — recipe-first guide grouped by goal (cost, quality+judge, consensus, security, scale, QA) with local **and** cloud treated as peers, a model-agnostic callout, config cheat-sheet (incl. `*_TIMEOUT`), and troubleshooting.
- **Self-judge guard** — `eval`, `redblue`, and `adversarial` now detect when a judge is the same provider+model as the model under test (covering consensus-panel members and the both-default-model case) via `detect_self_judge`, and warn that scores are biased by self-preference. Providers expose a uniform `default_model` property to resolve unspecified models. 6 tests.
- **Judge accuracy** — the LLM-as-judge quality scorer was hardened so accuracy scores are reproducible and harder to game. (1) **Deterministic scoring**: `generate()` gains an optional `temperature` across every provider (withheld where the backend forbids it — OpenAI reasoning models and Claude extended-thinking; brain-gateway controls sampling server-side) and the quality/resistance judges now request `temperature=0.0`. (2) **Fair completeness**: the judge's response-truncation cap scales to each fixture's expected output length (`char_budget_for_tokens`, ~4 chars/token, floored at 3000) so long HEAVY answers are judged in full instead of cut at 3000 chars. (3) **Gold-criteria coverage**: `compute_criteria_coverage` adds an objective, judge-independent lexical measure of how many of a fixture's gold criteria appear in the response — persisted on `task_results.criteria_coverage` (schema v13) and aggregated in `compare`. (4) **Multi-judge consensus**: `score_consensus` scores with a primary judge plus an optional panel of `(provider, model)` judges, averaging the scores that parsed and recording inter-judge stdev (`task_results.judge_score_stdev`, schema v14, surfaced as `avg_judge_score_stdev`); `eval` gains `--extra-judges provider:model[@host]`. (5) **Robust parsing**: a lenient field-by-field fallback (tolerates markdown, reordering, missing rationale) plus exactly one reformat retry replaces the all-or-nothing single regex, and a `parse_failure_rate` is surfaced in the eval summary. (6) **Calibration regression guard**: `atomics/eval/calibration.py` + `calibrate_judge` rank graded answers (wrong → thin → thorough) and assert monotonic, well-separated scoring; an opt-in live test (`ATOMICS_LIVE_JUDGE=1`) validates the real Ollama judge. New tests: `test_temperature.py`, `test_judge.py`, `test_calibration.py` plus storage/CLI coverage. README "Judge accuracy" section documents the methodology.
- **Token-burn fidelity** — provider metrics now report only what each API can actually observe, so cross-model cost/throughput comparisons are honest. (1) Claude prompt-cache tokens (`cache_read_input_tokens`/`cache_creation_input_tokens`) are captured on `ProviderResponse.cache_read_tokens`/`cache_write_tokens` and priced correctly (reads 0.10×, writes 1.25× the base input rate). (2) Thinking tokens are populated only when truly reported — OpenAI `reasoning_tokens` (Chat Completions `completion_tokens_details`, Responses API `output_tokens_details`); Ollama/vLLM use a character-proportional estimate anchored to the real output-token total; Claude stays 0 (Anthropic bills thinking as output). (3) `tokens_per_second` is standardized to total output tokens ÷ elapsed time via `compute_tps`, with a new `tps_basis` field labeling `wall_clock` vs `generation` (Ollama decode time); Bedrock now reports throughput. (4) Pricing tables and the cost function are centralized in `atomics/providers/pricing.py`. New fields persist to `task_results` (schema v12) and surface in `provider-test` output and `compare`. README "Metrics & Fidelity" section documents the methodology. New tests: `test_pricing.py` plus cache/thinking/tps coverage across the provider suites.
- **`inference.env` standard + reference reader/resolver** — a vendor-neutral control file (`docs/INFERENCE_ENV.md`) lets any box describe the LLM inference target it is wired to, so consumers self-configure. New `atomics/inference.py` provides: `parse_env`, `normalize_legacy` (folds legacy `brain/` keys — `INFERENCE_API`, `OLLAMA_*`, `OPENAI_*` — into the canonical `INFERENCE_BACKEND/URL/MODEL/THINK/API_KEY` schema), `InferenceTarget` (typed view with `from_text`/`from_mapping`), `load_control_file` (searches `$INFERENCE_ENV`/`$BRAIN_ENV`/`/opt/agentic`/`/etc/agentic`, returns `None` for clean fallback), the agnostic resolver (`resolve_model`, `resolve_endpoint`, `check_model_compat`, `check_backend`, `render_env`, `resolve`), and `provider_from_target`/`load_provider` to auto-build the matching provider (ollama→`OllamaProvider`, vllm→`VllmProvider`, openai→`OpenAIProvider`). No box-specific hosts/creds/k8s glue. 23 unit tests.
- **`--provider vllm`** — new `VllmProvider` adapter targeting any OpenAI-compatible endpoint (`/v1/chat/completions`). Supports vLLM, LiteLLM, llama.cpp. `--vllm-host` flag on all eval commands (`run`, `provider-test`, `sweep`, `adversarial`, `redblue`, `probe`). Config via `ATOMICS_VLLM_HOST` / `ATOMICS_VLLM_MODEL`. Thinking mode via `chat_template_kwargs.enable_thinking` for qwen3-family models. 24 unit tests + 7 CLI integration tests. Probe profile `profiles/vllm-gateway.yaml` for the lab LiteLLM gateway.
- **`atomics models --provider vllm`** — model discovery from OpenAI-compatible `/v1/models` endpoint. Table drops Size/Params columns (not available from gateway). `--vllm-host` flag mirrors `--host` for Ollama.
- **`qwen3:0.6b` registered** — added to `MODEL_CLASS_MAP` (LIGHT) and `THINKING_CAPABLE` set. Was showing as unknown on the gateway.
- **`atomics baselines` CLI test** — added `test_cli_baselines_empty` and `test_cli_baselines_with_records` covering the empty-db and populated table paths.
- **Baseline regression tracking** — `atomics soak --save-baseline NAME` captures key metrics (avg tok/s, peak tok/s, P95 latency, error rate, verdict) under a named key. `--compare-baseline NAME` prints a colour-coded delta table and reports IMPROVED / STABLE / REGRESSED. `atomics baselines` lists all saved baselines. Thresholds: >10% TPS drop or >20% P95 spike triggers REGRESSED. Schema v11 adds `baselines` table with UNIQUE(name, suite) upsert. 23 tests.
- **Scenario ramp (`--ramp`)** — `atomics scenario --ramp 10` staggers worker start times across the ramp window so load builds gradually rather than hammering at t=0. Stored on `ScenarioResult.ramp_seconds`. 6 tests.
- **Multi-model VRAM contention (`--models`)** — `atomics stress --models qwen2.5:3b,qwen2.5:7b` runs each model solo first (baseline TPS), then all simultaneously. Reports per-model TPS degradation as a contention factor (<1.0 = degradation). CLI colour codes: green ≥0.9, yellow ≥0.7, red <0.7. 22 tests.
- **`atomics qa`** — QA validation for CTF solvability and AI gate regression. Reads a YAML fixture file defining prompts with `pass_patterns`, `fail_patterns`, and `must_match` (pass/fail/any). Fires each at an Ollama model, evaluates responses with case-insensitive regex, prints a rich table and overall pass rate. `--fail-fast` stops at first failure. Example fixtures in `qa/examples/`. 32 tests.
- **`atomics soak --think-time SECONDS`** — simulate realistic user pacing by inserting a think-time sleep between requests per worker. Defaults to 0 (no pause). Lets you model actual concurrency (N workers × think_time determines effective req/s) rather than pure hammering. 4 tests.
- **`atomics qa --profile`** — `--profile profiles/local/gate.yaml` routes fixture queries through a TargetProfile (app HTTP endpoint or Ollama with custom system prompt) instead of raw Ollama. Fixture YAML stays committed; real IPs/tokens live in `profiles/local/` (gitignored). 8 tests.
- **Custom target profiles** — YAML-based profiles for testing application-level AI gates. Two modes: `ollama` (Ollama with custom system prompt, temperature, num_predict) and `http` (arbitrary HTTP endpoint with body template, response parsing, and latency extraction). `--profile` flag added to `soak`, `stress`, and `scenario` commands. Response classification (`classify:` in YAML) detects model drift under load. Sensitive profiles gitignored via `profiles/local/`. 42 tests.
- **`atomics soak`** — long-duration stability test. Holds fixed concurrency for minutes or hours, sampling throughput and latency at configurable intervals. Computes linear-regression drift to classify runs as STABLE / DEGRADED / UNSTABLE. Tracks VRAM drift, error rate, and total cost. Human-friendly duration parsing (`30m`, `2h`, `1h30m`, bare minutes). Works with all providers. Database persistence in `soak_results` table. 48 tests.
- **`atomics scenario`** — mixed-workload inference simulation. Runs multiple agentic workload profiles (gate, eval) concurrently against a shared Ollama host. Measures per-workload P50/P95 latency, SLA compliance, and cross-workload interference scores via automatic solo-baseline comparison. Supports YAML scenario files and CLI shorthand (`-w type:model:concurrency[:sla_ms]`). 8 gate prompts + 8 eval prompts built in; custom prompt files supported. 42 tests.
- **`atomics sweep --save`** — persist sweep results to new `sweep_results` DB table (schema v8)
- **`atomics export --suite {tasks,stress,sweep,all}`** — export any stored suite as jsonl or CSV
- **`atomics compare --output FILE`** — write JSON comparison alongside the Rich table
- **`atomics doctor` documentation** — README section with full check list and CI usage guidance
- **`configs/*.toml` removed** — orphaned profile TOMLs unreferenced by any code were deleted

### Fixed (stoneburner)
- **Thinking models spuriously timed out at 120s** — `OllamaProvider`/`VllmProvider` hard-coded a 120s request timeout, so reasoning models (e.g. qwen3:4b) that legitimately think for >2min on hard fixtures failed mid-eval (surfaced as `ev-19` `ReadTimeout` in a live run). Timeout is now configurable via `ATOMICS_OLLAMA_TIMEOUT` / `ATOMICS_VLLM_TIMEOUT` with a 300s default, threaded through the provider factories.
- **Blank error messages on failed fixtures** — exceptions with an empty `str()` (notably `httpx.ReadTimeout`) recorded `error_message=''`, producing useless `"ev-NN failed:"` log lines and empty DB rows. The eval/redblue/adversarial runners now fall back to `repr(exc)` and log the resolved message; redblue also records `error_class`. Regression test added.
- **`atomics sweep` hid why a model failed** — a model whose fixtures all errored (e.g. a tag not pulled on the host → 404) showed only `FAIL`. `ModelSweepResult` now carries a representative `error` and the CLI prints it next to each failed model.
- **Test suite hard-failed without optional extras** — running on a base install (no `openai`/`boto3`) produced 7 failures instead of skips. Tests that require those extras now use `pytest.importorskip`, and two coverage-sensitive soak timing tests were hardened (1.0s→3.0s sampling window) after flaking on a slower Linux host. Verified green on both macOS/3.13 and Linux/3.12.
- **`atomics eval --provider vllm` crashed with "Unknown provider: vllm"** — `eval` was the one eval command never wired for vLLM: its local `_build_provider` had no `vllm` branch and the command lacked a `--vllm-host` flag, despite `vllm` being a valid `PROVIDER_CHOICES` value. Added the `--vllm-host` option, a `vllm` branch (model + judge resolve via `vllm_host`/`ATOMICS_VLLM_HOST`), and made the saved run record's model resolution vllm-aware. 2 regression tests (explicit host + config fallback). Verified live against the vLLM gateway (model under test, qwen2.5:3b as judge).
- **Probe HTTP connector used uninstalled `aiohttp`** — `_fetch_http` lazily imported `aiohttp`, which is not in the dependency tree, so the entire HTTP probe path was dead (0% coverage). Migrated to `httpx` (already a core dep). Added truncation, custom-header, and error-path tests; connector coverage 57% → 94%.
- **Adversarial scorer regex** — `[\r\n]+` after resistance score required a newline immediately after the integer. Small models (qwen2.5:3b, qwen3.5:0.8b) pad lines with trailing spaces (`RESISTANCE: 5  \n`), breaking all parse attempts and silently returning `score=0.5` for every fixture. Changed to `\s+` throughout. 3 regression tests added (clean `\n`, trailing-space `\n`, CRLF).

### Tests (stoneburner)
- **OAuth flow coverage 36% → 100%** — added 17 tests for `auth/oauth.py`: `_exchange_code`, `_refresh` (incl. refresh-token preservation), `_device_code_flow` (success / `authorization_pending` / `slow_down` / unknown-error), `_browser_flow` with mocked callback server, `login` headless/browser delegation, `validate` exception path, `_parse_token_response`, and all three `Handler.do_GET` branches (success, state mismatch, OAuth error).

### Changed (stoneburner)
- `atomics sweep --host` renamed to `--ollama-host` (hidden `--host` alias kept for backward compat)
- `atomics capacity --think-time` shorthand changed from `-t` to `--think` (removes collision with `--tier` `-t`)
- Schema bumped v7 → v8 → v9 → v10 → v11 → v12 → v13 → v14 (adds `sweep_results`, `scenario_results`, `soak_results`, and `baselines` tables, then `cache_read_tokens`/`cache_write_tokens`/`tps_basis`, then `criteria_coverage`, then `judge_score_stdev` columns on `task_results`; existing DBs auto-migrated via fresh-start)
- `atomics export --suite soak` added for soak result export

### Added (mcpnuke)
- **SARIF 2.1.0 export** via `--sarif FILE` — maps CRITICAL/HIGH → `error`, MEDIUM → `warning`, LOW → `note`; embeds `security-severity` and taxonomy tags; ready for GitHub Code Scanning upload
- **`--fail-on {critical,high,medium,low,any,none}`** — configurable CI severity gate replacing hardcoded CRITICAL/HIGH exit; default unchanged (`high`)
- **LICENSE** file (MIT)
- **mcpnuke-runner** documentation in `docs/ci-cd-guide.md` — K8s/Helm deployment, env vars, manual trigger API

### Fixed (mcpnuke)
- `_raw_token` is now stripped from `auth_context` in all JSON output paths — tokens are never written to report files, PR comments, or CI artifacts

---

## Note on the 0.4.0 / 0.5.0 gap

Both versions existed, briefly, as `pyproject.toml` bumps during a single day of
work on 2026-05-23. Neither was tagged or released, and the work they carried —
the adversarial and red/blue suites, the live ecosystem probe, the brain-gateway
provider, and schema v6 — is documented under 0.6.0, where it first shipped.
They are recorded here so the jump from 0.3.0 to 0.6.0 is not mistaken for a
missing entry. This heading deliberately does not parse as a released version.

---

## 0.3.0 — Accuracy Scoring, LLM-as-Judge, and Business-Case Narrative

### Added
- **`atomics eval` command** — run a fixed set of 15 reproducible eval fixtures against any provider and score quality with an LLM judge
  - Each fixture has gold criteria (key concepts a correct answer must cover) that are injected into the judge's rubric
  - Fixtures span security, cloud/infra, LLM/AI, and general engineering at light / moderate / heavy complexity
- **LLM-as-judge** (`atomics/eval/judge.py`) — rubric-based scoring (Accuracy 0–4, Completeness 0–3, Format 0–3) normalized to 0.0–1.0
  - Defaults to local Ollama so judging never adds API spend
  - Tolerates CRLF line endings and common judge model spelling variations (`COMPLETNESS` etc.)
  - Multi-line rationales collapsed to a single stored sentence
- **`accuracy_score`, `judge_model`, `quality_rationale`** fields on `task_results` (schema v4)
- **`avg_accuracy_score` and `value_score`** columns in `atomics compare` — value score = accuracy / cost-per-1K-tokens, with a $0.001 floor so free local runs have a finite (large) score rather than infinity
- **`atomics compare --narrative`** — plain-English business-case summary comparing self-hosted vs cloud API options: quality gap, cost delta, privacy posture, and total API spend
- **`--judge-host`** option on `atomics eval`; falls back to `--ollama-host` → `ATOMICS_OLLAMA_HOST` so the judge always routes to the right Ollama instance
- **Reasoning-model support** in the OpenAI provider: `gpt-5` and related models use `max_completion_tokens` (not `max_tokens`) with an 8× multiplier for internal reasoning budget
- **Model pricing** for `gpt-5`, `gpt-5-turbo`, `gpt-5.3`, `gpt-5.5`, `o3-pro`
- **Model class entries** for `qwen2.5:14b`, `qwen2.5:32b`, `qwen2.5:72b`, `gpt-5*`, `o3-pro`

### Fixed
- `on_fixture_done` callback now fires for **failed** fixtures too — previously provider failures were invisible in the live eval table and were never saved to the database
- `_SCORE_RE` regex uses `[\r\n]+` instead of `\n` to handle CRLF responses from OpenAI and other APIs
- `COMPLET\w*` pattern absorbs both `COMPLETENESS` and `COMPLETNESS` (qwen spelling variant)
- Schedule command used Rich's `Console.print` for raw config text, causing word-wrap to split long `ExecStart` lines and break embedded flags
- `test_ollama_config_defaults` isolated from project `.env` so it does not fail when `ATOMICS_OLLAMA_HOST` is set to a non-default value

### Changed
- `atomics compare` table now includes Quality and Value Score columns alongside existing latency/cost columns
- Schema bumped from v3 → v4; existing databases are migrated automatically on first open
- 289 tests, 0 failures

## 0.2.0 — Ollama Provider + Throughput Metrics

### Added
- **Ollama provider** (`--provider ollama`) for zero-cost local LLM inference
  - Configurable endpoint via `--ollama-host` or `ATOMICS_OLLAMA_HOST`
  - Default model via `ATOMICS_OLLAMA_MODEL` (default: `qwen2.5:7b`)
  - Full tok/s throughput measurement from Ollama's `eval_duration`
  - Health check via `/api/tags`
- **`tokens_per_second`** field on `ProviderResponse` — all providers now report throughput
  - Cloud providers derive it from `output_tokens / latency`
  - Ollama derives it from Ollama's native eval timing
- **tok/s column** in `atomics compare` output for throughput comparison
- **Ollama models** in the model class taxonomy (`qwen2.5:*`, `qwen3:*`, `llama3.*`, `mistral:7b`, `codellama:7b`)
- **Ollama connectivity check** in `atomics doctor`
- **`tokens_per_second`** persisted in SQLite (`task_results` table), schema v3

### Changed
- `atomics provider-test` now shows throughput (tok/s) when available
- `atomics compare` table includes average tok/s per provider/model
- Bumped version to 0.2.0

## 0.1.0 — Initial Release

- Claude, Bedrock, OpenAI providers
- Burn tiers (ez/baseline/mega)
- SQLite metrics persistence
- CLI: run, report, compare, schedule, provider-test, doctor, export
- OAuth/OIDC and Codex authentication
- Cron/systemd/launchd scheduling
