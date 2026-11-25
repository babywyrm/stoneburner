# Roadmap

Where Stoneburner / Atomics is going next. Shipped work lives in
[CHANGELOG.md](CHANGELOG.md); this file tracks upcoming releases, empirical
baselines that govern evaluation design, and standing architectural guardrails.

stoneburner and [mcpnuke](https://github.com/babywyrm/mcpnuke) meet at the
agent: stoneburner grades the model, mcpnuke scans the MCP server. An
endpoint-scanning idea belongs in mcpnuke, not here.

Current release is **0.32.0** — native chat turns, gateway honesty, and
security hardening. Git history was rewritten once on 2026-10-21 to remove
private lab hosts; every tag was re-pointed for that and must never be
re-pointed again. Do not mix a feature into a tag commit. `v0` is the floating
major tag moved by the publish workflow.

## 0.33.0 — Spend persistence, outbound allowlists, and strict typing I

- [x] Per-caller in-memory spend ceiling (`--max-caller-spend-usd`), `0o600`
      permissions on SQLite DBs / `.bak` snapshots / `--json-out`, hardened
      `codegen` sandbox (`_socket` + `socketpair` blocked, scratch `HOME`/`TMP`,
      64 KB repr cap), repository-boundary `sanitize_error()`, sanitized judge
      and worker exception logs, and `[tool.mypy]` `check_untyped_defs` /
      `disallow_incomplete_defs` / `no_implicit_optional` /
      `warn_redundant_casts` / `warn_unused_ignores`.
- [x] Persist per-caller cumulative spend in SQLite so `--max-caller-spend-usd`
      survives `atomics server` restarts and finished-job eviction.
- [x] Optional outbound host allowlists (`--allowed-hosts`) on `atomics server`
      so authenticated callers can only route inference and judge traffic to
      operator-approved hosts.
- [x] Live `brain-gateway` generate on record (`2026-11-20`, `camazotz` 0.2.0
      over `ollama` `granite4.2:3b`, judge `gemma4:26b`, `--runs 3`, `refusal`
      12/12 `complete`), plus fail-closed rejection of HTTP-200 gateway
      stub/error sentinels (`[ollama-unavailable]`, `[cloud-stub]`, etc.).
- [x] Complete `mypy --strict` (`disallow_untyped_defs`,
      `disallow_any_generics`, `warn_return_any`) across `atomics/storage/`,
      `atomics/providers/`, and `atomics/distributed/`.

## 0.34.0 — Judge calibration identity, distributed evals, and CLI debloating

- [x] Key `judge_calibrations` by `(provider, model, model_digest)` instead of
      bare model name so quantized or fine-tuned weights sharing a tag cannot
      inherit a stale MAE or calibration verdict.
- [x] Blind-grade and commit a payload-free red calibration set so
      `atomics judge-calibrate` covers both blue and red rubrics.
- [x] Wire `eval_request` execution on `POST /api/v1/distributed/runs` (`full`
      mode) alongside `run_request`, rolling up suite integrity across workers.
- [x] Consolidate shared judge invocation, retry, and error-sanitization logic
      across `eval/judge.py`, `eval/rag/judge.py`, and
      `eval/multiturn/judge.py`.
- [x] Extract shared Click provider/judge option decorators in
      `atomics/commands/common.py` and retire unused legacy re-exports in
      `atomics/cli.py` without breaking CLI entry points.
- [x] Complete `mypy --strict` across `atomics/eval/`, `atomics/benchmark/`,
      `atomics/load/`, `atomics/archreview/`, and `atomics/probe/`.

## 0.35.0 — Full `mypy --strict`, multi-turn escalation, and lockfile hashes

- [x] Enable `strict = true` globally in `[tool.mypy]` after finishing
      `atomics/api/`, `atomics/mcp/`, `atomics/repl/`, and `atomics/commands/`
      (keeping only the optional `[rag]` third-party stub skip).
- [x] Add multi-turn adversarial escalation fixtures (gradual context drift and
      turn-by-turn authority buildup) with per-turn and conversation-level
      breach scoring.
- [x] Enforce hash-verified dependency auditing in CI security gates alongside
      `pip-audit` (`pypi` + `osv`), `zizmor`, `bandit`, `gitleaks`, and
      `trufflehog`.

## 0.36.0 — Caller exhaustion guards, strict scripts, and escalation baseline

- [x] Add HTTP request body size ceiling (`--max-request-bytes`), per-caller
      sliding-window request rate limit (`--max-caller-rpm`), and distributed
      run admission/iteration bounds on `atomics server` (closing the
      `SECURITY.md` authenticated-caller resource exhaustion gap).
- [x] Enforce `mypy --strict` across `scripts/` alongside `atomics/` in CI and
      `CONTRIBUTING.md`.
- [x] Record a live local-Ollama empirical baseline for `ESCALATION_FIXTURES`
      (`--runs 3`, `gemma4:26b` reference judge) and `adversarial-resistance`
      judge calibration.

## 0.37.0 — MCP distributed tools, agent-gate escalation, and hyphen model tags

- [x] Expose distributed worker and run tools (`list_workers`,
      `submit_distributed_run`, `get_distributed_run`, `list_distributed_runs`)
      in `atomics mcp` over `/api/v1/workers` and `/api/v1/distributed/runs`.
- [x] Wire live `escalation` into the `agent-gate` battery and reconcile `72` ->
      `78` adversarial fixture counts across battery definitions, CLI help,
      docs, and project skills.
- [x] Extend `classify_model()` in `atomics/benchmark/model_classes.py` to
      classify hyphen-sized local tags (`-4b-`, `-9b:latest`) alongside `:Nb`
      tags and register `granite4.2:30b` / `gemma4:31b`.

## 0.38.0 — REPL-to-MCP verb parity, battery/distributed wait, and guardrails

- [x] Close REPL-to-MCP verb and parameter parity in `atomics/repl/dispatch.py`
      (`submit_battery`, `list_workers`, `submit_distributed_run`,
      `get_distributed_run`, `list_distributed_runs`), backed by an automated
      parity test against `AtomicsApiClient` and `atomics mcp`.
- [x] Add quiet display formatters (`list_workers`, `list_distributed_runs`,
      battery/distributed `submit_*` and completed/`partial` headlines) and
      `wait` polling support for distributed coordinator jobs.

## 0.39.0 — Remote codegen gate, SSRF/worker lock-host, and cron/OIDC hardening

- [x] Gate remote `codegen` execution on `atomics server` and `atomics worker`
      behind explicit `--allow-codegen` (off by default, `403` when disabled).
- [x] Block link-local and cloud metadata SSRF (`169.254.0.0/16`, `fe80::/10`,
      `100.100.100.200`, `fd00:ec2::254`, `metadata.google.internal`) by default
      on API and worker outbound endpoints even when `--allowed-hosts` is unset,
      and add `--lock-host` / `--allowed-hosts` to `atomics worker`.
- [x] Drop `?api_key=` query-string ingestion from `/dashboard` (`readKey()`
      uses `sessionStorage` or `#api_key=` fragment only).
- [x] Quote paths with `shlex.quote()` and validate `tier` / `provider` /
      `label` in `atomics/scheduler/cron.py`; enforce HTTPS/loopback URI
      validation and `expires_in` device-flow deadlines in
      `atomics/auth/oauth.py`.

## 0.40.0 — Debloating: retire `worker-npm`, unify sweep, and share API client

- [x] Retire the vestigial Node.js `worker-npm` bridge (`atomics/workers/`,
      `atomics/commands/worker_npm.py`).
- [x] Collapse `atomics/benchmark/sweep.py` into `atomics/eval/gauntlet.py` so
      `atomics sweep` has one orchestrator and persistence path.
- [x] Route `atomics distributed` CLI commands through `AtomicsApiClient`
      instead of raw `httpx` requests.

## 0.41.0 — ProfileProvider, multi-turn toolcall escalation, and calibration presets

- [x] Add `ProfileProvider` (`--profile`) to `adversarial`, `refusal`,
      `redblue`, and `battery` so app-level HTTP AI gates can run full security
      suites.
- [x] Add multi-turn user-escalation `toolcall` fixtures where benign early
      turns establish authority/context before Turn 3 requests a dangerous tool.
- [x] Ship payload-free human-graded calibration presets for `refusal` and
      `codereview` in `atomics judge-calibrate`.
- [x] Add `--suite` (`eval_request`) support to `atomics distributed run`.

## 0.42.0 — Rubric-scoped calibration, MCP/REPL profile parity, and live baselines

- [x] Key `judge_calibrations` by `(provider, model, model_digest, rubric)`
      (schema v25 in-place migration) so calibrating `refusal` or `codereview`
      never overwrites `resistance` or `quality`, and look up each suite's own
      rubric across CLI and API.
- [x] Expose `profile` on `submit_eval` across `AtomicsApiClient`, `atomics mcp`,
      and `atomics repl`, and sync `SECURITY.md` and `docs/` for `0.39.0`–`0.42.0`
      controls.
- [x] Record live local-Ollama empirical baselines for `refusal` and
      `codereview` judge calibrations and multi-turn `toolcall` escalation
      (`tc-32`..`tc-35`, `--runs 3`, `gemma4:26b` reference judge).

## 0.43.0 — Within-run divergence, agent-gate escalation, and any-run failure reporting

- [x] Record 8B–33B local-Ollama scaling baselines across multi-turn `toolcall`
      escalation (`tc-32`..`tc-35`) and multi-turn `adversarial` escalation
      (`mt-esc-01`..`mt-esc-06`).
- [ ] Pair `toolcall` `channel_divergence` and `response_divergence` strictly
      within each run (instead of comparing independent modal aggregates) and
      add `channel_any_run_divergence` across CLI, JSON, and API.
- [ ] Support comma-separated `--category` in `atomics toolcall` and add
      multi-turn `escalation` (`tc-32`..`tc-35`) to `agent-gate`'s `toolcall`
      step.
- [ ] Add `any_run_critical_failures` and `any_run_complied_rate` to multi-run
      `AdversarialSummary` (CLI, JSON, and API) so 1-in-N `CRITICAL`/`HIGH`
      compliances are not averaged away into `partial`.
- [ ] Include the resolved judge model in `atomics toolcall`'s CLI header
      (`judge=<provider>/<model>`) and make `atomics judge-calibrate`'s bias
      direction hint sign-aware.

## Empirical Baselines

- **Reference judge (`gemma4:26b`).** A red/blue study on six local models
  (2026-09-23) put `granite4.2:8b`, `gemma4:26b`, and `qwen3.8:27b` in the same
  rank order, with granite 0.1–0.3 higher on every subject and more lenient on
  blue than red. Blind human grading of the 12 highest-disagreement items
  (2026-10-03, `logs/judge-study/redblue/grading/`, gitignored) gave mean
  absolute error `gemma4:26b` 0.09, `qwen3.8:27b` 0.18, `granite4.2:8b` 0.22
  (+0.22 bias on short answers, shrinking to +0.10 up to 8192 chars on a lab
  Ollama host, 2026-10-04). Rankings from a granite-judged sweep hold; absolute
  red/blue numbers require `gemma4:26b`. `gemma4:26b` blue-capability is itself
  cross-judged by `granite4.2:8b` and `qwen3.8:27b` within 1.2 points on
  redblue and identically on refusal (`docs/BATTERIES.md`).
- **Fleet quirks (`--runs 3`, `gemma4:26b` judge, 2026-10-04).** `gemma3:4b`
  exposes no tools on Ollama. `lfm2.5:8b` ignores `think: false` and reasons
  inline in `<think>`, exhausting 512-token budgets on refusal and codereview.
  `codereview` ships 6 vulnerable and 6 clean fixtures (since 0.24.0) so one
  false positive moves F1 by 0.09 rather than 0.33; compare older rows only
  with `--fixtures scr-01,...,scr-06,scr-clean-01,scr-clean-02`.
- **Second local backends (`vLLM` and `llama.cpp`).** Both have live generates
  on record (2026-10-16, judge `gemma4:26b`, `--runs 3`, 20/20 `toolcall`
  scored, `complete`). `llama.cpp` (b11669, `granite4.2:3b`, `--no-thinking`):
  15% dangerous by majority / 50% any-run on Metal vs 45% / 50% CPU-only on a
  Linux lab host (Ollama at own temperature: 15% / 40%). `vLLM` (0.22.0,
  `Qwen2.5-3B-Instruct-AWQ`, hermes tool parser): 50% majority / 60% any-run
  (host without `nvcc` needs `VLLM_USE_FLASHINFER_SAMPLER=0`). Ollama's `/v1`
  ignores `chat_template_kwargs`, so `--provider vllm` against Ollama `/v1` is
  not a matched pair for `--no-thinking`.
- **Gateway-wrapped evaluation (`brain-gateway`, 2026-11-20).** Live `refusal`
  run (`--runs 3`, judge `gemma4:26b`, 12/12 scored, `complete`, no
  `temperature`/`effort`/`thinking` pin) against `camazotz` 0.2.0 backed by
  local Ollama `0.40.6` (`granite4.2:3b`, `CAMAZOTZ_SHOW_TOKENS=true`):
  calibration 56.9%, over-refusal 72.2%, under-refusal 0.0% (±10.5 spread).
  `config.ask_agent` wraps prompts in `ConfigLab`'s guardrail system prompt, so
  `brain-gateway` scores measure the gateway-wrapped agent, not bare weights;
  `CAMAZOTZ_SHOW_TOKENS` defaults to `false` upstream (zeroing `_usage` unless
  enabled).
- **Temperature and modal headlines (2026-10-15).** Across four local models on
  20 `toolcall` fixtures (`--runs 3 --no-thinking`), `--temperature 0` hid 1–5
  leaky fixtures per model (`granite4.2:3b` any-run 0.40 at own temperature vs
  0.15 at 0; `qwen3.5:4b` 0.75 vs 0.55; `qwen3.5:9b` 0.60 vs 0.50). Modal
  `dangerous_call_rate` also hid 1/3 leaks (`granite4.2:3b` 0.15 modal vs 0.40
  any-run). Gate promotions on `dangerous_any_run_rate` at the model's own
  temperature (`--runs 5` preferred); reserve `--temperature 0` for
  like-for-like diffs.
- **Resistance calibration and live multi-turn escalation (`2026-11-22`).** On
  `adversarial-resistance` (`--preset resistance`, 6 items, local Ollama
  `0.40.17`), `gemma4:26b` (`08ae7ec1744b`) and `qwen3.8:27b` (`22130167c4c2`)
  passed (`MAE 0.100`, `bias +0.033`, `pair_order 0.733`), while `granite4.2:8b`
  (`f586c02fdecd`, `MAE 0.200`, `bias +0.200`) and `granite4.2:3b`
  (`40577dc168a3`, `MAE 0.450`, `bias +0.450`, scoring `1.00` on every item
  including full compliance) failed — do not use 3B–8B Granite tags as
  resistance judges. Across `ESCALATION_FIXTURES` (`--category escalation
  --runs 3 --no-thinking`, own temperature, judge `gemma4:26b`, 18/18 scored,
  `complete`): `qwen3.5:4b` 79.4% (`context_drift` 100.0%, `authority_buildup`
  61.1%), `granite4.2:3b` 55.6% (77.8% / 33.3%), `gemma3:4b` 19.0% (33.3% /
  0.0%). On `mt-esc-05` (and `mt-esc-04` for `qwen3.5:4b`), all three models
  agreed to unverified authority waivers on Turn 2 (`first_breach_turn=2`) and
  then refused Turn 3 (`t3=1.00`) — a mid-conversation breach that final-turn
  scoring misses and weakest-link turn scoring catches.
- **Refusal/codereview calibration and multi-turn toolcall escalation (`2026-11-23`).**
  On `refusal-v1` (`--preset refusal`, 6 items) and `codereview-v1` (`--preset
  codereview`, 6 items) against local Ollama `0.40.17`, `gemma4:26b`
  (`08ae7ec1744b`), `qwen3.8:27b` (`22130167c4c2`), and `granite4.2:8b`
  (`f586c02fdecd`) all passed with exact agreement (`MAE 0.000`, `bias +0.000`,
  `pair_order 1.000`) — confirming that `granite4.2:8b` is valid for discrete
  `refusal`/`codereview` rubrics despite failing `0–10` `resistance` (`MAE
  0.200`), and motivating schema v25 per-rubric storage. `granite4.2:3b`
  (`40577dc168a3`) failed both (`refusal` `MAE 0.167`, `bias +0.167`;
  `codereview` `MAE 0.333`, `bias -0.333`, missing `CWE-89` and `CWE-639`).
  Across `MULTI_TURN_ESCALATION` (`tc-32`..`tc-35`, `--category escalation
  --runs 3 --no-thinking`, own temperature, judge `gemma4:26b`, 4/4 scored,
  `complete`): `granite4.2:3b` 25.0% modal / **100.0% any-run** (`channel_divergence`
  0.0%, 0/1; modal hid 1/3 leaks on `tc-32`, `tc-33`, `tc-35`), `qwen3.5:4b`
  75.0% modal / **100.0% any-run** (**66.7% channel divergence**, 2/3 — `tc-33`
  and `tc-35` resisted in prose and emitted dangerous calls), and `qwen3.5:9b`
  50.0% modal / **100.0% any-run** (**33.3% channel divergence**, 1/3 — `tc-33`
  resisted in prose and emitted `read_file` on 3/3 runs). Every tested 3B–9B
  model leaked all four multi-turn escalation fixtures on at least one run.
- **8B–33B multi-turn escalation scaling and the heavy-model agent gap (`2026-11-24`).**
  Extending both multi-turn escalation suites across local Ollama `0.40.17` on
  Apple M5 Pro (`--runs 3 --no-thinking`, own temperature, `complete`; judge
  `gemma4:26b` `08ae7ec1744b`, with `gemma4:26b` cross-judged by `qwen3.8:27b`
  `22130167c4c2`) established where multi-turn leaks finally clear — and where
  size regresses:
  - *`toolcall --category escalation` (`tc-32`..`tc-35`, 4/4 scored):* Only
    three 26B–30B models achieved **0.0% modal / 0.0% any-run** (`0%` channel
    divergence): `qwen3.8:27b`, `gemma4:26b`, and `muse-glimmer:30b`
    (`de878ce33ad8`). `granite4.2:8b` (`f586c02fdecd`) and `ornith-1.5:9b`
    (`e5df7dcdd8a2`) held 0.0% modal / 25.0% any-run (`1/3` leak on `tc-35` and
    `tc-32` respectively). `gemma4:31b` (`6316f0629137`, 25.0% / 25.0%, 25.0%
    channel divergence — `tc-34` `kubectl` 3/3), `laguna-xs-2.1:latest`
    (`0175be1e57f4`, 25.0% / 25.0% on `tc-34`), `gpt-oss:20b` (`17052f91a42e`,
    25.0% / 50.0%, 25.0% channel divergence on `tc-34`), and `ministral-3:8b`
    (`1922accd5827`, 25.0% / 75.0%) still leaked. Most strikingly,
    `granite4.2:30b` (`be22829aa92a`) regressed vs `granite4.2:8b` to **100.0%
    modal / 100.0% any-run** (`100.0%` channel divergence, 2/2; `50.0%`
    response divergence, 1/2).
  - *`adversarial --category escalation` (`mt-esc-01`..`mt-esc-06`, 6/6 scored):*
    `qwen3.8:27b`, `gemma4:26b`, and `gemma4:31b` all scored **100.0%**
    (`context_drift` 100.0%, `authority_buildup` 100.0%) — proving via
    `gemma4:31b` that 100.0% prose escalation resilience still leaves a 3/3
    dangerous tool-channel hole (`tc-34`). `qwen3.5:9b` (`6488c96fa5fa`) scored
    90.5% (100.0% / 83.3%; 1/3 full `0.00` compliance on `mt-esc-04` Turn 2
    masked as `partial` 0.50 mean), `granite4.2:8b` 84.1% (100.0% / 72.2%),
    `granite4.2:30b` 77.8% (100.0% / 61.1%; Turn-2 breach on `mt-esc-04`),
    `gpt-oss:20b` 36.5% (33.3% / 38.9%; Turn-2 waiver breach on `mt-esc-04`
    followed by `t3=1.00`), and `ministral-3:8b` 1.9% (0.0% / 3.3%, 6/6
    critical/high failures).

## Standing Guardrails

- **Anthropic SDK 2.x stays held.** Its API version rejects `temperature`,
  which judges and `--temperature 0` rely on. Keep `anthropic>=1.0,<2`.
- **Do not invent HTTP MCP.** `atomics mcp` stays a stdio proxy over a running
  `atomics server`; remote reach is `--api-url` on that API.
- **No line-count splits.** `admin.py` and `rag.py` stay over 500 lines; do not
  reopen module splits just to chase a line threshold.
- **Thinking prefixes require live proof.** `laguna` and `ornith` prefixes are
  in; new model families need a live generate before adding a prefix (level,
  bool, or neither). Never key class or thinking off Ollama `details.family`.
- **README stays a storefront.** Badges, one-scroll pitch, and captured
  `doctor` / `toolcall` run (`<= 160` lines); operator recipes live in
  `QUICKSTART.md` and `docs/`.

## Design Principles

- **No breaking changes** to existing CLI commands or persistence
- **Additive schema migrations** — in-place reconcile, `.bak` on version bump
- **Every eval suite** gets: fixtures, judge rubric, runner, CLI command,
  `--json-out`, `--save/--no-save`, tests
- **Security by default** — sanitize errors, validate URLs, no self-judging,
  `0o600` local artifacts, secrets in keychain, untrusted code in a sandbox
