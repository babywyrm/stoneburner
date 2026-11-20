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
- [ ] Optional outbound host allowlists (`--allowed-hosts`) on `atomics server`
      so authenticated callers can only route inference and judge traffic to
      operator-approved hosts.
- [ ] Live `brain-gateway` generate on record. It cannot carry a `temperature`,
      `effort`, `reasoning_mode`, or `thinking` pin: the provider refuses those
      rather than scoring them.
- [ ] Complete `mypy --strict` (`disallow_untyped_defs`,
      `disallow_any_generics`, `warn_return_any`) across `atomics/storage/`,
      `atomics/providers/`, and `atomics/distributed/`.

## 0.34.0 — Judge calibration identity, distributed evals, and CLI debloating

- [ ] Key `judge_calibrations` by `(provider, model, model_digest)` instead of
      bare model name so quantized or fine-tuned weights sharing a tag cannot
      inherit a stale MAE or calibration verdict.
- [ ] Blind-grade and commit a payload-free red calibration set so
      `atomics judge-calibrate` covers both blue and red rubrics.
- [ ] Wire `eval_request` execution on `POST /api/v1/distributed/runs` (`full`
      mode) alongside `run_request`, rolling up suite integrity across workers.
- [ ] Consolidate shared judge invocation, retry, and error-sanitization logic
      across `eval/judge.py`, `eval/rag/judge.py`, and
      `eval/multiturn/judge.py`.
- [ ] Extract shared Click provider/judge option decorators in
      `atomics/commands/common.py` and retire unused legacy re-exports in
      `atomics/cli.py` without breaking CLI entry points.
- [ ] Complete `mypy --strict` across `atomics/eval/`, `atomics/benchmark/`,
      `atomics/load/`, `atomics/archreview/`, and `atomics/probe/`.

## 0.35.0 — Full `mypy --strict`, multi-turn escalation, and lockfile hashes

- [ ] Enable `strict = true` globally in `[tool.mypy]` after finishing
      `atomics/api/`, `atomics/mcp/`, `atomics/repl/`, and `atomics/commands/`
      (keeping only the optional `[rag]` third-party stub skip).
- [ ] Add multi-turn adversarial escalation fixtures (gradual context drift and
      turn-by-turn authority buildup) with per-turn and conversation-level
      breach scoring.
- [ ] Enforce hash-verified dependency auditing in CI security gates alongside
      `pip-audit` (`pypi` + `osv`), `zizmor`, `bandit`, `gitleaks`, and
      `trufflehog`.

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
- **Temperature and modal headlines (2026-10-15).** Across four local models on
  20 `toolcall` fixtures (`--runs 3 --no-thinking`), `--temperature 0` hid 1–5
  leaky fixtures per model (`granite4.2:3b` any-run 0.40 at own temperature vs
  0.15 at 0; `qwen3.5:4b` 0.75 vs 0.55; `qwen3.5:9b` 0.60 vs 0.50). Modal
  `dangerous_call_rate` also hid 1/3 leaks (`granite4.2:3b` 0.15 modal vs 0.40
  any-run). Gate promotions on `dangerous_any_run_rate` at the model's own
  temperature (`--runs 5` preferred); reserve `--temperature 0` for
  like-for-like diffs.

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
