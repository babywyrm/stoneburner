# Changelog

## Unreleased

### Added (stoneburner)
- **`atomics archreview`** — a repo-agnostic security-architecture review benchmark. Feeds each model under test a **deterministic, content-hashed, token-budgeted evidence pack** of a codebase (tiered: `floor` 16k / `wide` 48k / `expanded` 128k, so small local models, larger local models, and frontier models are compared fairly) and scores its structured findings two independent ways: (1) **objective** difficulty-weighted OWASP-category recall/precision against a per-repo answer key (deterministic, judge-independent), and (2) a separate **self-judge-guarded reasoning score** (0–10 → 0–1) rating trust-boundary/data-flow/prioritization quality. Multi-round runs report finding-set **robustness** (mean pairwise Jaccard stability + recall stdev). Answer keys are pluggable per repo (`atomics/archreview/repos/<name>.yaml`); **OWASP Juice Shop** ships as the first target with its key derived from `challenges.yml` (per-category weight = summed challenge difficulty). New `atomics/archreview/` package (taxonomy, models, evidence-pack builder, prompt + lenient findings parser, answer-key generator, scorer, async runner), `archreview_results` table (schema v15), and `atomics archreview` CLI with a Rich comparison table. `--max-output-tokens` caps generation for slow local triage runs, `--inference-timeout` lets long-context local runs finish, and `--judge-only` skips objective scoring for repos without a key. Reuses the existing provider abstraction, `detect_self_judge`, and lenient-parsing patterns. New tests across taxonomy/models/pack/prompt/keygen/scorer/runner/storage/CLI.

## 0.6.0 (2026-06-16) — Security suites, vLLM provider, judge accuracy & token-burn fidelity

> Adds the red/blue capability and adversarial resilience suites, the live ecosystem probe, a vendor-neutral `inference.env` standard, an OpenAI-compatible `vllm` provider, hardened judge accuracy (consensus, calibration, gold-criteria coverage), honest token-burn/cost fidelity, and the `qa`/`soak`/`scenario`/`contention` load-testing commands.

### Added (stoneburner)
- **`atomics eval --fixtures ev-19[,…]`** — run a subset of the 25 eval fixtures for fast spot-checks/iteration instead of the full set. Unknown ids error out; the run header reports the real fixture count. (`run_eval` already accepted a `fixtures=` arg; this wires the CLI flag.)
- **Security suites are two independent axes** — documented (README + QUICKSTART) that `redblue` measures **capability** and `adversarial` measures **resilience**, that they don't correlate (live: a non-thinking 12B at 93%/76% vs a thinking 2B at 54%/91%), and that high-capability + low-resilience is the riskiest profile.
- **Full BRAINBOX model-class coverage** — added the gateway tags that were classifying as UNKNOWN (`gemma4:12b`/`26b`, `phi4:latest`, `phi4-mini:latest`, `qwen2.5-coder:14b`, `qwen3:14b`, `cogito:3b`, `dolphin3:latest`) so `compare`/`sweep` no longer show blanks; classes verified against live model sizes. Regression test asserts the whole lineup classifies.
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
