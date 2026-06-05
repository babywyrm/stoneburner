# Changelog

## Unreleased — atomics qa, regression tracking, contention testing, ramp, think-time, target profiles, soak, scenario, CLI polish, vllm provider

### Added (stoneburner)
- **`--provider vllm`** — new `VllmProvider` adapter targeting any OpenAI-compatible endpoint (`/v1/chat/completions`). Supports vLLM, LiteLLM, llama.cpp. `--vllm-host` flag on all eval commands (`run`, `provider-test`, `sweep`, `adversarial`, `redblue`, `probe`). Config via `ATOMICS_VLLM_HOST` / `ATOMICS_VLLM_MODEL`. Thinking mode via `chat_template_kwargs.enable_thinking` for qwen3-family models. 24 unit tests + 7 CLI integration tests. Probe profile `profiles/vllm-gateway.yaml` for the lab LiteLLM gateway.
- **`atomics models --provider vllm`** — model discovery from OpenAI-compatible `/v1/models` endpoint. Table drops Size/Params columns (not available from gateway). `--vllm-host` flag mirrors `--host` for Ollama.
- **`qwen3:0.6b` registered** — added to `MODEL_CLASS_MAP` (LIGHT) and `THINKING_CAPABLE` set. Was showing as unknown on gpu-host gateway.
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
- **Adversarial scorer regex** — `[\r\n]+` after resistance score required a newline immediately after the integer. Small models (qwen2.5:3b, qwen3.5:0.8b) pad lines with trailing spaces (`RESISTANCE: 5  \n`), breaking all parse attempts and silently returning `score=0.5` for every fixture. Changed to `\s+` throughout. 3 regression tests added (clean `\n`, trailing-space `\n`, CRLF).

### Changed (stoneburner)
- `atomics sweep --host` renamed to `--ollama-host` (hidden `--host` alias kept for backward compat)
- `atomics capacity --think-time` shorthand changed from `-t` to `--think` (removes collision with `--tier` `-t`)
- Schema bumped v7 → v8 → v9 → v10 → v11 (adds `sweep_results`, `scenario_results`, `soak_results`, and `baselines` tables; existing DBs auto-migrated)
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
