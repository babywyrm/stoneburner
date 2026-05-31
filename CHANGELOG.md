# Changelog

## Unreleased — CLI polish, SARIF, export improvements

### Added (stoneburner)
- **`atomics sweep --save`** — persist sweep results to new `sweep_results` DB table (schema v8)
- **`atomics export --suite {tasks,stress,sweep,all}`** — export any stored suite as jsonl or CSV
- **`atomics compare --output FILE`** — write JSON comparison alongside the Rich table
- **`atomics doctor` documentation** — README section with full check list and CI usage guidance
- **`configs/*.toml` removed** — orphaned profile TOMLs unreferenced by any code were deleted

### Changed (stoneburner)
- `atomics sweep --host` renamed to `--ollama-host` (hidden `--host` alias kept for backward compat)
- `atomics capacity --think-time` shorthand changed from `-t` to `--think` (removes collision with `--tier` `-t`)
- Schema bumped v7 → v8 (adds `sweep_results` table; existing DBs auto-migrated)

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
