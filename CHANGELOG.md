# Changelog

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
