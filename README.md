# Stoneburner

> **Atomics** — Agentic token usage benchmarking platform

A continuous, cron-schedulable benchmarking harness that runs realistic everyday tasks against LLM providers to measure token consumption, cost, and performance trends over time. Supports tiered usage profiles and multiple providers.

## Quick Start

```bash
# Install with uv
uv sync

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Test the provider connection
uv run atomics provider-test

# Run 5 benchmark tasks on the default (baseline) tier
uv run atomics run -n 5

# View reports
uv run atomics report
```

## Burn Tiers

Atomics supports three usage tiers that control task complexity, model selection, cadence, and budget:

| Tier | Tasks | Model | Interval | Budget | Tokens/hr |
|------|-------|-------|----------|--------|-----------|
| **ez** | Light only | Haiku 4.5 | 300s | $5 | 15k |
| **baseline** | Light + Moderate | Sonnet 4.6 | 120s | $50 | 100k |
| **mega** | All (incl. Heavy) | Sonnet 4.6 | 30s | $250 | 500k |

```bash
# Quick local test on each tier
uv run atomics run --tier ez -n 3 -i 5
uv run atomics run --tier baseline -n 3 -i 5
uv run atomics run --tier mega -n 3 -i 5

# List tier profiles
uv run atomics tiers
```

## Architecture

```
stoneburner/
├── atomics/              # Core Python package
│   ├── core/             # Loop engine, task runner, rate/budget guard
│   ├── providers/        # LLM adapters (Claude, Bedrock)
│   ├── tasks/            # Task catalog with weighted, tiered selection
│   ├── storage/          # SQLite metrics persistence
│   ├── scheduler/        # Cron/systemd/launchd generation and installation
│   ├── workers/          # Optional npm worker bridge (Phase 3)
│   ├── tiers.py          # Burn tier profiles (ez/baseline/mega)
│   ├── cli.py            # Click CLI entry point
│   └── reporting.py      # Rich table trend reports
├── configs/              # Rate/budget profiles (default, aggressive, conservative)
├── tests/                # 63 tests at 73% coverage
└── workers/npm/          # Optional Node.js workers (Phase 3)
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `atomics run` | Start the benchmarking loop (continuous or bounded) |
| `atomics run --tier mega -n 10` | Run 10 mega-tier tasks |
| `atomics run --provider bedrock` | Use AWS Bedrock instead of Claude API |
| `atomics run -b 5.0` | Run with $5 budget cap |
| `atomics run -i 10` | Override interval to 10 seconds |
| `atomics report` | Display usage reports and trends |
| `atomics tiers` | Show available burn tiers and their profiles |
| `atomics provider-test` | Health check the configured provider |
| `atomics schedule` | Generate scheduler configs |
| `atomics schedule --install` | Install schedule on this system |
| `atomics schedule --uninstall` | Remove installed schedule |

## Providers

| Provider | Status | Flag |
|----------|--------|------|
| **Claude** (Anthropic API) | Production-ready | `--provider claude` (default) |
| **Bedrock** (AWS) | Implemented | `--provider bedrock --region us-east-1` |

Bedrock requires `boto3`: `uv sync --extra bedrock`

## Scheduling

```bash
# Auto-detect best scheduler for this OS and install
uv run atomics schedule --tier ez -n 5 -i 15 --install

# Generate without installing (preview)
uv run atomics schedule --tier baseline --format crontab
uv run atomics schedule --tier mega --format systemd
uv run atomics schedule --tier ez --format launchd

# Remove installed schedule
uv run atomics schedule --tier ez --uninstall
```

Supports crontab (Linux/macOS), systemd timers (Linux), and launchd (macOS). Auto-detection picks the best option for the current platform.

## Configuration

Set via environment variables (prefix `ATOMICS_`) or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required for Claude provider |
| `ATOMICS_DEFAULT_MODEL` | `claude-sonnet-4-6` | Model to benchmark |
| `ATOMICS_LOOP_INTERVAL_SECONDS` | `120` | Seconds between tasks |
| `ATOMICS_MAX_TOKENS_PER_HOUR` | `100000` | Hourly token cap |
| `ATOMICS_MAX_REQUESTS_PER_MINUTE` | `30` | Request rate limit |
| `ATOMICS_BUDGET_LIMIT_USD` | `50.00` | Total cost cap per run |
| `ATOMICS_DB_PATH` | `data/atomics.db` | SQLite database location |

CLI flags (`--tier`, `--budget`, `--interval`) override these defaults at runtime.

## Running Tests

```bash
uv sync --extra dev
uv run python -m pytest -v
uv run python -m pytest --cov=atomics --cov-report=term-missing
```

## License

MIT
