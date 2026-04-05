# Stoneburner

> **Atomics** — Agentic token usage benchmarking platform

A continuous, cron-schedulable benchmarking harness that runs realistic everyday tasks against LLM providers to measure token consumption, cost, and performance trends over time.

## Quick Start

```bash
# Install with uv
uv sync

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Test the provider connection
uv run atomics provider-test

# Run 5 benchmark tasks
uv run atomics run -n 5

# View reports
uv run atomics report
```

## Architecture

```
stoneburner/
├── atomics/              # Core Python package
│   ├── core/             # Loop engine, task runner, rate/budget guard
│   ├── providers/        # LLM adapters (Claude, Bedrock scaffold)
│   ├── tasks/            # Task catalog with weighted selection
│   ├── storage/          # SQLite metrics persistence
│   ├── scheduler/        # Cron/systemd/launchd generation
│   ├── workers/          # Optional npm worker bridge (Phase 3)
│   ├── cli.py            # Click CLI entry point
│   └── reporting.py      # Rich table trend reports
├── configs/              # Rate/budget profiles (default, aggressive, conservative)
├── tests/                # Unit and integration tests
└── workers/npm/          # Optional Node.js workers (Phase 3)
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `atomics run` | Start the benchmarking loop (continuous or bounded) |
| `atomics run -n 10` | Run exactly 10 tasks then stop |
| `atomics run -b 5.0` | Run with $5 budget cap |
| `atomics report` | Display usage reports and trends |
| `atomics provider-test` | Health check the configured provider |
| `atomics schedule` | Generate crontab/systemd/launchd configs |

## Scheduling

```bash
# Generate a crontab entry (runs every 30 min, 10 tasks per run)
uv run atomics schedule --format crontab -i 30 -n 10

# Generate systemd timer units
uv run atomics schedule --format systemd -i 60 -n 20

# Generate macOS launchd plist
uv run atomics schedule --format launchd -i 30 -n 10
```

## Configuration

Set via environment variables (prefix `ATOMICS_`) or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required for Claude provider |
| `ATOMICS_DEFAULT_MODEL` | `claude-sonnet-4-20250514` | Model to benchmark |
| `ATOMICS_LOOP_INTERVAL_SECONDS` | `120` | Seconds between tasks |
| `ATOMICS_MAX_TOKENS_PER_HOUR` | `100000` | Hourly token cap |
| `ATOMICS_MAX_REQUESTS_PER_MINUTE` | `30` | Request rate limit |
| `ATOMICS_BUDGET_LIMIT_USD` | `50.00` | Total cost cap per run |

## Providers

- **Claude** (v1) — Anthropic API, production-ready
- **Bedrock** (Phase 2) — AWS Bedrock, scaffolded

## Running Tests

```bash
uv sync --extra dev
uv run pytest -v
```

## License

MIT
