# Schedule Status & Provider Comparison

## Problem

Runs don't record which provider, model, tier, or trigger source was used —
making it impossible to compare providers or track scheduled run health.
Installed schedules exist only as OS artifacts with no visibility from atomics.

## Schema v2

Fresh start: bump `SCHEMA_VERSION` to 2, drop and recreate all tables on
mismatch. No migration path for v1 data.

### `runs` table — new columns

| Column | Type | Purpose |
|--------|------|---------|
| `tier` | TEXT NOT NULL | ez/baseline/mega |
| `provider` | TEXT NOT NULL | claude/bedrock/openai |
| `model` | TEXT NOT NULL | Specific model used |
| `trigger` | TEXT NOT NULL DEFAULT 'manual' | manual/scheduled/test |

### New `schedules` table

| Column | Type | Purpose |
|--------|------|---------|
| `schedule_id` | TEXT PK | `{format}.{tier}.{provider}` |
| `format` | TEXT NOT NULL | crontab/systemd/launchd |
| `tier` | TEXT NOT NULL | ez/baseline/mega |
| `provider` | TEXT NOT NULL | claude/bedrock/openai |
| `model` | TEXT | Model override (nullable) |
| `interval_minutes` | INTEGER NOT NULL | Minutes between runs |
| `max_iterations` | INTEGER NOT NULL | Tasks per invocation |
| `installed_at` | TEXT NOT NULL | ISO timestamp |
| `last_run_at` | TEXT | Updated after each scheduled run |
| `last_status` | TEXT | success/failed |

## CLI Changes

### `atomics run` — new flag

`--trigger` (choices: manual, scheduled, test; default: manual). Generated
scheduler commands include `--trigger scheduled`. Engine passes tier, provider,
model, and trigger through to `repository.create_run()`.

### `atomics schedule --install` / `--uninstall`

Install writes a row to `schedules`. Uninstall deletes it. Generated commands
now include `--trigger scheduled`.

### `atomics schedule status` — new subcommand

Reads `schedules` table, then for each entry:
- Checks OS state (launchctl list, systemctl --user is-active, crontab -l grep)
- Computes next expected run from last_run_at + interval_minutes
- Renders a Rich table: format, tier, provider, interval, installed, last run,
  last status, OS health (alive/missing), next run

### `atomics compare` — new command

Groups `task_results` by provider (default) or `--by model`. Rich table with:
avg cost/task, avg latency, avg tokens, success rate, total tasks.

Filters: `--since-hours`, `--tier`, `--category`. Sorts by cost ascending.

### `atomics report` — enrichment

New "Runs by Provider" summary table showing per-provider aggregates.

## Commit Plan

1. Schema v2 + run metadata (schema, engine, CLI `--trigger`, repository, tests)
2. `schedule status` (CLI command, OS health checks, install/uninstall registry, tests)
3. `compare` + report enrichment (comparison command, report section, tests)

## Testing

- Schema: v2 drop/recreate, new columns populated on create_run/complete_run
- Schedule status: mock OS commands, verify alive/missing detection
- Compare: seed multi-provider data, verify grouping/filtering/sorting
- Coverage target: maintain 86%+
