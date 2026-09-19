# API/MCP submit_battery

Date: 2026-09-18. Slice: roadmap Next. Not a retag.

## Intent

`atomics battery` is CLI-only. An agent that can `submit_eval` and
`submit_sweep` cannot run a named battery without hand-building the
steps. Add `POST /batteries` and MCP `submit_battery` so one request
runs a battery as one job.

## Shape

- Route: `POST /batteries` → 202 `JobResponse`, kind `battery`.
- MCP: `submit_battery(name, provider, model, budget_usd, ...)`.
- Body (`BatteryRequest`, `extra="forbid"`):
  - `name` — one of the five; unknown → 400.
  - `provider`, `model`, `budget_usd` (required, like sweep).
  - Optional: `judge_model`, `judge_host`, `host`, `thinking`,
    `effort`, `reasoning_mode`, `runs` (1–3), `profile`.
- No `archreview` in v1 (needs a repo pack; stays CLI).

## Execution

Reuse the sweep runner shape. Expand `get_battery(name)` into its
visible steps; run them in order in one job under one `BudgetMeter`.
Each step is the suite the CLI would call. `progress.total` is the
step count; `in_flight` is `{step, suite}`; `result.steps` grows one
entry per finished step with that suite's `result.fixtures` and a
per-step `ok`. Job fails when a step fails (same as CLI without
`--keep-going`); no keep-going flag in v1.

## Constraints

- Budget required (paid judge possible). `runs` 1–3.
- `thinking` / `effort` / `reasoning_mode` forward onto every step
  (the CLI bug fixed this week). Battery default stays `--no-thinking`
  when `thinking` is unset.
- `profile` enables qa on non-Ollama providers, same as CLI.

## Tests

- `test_api_*`: unknown name 400; budget required; steps expand in
  order; thinking/effort reach the step runner.
- `test_mcp_*`: tool exists, forwards fields, returns job id.
- No live provider.

## Out of scope

`archreview` step, `--keep-going`, per-step model overrides, retag.
