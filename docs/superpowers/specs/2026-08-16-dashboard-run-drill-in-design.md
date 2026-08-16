# Dashboard Run Drill-In

**Date:** 2026-08-16
**Status:** Approved 2026-08-16

## Goal

An operator on `/dashboard` can open one persisted run and see its fixtures.
Today the recent-runs card is a truncated id and a status. The JSON export is
the only way to see what happened.

## Scope

1. `GET /api/v1/runs/{run_id}` — parent row plus sanitized fixture rows.
   Auth. `404` if missing.
2. Dashboard: the run id is a button. Click (or `#run=<id>`) opens a detail
   panel. Back clears the hash.
3. Docs: API_SERVER dashboard section.

It will not:

- add historical trend charts;
- poll in-memory `/jobs/{id}` as a live progress view;
- return `prompt`, `response`, `result_json`, `attempts_json`, or judge
  rationale (eval evidence is sensitive);
- add an MCP tool in this milestone;
- change CSP, `textContent`-only rendering, or key handling.

## Architecture

`MetricsRepository.get_run_detail(run_id)` assembles:

- `run`: the `runs` row
- `fixtures`: evaluation_results, adversarial_results, and task_results,
  normalized to `{id, kind, suite, score, label, status, latency_ms, cost_usd}`

The dashboard stays a single HTML file. No `innerHTML`.
