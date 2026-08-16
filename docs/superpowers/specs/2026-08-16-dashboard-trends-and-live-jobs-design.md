# Dashboard Trends and Live Job Progress

**Date:** 2026-08-16
**Status:** Approved 2026-08-16 (continue)

## Goal

An operator on `/dashboard` can see spend over time and watch an in-memory
API job move from pending to completed. Today compare is a snapshot, and
`GET /api/v1/jobs/{id}` is only useful if you already have the id.

## Testing honesty

What we already cover well: JobManager lifecycle, auth on existing routes,
run-detail sanitization, static `.innerHTML` absence.

What we do not: the dashboard script is never executed; `get_token_usage_by_hour`
has no aggregation assertion and only reads `task_results`, so a refusal-only
database looks empty. This milestone tests the Python and HTTP layers until
those holes are closed. No browser runner ships in this repo; we will not
pretend the JS ran.

## Scope

1. `GET /api/v1/reports/trends?hours=24` — hourly token/cost/count series.
   Auth. `hours` is 1–168. Uses `request.app.state.settings.db_path`.
2. Widen `get_token_usage_by_hour` so evaluation and adversarial fixtures
   count, and the window compares ISO timestamps reliably.
3. `GET /api/v1/jobs` — in-memory jobs, newest first, **no `result`**.
   `GET /api/v1/jobs/{id}` is unchanged (clients still poll for the payload).
4. Dashboard: Trends card (bars, `textContent` only). API jobs card; click
   or `#job=<id>` opens a status panel. The 10s refresh reloads an open job.
   The panel never writes `result` into the DOM.

It will not:

- add Chart.js or SVG charts;
- change MCP;
- poll faster than the existing 10s refresh;
- dump `prompt`, `response`, `result_json`, or job `result`;
- add `.innerHTML`;
- confuse these jobs with `/distributed/runs`.

## Architecture

`get_token_usage_by_hour(hours)` UNIONs `task_results` (`started_at`),
`evaluation_results` (`timestamp`), and `adversarial_results` (`timestamp`).
Window: `col >= strftime('%Y-%m-%dT%H:%M:%S', 'now', '-N hours')` so an
ISO `T` timestamp is not compared to SQLite's space-separated `datetime()`.

`JobManager.list_jobs()` returns current jobs, newest `created_at` first.
The list route maps each job through a summary that omits `result`.
