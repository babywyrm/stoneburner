# Dashboard Trends and Live Jobs Implementation Plan

> Executed inline 2026-08-16.

**Goal:** Hourly spend on the dashboard, and a live view of in-memory API jobs.

**Architecture:** Widen `get_token_usage_by_hour`, expose `/reports/trends` and
`GET /jobs` (no `result`). Dashboard cards poll on the existing 10s refresh.
