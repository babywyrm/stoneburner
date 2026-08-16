# Dashboard Run Drill-In Implementation Plan

> Executed inline 2026-08-16.

**Goal:** Click a recent run and see sanitized fixture scores.

**Architecture:** `get_run_detail` on the repository, `GET /api/v1/runs/{id}`, dashboard hash panel.
