# Fleet Mode Implementation Plan

**STATUS: COMPLETED** — all seven tasks landed. Final state: 1832 passed,
24 skipped, 87.41% coverage, mypy clean across 154 files.

Task 5 grew during implementation: it needed worker liveness detection and
bounded retries, neither of which the spec anticipated. See the spec's
"What changed during implementation".

Spec: `docs/superpowers/archive/specs/2026-07-25-fleet-mode-design.md`

**Goal:** Implement `--mode fleet` as a broadcast — every label-matched worker
runs the identical task set — with a per-host rollup, and close the three
unauthenticated coordinator endpoints on the way.

**Architecture:** A nullable `target_worker_id` on `distributed_assignments`
turns one FIFO claim path into a pinned one without changing split mode
(`NULL` = any worker). Pinned assignments are never reassigned across hosts, so a
dead worker fails its own slice and the job reports `PARTIAL`. The column reaches
existing databases through new additive reconciliation in `init_db` rather than a
destructive version bump.

**Tech stack:** Python 3.11+, SQLite, FastAPI, Click, pytest, `uv`.

**Regression baseline:** 1770 passed, 24 skipped, 87.22% coverage, mypy clean
across 152 files. Split-mode tests must pass untouched throughout.

**Verification per task:** `uv run pytest -q --override-ini="addopts="` for the
touched files, then `uv run mypy atomics/` and
`uv run ruff check atomics/ tests/ --ignore E501`. Full suite with
`--cov-fail-under=85` before each commit.

---

### Task 1: Additive column reconciliation in `init_db`

Independent of fleet mode and ships value alone: every future nullable column
stops requiring a data wipe.

**Files:**
- Modify: `atomics/storage/schema.py` (`init_db`, new helper)
- Test: `tests/test_storage_schema.py`

- [x] **Step 1: Write failing tests**

`test_init_db_adds_a_missing_nullable_column_in_place` — create a DB, drop a
nullable column by rebuilding the table without it, insert a row, reopen with
`init_db`, assert the column exists and the row survived.
`test_column_reconciliation_is_idempotent` — call `init_db` three times, assert
no error and one column.
`test_reconciliation_leaves_a_current_database_untouched` — assert no `ALTER` is
issued when the schema already matches.

- [x] **Step 2: Run and confirm they fail**

- [x] **Step 3: Implement**

Read expected columns by materializing `SCHEMA_SQL` in a scratch in-memory
database and asking `PRAGMA table_info`, rather than parsing the DDL by hand — a
hand-maintained second list would drift from the first. Compare against the real
database's `table_info` and issue `ALTER TABLE <table> ADD COLUMN` for each
missing column, carrying over a constant `DEFAULT`. Guard rail: skip and log
`NOT NULL` or `PRIMARY KEY` columns, which SQLite cannot add to a populated
table; those still require a `SCHEMA_VERSION` bump. Runs after
`_execute_sql_statements(conn, SCHEMA_SQL)`, inside the existing transaction.
`SCHEMA_VERSION` stays 20.

- [x] **Step 4: Tests pass; full suite green**
- [x] **Step 5: Commit** — `feat(storage): add missing nullable columns in place`

---

### Task 2: Authenticate the three open coordinator endpoints

**Files:**
- Create: `atomics/api/dependencies.py`
- Modify: `atomics/api/routes.py` (remove `get_auth`/`require_auth`, import them)
- Modify: `atomics/distributed/routes.py` (apply both dependencies)
- Test: `tests/test_distributed_api.py`, `tests/test_api_auth.py` (or nearest)

- [x] **Step 1: Write failing tests**

For each of `POST /api/v1/distributed/runs`,
`GET /api/v1/distributed/runs/{job_id}`, `POST /api/v1/workers/register`:
`..._requires_a_key` (401 without) and `..._succeeds_with_a_key` (200/202 with).
Plus `test_no_auth_backend_permits_the_distributed_endpoints` for the `--no-auth`
dev path, and
`test_worker_registration_rejects_a_non_worker_key` proving registration checks
the *worker* backend, not the client one.

- [x] **Step 2: Run and confirm they fail** (currently all pass anonymously)

- [x] **Step 3: Implement**

Move `get_auth` and `require_auth` verbatim from `atomics/api/routes.py` into
`atomics/api/dependencies.py`, alongside the existing `get_worker_auth` /
`require_worker_auth` from `atomics/distributed/routes.py`, so one module owns
the dependencies and neither routes module imports the other. Both routers import
from it. Then add `_: None = Depends(require_auth)` to `start_distributed_run` and
`get_job`, and `_: None = Depends(require_worker_auth)` to `register_worker`.

- [x] **Step 4: Update the existing anonymous tests**

The Phase 1 tests call these endpoints with no key. Each becomes authenticated
rather than deleted — the assertion it was making about job creation or lookup is
still worth making. Confirm the count of updated tests matches the count of
call sites so none is quietly dropped.

- [x] **Step 5: Full suite green; mypy; ruff**
- [x] **Step 6: Commit** — `fix(distributed): require auth on the coordinator endpoints`

---

### Task 3: `target_worker_id` and pinned claiming

**Files:**
- Modify: `atomics/storage/schema.py` (`SCHEMA_SQL` — add the column)
- Modify: `atomics/distributed/models.py` (`TaskAssignment.target_worker_id`)
- Modify: `atomics/distributed/coordinator.py` (`claim_assignment`,
  `_row_to_assignment`, `create_split_job` insert)
- Test: `tests/test_distributed_coordinator.py`

- [x] **Step 1: Write failing tests**

`test_a_pinned_assignment_is_not_claimable_by_another_worker`,
`test_a_worker_claims_its_own_pinned_assignment`,
`test_unpinned_assignments_remain_claimable_by_any_worker` (the split-mode
regression guard).

- [x] **Step 2: Run and confirm they fail**

- [x] **Step 3: Implement**

Add `target_worker_id TEXT REFERENCES workers(worker_id)` to
`distributed_assignments` in `SCHEMA_SQL`; Task 1's reconciliation delivers it to
existing databases. Add the field to `TaskAssignment` (default `None`) and to
`_row_to_assignment`. Change the `claim_assignment` inner select to
`WHERE status = ? AND (target_worker_id IS NULL OR target_worker_id = ?)` and add
the `worker_id` parameter. Extend the `RETURNING` list and the row unpacking.

- [x] **Step 4: Tests pass; full suite green** (split behavior must not move)
- [x] **Step 5: Commit** — `feat(distributed): allow assignments pinned to a worker`

---

### Task 4: Fleet job creation

**Files:**
- Modify: `atomics/distributed/coordinator.py` (`create_fleet_job`,
  `matching_workers`)
- Modify: `atomics/distributed/routes.py` (accept fleet, reject zero matches)
- Modify: `atomics/commands/distributed.py` (`--mode fleet`, un-reject `--label`)
- Test: `tests/test_distributed_coordinator.py`, `tests/test_distributed_api.py`,
  `tests/test_distributed_cli.py`

- [x] **Step 1: Write failing tests**

Coordinator: `test_fleet_job_creates_one_assignment_per_worker_per_task`,
`test_every_worker_receives_the_identical_task_set` (compare prompt lists per
worker — not just counts; guards the `get_weighted_task` resampling trap),
`test_selector_matches_only_workers_with_every_label_pair`,
`test_empty_selector_matches_all_online_workers`,
`test_offline_workers_are_excluded_from_the_snapshot`.
API: `test_fleet_run_with_no_matching_workers_is_rejected` (400, and assert no
job row was created), `test_worker_selector_is_still_rejected_for_split_mode`.
CLI: `test_fleet_mode_sends_the_label_selector`,
`test_label_is_still_rejected_for_split_mode`.

- [x] **Step 2: Run and confirm they fail**

- [x] **Step 3: Implement**

`matching_workers(selector)` returns online workers whose `labels` superset the
selector. `create_fleet_job(request, task_specs, workers)` records
`mode=JobMode.FLEET` and inserts `len(workers) * len(task_specs)` rows, each
pinned. Build task specs **once** and replicate — do not call
`_build_task_specs` per worker. In `routes.py`, allow `JobMode.FLEET`, keep
rejecting `FULL`, keep rejecting `worker_selector` for `SPLIT`, and 400 when
`matching_workers` is empty with a message naming the selector. In the CLI, add
`fleet` to the `--mode` choices and gate the existing `--label` rejection on
`mode == "split"`.

- [x] **Step 4: Tests pass; full suite green**
- [x] **Step 5: Commit** — `feat(distributed): add fleet mode broadcast runs`

---

### Task 5: Pinned-assignment failure handling

The riskiest behavior change. Do it after claiming works, so the tests can
observe real pinned rows.

**Files:**
- Modify: `atomics/distributed/coordinator.py` (`_requeue_stale_assignments`)
- Test: `tests/test_distributed_coordinator.py`

- [x] **Step 1: Write failing tests**

`test_pinned_assignment_fails_when_its_worker_goes_offline`,
`test_a_dead_hosts_tasks_are_never_given_to_another_worker` (the whole point:
take a worker offline, then have a second worker poll repeatedly and assert it
receives nothing from that slice),
`test_pinned_assignment_that_timed_out_keeps_its_pin_while_the_worker_lives`,
`test_unpinned_stale_assignments_still_requeue_to_anyone` (split guard),
`test_job_reports_partial_when_one_host_drops_out`.

- [x] **Step 2: Run and confirm they fail**

- [x] **Step 3: Implement**

In the stale loop, select `target_worker_id` alongside the existing columns.
Branch: unpinned → today's behavior (`pending`, `worker_id = NULL`). Pinned and
worker offline/unknown → `status = 'failed'`, `completed_at = now`, then let
`_update_job_status` run. Pinned and worker online but timed out → `pending`,
`worker_id = NULL`, `target_worker_id` preserved.

Note `_requeue_stale_assignments` is called from `claim_assignment`, so failing a
row must call `_update_job_status` for that job or a fleet job whose last
assignment fails will never reach a terminal status.

- [x] **Step 4: Tests pass; full suite green**
- [x] **Step 5: Commit** — `fix(distributed): never reassign a pinned assignment across hosts`

---

### Task 6: Per-worker aggregation and `distributed status`

**Files:**
- Create: `atomics/distributed/rollup.py` (pure aggregation, no DB access)
- Modify: `atomics/distributed/coordinator.py` (write `summary_json` on terminal)
- Modify: `atomics/commands/distributed.py` (`status` table, `--json-out`)
- Modify: `atomics/distributed/routes.py` if the job response needs the summary
- Test: `tests/test_distributed_rollup.py`, `tests/test_distributed_cli.py`

- [x] **Step 1: Write failing tests**

Rollup as a pure function over `(assignment, result_json)` pairs:
`test_rollup_groups_results_by_worker`,
`test_rollup_reports_mean_and_p95_latency_from_known_inputs`,
`test_rollup_counts_failures_per_worker`,
`test_rollup_totals_tokens_and_cost`,
`test_rollup_handles_a_worker_with_only_failures` (no division by zero).
Coordinator: `test_summary_json_is_written_when_the_job_completes`.
CLI: `test_status_prints_a_row_per_host`, `test_status_json_out_writes_the_rollup`.

Keeping the math in a pure module is what makes it testable without a coordinator
or a worker, and mirrors how the eval suites separate scoring from running.

- [x] **Step 2: Run and confirm they fail**

- [x] **Step 3: Implement**

`rollup.py` builds per-worker records (`worker_id`, `labels`, `completed`,
`failed`, `input_tokens`, `output_tokens`, `mean_latency_ms`, `p95_latency_ms`,
`mean_tokens_per_second`, `estimated_cost_usd`) plus job totals, reading the
fields workers already serialize into `result_json`. `_update_job_status` writes
`json.dumps(...)` into `summary_json` when it sets a terminal status. `status`
renders a Rich table for fleet jobs, keeping the current output for split.

- [x] **Step 4: Tests pass; full suite green**
- [x] **Step 5: Commit** — `feat(distributed): roll up fleet results per worker`

---

### Task 7: End-to-end run and documentation

**Files:**
- Modify: `tests/test_distributed_e2e.py`
- Modify: `docs/CLI_REFERENCE.md`, `docs/API_SERVER.md`, `ROADMAP.md`,
  `CHANGELOG.md`
- Move: spec → `docs/superpowers/archive/specs/`, plan →
  `docs/superpowers/archive/plans/`, add rows to
  `docs/superpowers/archive/README.md`

- [x] **Step 1: Write the end-to-end test**

`test_end_to_end_fleet_run`: register two workers with distinct labels, submit a
fleet run with an authenticated client, drain both workers through the real HTTP
routes, assert each executed the full task set, the job is `completed`, and the
rollup reports both hosts separately.

- [x] **Step 2: Run; fix what it exposes**

- [x] **Step 3: Update docs**

`CLI_REFERENCE.md`: `--mode fleet`, `--label` now meaningful for fleet and still
rejected for split. `API_SERVER.md`: fleet accepted, `worker_selector` semantics
per mode, and that all distributed endpoints now require a key. `ROADMAP.md`:
move fleet out of Phase 2 into shipped, and fix the stale unchecked "Multi-turn
conversation eval fixtures" box, which contradicts the checked entry above it.
`CHANGELOG.md`: Unreleased entries for fleet mode, the auth fix, and the
non-destructive migration.

- [x] **Step 4: Full suite, mypy, ruff, coverage gate**
- [x] **Step 5: Secret scan** per `SECURITY.md` before pushing
- [x] **Step 6: Commit and push**

---

## Self-review against the spec

- Semantics → Task 4 (matching, identical task set, zero-match rejection,
  snapshot-at-submit via `matching_workers` at creation time).
- Data model → Task 3 (column, model field) and Task 1 (migration path).
- Claiming → Task 3.
- Failure handling → Task 5, including the explicit no-cross-host test.
- Aggregation and output → Task 6.
- Auth → Task 2.
- Testing → distributed across tasks; the spec's end-to-end item is Task 7.
- Regression safety → split-mode guards in Tasks 3 and 5, honest test updates in
  Task 2, non-destructive migration in Task 1.

No spec requirement is unassigned. `FULL` mode stays rejected, matching the
non-goal. Names used consistently across tasks: `target_worker_id`,
`matching_workers`, `create_fleet_job`, `rollup.py`.
