# Fleet mode (distributed Phase 2) — design

**STATUS: COMPLETED** — shipped unreleased, after 0.12.0.

Date: 2026-07-25
Follows: `2026-07-21-distributed-runs-design.md` (Phase 1, split mode)

## What changed during implementation

Two things this design got wrong, recorded because the reasoning is the point of
keeping it:

1. **The stated risk was already covered.** The design justified pinning by
   claiming a dead host's tasks would otherwise be executed by another host. Once
   `target_worker_id` was in the claim query that could not happen. The real defect
   was narrower and worse: the job never terminated at all, because pinned work
   returned to `pending` where only an absent worker could claim it.
2. **Failure handling needed liveness, which did not exist.** The design keyed
   failure on the worker being offline. Nothing in the coordinator ever set that
   status, so the branch was unreachable — a killed worker stayed `online` forever.
   A host that died before claiming anything was worse still: its assignments sat
   in `pending`, which the stale-assignment scan never examines. Implementation
   added a 120-second heartbeat sweep and bounded retries via the previously unused
   `max_retries`, both beyond the scope agreed here.

## Problem

Phase 1 shipped split mode: the coordinator divides a run into task assignments
and any worker claims the next one. Two things were left dangling.

`JobMode` already declares `SPLIT`, `FULL`, and `FLEET`, but only `SPLIT` is
implemented and no document ever defined what the other two mean. `Worker.labels`
is accepted at registration and persisted, then never consulted by anything. As
of the previous change, `--label` and `worker_selector` are rejected outright
rather than silently ignored, with the error text pointing users at "fleet mode"
— a mode that did not exist even in specified form.

There is also no results aggregation. `distributed_jobs.summary_json` exists as a
column and nothing has ever written to it, so a finished distributed job reports
only a status, never numbers.

## Goals

- Define and implement `--mode fleet` as a broadcast: every matching worker runs
  the full task set, so the same suite can be compared across hosts.
- Make `--label` meaningful for fleet runs.
- Produce a per-worker rollup so a completed fleet run answers "which host was
  faster, cheaper, more reliable".
- Close the three unauthenticated coordinator endpoints.
- Add no regressions to split mode, and do not destroy existing local data.

## Non-goals

- `FULL` mode (one worker runs an entire run). It stays unimplemented and keeps
  its current explicit rejection. No reason to build two modes at once.
- Integrating fleet results into the existing `compare` / leaderboard machinery.
  The rollup is persisted in a shape that allows this later.
- Worker autoscaling, scheduling policy, or capability-based matching beyond
  exact label equality.

## Semantics

```
atomics distributed run --mode fleet --label gpu=4090 -t baseline -n 20
```

- The coordinator snapshots **online** workers matching the selector at submit
  time, then creates one assignment per (worker × task). 3 workers × 20 tasks
  produces 60 assignments.
- Matching is exact key/value equality on `Worker.labels`. Every selector pair
  must match for a worker to be eligible. An empty selector matches every online
  worker.
- A selector matching zero online workers is rejected at submit with `400`
  rather than creating a job that can never progress. This mirrors the decision
  to reject rather than silently no-op.
- `worker_selector` remains rejected for `split` mode: split assigns each task to
  the next available worker, so a selector still cannot be honored there.
- Snapshot-at-submit is deliberate. A worker registering mid-run does not join
  the job, because the comparison is between the hosts that were present when the
  run was defined.
- **Every worker must receive the identical task set.** `_build_task_specs` calls
  `get_weighted_task(tier)`, which samples randomly per call, so building specs
  per worker would give each host different prompts and make the comparison
  meaningless while still looking correct. The task set is built once and then
  replicated across workers, and this gets its own test.
- `create_split_job` hardcodes `mode=JobMode.SPLIT`. Fleet needs a sibling that
  records the requested mode and sets `target_worker_id`; `parent_run_id` is
  derived the same way split derives it, so one run id spans every host in the
  job and the rollup can group by it.

## Data model

`distributed_assignments` gains one column:

```sql
target_worker_id TEXT REFERENCES workers(worker_id)
```

Nullable, and that is what keeps the change cheap:

- `NULL` — any worker may claim. Exactly today's split-mode behavior.
- set — pinned to that worker, and only that worker.

### Migration

`init_db` currently responds to any `SCHEMA_VERSION` bump by snapshotting the
database to a timestamped `.bak` and dropping every table (the documented
pre-1.0 fresh-start policy). For a single nullable column that would reset local
run history, schedules, and the evaluation ledger — an avoidable regression.

Instead, `init_db` gains additive column reconciliation: after applying
`SCHEMA_SQL`, compare each table's `PRAGMA table_info` against the expected
columns and `ALTER TABLE ... ADD COLUMN` whatever is missing. `SCHEMA_VERSION`
stays at 20 for this change, no data is touched, and every future nullable-column
addition gets the same non-destructive path.

Constraints: additions of nullable columns only. A constant `DEFAULT` is carried
over, since SQLite applies one to existing rows without trouble. Anything
requiring a type change, a `NOT NULL`, a primary key, or a drop still goes through
a version bump and the existing backup-then-reset path, and is logged as such
rather than silently skipped.

## Claiming

`Coordinator.claim_assignment` changes one clause:

```sql
WHERE status = 'pending' AND (target_worker_id IS NULL OR target_worker_id = ?)
```

Both modes then share one claim path. Split-mode rows are all `NULL`, so their
behavior is unchanged.

## Failure handling

`_requeue_stale_assignments` today clears `worker_id` and returns the row to the
pool for anyone to claim. For a pinned row that is wrong in a way that is worse
than a crash: if `host-a` dies midway, its remaining tasks get executed by
`host-b`, the job still reports `completed`, and the per-host comparison silently
becomes a blend of two machines with nothing in the output saying so.

New behavior for pinned assignments:

- Timed out but the target worker is still online — return to `pending` with
  `target_worker_id` preserved, so only that worker can pick it up again. Retry
  accounting is unchanged.
- Target worker offline or unknown — mark the assignment `FAILED`. Its tasks are
  never reassigned to another host.

Unpinned (split) assignments keep their existing requeue behavior exactly.

The job then resolves through the existing `_update_job_status`, which already
treats `completed + failed == total` as terminal and reports `PARTIAL` when any
assignment failed. A fleet run where one host dropped out is therefore `PARTIAL`,
with that host's failures visible in the rollup.

## Aggregation and output

When a job reaches a terminal status the coordinator writes
`distributed_jobs.summary_json`. Workers already serialize a full `TaskResult`
per assignment into `result_json`, so every number below is a rollup of data
already persisted, not new measurement.

Per worker: `worker_id`, `labels`, assignments completed and failed, total input
and output tokens, mean and p95 latency, mean tokens/sec, and estimated cost.
Plus job-level totals and the job's mode and selector.

`atomics distributed status <job_id>` prints a per-host table for fleet jobs and
accepts `--json-out` to write the artifact, consistent with every other suite
command.

## Auth

Three coordinator endpoints currently have none, while the three worker
lifecycle endpoints require a worker key:

```
OPEN   POST  /workers/register
OPEN   POST  /distributed/runs
OPEN   GET   /distributed/runs/{job_id}
```

Anyone who can reach the port can submit jobs that consume GPU time and cloud
spend, read job status, and register phantom workers. Fleet mode makes this worse
by adding label-targeted submission, so it is closed as part of this work.

`atomics/api/routes.py` already defines a `require_auth` client-auth dependency
and applies it to all five of its own endpoints. Rather than have one routes
module import another, `get_auth` / `require_auth` move to a new
`atomics/api/dependencies.py` that both routers import. Then:

- `POST /distributed/runs`, `GET /distributed/runs/{job_id}` — client auth.
  These are submitter-facing, like the rest of the API.
- `POST /workers/register` — worker auth. A worker has no ID yet at
  registration, so the worker key is the only credential it can present.

`NoAuth` continues to satisfy both, so the `--no-auth` development path is
unchanged. The CLI already sends `X-API-Key`, so no client changes are needed.

## Testing

Job creation
- N workers × M tasks assignments, each with `target_worker_id` set.
- Every worker receives an identical task set, asserted by comparing the prompts
  per worker rather than only counting assignments.
- Label matching selects exactly the intended subset; multi-pair selectors
  require every pair; an empty selector matches all online workers.
- Offline workers are excluded from the snapshot.
- Zero matches → `400`, and no job row is created.
- `worker_selector` on a split run is still rejected.

Claiming
- A pinned assignment is not claimable by a different worker.
- A worker claims its own pinned assignment.
- Split-mode (`NULL`) assignments remain claimable by any worker.

Failure handling
- Pinned assignment whose worker went offline becomes `FAILED`, and the job
  becomes `PARTIAL`.
- **A dead host's tasks are never handed to another worker.** This is the
  regression the design exists to prevent, so it gets an explicit test rather
  than being implied by the two above.
- Pinned assignment that timed out while its worker is still online returns to
  `pending` with the pin intact.
- Unpinned stale assignments still requeue to anyone.

Migration
- Opening a database that predates the column adds it and preserves every
  existing row.
- Reconciliation is idempotent across repeated `init_db` calls.
- A database already carrying the column is untouched.

Aggregation
- `summary_json` shape and per-worker numbers, including a host with failures.
- Percentile and mean math against known inputs.

Auth
- Each of the three endpoints returns `401` without a key and succeeds with one.
- The `NoAuth` backend permits all three.
- Worker registration rejects a client key that is not a valid worker key.

End to end
- Extend `tests/test_distributed_e2e.py` with a two-worker fleet run: register
  both, submit, drain both workers, assert each executed the full task set and
  the rollup reports both hosts separately.

## Regression safety

Split mode is the existing product surface and must not move. Concretely:
`NULL` target keeps its claim behavior and its requeue behavior; the Phase 1
tests are expected to pass untouched, and any that do not indicates a real
regression rather than a test to relax.

Two intentional contract changes will require updating existing tests, and each
updated test must keep proving something real:

1. The three endpoints above now require a key, so tests that called them
   anonymously and expected success will expect `401`, with authenticated
   equivalents added.
2. `worker_selector` moves from "always rejected" to "rejected for split,
   honored for fleet".

Baseline before starting: 1770 passed, 24 skipped, 87.22% coverage, mypy clean
across 152 files.
