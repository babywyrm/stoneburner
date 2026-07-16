# CLI Foundation and Suite Convergence Design

**Date:** 2026-07-15  
**Status:** Approved design, pending written-spec review

## Goal

Make Stoneburner's command-line foundation easier to maintain and make
`refusal` and `codereview` behave like first-class evaluation suites.

This milestone is behavior-preserving for existing commands except where current
behavior incorrectly presents provider or judge failures as valid scores. Those
cases become explicitly indeterminate and produce honest run-integrity status.

## Scope

This milestone will:

- establish a small `atomics.commands` package;
- extract the `refusal` and `codereview` Click commands from `atomics/cli.py`;
- centralize reusable CLI result writing, progress, model attribution, repository
  lifetime, and integrity-exit behavior;
- converge both suites on the shared evaluation outcome contracts;
- add lossless per-fixture persistence and parent-run finalization;
- preserve existing command names and accepted flags;
- add regression tests before and during extraction.

It will not regroup root commands, redesign reports, split every command, split
the entire repository, or change unrelated suite behavior.

## Architecture

### Root CLI

`atomics/cli.py` remains the public entry point and owns:

- the root Click group;
- global logging, verbosity, and progress settings;
- command registration;
- compatibility re-exports needed by existing tests or integrations.

It will import and register the extracted `refusal` and `codereview` commands.
No wrapper subprocesses or circular imports are introduced.

### Command package

Create:

- `atomics/commands/__init__.py`
- `atomics/commands/common.py`
- `atomics/commands/refusal.py`
- `atomics/commands/codereview.py`

`common.py` contains only command-layer concerns:

- `FixtureProgress`;
- safe JSON output using `Summary.to_dict()`;
- effective model resolution;
- the existing `_make_provider()` implementation;
- repository ownership/finalization helpers;
- shared rendering primitives;
- shared integrity exit policy.

Provider construction continues through one `_make_provider()` implementation.
It moves unchanged to `commands/common.py`; `atomics/cli.py` re-exports it for
compatibility. This avoids a command-to-root circular import and does not add a
second provider switch.

### Evaluation runners

`RefusalSummary` and `CodeReviewSummary` keep their existing `results` constructor
field so Python callers do not break. A read-only `fixture_results` property
provides the documented cross-suite interface.

Their serialized output includes canonical `fixture_results` and a deprecated
`results` alias with identical data for one compatibility cycle. This prevents
existing JSON consumers from breaking while establishing the documented target
shape.

Each fixture result retains:

- the model-under-test provider outcome;
- the judge outcome;
- the model response and judge-call evidence needed to audit the verdict;
- latency, token, and cost data when available;
- suite-specific classification or verdict;
- sanitized failure details;
- whether the fixture contributed to aggregate metrics.

Judge parse failures, judge provider failures, empty responses, timeouts, and
model-generation failures remain distinct. None are converted into successful
scores or optimistic default verdicts.

Both summaries expose `RunIntegrity` with complete, partial, and
infrastructure-invalid states. Aggregate rates use scored fixtures only and
always report their coverage denominator.

## Persistence

Add one generic `evaluation_results` table rather than one table per new suite.
It stores:

- `run_id`, `suite`, and `fixture_id` as the logical identity;
- fixture status and optional score;
- generation and judge statuses;
- latency and token totals;
- sanitized error class and message;
- a lossless `result_json` payload.

The logical identity is unique so retries upsert instead of creating duplicate
fixture rows.

The repository gains a typed `EvaluationResultRecord` boundary plus:

- `save_evaluation_result()`;
- `get_evaluation_results()`;
- `complete_evaluation_run()`.

Storage remains independent of eval modules. Suite runners or command adapters
construct the storage record; the repository does not import runner classes.

Each command creates its parent run before fixture execution, saves completed
fixture callbacks incrementally, and finalizes the parent in `finally`.
Repository connections close on every exit path, including JSON-write or
callback failures.

## CLI behavior

Both commands consistently support:

- existing provider, model, judge, host, and thinking options;
- `--save/--no-save`, defaulting to save;
- `--json-out PATH`;
- group-level progress and verbosity;
- `--allow-partial`.

Default terminal output contains:

- selected provider/model and judge attribution;
- fixture progress;
- suite-specific aggregate metrics;
- scored-fixture coverage;
- generation and judge failure counts;
- run-integrity status;
- saved run ID when persistence is enabled.

JSON is written and database rows are finalized before integrity policy affects
the process exit code. Partial or infrastructure-invalid runs exit nonzero unless
`--allow-partial` is present.

No command is renamed, and current invocations remain valid.

## Error handling and security

- All persisted and displayed exception text passes through `sanitize_error()`.
- Raw model responses and judge evidence are stored in JSON and SQLite but are
  not printed by default. Documentation identifies these artifacts as sensitive.
- Provider and judge failures are never silently changed into score zero,
  `clean`, or another valid verdict.
- JSON-write failures produce a Click error after database finalization.
- Persistence failures finalize what can be finalized, close the connection, and
  produce a nonzero exit.
- No fixture, endpoint, or secret value is added to logs solely by this refactor.

## Regression strategy

Tests are added before moving command code and cover the existing public command
surface. The implementation must pass:

1. command help and invocation tests for both extracted commands;
2. existing flag compatibility and model-attribution tests;
3. success, provider failure, judge failure, parse failure, empty output, mixed
   outcome, and all-failed runner tests;
4. progress enabled/disabled tests;
5. JSON schema compatibility tests for `fixture_results` and `results`;
6. save/no-save, incremental callback, upsert, parent finalization, and storage
   round-trip tests;
7. exit-code tests for complete, partial, infrastructure-invalid, and
   `--allow-partial` runs;
8. schema migration and backup-safety tests;
9. full `pytest`, `mypy atomics`, and Ruff checks for all changed files;
10. existing CLI, adversarial-integrity, storage, provider, and reporting tests.

Tests must not contact live providers. Provider and judge behavior is represented
with typed fakes and realistic `ProviderResponse` objects.

## Commit structure

Keep each commit independently understandable and tested:

1. add CLI and runner regression characterization;
2. add shared command-layer primitives;
3. converge refusal outcomes and extract its command;
4. converge code-review outcomes and extract its command;
5. add generic evaluation persistence and schema migration;
6. update architecture, CLI documentation, and changelog.

If a dependency requires a different order, preserve the same conceptual
separation and avoid mixing unrelated cleanup into these commits.

## Success criteria

- Existing command lines continue to work.
- `atomics/cli.py` becomes smaller and no longer contains either extracted
  command body.
- No duplicate provider factory, JSON writer, progress implementation, or
  integrity-exit implementation is introduced.
- Refusal and code-review failures cannot improve or silently depress reported
  model quality; they are reported as indeterminate with explicit coverage.
- Both suites persist lossless fixture records and finalize accurate parent runs.
- Full regression, typing, and changed-file lint checks pass.
- Documentation accurately describes the resulting architecture and behavior.

## Deferred work

- Extracting the remaining commands from `atomics/cli.py`.
- Root command grouping or aliases.
- Provider-outcome normalization across every backend.
- Splitting `MetricsRepository` into persistence components.
- Unified cross-suite reporting, comparison, and export.
- Wiring `inference.env` into CLI defaults and `doctor`.

