# Per-Pass Live Lines for `--runs` Suites

**Date:** 2026-08-16
**Status:** Approved 2026-08-16

## Goal

`redblue` and `adversarial` already score every pass when `--runs` is greater
than one. The CLI only prints the mean. A 40% / 90% / 90% fixture looks like a
clean 73% until the JSON is opened. Port the `toolcall` `on_run_done` hook so
each pass is visible while it happens.

Refusal and codereview have no `--runs` loop. They are out of scope.

## Why now

Toolcall hid 1/3 leaks behind a modal row until `on_run_done` landed. The same
class of lie exists on the two suites that already have a pass loop. Overnight
`--runs 3` work is how we publish capability and resilience; operators should
see a bad pass without opening the export.

## Scope

This milestone will:

1. Add `on_run_done(index, fixture, run_number, runs, record)` to
   `run_redblue` and `run_adversarial`, matching `run_toolcall_suite`.
2. Fire the hook after **every** pass, including generate failures and
   unscoreable attempts. A silent failed pass is the bug.
3. Await the callback when it returns an awaitable (same as toolcall).
4. Print `id run 2/3 — …` from the CLI only when `--runs > 1`. The
   fixture-done line stays the mean / aggregate.
5. Pass the same hook on `adversarial --compare` so model B is not silent.
6. Document the lines in `SECURITY_SUITES` and `CLI_REFERENCE`.

It will not:

- add `--runs` to refusal or codereview;
- change score formulas, integrity accounting, or persistence (still one
  aggregated fixture row);
- change the fixture-done line;
- run another judged 27B suite as product work.

## Architecture

No new subsystem. The record is a small dict so the CLI does not import
`TaskResult` / `AttemptResult`:

Redblue:

```
{"score": float | None, "status": "scored" | "failed" | "parse_failed"}
```

Adversarial:

```
{"score": float | None, "label": str | None,
 "status": "scored" | "failed" | "parse_failed" | "unscored"}
```

CLI formats only:

- redblue scored → `40%` (green ≥80, yellow ≥60, else red)
- adversarial scored → `complied  0.12` (green / yellow / red by label)
- anything else → `ERROR` / `PARSE` / `UNSCORED`

## Tests

- Runner: hook fires once per pass; `run_number` is 0-based; `runs` is N.
- Runner: generate failure still fires, `status=failed`.
- Runner: async callback is awaited.
- CLI: `--runs 3` prints `run 1/3` … `run 3/3`; `--runs 1` does not.
