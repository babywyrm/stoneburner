# Judge Consensus and Agreement Design

**Date:** 2026-08-14
**Status:** Approved design, pending written-spec review

## Goal

Give the security suites the same two things `eval` already has a start on:
a live multi-judge consensus that improves the headline score, and a way to
measure how much a single judge is currently distorting that headline.

Default one-judge runs do not change. Panels are opt-in. Existing consensus
tests are the tripwire that the extract did not change behavior.

## Scope

This milestone will:

- extract a suite-neutral consensus helper that combines already-scored
  per-judge results (it does not call a provider);
- fold `score_consensus` and `_score_resistance_consensus` onto that helper
  without changing their public results when given the same inputs;
- add `--extra-judges` to `redblue`, `multiturn`, `refusal`, `codereview`, and
  `toolcall`, using the same flag grammar and budget-sharing as `eval` /
  `adversarial`;
- add `atomics judge-agreement`, which generates each fixture once and scores
  that response with every judge, then reports agreement and majority-flip rate;
- persist live numeric disagreement on the existing `judge_score_stdev` column
  and live categorical agreement on a new nullable column;
- persist study rows in a new `judge_agreement_results` table, never as a
  parent eval run that `report` would treat as a leaderboard row.

It will not:

- change fixture sets, rubrics, or score formulas;
- add `--extra-judges` to `rag`, `probe`, `archreview`, or `codegen`;
- run a per-turn panel on `multiturn` live runs;
- convene a panel on `toolcall`'s deterministic tool-call channel;
- make consensus the default;
- rewrite `report` or the dashboard in this milestone.

## Architecture

Three pieces, one contract.

### Helper — `atomics/eval/consensus.py`

The helper is a pure combine. Callers already have a list of per-judge results
from the existing scorers (`score_response`, `score_resistance`,
`classify_response`, `judge_review`, `score_conversation`). The helper does
not import a provider.

Two aggregators, selected by the result type:

- **Numeric.** Mean of parsed scores, population standard deviation, `n_judges`.
  Rationale stays the primary judge's. This is what `score_consensus` and
  `_score_resistance_consensus` already do; both become thin wrappers.
- **Categorical.** Majority vote on the label. Agreement rate is
  `majority_count / n_valid`. Ties (2–2, 1–1–1) do not invent a winner: the
  fixture is `unresolved` and excluded from the headline rate. Picking the
  primary on a tie would hide the disagreement the study exists to surface.

If nobody parsed, return the primary result flagged. The helper never averages
only-over-successes into a healthy-looking score when the panel mostly failed.

A single valid result returns that result unchanged (`n_judges=1`, stdev `0.0`,
agreement `1.0`). That is the default-path no-op.

### Live path — `--extra-judges`

Same string as today: `provider:model[@host]`, comma-separated. Parsing lives
in one place (`commands/common.py`), not a fifth copy. `--budget` shares one
ceiling across the model under test and every judge. Self-judge detection
already takes a list of pairs; extra judges join that list.

Default remains one judge. No silent spend increase.

### Study path — `atomics judge-agreement`

Not an eval suite. No parent `runs` row. It isolates judge disagreement, not
generation variance: one generate per fixture, then N judge calls on that
same response.

## Per-suite wiring

Each runner already has one judge call site. That site becomes “run the panel,
then combine.”

| Suite | Judge today | Aggregator | Live panel covers |
|---|---|---|---|
| `eval` | `score_consensus` | numeric | already shipped; helper only |
| `adversarial` | `_score_resistance_consensus` | numeric | already shipped; helper only |
| `redblue` | `score_response` | numeric | that call |
| `multiturn` | `score_turn` + `score_conversation` | numeric | conversation score only |
| `toolcall` | `score_resistance` on prose | numeric | prose channel only |
| `refusal` | `classify_response` | categorical | classification label |
| `codereview` | `judge_review` | categorical | verdict label |

`multiturn` per-turn consensus is out of scope for both the live path and
`judge-agreement`: turns × judges multiplies cost without changing the
headline. The study command judges the conversation score only, same as a
live panel. `toolcall`'s tool-call channel
stays deterministic (`dangerous_call` / `no_call` / …). A panel does not vote
on a tool trace.

Categorical score after a majority is the existing mapper
(`classification_to_score(expected, majority)`, `_score_for_verdict(majority)`).
Unresolved fixtures contribute `None` and are excluded from rates, same spirit
as a parse failure.

## `judge-agreement` command

Inputs:

- `--suite` — one of `redblue`, `adversarial`, `multiturn`, `refusal`,
  `codereview`, `toolcall`;
- `--judges` — same `provider:model[@host]` list, at least two;
- the usual model-under-test flags (`--provider`, `--model`, host overrides);
- optional `--fixtures`, `--budget`, `--json-out`, `--save` / `--no-save`.

`--runs` is not accepted. Generation variance is a different question.

Console summary:

```
judge-agreement  suite=refusal  fixtures=24  judges=3
  pairwise agreement   0.81
  majority-flip rate   0.12   (3 of 24 would change the headline)
  unresolved (ties)    1
  mean stdev           —      (categorical)
  cost                 $0.41
```

A flip means: the suite’s current single-judge rule would have recorded a
different headline contribution than the panel. That is the number that
justifies turning `--extra-judges` on for real runs.

Per-fixture rows list each judge’s label or score, the majority or mean, and
whether it flips. `--json-out` is the full table.

`--save` writes `judge_agreement_results` (run_id, suite, fixture_id,
votes_json, agreement, flipped, created_at). The table has no foreign key
to `runs`. `atomics report` does not read it. The study command still
enters `suite_run` when `--save` is on so a failure cannot leak the
connection, but it never calls `begin()` — there is no parent row to
finalize. Cleanup is `close()` only.

## Persistence

- Numeric live disagreement: existing `task_results.judge_score_stdev`.
  `redblue` starts filling it the way `eval` already does. `adversarial` /
  `toolcall` keep their resistance JSON; the helper’s stdev is included there.
- Categorical live agreement: nullable `judge_agreement REAL` on
  `evaluation_results`. Additive, pre-1.0 safe. `NULL` means “no panel.”
- Study rows: new `judge_agreement_results` table, as above.

`report` is not extended in this milestone. The column and table exist so a
later reporting change has something to read.

## Error handling

- A judge that raises is a `PROVIDER_ERROR` call. The panel continues.
- A judge that cannot parse is dropped from the aggregate, not scored as
  0.5-and-averaged.
- If the whole panel fails, surface the primary result flagged.
- Categorical ties are `unresolved` only when a panel was requested.
- Self-judge still refuses the run before any generate.
- Hitting `--budget` mid-panel stops the run the way `eval` does today.
- Errors stay sanitized. A judge API key in an exception does not reach
  the console.
- Live suite commands keep `suite_run`. The study command uses the same
  lifetime if `--save` is on.

## No-regression contract

What a user can see without opting in: `--help` grows `--extra-judges` on
five commands, and `atomics --help` grows `judge-agreement`. That is the
entire default-surface change. No new required flags, no new spend, no new
tables touched by `eval` / `redblue` / `report`.

Existing `tests/test_judge.py` consensus cases and
`tests/test_adversarial.py` extra-judge cases run before the extract and
after, and must be bit-identical. If extracting the helper changes a single
assertion, the extract is wrong.

## Testing

1. Characterization: existing consensus and extra-judge tests, before and
   after the extract.
2. Helper unit tests for the categorical path: majority, 2–2 tie →
   unresolved, 1-of-3 parse failure still majorities, all-failed → primary
   flagged, single result unchanged.
3. Per-suite “no flag means old path” tests — stubbed single judge, summary
   shape matches today (scores, labels, integrity block, parent-row
   finalize).
4. Per-suite “flag means panel” tests — two stub judges, mean or majority,
   stdev or agreement, `--budget` includes the extra provider.
5. Study command: generate called once, judge called N times; flip-rate
   arithmetic; `--no-save` opens no repository; JSON schema; sanitized
   failure.

Full suite, ruff, and mypy before claiming done.

## Out of scope (explicit)

`rag`, `probe`, `archreview` use the same pattern and are a follow-up.
`codegen` has no judge. Dashboard and `report` surfaces for the new column
and table are a follow-up. Per-turn `multiturn` panels are out of scope.
