# Judge Consensus and Agreement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in multi-judge consensus to the remaining security suites and a `judge-agreement` study command, without changing default one-judge runs.

**Architecture:** A pure combine helper (`atomics/eval/consensus.py`) aggregates already-scored votes. Live `--extra-judges` and `atomics judge-agreement` are two consumers of that helper. Existing `score_consensus` tests are the extract tripwire.

**Tech Stack:** Python 3.13, pytest, Click, SQLite (`CREATE TABLE IF NOT EXISTS` + nullable column reconcile, no `SCHEMA_VERSION` bump).

**Spec:** `docs/superpowers/specs/2026-08-14-judge-consensus-and-agreement-design.md`

---

## File map

| File | Role |
|---|---|
| `atomics/eval/consensus.py` | Numeric and categorical combine. No provider imports. |
| `tests/test_consensus.py` | Helper unit tests, including categorical path. |
| `atomics/eval/judge.py` | `score_consensus` calls the helper after scoring. |
| `atomics/eval/adversarial/runner.py` | `_score_with_all_judges` calls the helper. |
| `atomics/commands/common.py` | `extra_judges_option`, `parse_extra_judges`. |
| `atomics/storage/schema.py` | `judge_agreement` column; `judge_agreement_results` table. |
| `atomics/storage/records.py` | `judge_agreement` on `EvaluationResultRecord`. |
| `atomics/storage/repository.py` | Persist the new column and study rows. |
| `atomics/eval/redblue/runner.py` | `extra_judges` → `score_consensus`. |
| `atomics/eval/multiturn/runner.py` | Panel on `score_conversation` only. |
| `atomics/eval/toolcall/runner.py` | Panel on prose `score_resistance` only. |
| `atomics/eval/refusal/runner.py` | Categorical panel on `classify_response`. |
| `atomics/eval/codereview/runner.py` | Categorical panel on `judge_review`. |
| Matching `cmd_*.py` / `toolcall.py` | `--extra-judges` flag, budget share, parse. |
| `atomics/eval/agreement.py` | Study math: pairwise agreement, flip rate. |
| `atomics/commands/agreement.py` | `judge-agreement` command. |
| `atomics/cli.py` | Register the command. |
| `CHANGELOG.md`, `ROADMAP.md`, `docs/CLI_REFERENCE.md` | Docs. |

Out of scope: `rag`, `probe`, `archreview`, `codegen`, `report`, dashboard, per-turn multiturn panels.

---

### Task 1: Numeric combine helper (characterization first)

**Files:**
- Create: `atomics/eval/consensus.py`
- Create: `tests/test_consensus.py`

The existing `tests/test_judge.py` consensus cases stay untouched until Task 3. This task only adds the pure helper and its own tests.

- [ ] **Step 1: Write the failing numeric tests**

```python
# tests/test_consensus.py
from atomics.eval.consensus import NumericVote, combine_numeric


def test_single_numeric_vote_is_unchanged() -> None:
    result = combine_numeric(
        [NumericVote(score=0.8, parse_failed=False, judge_model="a", rationale="ok")]
    )
    assert result.score == 0.8
    assert result.n_judges == 1
    assert result.score_stdev == 0.0
    assert result.parse_failed is False
    assert result.rationale == "ok"


def test_numeric_mean_and_pstdev() -> None:
    result = combine_numeric(
        [
            NumericVote(score=1.0, parse_failed=False, judge_model="a", rationale="great"),
            NumericVote(score=0.4, parse_failed=False, judge_model="b", rationale="weak"),
        ]
    )
    assert result.score == 0.7
    assert result.score_stdev == 0.3
    assert result.n_judges == 2
    assert result.rationale == "great"
    assert result.judge_model == "a, b"


def test_numeric_drops_parse_failures() -> None:
    result = combine_numeric(
        [
            NumericVote(score=1.0, parse_failed=False, judge_model="good", rationale="ok"),
            NumericVote(score=0.5, parse_failed=True, judge_model="bad", rationale="nope"),
        ]
    )
    assert result.n_judges == 1
    assert result.score == 1.0


def test_numeric_all_failed_returns_flagged_primary() -> None:
    result = combine_numeric(
        [
            NumericVote(score=0.5, parse_failed=True, judge_model="a", rationale="nope"),
            NumericVote(score=0.5, parse_failed=True, judge_model="b", rationale="also"),
        ]
    )
    assert result.parse_failed is True
    assert result.n_judges == 2
    assert result.rationale == "nope"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_consensus.py -q --no-cov`
Expected: FAIL with `ModuleNotFoundError: atomics.eval.consensus`

- [ ] **Step 3: Implement the helper**

```python
# atomics/eval/consensus.py
"""Combine already-scored judge votes. Does not call a provider."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class NumericVote:
    score: float
    parse_failed: bool
    judge_model: str
    rationale: str


@dataclass(frozen=True)
class NumericConsensus:
    score: float
    score_stdev: float
    n_judges: int
    parse_failed: bool
    judge_model: str
    rationale: str
    valid_scores: tuple[float, ...]


def combine_numeric(votes: Sequence[NumericVote]) -> NumericConsensus:
    """Mean and population stdev of votes that parsed.

    `votes[0]` is the primary. If nobody parsed, return the primary flagged
    with `n_judges` equal to the panel size.
    """
    if not votes:
        raise ValueError("combine_numeric requires at least one vote")
    primary = votes[0]
    valid = [v for v in votes if not v.parse_failed]
    if not valid:
        return NumericConsensus(
            score=primary.score,
            score_stdev=0.0,
            n_judges=len(votes),
            parse_failed=True,
            judge_model=primary.judge_model,
            rationale=primary.rationale,
            valid_scores=(),
        )
    scores = [v.score for v in valid]
    stdev = round(statistics.pstdev(scores), 3) if len(scores) > 1 else 0.0
    return NumericConsensus(
        score=round(sum(scores) / len(scores), 3),
        score_stdev=stdev,
        n_judges=len(valid),
        parse_failed=False,
        judge_model=", ".join(v.judge_model for v in valid),
        rationale=valid[0].rationale,
        valid_scores=tuple(scores),
    )
```

Leave categorical types for Task 2 so this commit is one idea.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_consensus.py -q --no-cov`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add atomics/eval/consensus.py tests/test_consensus.py
git commit -m "Add a numeric judge-consensus helper

Pure combine: mean and population stdev of votes that parsed. A single
vote is unchanged. An all-failed panel returns the primary flagged.
Does not call a provider."
```

---

### Task 2: Categorical combine helper

**Files:**
- Modify: `atomics/eval/consensus.py`
- Modify: `tests/test_consensus.py`

- [ ] **Step 1: Write the failing categorical tests**

Append to `tests/test_consensus.py`:

```python
from atomics.eval.consensus import CategoricalVote, combine_categorical


def test_categorical_majority() -> None:
    result = combine_categorical(
        [
            CategoricalVote(label="refuse", parse_failed=False, judge_model="a", rationale="no"),
            CategoricalVote(label="refuse", parse_failed=False, judge_model="b", rationale="nope"),
            CategoricalVote(label="comply", parse_failed=False, judge_model="c", rationale="yes"),
        ]
    )
    assert result.label == "refuse"
    assert result.agreement == 2 / 3
    assert result.unresolved is False
    assert result.n_judges == 3
    assert result.rationale == "no"


def test_categorical_tie_is_unresolved() -> None:
    result = combine_categorical(
        [
            CategoricalVote(label="refuse", parse_failed=False, judge_model="a", rationale="no"),
            CategoricalVote(label="comply", parse_failed=False, judge_model="b", rationale="yes"),
        ]
    )
    assert result.label is None
    assert result.unresolved is True
    assert result.agreement == 0.5
    assert result.parse_failed is False


def test_categorical_one_each_among_valid_is_a_tie() -> None:
    result = combine_categorical(
        [
            CategoricalVote(label="detected", parse_failed=False, judge_model="a", rationale="hit"),
            CategoricalVote(label="missed", parse_failed=False, judge_model="b", rationale="miss"),
            CategoricalVote(label="unknown", parse_failed=True, judge_model="c", rationale="?"),
        ]
    )
    assert result.label == "detected"
    assert result.n_judges == 2
    assert result.agreement == 0.5
    assert result.unresolved is True  # 1-1 among valid is a tie


def test_categorical_two_of_three_is_majority() -> None:
    result = combine_categorical(
        [
            CategoricalVote(label="detected", parse_failed=False, judge_model="a", rationale="hit"),
            CategoricalVote(label="detected", parse_failed=False, judge_model="b", rationale="hit2"),
            CategoricalVote(label="unknown", parse_failed=True, judge_model="c", rationale="?"),
        ]
    )
    assert result.label == "detected"
    assert result.unresolved is False
    assert result.agreement == 1.0
    assert result.n_judges == 2


def test_categorical_all_failed_returns_flagged_primary() -> None:
    result = combine_categorical(
        [
            CategoricalVote(label="unknown", parse_failed=True, judge_model="a", rationale="nope"),
            CategoricalVote(label="unknown", parse_failed=True, judge_model="b", rationale="also"),
        ]
    )
    assert result.parse_failed is True
    assert result.unresolved is False
    assert result.label is None
    assert result.n_judges == 2
    assert result.rationale == "nope"
```

Note the 1–1-among-valid case: that is a tie (`unresolved=True`). Two matching valid votes is a majority even if a third failed.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_consensus.py -q --no-cov -k categorical`
Expected: FAIL with `ImportError: CategoricalVote`

- [ ] **Step 3: Implement categorical combine**

Append to `atomics/eval/consensus.py`:

```python
@dataclass(frozen=True)
class CategoricalVote:
    label: str
    parse_failed: bool
    judge_model: str
    rationale: str


@dataclass(frozen=True)
class CategoricalConsensus:
    label: str | None
    agreement: float | None
    n_judges: int
    unresolved: bool
    parse_failed: bool
    judge_model: str
    rationale: str
    votes: tuple[str, ...]


def combine_categorical(votes: Sequence[CategoricalVote]) -> CategoricalConsensus:
    """Majority vote. Ties do not invent a winner."""
    if not votes:
        raise ValueError("combine_categorical requires at least one vote")
    primary = votes[0]
    valid = [v for v in votes if not v.parse_failed]
    if not valid:
        return CategoricalConsensus(
            label=None,
            agreement=None,
            n_judges=len(votes),
            unresolved=False,
            parse_failed=True,
            judge_model=primary.judge_model,
            rationale=primary.rationale,
            votes=(),
        )
    counts = Counter(v.label for v in valid)
    top = counts.most_common()
    majority_label, majority_count = top[0]
    tied = len(top) > 1 and top[1][1] == majority_count
    agreement = majority_count / len(valid)
    if tied:
        return CategoricalConsensus(
            label=None,
            agreement=agreement,
            n_judges=len(valid),
            unresolved=True,
            parse_failed=False,
            judge_model=", ".join(v.judge_model for v in valid),
            rationale=valid[0].rationale,
            votes=tuple(v.label for v in valid),
        )
    return CategoricalConsensus(
        label=majority_label,
        agreement=agreement,
        n_judges=len(valid),
        unresolved=False,
        parse_failed=False,
        judge_model=", ".join(v.judge_model for v in valid),
        rationale=valid[0].rationale,
        votes=tuple(v.label for v in valid),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_consensus.py -q --no-cov`
Expected: all passed (9)

- [ ] **Step 5: Commit**

```bash
git add atomics/eval/consensus.py tests/test_consensus.py
git commit -m "Add categorical majority consensus

Ties are unresolved rather than a silent pick of the primary. Parse
failures drop out of the vote; an all-failed panel returns the primary
flagged."
```

---

### Task 3: Fold `score_consensus` onto the helper

**Files:**
- Modify: `atomics/eval/judge.py` (`score_consensus`, around lines 374–436)
- Test: `tests/test_judge.py` (do not edit; these are the tripwire)

- [ ] **Step 1: Run the existing consensus tests (must already pass)**

Run: `uv run pytest tests/test_judge.py -q --no-cov -k consensus`
Expected: 4 passed

- [ ] **Step 2: Rewrite `score_consensus` to combine via the helper**

Keep the function signature and the `score_response` loop. After collecting `results: list[JudgeResult]`, map and combine:

```python
from atomics.eval.consensus import NumericVote, combine_numeric

# inside score_consensus, replace the valid/mean/stdev block:
votes = [
    NumericVote(
        score=r.score,
        parse_failed=r.parse_failed,
        judge_model=r.judge_model,
        rationale=r.rationale,
    )
    for r in results
]
combined = combine_numeric(votes)
if combined.parse_failed:
    primary = results[0]
    primary.n_judges = combined.n_judges
    primary.score_stdev = combined.score_stdev
    return primary
valid = [r for r in results if not r.parse_failed]
return JudgeResult(
    score=combined.score,
    accuracy=round(statistics.mean(r.accuracy for r in valid)),
    completeness=round(statistics.mean(r.completeness for r in valid)),
    format_score=round(statistics.mean(r.format_score for r in valid)),
    rationale=combined.rationale,
    judge_model=combined.judge_model,
    criteria_coverage=results[0].criteria_coverage,
    score_stdev=combined.score_stdev,
    n_judges=combined.n_judges,
)
```

Do not change `_parse_rubric`, `score_response`, or `JudgeResult`.

- [ ] **Step 3: Re-run the tripwire**

Run: `uv run pytest tests/test_judge.py tests/test_consensus.py -q --no-cov`
Expected: all passed, bit-identical assertions

- [ ] **Step 4: Commit**

```bash
git add atomics/eval/judge.py
git commit -m "Fold score_consensus onto the shared numeric helper

Existing test_judge consensus cases are the contract. Rubric parsing
and per-judge score_response calls are unchanged."
```

---

### Task 4: Fold adversarial panel onto the helper

**Files:**
- Modify: `atomics/eval/adversarial/runner.py` (`_score_with_all_judges`, around lines 256–334)
- Test: `tests/test_adversarial.py` extra-judge cases (do not edit)

- [ ] **Step 1: Run the existing extra-judge tests**

Run: `uv run pytest tests/test_adversarial.py -q --no-cov -k extra_judges`
Expected: passed (current count; do not change assertions)

- [ ] **Step 2: Replace the mean/label block with the helper**

Keep the `score_resistance` loop, cost sum, and `judge_calls` flatten. After `all_results` is built:

```python
from atomics.eval.consensus import NumericVote, combine_numeric
from atomics.eval.adversarial.scorer import _label_from_score

votes = [
    NumericVote(
        score=r.score,
        parse_failed=r.parse_failed,
        judge_model=r.judge_model,
        rationale=r.rationale,
    )
    for r in all_results
]
combined = combine_numeric(votes)
# keep the existing all-failed ResistanceResult construction when
# combined.parse_failed, using the same provider_failed / representative logic
# as today. When it succeeds:
return ResistanceResult(
    score=combined.score,
    label=_label_from_score(combined.score),
    rationale=combined.rationale,
    judge_model=combined.judge_model,
    judge_cost_usd=total_judge_cost,
    judge_scores=list(combined.valid_scores),
    judge_calls=all_judge_calls,
    judges_expected=len(all_results),
    judges_scored=combined.n_judges,
)
```

Consensus label stays `_label_from_score(mean)`, not a majority of labels. That is today's behavior.

- [ ] **Step 3: Re-run the tripwire**

Run: `uv run pytest tests/test_adversarial.py tests/test_judge.py tests/test_consensus.py -q --no-cov`
Expected: all passed

- [ ] **Step 4: Commit**

```bash
git add atomics/eval/adversarial/runner.py
git commit -m "Fold adversarial consensus onto the shared numeric helper

Label still comes from the mean score, matching the existing extra-judge
tests. Cost and judge_calls stay flattened across the panel."
```

---

### Task 5: Shared `--extra-judges` parsing

**Files:**
- Modify: `atomics/commands/common.py`
- Create: `tests/test_commands_extra_judges.py`
- Modify: `atomics/commands/eval.py` and `atomics/commands/security/cmd_adversarial.py` to call the shared parser

- [ ] **Step 1: Write the failing parser tests**

```python
# tests/test_commands_extra_judges.py
from types import SimpleNamespace

from atomics.commands.common import parse_extra_judges


def test_empty_spec_is_no_panel() -> None:
    assert parse_extra_judges(None, build=lambda *_a, **_k: None) == []
    assert parse_extra_judges("", build=lambda *_a, **_k: None) == []


def test_parses_provider_model_and_optional_host() -> None:
    built: list[tuple] = []

    def build(name, model, host):
        built.append((name, model, host))
        return SimpleNamespace(name=name)

    pairs = parse_extra_judges(
        "claude:claude-sonnet-4-6,ollama:deepseek-r1:14b@http://gpu-host:11434",
        build=build,
        default_host="http://fallback:11434",
    )
    assert [p[1] for p in pairs] == ["claude-sonnet-4-6", "deepseek-r1:14b"]
    assert built[0] == ("claude", "claude-sonnet-4-6", "http://fallback:11434")
    assert built[1] == ("ollama", "deepseek-r1:14b", "http://gpu-host:11434")
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_commands_extra_judges.py -q --no-cov`
Expected: FAIL, `parse_extra_judges` missing

- [ ] **Step 3: Implement parser and decorator**

Add to `atomics/commands/common.py`:

```python
from collections.abc import Callable
from typing import TypeVar

_P = TypeVar("_P")


def extra_judges_option(fn: Callable) -> Callable:
    return click.option(
        "--extra-judges",
        type=str,
        default=None,
        help="Comma-separated extra judges for consensus scoring. "
        "Format: provider:model[@host] "
        "(e.g. claude:claude-sonnet-4-6,ollama:deepseek-r1:14b@http://gpu-host:11434).",
    )(fn)


def parse_extra_judges(
    spec: str | None,
    *,
    build: Callable[[str, str | None, str | None], _P],
    default_host: str | None = None,
) -> list[tuple[_P, str | None]]:
    """Parse the --extra-judges string into (provider, model) pairs."""
    if not spec:
        return []
    pairs: list[tuple[_P, str | None]] = []
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        host = default_host
        if "@" in item:
            item, host = item.rsplit("@", 1)
        name, _, model = item.partition(":")
        model_or_none = model or None
        pairs.append((build(name, model_or_none, host), model_or_none))
    return pairs
```

Replace the duplicated parse loops in `eval.py` and `cmd_adversarial.py` with `parse_extra_judges(...)` using each command's existing `_build_provider` / `_make_provider` as `build`. Keep the `@click.option("--extra-judges"...` on those two commands for this task (decorator swap can happen when touching each command later) **or** swap them to `extra_judges_option` now if the help text change is acceptable. Prefer swapping now so help text is one string.

- [ ] **Step 4: Run parser tests plus eval/adversarial CLI tests**

Run: `uv run pytest tests/test_commands_extra_judges.py tests/test_cli_eval_budget.py tests/test_adversarial.py -q --no-cov -k "extra_judges or consensus or budget"`
Expected: passed

- [ ] **Step 5: Commit**

```bash
git add atomics/commands/common.py atomics/commands/eval.py \
  atomics/commands/security/cmd_adversarial.py tests/test_commands_extra_judges.py
git commit -m "Share --extra-judges parsing across commands

One grammar: provider:model[@host]. eval and adversarial call the same
parser the remaining suites will use."
```

---

### Task 6: Persistence for agreement

**Files:**
- Modify: `atomics/storage/schema.py`
- Modify: `atomics/storage/records.py`
- Modify: `atomics/storage/repository.py`
- Test: `tests/test_storage.py` (add cases)

Do **not** bump `SCHEMA_VERSION`. New nullable column is reconciled in place; new table is `CREATE TABLE IF NOT EXISTS`.

- [ ] **Step 1: Write failing storage tests**

```python
def test_evaluation_result_persists_judge_agreement(tmp_path):
    from atomics.storage.records import EvaluationResultRecord
    from atomics.storage.repository import MetricsRepository

    db = tmp_path / "agree.db"
    repo = MetricsRepository(db)
    repo.create_run("r1", tier="refusal", provider="mock", model="m")
    repo.save_evaluation_result(
        EvaluationResultRecord(
            run_id="r1",
            suite="refusal",
            fixture_id="rf-01",
            status="complete",
            generation_status="ok",
            judge_status="scored",
            latency_ms=1.0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            result_json={},
            score=1.0,
            judge_agreement=2 / 3,
        )
    )
    rows = repo.get_evaluation_results(suite="refusal")
    repo.close()
    assert rows[0]["judge_agreement"] == pytest.approx(2 / 3)


def test_save_agreement_study_row(tmp_path):
    from atomics.storage.repository import MetricsRepository

    db = tmp_path / "study.db"
    repo = MetricsRepository(db)
    repo.save_agreement_result(
        run_id="s1",
        suite="refusal",
        fixture_id="rf-01",
        votes={"a": "refuse", "b": "refuse", "c": "comply"},
        agreement=2 / 3,
        flipped=True,
    )
    rows = repo.get_agreement_results(run_id="s1")
    repo.close()
    assert rows[0]["flipped"] == 1
    assert rows[0]["agreement"] == pytest.approx(2 / 3)
```

Look at an existing `save_evaluation_result` test in `tests/test_storage.py` and match its `create_run` / record construction exactly; add `judge_agreement=` only.

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_storage.py -q --no-cov -k "judge_agreement or agreement_study"`
Expected: FAIL (unknown column / missing method)

- [ ] **Step 3: Schema and repository**

Add to `evaluation_results` in `SCHEMA_SQL`:

```sql
    judge_agreement         REAL DEFAULT NULL,
```

Add table (no FK to `runs`):

```sql
CREATE TABLE IF NOT EXISTS judge_agreement_results (
    result_id     TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    suite         TEXT NOT NULL,
    fixture_id    TEXT NOT NULL,
    votes_json    TEXT NOT NULL DEFAULT '{}',
    agreement     REAL DEFAULT NULL,
    flipped       INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    UNIQUE (run_id, suite, fixture_id)
);
```

Add `judge_agreement: float | None = None` to `EvaluationResultRecord`. Thread it through `save_evaluation_result` INSERT (read the current column list and append). Add `save_agreement_result` / `get_agreement_results`.

- [ ] **Step 4: Run storage tests**

Run: `uv run pytest tests/test_storage.py tests/test_storage_schema.py -q --no-cov`
Expected: passed

- [ ] **Step 5: Commit**

```bash
git add atomics/storage/schema.py atomics/storage/records.py \
  atomics/storage/repository.py tests/test_storage.py
git commit -m "Persist live judge agreement and study rows

Nullable evaluation_results.judge_agreement, reconciled in place.
judge_agreement_results has no foreign key to runs, so a study cannot
show up as an in-progress eval."
```

---

### Task 7: `redblue` live panel

**Files:**
- Modify: `atomics/eval/redblue/runner.py`
- Modify: `atomics/commands/security/cmd_redblue.py`
- Test: `tests/test_redblue.py` and/or `tests/test_cli_suite_persistence.py`

- [ ] **Step 1: Characterization — no flag means old path**

Add a test that stubs `score_response` (or the runner's provider) with one judge and asserts the summary score equals today's single-judge score. If `tests/test_redblue.py` already covers a one-judge run, run it first and keep it green after the change.

Add a panel test:

```python
@pytest.mark.asyncio
async def test_redblue_extra_judges_averages_scores(monkeypatch):
    # stub provider generate; stub two judges via extra_judges
    # assert fixture score is the mean and judge_score_stdev is set
```

Use the existing redblue test doubles in `tests/test_redblue.py` rather than inventing a new harness.

- [ ] **Step 2: Run new tests (panel test fails; old tests pass)**

- [ ] **Step 3: Wire the runner**

Add `extra_judges: list[tuple[BaseProvider, str | None]] | None = None` to `run_redblue`. Include them in `detect_self_judge`. Replace `score_response(...)` with `score_consensus(..., extra_judges=extra_judges or [])`. Copy `judge.score_stdev` onto `task_result.judge_score_stdev` (the field already exists on `TaskResult`).

CLI: `@extra_judges_option`, `parse_extra_judges`, pass into `share_budget` and `run_redblue`.

- [ ] **Step 4: Run redblue tests plus persistence tests**

Run: `uv run pytest tests/test_redblue.py tests/test_cli_suite_persistence.py -q --no-cov -k redblue`
Expected: passed

- [ ] **Step 5: Commit**

```bash
git add atomics/eval/redblue/runner.py atomics/commands/security/cmd_redblue.py tests/
git commit -m "Add --extra-judges to redblue

One-judge runs still call score_consensus with an empty panel, which
returns the primary unchanged. judge_score_stdev starts getting filled."
```

---

### Task 8: `multiturn` live panel (conversation only)

**Files:**
- Modify: `atomics/eval/multiturn/runner.py`
- Modify: `atomics/eval/multiturn/judge.py` (add `score_conversation_consensus` or accept extra judges in the runner)
- Modify: `atomics/commands/security/cmd_multiturn.py`
- Test: `tests/test_multiturn.py`

- [ ] **Step 1: Write failing tests**

One-judge conversation score unchanged. Two-judge panel: `score_conversation` invoked twice on the same transcript; headline is the mean; per-turn `score_turn` still called once.

- [ ] **Step 2: Run to verify the panel test fails**

- [ ] **Step 3: Implement**

In the runner, after the transcript is complete, loop extra judges on `score_conversation` only. Map each `ConversationJudgeResult` to `NumericVote(score=r.score, parse_failed=r.parse_failed, ...)`. Combine. Write the combined score as the conversation headline. Do not panel `score_turn`.

CLI: same flag / parse / budget pattern as redblue.

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_multiturn.py tests/test_cli_suite_persistence.py -q --no-cov -k multiturn`
Expected: passed

- [ ] **Step 5: Commit**

```bash
git add atomics/eval/multiturn/ atomics/commands/security/cmd_multiturn.py tests/test_multiturn.py
git commit -m "Add --extra-judges to multiturn conversation scoring

Per-turn judges stay single. The headline conversation score is the
panel mean when extra judges are supplied."
```

---

### Task 9: `toolcall` live panel (prose only)

**Files:**
- Modify: `atomics/eval/toolcall/runner.py` (`_judge`)
- Modify: `atomics/commands/toolcall.py`
- Test: `tests/test_toolcall_runner.py`, `tests/test_toolcall_cli.py`

- [ ] **Step 1: Write failing tests**

Tool-channel outcome unchanged when extra judges are present. Prose label / resistance score is the panel mean. `_judge` called once per extra judge.

- [ ] **Step 2: Run to verify fail**

- [ ] **Step 3: Implement**

Thread `extra_judges` into `run_toolcall_suite` and `_judge`. When extras exist, call `score_resistance` for primary + extras and combine via `combine_numeric`, then rebuild `ResistanceResult` the same way Task 4 does (including `_label_from_score`). When extras is empty, keep the single `score_resistance` call.

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_toolcall_runner.py tests/test_toolcall_cli.py tests/test_toolcall_scorer.py -q --no-cov`
Expected: passed

- [ ] **Step 5: Commit**

```bash
git add atomics/eval/toolcall/runner.py atomics/commands/toolcall.py tests/test_toolcall_*.py
git commit -m "Add --extra-judges to the toolcall prose channel

The tool-call channel stays deterministic. Only score_resistance is
convened as a panel."
```

---

### Task 10: `refusal` live panel

**Files:**
- Modify: `atomics/eval/refusal/runner.py`
- Modify: `atomics/eval/refusal/scorer.py` (optional thin `classify_consensus`)
- Modify: `atomics/commands/security/cmd_refusal.py`
- Test: `tests/test_cli_refusal.py`, `tests/test_refusal_*.py` if present

- [ ] **Step 1: Write failing tests**

No extras: classification and score match today. Two judges `refuse`/`refuse`/`comply` (if three): majority `refuse`. Two-judge `refuse`/`comply`: `unresolved`, score `None`, excluded from rates. Persist `judge_agreement` when saving.

- [ ] **Step 2: Run to verify fail**

- [ ] **Step 3: Implement**

After generate, call `classify_response` for primary + extras. Map to `CategoricalVote(label=r.classification, parse_failed=r.status is not SCORED, ...)`. Combine. If `unresolved` or `parse_failed`, judge outcome is unscored (`score=None`). Else `classification_to_score(expected, combined.label)` and `classification_to_judge_outcome` on a reconstructed `ClassificationResult`.

CLI: flag / parse / budget / self-judge list.

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_cli_refusal.py tests/test_refusal_calibration.py tests/test_eval_refusal.py -q --no-cov`
(Use whatever refusal test files exist: `ls tests/*refusal*`)

Expected: passed

- [ ] **Step 5: Commit**

```bash
git add atomics/eval/refusal/ atomics/commands/security/cmd_refusal.py tests/
git commit -m "Add --extra-judges to refusal as a majority vote

Ties are unresolved and drop out of calibration rates. One-judge runs
still classify once."
```

---

### Task 11: `codereview` live panel

**Files:**
- Modify: `atomics/eval/codereview/runner.py`
- Modify: `atomics/commands/security/cmd_codereview.py`
- Test: `tests/test_cli_codereview.py`, `tests/test_codereview_scorer.py`

Same shape as Task 10, voting on `verdict`. Unresolved fixtures drop out of detection / false-positive rates.

- [ ] **Step 1: Write failing tests** (majority `detected`, 1–1 tie unresolved, no-flag unchanged)

- [ ] **Step 2: Run to verify fail**

- [ ] **Step 3: Implement** analogously, using `judge_review` + `combine_categorical` + `_score_for_verdict` / `verdict_to_judge_outcome`

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_cli_codereview.py tests/test_codereview_scorer.py tests/test_codereview_runner.py -q --no-cov`
Expected: passed

- [ ] **Step 5: Commit**

```bash
git add atomics/eval/codereview/ atomics/commands/security/cmd_codereview.py tests/
git commit -m "Add --extra-judges to codereview as a majority vote

Unresolved verdicts are excluded from detection and false-positive
rates, matching parse failures."
```

---

### Task 12: `judge-agreement` study command

**Files:**
- Create: `atomics/eval/agreement.py`
- Create: `atomics/commands/agreement.py`
- Create: `tests/test_agreement.py`
- Create: `tests/test_cli_agreement.py`
- Modify: `atomics/cli.py` (register)

- [ ] **Step 1: Write failing math tests**

```python
# tests/test_agreement.py
from atomics.eval.agreement import majority_flip, pairwise_agreement


def test_pairwise_agreement_all_match() -> None:
    assert pairwise_agreement(["refuse", "refuse", "refuse"]) == 1.0


def test_pairwise_agreement_two_of_three() -> None:
    # pairs: (a,b) match, (a,c) differ, (b,c) differ → 1/3
    assert pairwise_agreement(["refuse", "refuse", "comply"]) == pytest.approx(1 / 3)


def test_flip_when_primary_loses_majority() -> None:
    assert majority_flip(primary="comply", majority="refuse") is True
    assert majority_flip(primary="refuse", majority="refuse") is False
    assert majority_flip(primary="refuse", majority=None) is False  # unresolved is not a flip
```

A flip is: the suite’s single-judge rule (the primary vote) would have recorded a different headline contribution than the panel majority/mean. Unresolved is not a flip; it is counted separately.

For numeric suites, a flip is: `_headline(primary_score) != _headline(mean_score)` where `_headline` is the suite’s existing rounding/label rule. For `redblue`/`multiturn`, compare the scores rounded to 3 decimals. For `adversarial`/`toolcall` prose, compare `_label_from_score(primary)` vs `_label_from_score(mean)`.

- [ ] **Step 2: Run to verify fail**

- [ ] **Step 3: Implement math in `atomics/eval/agreement.py`**

```python
from itertools import combinations


def pairwise_agreement(labels: list[str]) -> float | None:
    pairs = list(combinations(labels, 2))
    if not pairs:
        return None
    return sum(a == b for a, b in pairs) / len(pairs)


def majority_flip(*, primary: str | None, majority: str | None) -> bool:
    if majority is None or primary is None:
        return False
    return primary != majority
```

Plus `numeric_flip(primary: float, mean: float, *, label_fn=None) -> bool` for the numeric suites.

- [ ] **Step 4: Pass math tests**

- [ ] **Step 5: Write failing CLI tests**

In `tests/test_cli_agreement.py`:

- `--help` lists `judge-agreement` and requires `--suite` and `--judges`
- `--judges` with one entry exits nonzero
- stubbed refusal study: generate called once per fixture, classify called N times
- flip-rate on the console
- `--no-save` opens no repository (reuse `_tracking_repository` from `tests/test_cli_suite_persistence.py`)
- `--json-out` writes the table
- a raised judge error is sanitized (no `api_key=` in output)

- [ ] **Step 6: Implement `atomics/commands/agreement.py`**

Command name: `judge-agreement`. Options: `--suite` (choice of the six), `--judges` (required, same grammar, ≥2 after parse), `--provider`/`--model`/host flags, `--fixtures`, `--budget`, `--json-out`, `--save/--no-save` (default `--no-save` — a study should not write unless asked).

Dispatch: load that suite’s fixtures, generate once, score with every parsed judge, combine via the helper, compute flip vs primary. Print the summary from the spec. `--save` enters `suite_run` but never `begin()`; write rows with `save_agreement_result`.

Register: `from atomics.commands import agreement as agreement_commands` and `cli.add_command(agreement_commands.judge_agreement)` in `atomics/cli.py`.

- [ ] **Step 7: Run CLI + math tests**

Run: `uv run pytest tests/test_agreement.py tests/test_cli_agreement.py -q --no-cov`
Expected: passed

- [ ] **Step 8: Commit**

```bash
git add atomics/eval/agreement.py atomics/commands/agreement.py \
  atomics/cli.py tests/test_agreement.py tests/test_cli_agreement.py
git commit -m "Add atomics judge-agreement

Generates each fixture once and scores the same response with every
judge. Reports pairwise agreement and majority-flip rate. Does not
write a parent eval run."
```

---

### Task 13: Docs

**Files:**
- Modify: `CHANGELOG.md` (Unreleased)
- Modify: `ROADMAP.md` (Judge quality bullet)
- Modify: `docs/CLI_REFERENCE.md`

- [ ] **Step 1: Write the changelog / roadmap / CLI rows**

CHANGELOG under Unreleased → Added:

- `--extra-judges` on `redblue`, `multiturn`, `refusal`, `codereview`, `toolcall`. Default remains one judge. Categorical suites majority-vote; ties are unresolved.
- `atomics judge-agreement` — generate once, judge N times, report flip rate. Not a leaderboard row.

ROADMAP “Judge quality” bullet: mark the security-suite half done; note `rag` / `probe` / `archreview` still single-judge.

CLI_REFERENCE: one row for `judge-agreement`, and `--extra-judges` on the five commands.

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md ROADMAP.md docs/CLI_REFERENCE.md
git commit -m "Document consensus judging and judge-agreement"
```

---

### Task 14: Full regression

- [ ] **Step 1: Run the tripwires, then everything**

```bash
uv run pytest tests/test_judge.py tests/test_adversarial.py tests/test_consensus.py -q --no-cov
uv run pytest -q
uv run ruff check atomics/ tests/ --ignore E501
uv run mypy atomics/
```

Expected: 4 consensus + extra-judge tripwires still green; full suite green; ruff/mypy clean.

- [ ] **Step 2: If anything failed, fix it in the task that owns it. Do not “clean up” across tasks.**

- [ ] **Step 3: No extra commit unless a fix was needed.**

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|---|---|
| Pure combine helper, no provider | 1, 2 |
| Numeric mean + pstdev; single vote unchanged; all-failed → primary | 1 |
| Categorical majority; ties unresolved | 2 |
| Fold `score_consensus` | 3 |
| Fold adversarial panel; label from mean | 4 |
| Shared `--extra-judges` grammar | 5 |
| `judge_agreement` column + study table, no SCHEMA_VERSION bump | 6 |
| `redblue` / `multiturn` (conversation) / `toolcall` (prose) | 7, 8, 9 |
| `refusal` / `codereview` majority | 10, 11 |
| `judge-agreement`: generate once, no parent run, flip rate | 12 |
| Default surface = help text only | 5, 7–12 |
| Existing consensus tests as tripwire | 3, 4, 14 |
| Out of scope: rag/probe/archreview/codegen/report | none, by design |
