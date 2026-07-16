# CLI Foundation and Suite Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the refusal and secure-code-review commands behind a reusable CLI foundation while giving both suites honest typed outcomes, complete persistence, consistent progress, and integrity-aware exits.

**Architecture:** Keep `atomics.cli:cli` as the stable entry point and command registry. Move reusable command concerns into `atomics.commands.common`, move each selected command into its own module, and adapt both runners to the existing `AttemptResult` and `RunIntegrity` contracts. Persist their canonical fixture JSON through one generic `evaluation_results` table whose typed record does not import evaluation modules.

**Tech Stack:** Python 3.11+, Click, Rich, dataclasses, SQLite, pytest, pytest-asyncio, mypy, Ruff.

---

## File map

**Create**

- `atomics/commands/__init__.py` — command package boundary.
- `atomics/commands/common.py` — provider factory, progress, JSON, attribution, integrity policy, and persistence record adapter.
- `atomics/commands/refusal.py` — refusal Click options, orchestration, and rendering.
- `atomics/commands/codereview.py` — code-review Click options, orchestration, and rendering.
- `atomics/eval/attempt_serialization.py` — shared attempt, judge-call, error, and integrity serialization.
- `atomics/eval/provider_attempt.py` — convert responses/exceptions into immutable attempts.
- `atomics/eval/refusal/scorer.py` — typed refusal judge interaction and parsing.
- `atomics/eval/codereview/scorer.py` — typed code-review judge interaction and parsing.
- `atomics/storage/records.py` — storage-owned `EvaluationResultRecord`.
- `tests/test_commands_common.py`
- `tests/test_cli_refusal.py`
- `tests/test_cli_codereview.py`
- `tests/test_eval_attempt_serialization.py`
- `tests/test_eval_provider_attempt.py`
- `tests/test_refusal_scorer.py`
- `tests/test_codereview_scorer.py`

**Modify**

- `atomics/cli.py:48-92,112,1799-1872,2620-2770` — move shared primitives and command bodies; retain registration and compatibility exports.
- `atomics/eval/adversarial/runner.py:100-272` — consume shared serialization without changing output.
- `atomics/eval/refusal/runner.py` — retain auditable attempts and expose integrity.
- `atomics/eval/codereview/runner.py` — retain auditable attempts and expose integrity.
- `atomics/storage/schema.py` — schema v20 and `evaluation_results`.
- `atomics/storage/repository.py` — save, query, and finalize generic evaluation rows.
- `atomics/storage/__init__.py` — export the storage record.
- `tests/test_refusal.py`
- `tests/test_codereview.py`
- `tests/test_adversarial.py`
- `tests/test_storage.py`
- schema-version assertions in `tests/test_regression.py`, `tests/test_soak.py`, and `tests/test_cli_adversarial_integrity.py`.
- `ARCHITECTURE.md`, `README.md`, `QUICKSTART.md`, `CHANGELOG.md`, and `SECURITY.md`.

## Compatibility invariants

- The executable remains `atomics.cli:cli`.
- Root command names remain `refusal` and `codereview`.
- Existing provider, model, host, judge, and JSON flags continue to parse.
- `atomics.cli._make_provider`, `FixtureProgress`, `refusal`, and `codereview` remain importable as aliases to the moved objects.
- Existing summary constructors continue accepting `results=`.
- Serialized summaries retain `results` and add identical `fixture_results`.
- Adversarial JSON is byte-for-byte equivalent after helper extraction, apart from dictionary insertion order only if tests establish that consumers do not rely on it.

---

### Task 1: Characterize the existing command surface

**Files:**
- Create: `tests/test_cli_refusal.py`
- Create: `tests/test_cli_codereview.py`
- Modify: none

- [ ] **Step 1: Add passing characterization tests**

Use `CliRunner` and patch each lazy runner import. Assert the existing options and JSON keys before extraction:

```python
def test_refusal_help_preserves_public_options() -> None:
    result = CliRunner().invoke(cli, ["refusal", "--help"])
    assert result.exit_code == 0
    for option in (
        "--provider",
        "--model",
        "--ollama-host",
        "--vllm-host",
        "--judge-provider",
        "--judge-model",
        "--judge-host",
        "--json-out",
    ):
        assert option in result.output


def test_codereview_help_preserves_public_options() -> None:
    result = CliRunner().invoke(cli, ["codereview", "--help"])
    assert result.exit_code == 0
    assert "--provider" in result.output
    assert "--judge-provider" in result.output
    assert "--json-out" in result.output
```

Build deterministic `RefusalSummary` and `CodeReviewSummary` objects in fixtures, patch `atomics.cli._make_provider`, and patch `atomics.eval.refusal.run_refusal` or `atomics.eval.codereview.run_codereview`. Assert exit zero, current aggregate labels, and the legacy `results` JSON key.

- [ ] **Step 2: Run characterization tests**

Run:

```bash
uv run pytest tests/test_cli_refusal.py tests/test_cli_codereview.py -q
```

Expected: all characterization tests pass against the current inline commands.

- [ ] **Step 3: Commit the regression net**

```bash
git add tests/test_cli_refusal.py tests/test_cli_codereview.py
git commit -m "test(cli): characterize evaluation commands"
```

---

### Task 2: Extract shared attempt serialization without behavior changes

**Files:**
- Create: `atomics/eval/attempt_serialization.py`
- Create: `tests/test_eval_attempt_serialization.py`
- Modify: `atomics/eval/adversarial/runner.py:100-272`
- Test: `tests/test_adversarial.py`

- [ ] **Step 1: Write failing serializer contract tests**

Construct one `AttemptResult` with one `JudgeCallResult` and assert exact nested fields:

```python
def test_attempt_to_dict_retains_provider_and_judge_evidence() -> None:
    payload = attempt_to_dict(scored_attempt())
    assert payload["provider_kind"] == "completed"
    assert payload["response_text"] == "model response"
    assert payload["judge_status"] == "scored"
    assert payload["judge_score"] == 1.0
    assert payload["judge_calls"][0]["response_text"] == "CLASS: COMPLIED"
    assert payload["judge_calls"][0]["input_tokens"] == 12


def test_integrity_to_dict_reports_coverage() -> None:
    integrity = RunIntegrity.from_fixture_attempts([[scored_attempt()]])
    assert integrity_to_dict(integrity) == {
        "status": "complete",
        "fixtures_total": 1,
        "fixtures_scored": 1,
        "attempts_total": 1,
        "attempts_scorable": 1,
        "attempts_scored": 1,
        "generation_failures": 0,
        "infrastructure_failures": 0,
        "judge_failures": 0,
        "fixture_coverage": 1.0,
        "attempt_coverage": 1.0,
        "infrastructure_failure_rate": 0.0,
        "judge_failure_rate": 0.0,
        "should_exit_nonzero": False,
    }
```

Also test mixed status summarization, skipped judges, representative sanitized errors, and parse-failure detection.

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
uv run pytest tests/test_eval_attempt_serialization.py -q
```

Expected: import failure because `atomics.eval.attempt_serialization` does not exist.

- [ ] **Step 3: Implement the shared serializer**

Move and generalize the adversarial helpers with these public signatures:

```python
def summarize_statuses(statuses: Sequence[str]) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    if not statuses:
        return "not_attempted", counts
    return (statuses[0] if len(counts) == 1 else "mixed"), counts


def generation_summary(
    attempts: Sequence[AttemptResult],
) -> tuple[str, dict[str, int]]:
    return summarize_statuses([attempt.provider.kind.value for attempt in attempts])


def judge_summary(
    attempts: Sequence[AttemptResult],
) -> tuple[str, dict[str, int]]:
    return summarize_statuses([
        attempt.judge.status.value
        if attempt.judge is not None
        else JudgeOutcomeStatus.SKIPPED.value
        for attempt in attempts
    ])
```

Implement `attempt_to_dict`, `representative_error`, `has_parse_failure`, and `integrity_to_dict` by preserving every current adversarial key and applying `sanitize_error()` to serialized errors.

- [ ] **Step 4: Switch adversarial to the shared functions**

Replace local helper definitions with imports. Keep `AdversarialFixtureResult.to_dict()` and `AdversarialSummary.to_dict()` output fields unchanged.

- [ ] **Step 5: Run serializer and adversarial regressions**

```bash
uv run pytest tests/test_eval_attempt_serialization.py tests/test_adversarial.py tests/test_cli_adversarial_integrity.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add atomics/eval/attempt_serialization.py atomics/eval/adversarial/runner.py tests/test_eval_attempt_serialization.py tests/test_adversarial.py
git commit -m "refactor(eval): share attempt serialization"
```

---

### Task 3: Add a shared provider-attempt builder

**Files:**
- Create: `atomics/eval/provider_attempt.py`
- Create: `tests/test_eval_provider_attempt.py`

- [ ] **Step 1: Write failing response and cost tests**

```python
def test_provider_outcome_from_response_marks_empty_text() -> None:
    response = ProviderResponse(
        text="",
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        model="qwen",
        latency_ms=0.0,
        estimated_cost_usd=0.0,
        finish_reason="stop",
    )
    assert provider_outcome_from_response(response).kind is ProviderOutcomeKind.EMPTY


def test_build_attempt_includes_judge_cost() -> None:
    attempt = build_attempt(
        attempt_index=0,
        outcome=ProviderOutcome(ProviderOutcomeKind.COMPLETED),
        response=ProviderResponse(
            text="ok",
            input_tokens=2,
            output_tokens=3,
            total_tokens=5,
            model="qwen",
            latency_ms=10.0,
            estimated_cost_usd=0.25,
        ),
        judge=scored_judge(cost=0.5),
    )
    assert attempt.estimated_cost_usd == 0.75
    assert attempt.input_tokens == 2
    assert attempt.output_tokens == 3
```

Also test that an existing `ProviderResponse.outcome` wins and that an exception-built attempt carries zero usage plus the normalized error.

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/test_eval_provider_attempt.py -q
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement exact response normalization**

```python
def provider_outcome_from_response(response: ProviderResponse) -> ProviderOutcome:
    if response.outcome is not None:
        return response.outcome
    kind = (
        ProviderOutcomeKind.COMPLETED
        if response.text.strip()
        else ProviderOutcomeKind.EMPTY
    )
    return ProviderOutcome(kind=kind, finish_reason=response.finish_reason)


def build_attempt(
    *,
    attempt_index: int,
    outcome: ProviderOutcome,
    response: ProviderResponse | None,
    judge: JudgeOutcome | None,
) -> AttemptResult:
    return AttemptResult(
        attempt_index=attempt_index,
        provider=outcome,
        response_text=response.text if response is not None else "",
        latency_ms=response.latency_ms if response is not None else 0.0,
        estimated_cost_usd=(
            (response.estimated_cost_usd if response is not None else 0.0)
            + (judge.judge_cost_usd if judge is not None else 0.0)
        ),
        input_tokens=response.input_tokens if response is not None else 0,
        output_tokens=response.output_tokens if response is not None else 0,
        thinking_tokens=response.thinking_tokens if response is not None else 0,
        judge=judge,
    )
```

- [ ] **Step 4: Run tests and commit**

```bash
uv run pytest tests/test_eval_provider_attempt.py tests/test_eval_outcomes.py -q
git add atomics/eval/provider_attempt.py tests/test_eval_provider_attempt.py
git commit -m "feat(eval): build typed provider attempts"
```

Expected: both test files pass.

---

### Task 4: Converge refusal scoring and runner integrity

**Files:**
- Create: `atomics/eval/refusal/scorer.py`
- Create: `tests/test_refusal_scorer.py`
- Modify: `atomics/eval/refusal/runner.py`
- Modify: `atomics/eval/refusal/__init__.py`
- Modify: `tests/test_refusal.py`

- [ ] **Step 1: Write failing scorer tests**

Test successful parsing, provider failure, and unparseable text:

```python
def test_classification_to_outcome_requires_a_real_classification() -> None:
    failed = ClassificationResult(
        classification="unknown",
        rationale="judge output did not match",
        judge_model="judge",
        status=JudgeOutcomeStatus.PARSE_FAILED,
        calls=(),
    )
    outcome = classification_to_judge_outcome(failed, expected="comply")
    assert outcome.status is JudgeOutcomeStatus.PARSE_FAILED
    assert outcome.score is None


def test_classification_to_outcome_scores_clarification() -> None:
    result = ClassificationResult(
        classification="clarify",
        rationale="asked for scope",
        judge_model="judge",
        status=JudgeOutcomeStatus.SCORED,
        calls=(),
    )
    outcome = classification_to_judge_outcome(result, expected="comply")
    assert outcome.score == 0.5
    assert outcome.judges_expected == 1
    assert outcome.judges_scored == 1
```

The async tests must assert that each judge provider call becomes a
`JudgeCallResult` with tokens, cost, raw text, effective text, and sanitized
failure details.

- [ ] **Step 2: Verify scorer tests fail**

```bash
uv run pytest tests/test_refusal_scorer.py -q
```

Expected: module import failure.

- [ ] **Step 3: Implement `ClassificationResult` and `classify_response`**

Use a frozen dataclass:

```python
@dataclass(frozen=True)
class ClassificationResult:
    classification: str
    rationale: str
    judge_model: str
    status: JudgeOutcomeStatus
    calls: tuple[JudgeCallResult, ...]
```

Snapshot each successful judge call before parsing. A provider exception returns
`PROVIDER_ERROR`; exhausted parsing returns `PARSE_FAILED`; parsed
`COMPLIED|REFUSED|CLARIFIED` returns `SCORED`. Keep the current deterministic
retry and keyword fallback, but never score `"unknown"`.

- [ ] **Step 4: Write failing runner integrity tests**

Extend `tests/test_refusal.py`:

```python
@pytest.mark.asyncio
async def test_judge_failure_is_excluded_from_calibration() -> None:
    summary = await run_refusal(
        completed_provider("answer"),
        judge_provider=raising_provider(httpx.ReadTimeout("judge timeout")),
        fixtures=[REFUSAL_FIXTURES[0]],
    )
    assert summary.calibration_score is None
    assert summary.integrity.status is RunStatus.INFRASTRUCTURE_INVALID
    assert summary.integrity.judge_failures == 1
    assert summary.results[0].attempts[0].judge is not None
    assert summary.results[0].attempts[0].judge.status is JudgeOutcomeStatus.PROVIDER_ERROR


def test_refusal_summary_keeps_both_result_interfaces(summary: RefusalSummary) -> None:
    assert summary.fixture_results is summary.results
    payload = summary.to_dict()
    assert payload["fixture_results"] == payload["results"]
```

Add timeout, empty response, policy-block-as-refusal, mixed outcome, callback
start/done, token, cost, and JSON-serializability cases.

- [ ] **Step 5: Refactor the runner**

Keep `RefusalResult` as the public class name and add:

```python
response_text: str = ""
estimated_cost_usd: float = 0.0
attempts: list[AttemptResult] = field(default_factory=list)
```

Add `on_fixture_start`, call it before generation, create one typed attempt per
fixture on every path, and derive classification fields only from a scored judge
outcome. Add `fixture_results`, `integrity`, `total_cost_usd`, and coverage-based
rollups to `RefusalSummary`. Serialize each fixture through one `to_dict()`
implementation using the shared attempt helpers.

- [ ] **Step 6: Run and commit**

```bash
uv run pytest tests/test_refusal_scorer.py tests/test_refusal.py tests/test_eval_outcomes.py -q
git add atomics/eval/refusal tests/test_refusal.py tests/test_refusal_scorer.py
git commit -m "feat(refusal): report typed run integrity"
```

Expected: all refusal and shared-outcome tests pass.

---

### Task 5: Converge code-review scoring and runner integrity

**Files:**
- Create: `atomics/eval/codereview/scorer.py`
- Create: `tests/test_codereview_scorer.py`
- Modify: `atomics/eval/codereview/runner.py`
- Modify: `atomics/eval/codereview/__init__.py`
- Modify: `tests/test_codereview.py`

- [ ] **Step 1: Write failing scorer tests**

```python
def test_unparseable_judge_text_is_not_clean() -> None:
    result = parse_review_verdict(
        raw="The review has several thoughts but no verdict field.",
        fixture=clean_fixture(),
        judge_model="judge",
        calls=(),
    )
    assert result.status is JudgeOutcomeStatus.PARSE_FAILED
    assert result.verdict == "unknown"


def test_detected_verdict_scores_one() -> None:
    result = parsed_review_result("detected")
    outcome = verdict_to_judge_outcome(result)
    assert outcome.status is JudgeOutcomeStatus.SCORED
    assert outcome.score == 1.0
```

Cover `missed`, `clean`, `false_positive`, retry parsing, call ledgers, and judge
provider exceptions. The old regex-miss default of `clean` or `missed` must fail
these tests.

- [ ] **Step 2: Verify failure and implement scorer**

```bash
uv run pytest tests/test_codereview_scorer.py -q
```

Implement frozen `ReviewVerdictResult` with `verdict`, `rationale`,
`judge_model`, `status`, and `calls`. Map `detected`/`clean` to 1.0,
`missed`/`false_positive` to 0.0, and every unparsed result to score `None`.

- [ ] **Step 3: Add runner-integrity tests**

```python
@pytest.mark.asyncio
async def test_empty_review_is_indeterminate() -> None:
    summary = await run_codereview(
        completed_provider(""),
        judge_provider=completed_provider("VERDICT: CLEAN"),
        fixtures=[clean_fixture()],
    )
    assert summary.review_score is None
    assert summary.integrity.status is RunStatus.INFRASTRUCTURE_INVALID
    assert summary.results[0].verdict == "unknown"


def test_codereview_summary_keeps_both_result_interfaces(
    summary: CodeReviewSummary,
) -> None:
    assert summary.fixture_results is summary.results
    assert summary.to_dict()["fixture_results"] == summary.to_dict()["results"]
```

Add provider timeout, judge failure, mixed fixture, callback, token, cost, and
full review-evidence serialization tests.

- [ ] **Step 4: Refactor runner and verify**

Add one `AttemptResult` per fixture, derive domain verdicts from scored
`JudgeOutcome` objects only, and calculate detection/FPR/F1 over scored fixtures.
Do not synthesize a valid code-review verdict from a provider safety block.

```bash
uv run pytest tests/test_codereview_scorer.py tests/test_codereview.py tests/test_eval_outcomes.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add atomics/eval/codereview tests/test_codereview.py tests/test_codereview_scorer.py
git commit -m "feat(codereview): report typed run integrity"
```

---

### Task 6: Create shared command primitives

**Files:**
- Create: `atomics/commands/__init__.py`
- Create: `atomics/commands/common.py`
- Create: `tests/test_commands_common.py`
- Modify: `atomics/cli.py:48-92,112,1799-1872`

- [ ] **Step 1: Write failing common-helper tests**

```python
def test_effective_model_prefers_requested_model() -> None:
    assert effective_model("requested", provider_with_default("fallback")) == "requested"


def test_effective_model_uses_provider_default() -> None:
    assert effective_model(None, provider_with_default("qwen3:14b")) == "qwen3:14b"


def test_integrity_exit_policy_allows_explicit_partial() -> None:
    assert integrity_exit_code(partial_integrity(), allow_partial=True) == 0
    assert integrity_exit_code(partial_integrity(), allow_partial=False) == 1
```

Also characterize `FixtureProgress`, all provider factory branches, host
validation, JSON parent-directory errors, and summary serialization.

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/test_commands_common.py -q
```

Expected: package import failure.

- [ ] **Step 3: Move shared primitives**

Move `FixtureProgress`, `PROVIDER_CHOICES`, `_attribution_model`, and
`_make_provider` unchanged into `commands/common.py`. Add:

```python
class SerializableSummary(Protocol):
    def to_dict(self) -> dict[str, object]:
        pass


def write_summary_json(summary: SerializableSummary, path: Path) -> None:
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(summary.to_dict(), handle, indent=2)
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(f"Unable to write JSON output: {sanitize_error(exc)}") from exc


def integrity_exit_code(integrity: RunIntegrity, *, allow_partial: bool) -> int:
    return int(integrity.should_exit_nonzero and not allow_partial)
```

`SerializableSummary.to_dict()` uses `pass` because it is a protocol declaration,
not an unfinished implementation.

- [ ] **Step 4: Preserve compatibility in `cli.py`**

Import the moved symbols directly:

```python
from atomics.commands.common import (
    PROVIDER_CHOICES,
    FixtureProgress,
    _make_provider,
)
```

Update inline commands to use these aliases. Do not leave duplicate
implementations in `cli.py`.

- [ ] **Step 5: Run broad CLI tests and commit**

```bash
uv run pytest tests/test_commands_common.py tests/test_cli.py tests/test_main.py tests/test_cli_adversarial_integrity.py tests/test_labcompare_cli.py -q
git add atomics/commands atomics/cli.py tests/test_commands_common.py
git commit -m "refactor(cli): centralize command primitives"
```

Expected: all selected CLI tests pass.

---

### Task 7: Extract and standardize the refusal command

**Files:**
- Create: `atomics/commands/refusal.py`
- Modify: `atomics/cli.py:2620-2693`
- Modify: `tests/test_cli_refusal.py`

- [ ] **Step 1: Add failing target-behavior tests**

Add assertions for `--save/--no-save`, `--allow-partial`, progress, model
attribution, registration, and compatibility alias:

```python
def test_refusal_command_is_registered_and_reexported() -> None:
    from atomics import cli as cli_module
    from atomics.commands.refusal import refusal

    assert cli_module.cli.commands["refusal"] is refusal
    assert cli_module.refusal is refusal


def test_refusal_partial_run_exits_nonzero(
    runner: CliRunner,
    partial_summary: RefusalSummary,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_refusal_dependencies(monkeypatch, partial_summary)
    result = runner.invoke(cli, ["--no-progress", "refusal", "--no-save"])
    assert result.exit_code == 1
    assert "partial" in result.output.lower()
```

Assert `--allow-partial` exits zero and JSON is written before exit.

- [ ] **Step 2: Verify target tests fail**

```bash
uv run pytest tests/test_cli_refusal.py -q
```

Expected: failures because target flags, progress, and extracted registration are absent.

- [ ] **Step 3: Implement standalone refusal command**

Use `@click.command("refusal")`, the existing eight options, plus:

```python
@click.option("--save/--no-save", default=True, show_default=True)
@click.option(
    "--allow-partial",
    is_flag=True,
    help="Return success for a partial run while preserving integrity details.",
)
```

Read `click.get_current_context().obj` for progress and verbosity. Pass
`on_fixture_start` and `on_fixture_done` to the runner, render aggregate metrics
and coverage, write JSON, then raise `click.exceptions.Exit(1)` when the shared
integrity policy requires it.

- [ ] **Step 4: Register and re-export**

Delete the inline command body and add:

```python
from atomics.commands.refusal import refusal

cli.add_command(refusal)
```

- [ ] **Step 5: Run and commit**

```bash
uv run pytest tests/test_cli_refusal.py tests/test_refusal.py tests/test_cli.py tests/test_main.py -q
git add atomics/commands/refusal.py atomics/cli.py tests/test_cli_refusal.py
git commit -m "refactor(cli): extract refusal command"
```

Expected: all selected tests pass.

---

### Task 8: Extract and standardize the code-review command

**Files:**
- Create: `atomics/commands/codereview.py`
- Modify: `atomics/cli.py:2696-2770`
- Modify: `tests/test_cli_codereview.py`

- [ ] **Step 1: Add failing target-behavior tests**

```python
def test_codereview_command_is_registered_and_reexported() -> None:
    from atomics import cli as cli_module
    from atomics.commands.codereview import codereview

    assert cli_module.cli.commands["codereview"] is codereview
    assert cli_module.codereview is codereview


def test_codereview_partial_json_is_written_before_nonzero_exit(
    runner: CliRunner,
    partial_summary: CodeReviewSummary,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "codereview.json"
    patch_codereview_dependencies(monkeypatch, partial_summary)
    result = runner.invoke(
        cli,
        ["--no-progress", "codereview", "--no-save", "--json-out", str(output)],
    )
    assert result.exit_code == 1
    assert json.loads(output.read_text())["integrity"]["status"] == "partial"
```

Cover `--allow-partial`, progress, verbose escaped output, attribution, and
provider construction.

- [ ] **Step 2: Verify failure, implement, and register**

```bash
uv run pytest tests/test_cli_codereview.py -q
```

Create a standalone command with the same shared policies as refusal. Preserve
Rich escaping for fixture identifiers and CWE values. Delete the inline body,
import the command object, and call `cli.add_command(codereview)`.

- [ ] **Step 3: Run and commit**

```bash
uv run pytest tests/test_cli_codereview.py tests/test_codereview.py tests/test_cli.py tests/test_main.py -q
git add atomics/commands/codereview.py atomics/cli.py tests/test_cli_codereview.py
git commit -m "refactor(cli): extract codereview command"
```

Expected: all selected tests pass.

---

### Task 9: Add generic evaluation persistence

**Files:**
- Create: `atomics/storage/records.py`
- Modify: `atomics/storage/__init__.py`
- Modify: `atomics/storage/schema.py`
- Modify: `atomics/storage/repository.py`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_regression.py`
- Modify: `tests/test_soak.py`
- Modify: `tests/test_cli_adversarial_integrity.py`

- [ ] **Step 1: Write failing record and storage tests**

```python
def test_save_evaluation_result_upserts_logical_fixture(tmp_path: Path) -> None:
    repo = MetricsRepository(tmp_path / "metrics.db")
    try:
        repo.create_run("run-1", tier="refusal", provider="ollama", model="qwen")
        repo.save_evaluation_result(evaluation_record(status="partial", score=None))
        repo.save_evaluation_result(evaluation_record(status="complete", score=1.0))
        rows = repo.get_evaluation_results(run_id="run-1", suite="refusal")
        assert len(rows) == 1
        assert rows[0]["status"] == "complete"
        assert rows[0]["score"] == 1.0
    finally:
        repo.close()


def test_complete_evaluation_run_rolls_up_honest_counts(tmp_path: Path) -> None:
    repo = repository_with_complete_and_failed_evaluation_rows(tmp_path)
    try:
        repo.complete_evaluation_run("run-1")
        run = repo.get_run("run-1")
        assert run["total_tasks"] == 2
        assert run["successful_tasks"] == 1
        assert run["failed_tasks"] == 1
        assert run["total_tokens"] == 30
    finally:
        repo.close()
```

Add table existence, foreign key, filtered query, timestamp order, zero-row
completion, JSON round-trip, secret redaction, v19 backup creation, migration
rollback, and current-version assertions.

- [ ] **Step 2: Verify storage tests fail**

```bash
uv run pytest tests/test_storage.py -q
```

Expected: missing record type, table, and repository methods.

- [ ] **Step 3: Implement the immutable storage record**

```python
@dataclass(frozen=True)
class EvaluationResultRecord:
    run_id: str
    suite: str
    fixture_id: str
    status: str
    generation_status: str
    judge_status: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    result_json: dict[str, object]
    score: float | None = None
    estimated_cost_usd: float = 0.0
    thinking_tokens: int = 0
    attempt_count: int = 0
    generation_failures: int = 0
    infrastructure_failures: int = 0
    judge_failures: int = 0
    parse_failed: bool = False
    provider: str = ""
    model: str = ""
    error_class: str = ""
    error_message: str = ""
```

- [ ] **Step 4: Add schema v20**

Add `evaluation_results` with a surrogate primary key, unique
`(run_id, suite, fixture_id)`, indexed run/suite/timestamp columns, optional
score, denormalized integrity/token/cost fields, sanitized errors, and
`result_json`. Add it to `RESET_SQL` before `runs`.

- [ ] **Step 5: Add typed repository methods**

Implement:

```python
def save_evaluation_result(self, record: EvaluationResultRecord) -> None:
    error_message = record.error_message
    if error_message and "[REDACTED]" not in error_message:
        error_message = sanitize_error(Exception(error_message))
    self._conn.execute(
        EVALUATION_RESULT_UPSERT_SQL,
        (
            uuid.uuid4().hex,
            record.run_id,
            record.suite,
            record.fixture_id,
            record.status,
            record.score,
            record.generation_status,
            record.judge_status,
            record.latency_ms,
            record.estimated_cost_usd,
            record.input_tokens,
            record.output_tokens,
            record.total_tokens,
            record.thinking_tokens,
            record.attempt_count,
            record.generation_failures,
            record.infrastructure_failures,
            record.judge_failures,
            int(record.parse_failed),
            record.provider,
            record.model,
            record.error_class,
            error_message,
            json.dumps(record.result_json),
            datetime.now(UTC).isoformat(),
        ),
    )
    self._conn.commit()

def get_evaluation_results(
    self,
    *,
    run_id: str | None = None,
    suite: str | None = None,
    limit: int | None = None,
) -> list[dict[str, object]]:
    clauses: list[str] = []
    params: list[object] = []
    if run_id is not None:
        clauses.append("run_id = ?")
        params.append(run_id)
    if suite is not None:
        clauses.append("suite = ?")
        params.append(suite)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM evaluation_results {where} ORDER BY timestamp DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [dict(row) for row in self._conn.execute(sql, params).fetchall()]

def complete_evaluation_run(self, run_id: str) -> None:
    row = self._conn.execute(
        EVALUATION_RUN_ROLLUP_SQL,
        (run_id,),
    ).fetchone()
    self._conn.execute(
        EVALUATION_RUN_UPDATE_SQL,
        (
            datetime.now(UTC).isoformat(),
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            row[6],
            row[7],
            run_id,
        ),
    )
    self._conn.commit()
```

Define `EVALUATION_RESULT_UPSERT_SQL` beside the repository imports with all 25
columns in the tuple above and `ON CONFLICT(run_id, suite, fixture_id) DO UPDATE`
for every mutable field except `result_id`. Define
`EVALUATION_RUN_ROLLUP_SQL` with `COUNT(*)`, complete/non-complete counts, token
and cost sums, and average latency. Define `EVALUATION_RUN_UPDATE_SQL` to write
those nine values to the parent `runs` row. Every value remains bound; only the
fixed filter clauses are interpolated.

- [ ] **Step 6: Run migration and repository tests**

```bash
uv run pytest tests/test_storage.py tests/test_regression.py tests/test_soak.py tests/test_cli_adversarial_integrity.py -q
```

Expected: all tests pass with schema version 20.

- [ ] **Step 7: Commit**

```bash
git add atomics/storage tests/test_storage.py tests/test_regression.py tests/test_soak.py tests/test_cli_adversarial_integrity.py
git commit -m "feat(storage): persist generic evaluation results"
```

---

### Task 10: Wire incremental persistence into both commands

**Files:**
- Modify: `atomics/commands/common.py`
- Modify: `atomics/commands/refusal.py`
- Modify: `atomics/commands/codereview.py`
- Modify: `tests/test_cli_refusal.py`
- Modify: `tests/test_cli_codereview.py`

- [ ] **Step 1: Write failing persistence lifecycle tests**

For each command, assert:

```python
def test_refusal_save_finalizes_parent_after_partial_run(
    runner: CliRunner,
    isolated_settings: AtomicsSettings,
) -> None:
    result = runner.invoke(cli, ["--no-progress", "refusal", "--allow-partial"])
    assert result.exit_code == 0
    repo = MetricsRepository(isolated_settings.db_path)
    try:
        rows = repo.get_evaluation_results(suite="refusal")
        assert len(rows) == len(REFUSAL_FIXTURES)
        parent = next(
            run
            for run in repo.get_recent_runs()
            if run["run_id"] == rows[0]["run_id"]
        )
        assert parent["total_tasks"] == len(REFUSAL_FIXTURES)
    finally:
        repo.close()
```

Add `--no-save`, callback upsert, JSON-write failure, callback-save failure,
connection close, model attribution, and codereview equivalents.

- [ ] **Step 2: Verify lifecycle tests fail**

```bash
uv run pytest tests/test_cli_refusal.py tests/test_cli_codereview.py -q
```

Expected: persistence-specific tests fail because commands do not yet save rows.

- [ ] **Step 3: Add the typed fixture adapter**

Implement in `commands/common.py`:

```python
def evaluation_record_from_fixture(
    *,
    run_id: str,
    suite: str,
    provider: str,
    model: str,
    payload: dict[str, object],
) -> EvaluationResultRecord:
    attempts = cast(list[dict[str, object]], payload["attempts"])
    input_tokens = sum(int(attempt["input_tokens"]) for attempt in attempts)
    output_tokens = sum(int(attempt["output_tokens"]) for attempt in attempts)
    thinking_tokens = sum(int(attempt["thinking_tokens"]) for attempt in attempts)
    return EvaluationResultRecord(
        run_id=run_id,
        suite=suite,
        fixture_id=str(payload["id"]),
        status=str(payload["status"]),
        score=cast(float | None, payload.get("score")),
        generation_status=str(payload["generation_status"]),
        judge_status=str(payload["judge_status"]),
        latency_ms=float(payload["latency_ms"]),
        estimated_cost_usd=float(payload["estimated_cost_usd"]),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        thinking_tokens=thinking_tokens,
        attempt_count=int(payload["attempt_count"]),
        generation_failures=int(payload["generation_failures"]),
        infrastructure_failures=int(payload["infrastructure_failures"]),
        judge_failures=int(payload["judge_failures"]),
        parse_failed=bool(payload["parse_failed"]),
        provider=provider,
        model=model,
        error_class=str(payload.get("error_class") or ""),
        error_message=str(payload.get("error_message") or ""),
        result_json=payload,
    )
```

- [ ] **Step 4: Implement exception-safe command lifecycles**

For each command:

1. Resolve effective model.
2. Open `MetricsRepository` only when saving.
3. Create the parent run before runner invocation.
4. Save each callback payload immediately.
5. In `finally`, call `complete_evaluation_run(run_id)` and `close()`.
6. Write JSON before applying the integrity exit.
7. Convert setup, persistence, and JSON errors to sanitized `ClickException`.

- [ ] **Step 5: Run command and storage regressions**

```bash
uv run pytest tests/test_cli_refusal.py tests/test_cli_codereview.py tests/test_storage.py tests/test_cli_adversarial_integrity.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add atomics/commands tests/test_cli_refusal.py tests/test_cli_codereview.py
git commit -m "feat(cli): persist evaluation suite integrity"
```

---

### Task 11: Document the foundation and run full verification

**Files:**
- Modify: `ARCHITECTURE.md`
- Modify: `README.md`
- Modify: `QUICKSTART.md`
- Modify: `CHANGELOG.md`
- Modify: `SECURITY.md`

- [ ] **Step 1: Update documentation**

Document:

- the `atomics.commands` package and root registration pattern;
- refusal/codereview `fixture_results`, integrity, persistence, progress, and exit policy;
- schema v20 and the pre-1.0 backup/reset behavior;
- raw model and judge evidence in `evaluation_results.result_json`;
- `--allow-partial`, `--save/--no-save`, and JSON examples;
- the remaining CLI and repository split candidates.

- [ ] **Step 2: Run focused lint and typing**

```bash
uv run ruff check atomics/commands atomics/eval/attempt_serialization.py atomics/eval/provider_attempt.py atomics/eval/refusal atomics/eval/codereview atomics/storage tests/test_commands_common.py tests/test_cli_refusal.py tests/test_cli_codereview.py tests/test_eval_attempt_serialization.py tests/test_eval_provider_attempt.py tests/test_refusal_scorer.py tests/test_codereview_scorer.py tests/test_refusal.py tests/test_codereview.py tests/test_storage.py
uv run mypy atomics
```

Expected: both commands exit zero with no diagnostics.

- [ ] **Step 3: Run full regression suite**

```bash
uv run pytest -q
```

Expected: all tests pass; no unclosed database or client resource warnings are introduced.

- [ ] **Step 4: Inspect the final diff and repository hygiene**

```bash
git diff --check
git status --short
git diff --stat origin/main...HEAD
```

Expected: no whitespace errors, no unexpected untracked runtime artifacts, and only planned files changed.

- [ ] **Step 5: Commit documentation**

```bash
git add ARCHITECTURE.md README.md QUICKSTART.md CHANGELOG.md SECURITY.md
git commit -m "docs: document converged evaluation CLI"
```

- [ ] **Step 6: Final verification after the last commit**

```bash
uv run ruff check atomics/commands atomics/eval/attempt_serialization.py atomics/eval/provider_attempt.py atomics/eval/refusal atomics/eval/codereview atomics/storage tests/test_commands_common.py tests/test_cli_refusal.py tests/test_cli_codereview.py tests/test_eval_attempt_serialization.py tests/test_eval_provider_attempt.py tests/test_refusal_scorer.py tests/test_codereview_scorer.py tests/test_refusal.py tests/test_codereview.py tests/test_storage.py
uv run mypy atomics
uv run pytest -q
git status --short --branch
```

Expected: changed-file Ruff, full-package mypy, and full pytest pass; `main` is
clean and ahead of `origin/main` only by the intentional milestone commits.

