"""Multi-suite overnight sweep driver.

Today's `atomics sweep` is one eval family. The 08-14/15 night was a shell
script in `/tmp` that died twice to SIGPIPE because stdout was still the chat.
This module is the in-process loop: models × suites, a status file you can
`cat` after the terminal is gone, and a log that is not a pipe.
"""

from __future__ import annotations

import json
import logging
import signal
import statistics
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from atomics.eval.fixtures import EVAL_FIXTURES
from atomics.eval.intervals import paired_interval
from atomics.eval.outcomes import RunStatus
from atomics.eval.runner import EvalRunSummary
from atomics.eval.suite_integrity import headline_rate
from atomics.providers.base import BaseProvider
from atomics.validation import sanitize_error

logger = logging.getLogger("atomics.gauntlet")

KNOWN_SUITES = ("eval", "adversarial", "redblue", "refusal", "toolcall", "codereview")
CATEGORY_SUITES = frozenset({"adversarial", "toolcall", "redblue"})


class CategoryFixture(Protocol):
    """Fixture row exposing an ``id`` string for category and ID subsetting."""

    @property
    def id(self) -> str: ...


_F = TypeVar("_F", bound=CategoryFixture)


def category_catalog_for_suite(
    suite: str, parts: Sequence[str]
) -> tuple[list[CategoryFixture], set[str]]:
    """Return ``(matching_fixtures, valid_category_tokens)`` for a suite."""
    if suite == "adversarial":
        from atomics.eval.adversarial import ALL_FIXTURES as ADV_ALL
        from atomics.eval.adversarial import GROUP_ALIASES as ADV_ALIASES
        from atomics.eval.adversarial import select_fixtures as select_adversarial

        valid = set(ADV_ALIASES) | {f.category for f in ADV_ALL}
        return list(select_adversarial(parts)), valid
    if suite == "toolcall":
        from atomics.eval.toolcall.fixtures import (
            ALL_FIXTURES as TC_ALL,
        )
        from atomics.eval.toolcall.fixtures import (
            GROUP_ALIASES as TC_ALIASES,
        )
        from atomics.eval.toolcall.fixtures import (
            fixtures_for_category,
        )

        valid = set(TC_ALIASES) | {f.category for f in TC_ALL}
        return list(fixtures_for_category(parts)), valid
    if suite == "redblue":
        from atomics.eval.redblue.fixtures import ALL_FIXTURES as RB_ALL

        valid = {"red", "blue", "all"} | {f.category for f in RB_ALL}
        wanted = set(parts)
        if "all" in wanted:
            return list(RB_ALL), valid
        return [f for f in RB_ALL if f.team in wanted or f.category in wanted], valid
    return [], set()


def _full_catalog_for_suite(suite: str) -> Sequence[CategoryFixture]:
    if suite == "eval":
        return EVAL_FIXTURES
    if suite == "adversarial":
        from atomics.eval.adversarial import ALL_FIXTURES as ADV_ALL

        return ADV_ALL
    if suite == "redblue":
        from atomics.eval.redblue.fixtures import ALL_FIXTURES as RB_ALL

        return RB_ALL
    if suite == "refusal":
        from atomics.eval.refusal.fixtures import REFUSAL_FIXTURES

        return REFUSAL_FIXTURES
    if suite == "toolcall":
        from atomics.eval.toolcall.fixtures import ALL_FIXTURES as TC_ALL

        return TC_ALL
    if suite == "codereview":
        from atomics.eval.codereview.fixtures import SECURE_CODE_FIXTURES

        return SECURE_CODE_FIXTURES
    raise ValueError(f"unknown suite: {suite}")


def _subset_by_ids(catalog: Sequence[_F], fixture_ids: Sequence[str] | None) -> list[_F]:
    if fixture_ids is None:
        return list(catalog)
    cleaned = [fid.strip() for fid in fixture_ids if fid.strip()]
    if not cleaned:
        raise ValueError("fixtures must not be empty")
    by_id = {f.id: f for f in catalog}
    matched = [by_id[fid] for fid in cleaned if fid in by_id]
    if not matched:
        raise ValueError(f"unknown fixture IDs: {', '.join(cleaned)}")
    return matched


def validate_sweep_subset(
    suites: Sequence[str],
    *,
    category: str | None = None,
    fixture_ids: Sequence[str] | None = None,
) -> tuple[str | None, list[str] | None]:
    """Validate and normalize sweep ``category`` and ``fixture_ids`` across ``suites``."""
    parts: list[str] = []
    category_norm: str | None = None
    if category is not None:
        for suite in suites:
            if suite not in CATEGORY_SUITES:
                raise ValueError(
                    f"category is only valid for {sorted(CATEGORY_SUITES)}; "
                    f"{suite} does not support category filtering"
                )
        parts = [part.strip() for part in category.split(",") if part.strip()]
        if not parts:
            raise ValueError("category must not be empty")
        for suite in suites:
            _, valid = category_catalog_for_suite(suite, parts)
            unknown = [part for part in parts if part not in valid]
            if unknown:
                raise ValueError(
                    f"unknown {suite} category: {', '.join(unknown)}. "
                    f"Valid: {', '.join(sorted(valid))}"
                )
        category_norm = ",".join(parts)

    cleaned_ids: list[str] | None = None
    if fixture_ids is not None:
        cleaned_ids = [fid.strip() for fid in fixture_ids if fid.strip()]
        if not cleaned_ids:
            raise ValueError("fixtures must not be empty")
        by_suite: dict[str, set[str]] = {}
        for suite in suites:
            cat_rows = (
                category_catalog_for_suite(suite, parts)[0]
                if category_norm is not None
                else _full_catalog_for_suite(suite)
            )
            by_suite[suite] = {f.id for f in cat_rows}
        union_ids: set[str] = set().union(*by_suite.values())
        outside = [fid for fid in cleaned_ids if fid not in union_ids]
        if outside:
            if category_norm is not None:
                raise ValueError(
                    f"fixture IDs outside category {category_norm!r}: {', '.join(outside)}"
                )
            raise ValueError(f"unknown fixture IDs: {', '.join(outside)}")
        for suite in suites:
            if not any(fid in by_suite[suite] for fid in cleaned_ids):
                raise ValueError(f"no requested fixture IDs belong to suite {suite!r}")

    return category_norm, cleaned_ids


RunSuite = Callable[..., Awaitable["SuiteJobResult"]]
Persist = Callable[[str, str, str, object], None]
VsBest = Literal["best", "tied", "behind"]


@dataclass
class SuiteJobResult:
    model: str
    suite: str
    ok: bool
    headline: float | None = None
    error: str | None = None
    tool_capable: bool | None = None
    exit_code: int = 0
    stdev: float | None = None
    # Toolcall with runs > 1: share of fixtures dangerous on any run.
    any_run: float | None = None
    # Score per fixture id, for the paired comparison against the suite best.
    fixture_scores: dict[str, float] | None = None
    vs_best: VsBest | None = None
    # Paired-difference interval behind vs_best, in the leader's favor.
    vs_low: float | None = None
    vs_high: float | None = None


@dataclass
class ModelSweepResult:
    model: str
    fixtures_run: int
    overall_quality: float | None
    avg_latency_ms: float
    total_tokens: int
    total_cost_usd: float
    value_score: float | None
    eval_summary: EvalRunSummary | None
    error: str | None = None
    provider: str = ""


def _representative_error(summary: EvalRunSummary) -> str | None:
    """Most common per-fixture error message in a summary, or None."""
    msgs = [
        fr.task_result.error_message
        for fr in summary.fixture_results
        if fr.task_result.error_message
    ]
    if not msgs:
        return None
    return Counter(msgs).most_common(1)[0][0][:200]


class _RunScored(Protocol):
    @property
    def run_scores(self) -> list[float]: ...


class _MultiRun(Protocol):
    runs: int

    @property
    def results(self) -> Sequence[_RunScored]: ...


def run_mean_stdev(summary: _MultiRun) -> float | None:
    """Sample stdev of each run's mean over the same fixtures.

    This is how much the headline moves between passes. It is None unless
    every fixture scored every run, because unaligned runs are not
    comparable passes.
    """
    per_fixture = [result.run_scores for result in summary.results]
    if summary.runs < 2 or not per_fixture:
        return None
    if any(len(scores) != summary.runs for scores in per_fixture):
        return None
    means = [statistics.fmean(scores[k] for scores in per_fixture) for k in range(summary.runs)]
    return round(statistics.stdev(means), 3)


@dataclass
class GauntletProgress:
    started_at: str
    models: list[str]
    suites: list[str]
    current_model: str | None = None
    current_suite: str | None = None
    completed: list[dict[str, Any]] = field(default_factory=list)
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_suites(raw: str) -> list[str]:
    """Split a comma list into known suite names, preserving order."""
    parts = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("suites must not be empty")
    unknown = [part for part in parts if part not in KNOWN_SUITES]
    if unknown:
        raise ValueError(f"unknown suites: {', '.join(unknown)}")
    seen: list[str] = []
    for part in parts:
        if part not in seen:
            seen.append(part)
    return seen


def ignore_broken_pipe() -> None:
    """Keep going when the chat that launched the sweep is gone."""
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)


def format_job_log(result: SuiteJobResult) -> str:
    """One status line. Toolcall's number is a dangerous-call rate."""
    mark = "ok" if result.ok else "fail"
    base = f"{mark} {result.model} {result.suite}"
    if result.headline is None:
        if result.error:
            return f"{base} {result.error}"
        return base
    metric = "dangerous_call_rate" if result.suite == "toolcall" else "headline"
    line = f"{base} {metric}={result.headline:.3f}"
    if result.stdev is not None:
        line += f" stdev={result.stdev:.3f}"
    if result.any_run is not None:
        line += f" dangerous_any_run_rate={result.any_run:.3f}"
    return line


def format_headline_cell(result: SuiteJobResult) -> str:
    """Sweep table cell. Toolcall is named so a high rate is not a high score."""
    if result.headline is None:
        return "—"
    pct = f"{result.headline * 100:.1f}%"
    if result.stdev is not None:
        pct += f" ±{result.stdev * 100:.1f}"
    if result.suite == "toolcall":
        if result.any_run is not None:
            return f"dangerous {pct} · any run {result.any_run * 100:.1f}%"
        return f"dangerous {pct}"
    return pct


_LOWER_IS_BETTER = frozenset({"toolcall"})


def _ranked(job: SuiteJobResult) -> float | None:
    """The rate a repeated toolcall is gated on is any-run, not the modal headline."""
    return job.any_run if job.any_run is not None else job.headline


def _comparable(job: SuiteJobResult) -> bool:
    return job.ok and _ranked(job) is not None and job.fixture_scores is not None


def _better(job: SuiteJobResult, lead: SuiteJobResult) -> bool:
    score, lead_score = _ranked(job), _ranked(lead)
    if score is None or lead_score is None:
        return False
    if job.suite in _LOWER_IS_BETTER:
        return score < lead_score
    return score > lead_score


def _vs_best(
    job: SuiteJobResult, leader: SuiteJobResult | None
) -> tuple[VsBest | None, float | None, float | None]:
    if leader is None or not _comparable(job):
        return None, None, None
    if _ranked(job) == _ranked(leader):
        return "best", None, None
    if job.fixture_scores is None or leader.fixture_scores is None:
        return None, None, None
    sign = -1.0 if job.suite in _LOWER_IS_BETTER else 1.0
    shared = sorted(leader.fixture_scores.keys() & job.fixture_scores.keys())
    diffs = [sign * (leader.fixture_scores[k] - job.fixture_scores[k]) for k in shared]
    interval = paired_interval(diffs)
    if interval is None:
        return None, None, None
    mark: VsBest = "behind" if interval.low > 0 else "tied"
    return mark, interval.low, interval.high


def format_vs_best(result: SuiteJobResult) -> str:
    """Mark plus its interval, so two equal rates can still be told apart."""
    if result.vs_best is None:
        return "—"
    if result.vs_low is None or result.vs_high is None:
        return result.vs_best
    return f"{result.vs_best} {result.vs_low * 100:+.0f} to {result.vs_high * 100:+.0f}%"


def mark_vs_best(results: Sequence[SuiteJobResult]) -> list[SuiteJobResult]:
    """Mark each job against its suite's best headline. See docs/COMPARING.md#intervals."""
    leaders: dict[str, SuiteJobResult] = {}
    for job in filter(_comparable, results):
        lead = leaders.get(job.suite)
        if lead is None or _better(job, lead):
            leaders[job.suite] = job
    marked: list[SuiteJobResult] = []
    for job in results:
        vs_best, vs_low, vs_high = _vs_best(job, leaders.get(job.suite))
        marked.append(replace(job, vs_best=vs_best, vs_low=vs_low, vs_high=vs_high))
    return marked


def write_status(path: Path, progress: GauntletProgress) -> None:
    from atomics.paths import write_private_text

    write_private_text(path, json.dumps(progress.to_dict(), indent=2) + "\n", mkdir_parents=True)


def append_log(path: Path | None, line: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _pairs(models: Sequence[str], suites: Sequence[str]) -> list[tuple[str, str]]:
    return [(model, suite) for model in models for suite in suites]


def _resumable(
    status_path: Path | None,
) -> tuple[str, dict[tuple[str, str], SuiteJobResult]]:
    """Start time and ok jobs from an earlier status file, if there is one."""
    if status_path is None or not status_path.exists():
        return _now(), {}
    saved = json.loads(status_path.read_text(encoding="utf-8"))
    names = SuiteJobResult.__dataclass_fields__
    kept: dict[tuple[str, str], SuiteJobResult] = {}
    for entry in saved.get("completed") or []:
        job = SuiteJobResult(**{k: v for k, v in entry.items() if k in names})
        if job.ok:
            kept[(job.model, job.suite)] = job
    return str(saved.get("started_at") or _now()), kept


async def run_gauntlet(
    *,
    models: Sequence[str],
    suites: Sequence[str],
    run_suite: RunSuite,
    status_path: Path | None = None,
    log_path: Path | None = None,
    skip_incapable: bool = False,
    resume: bool = False,
) -> list[SuiteJobResult]:
    """Run each suite for each model, recording status after every job.

    With `resume`, jobs the status file already records as ok are kept and
    not re-run. Failed, crashed, and interrupted jobs run again.
    """
    started_at, prior = _resumable(status_path) if resume else (_now(), {})
    progress = GauntletProgress(
        started_at=started_at,
        models=list(models),
        suites=list(suites),
        completed=[asdict(prior[key]) for key in _pairs(models, suites) if key in prior],
    )
    results: list[SuiteJobResult] = []
    if status_path is not None:
        write_status(status_path, progress)
    append_log(log_path, f"started models={len(models)} suites={','.join(suites)}")
    if prior:
        append_log(log_path, f"resumed kept={len(progress.completed)}")

    stopped = False
    for model in models:
        if stopped:
            break
        for suite in suites:
            kept = prior.get((model, suite))
            if kept is not None:
                results.append(kept)
                continue
            progress.current_model = model
            progress.current_suite = suite
            if status_path is not None:
                write_status(status_path, progress)
            append_log(log_path, f"running {model} {suite}")
            result = await run_suite(
                model=model,
                suite=suite,
                skip_incapable=skip_incapable,
            )
            results.append(result)
            progress.completed.append(asdict(result))
            append_log(log_path, format_job_log(result))
            if result.error == "provider unreachable":
                append_log(log_path, "stopped provider unreachable")
                stopped = True
                break

    results = mark_vs_best(results)
    marked = {(job.model, job.suite): asdict(job) for job in results}
    progress.completed = [
        marked.get((row["model"], row["suite"]), row) for row in progress.completed
    ]
    progress.current_model = None
    progress.current_suite = None
    progress.finished_at = _now()
    if status_path is not None:
        write_status(status_path, progress)
    append_log(log_path, f"finished jobs={len(results)}")
    return results


def make_suite_runner(
    *,
    provider_factory: Callable[[str], BaseProvider],
    judge_provider: BaseProvider,
    judge_model: str | None,
    runs: int,
    thinking: bool | None,
    thinking_budget: int | None,
    temperature: float | None = None,
    fixture_ids: list[str] | None = None,
    category: str | None = None,
    effort: str | None = None,
    reasoning_mode: str | None = None,
    persist: Persist | None = None,
) -> RunSuite:
    """Build the in-process suite callback the CLI hands to `run_gauntlet`.

    `persist(suite, provider_name, model, summary)` runs after each suite
    that produced a summary. A save that raises fails the job rather than
    losing the night quietly.
    """

    async def run_suite(
        *,
        model: str,
        suite: str,
        skip_incapable: bool,
    ) -> SuiteJobResult:
        provider = provider_factory(model)
        try:
            result, summary = await _dispatch_suite(
                suite=suite,
                provider=provider,
                judge_provider=judge_provider,
                model=model,
                judge_model=judge_model,
                runs=runs,
                thinking=thinking,
                thinking_budget=thinking_budget,
                temperature=temperature,
                fixture_ids=fixture_ids,
                category=category,
                effort=effort,
                reasoning_mode=reasoning_mode,
                skip_incapable=skip_incapable,
            )
        except Exception as exc:
            err = sanitize_error(exc)[:200]
            logger.warning("[gauntlet] %s %s failed: %s", model, suite, err)
            return SuiteJobResult(
                model=model,
                suite=suite,
                ok=False,
                error=err,
                exit_code=1,
            )
        if persist is None or summary is None:
            return result
        try:
            persist(suite, provider.name, model, summary)
        except Exception as exc:
            err = sanitize_error(exc)[:200]
            logger.warning("[gauntlet] %s %s save failed: %s", model, suite, err)
            return replace(result, ok=False, error=f"save failed: {err}", exit_code=1)
        return result

    return run_suite


def _suite_job(
    *,
    model: str,
    suite: str,
    score: float | None,
    summary: object,
    stdev: float | None = None,
) -> SuiteJobResult:
    """Ok only when every fixture was scored.

    The summary average is over judges that parsed. A server that dies on
    the last fixture still has a number, and resume would keep it.
    """
    integrity = getattr(summary, "integrity", None)
    headline = score if integrity is None else headline_rate(score, integrity)
    ok = (not integrity.should_exit_nonzero) if integrity is not None else (headline is not None)
    error = None
    if integrity is not None and integrity.status is RunStatus.INFRASTRUCTURE_INVALID:
        error = "provider unreachable"
    elif integrity is not None and not ok:
        error = (
            f"{integrity.attempts_scored}/{integrity.attempts_total} attempts scored "
            f"({integrity.generation_failures} unanswered, "
            f"{integrity.judge_failures} unjudged)"
        )
    return SuiteJobResult(
        model=model,
        suite=suite,
        ok=ok,
        headline=headline,
        error=error,
        stdev=stdev,
        exit_code=0 if ok else 1,
        fixture_scores=_fixture_scores(summary) if ok else None,
    )


def _fixture_scores(summary: object, name: str = "fixture_scores") -> dict[str, float] | None:
    scores: dict[str, float] | None = getattr(summary, name, None)
    return scores


async def _dispatch_suite(
    *,
    suite: str,
    provider: BaseProvider,
    judge_provider: BaseProvider,
    model: str,
    judge_model: str | None,
    runs: int,
    thinking: bool | None,
    thinking_budget: int | None,
    fixture_ids: list[str] | None,
    skip_incapable: bool,
    category: str | None = None,
    temperature: float | None = None,
    effort: str | None = None,
    reasoning_mode: str | None = None,
    on_fixture_done: Callable[..., object] | None = None,
) -> tuple[SuiteJobResult, object | None]:
    cat_parts = (
        [part.strip() for part in category.split(",") if part.strip()]
        if category is not None
        else []
    )
    if suite == "eval":
        from atomics.eval.runner import run_eval

        eval_summary = await run_eval(
            provider,
            judge_provider=judge_provider,
            model=model,
            judge_model=judge_model,
            thinking=thinking,
            thinking_budget=thinking_budget,
            temperature=temperature,
            effort=effort,
            reasoning_mode=reasoning_mode,
            fixtures=_subset_by_ids(EVAL_FIXTURES, fixture_ids),
            on_fixture_done=on_fixture_done,
        )
        return SuiteJobResult(
            model=model,
            suite=suite,
            ok=eval_summary.overall_accuracy is not None,
            headline=eval_summary.overall_accuracy,
            error=(
                _representative_error(eval_summary)
                if eval_summary.overall_accuracy is None
                else None
            ),
            fixture_scores=_fixture_scores(eval_summary),
        ), eval_summary
    if suite == "adversarial":
        from atomics.eval.adversarial import ALL_FIXTURES as ADV_ALL
        from atomics.eval.adversarial import select_fixtures as select_adversarial
        from atomics.eval.adversarial.runner import run_adversarial

        adv_fixtures = None
        if cat_parts or fixture_ids is not None:
            adv_base = select_adversarial(cat_parts) if cat_parts else ADV_ALL
            adv_fixtures = _subset_by_ids(adv_base, fixture_ids)
        adv_summary = await run_adversarial(
            provider,
            judge_provider=judge_provider,
            model=model,
            judge_model=judge_model,
            fixtures=adv_fixtures,
            runs=runs,
            thinking=thinking,
            thinking_budget=thinking_budget,
            temperature=temperature,
            effort=effort,
            reasoning_mode=reasoning_mode,
        )
        return _suite_job(
            model=model,
            suite=suite,
            score=adv_summary.overall_resilience,
            summary=adv_summary,
            stdev=adv_summary.overall_resilience_stdev,
        ), adv_summary
    if suite == "redblue":
        from atomics.eval.redblue.fixtures import ALL_FIXTURES as RB_ALL
        from atomics.eval.redblue.runner import run_redblue

        rb_mode = (
            cat_parts[0]
            if len(cat_parts) == 1 and cat_parts[0] in {"red", "blue", "all"}
            else "all"
        )
        rb_fixtures = None
        if cat_parts or fixture_ids is not None:
            if cat_parts:
                wanted = set(cat_parts)
                rb_base = (
                    list(RB_ALL)
                    if "all" in wanted
                    else [f for f in RB_ALL if f.team in wanted or f.category in wanted]
                )
            else:
                rb_base = list(RB_ALL)
            rb_fixtures = _subset_by_ids(rb_base, fixture_ids)
        redblue_summary = await run_redblue(
            provider,
            judge_provider=judge_provider,
            mode=rb_mode,
            model=model,
            judge_model=judge_model,
            fixtures=rb_fixtures,
            runs=runs,
            thinking=thinking,
            thinking_budget=thinking_budget,
            temperature=temperature,
            effort=effort,
            reasoning_mode=reasoning_mode,
        )
        return _suite_job(
            model=model,
            suite=suite,
            score=redblue_summary.overall_quality,
            summary=redblue_summary,
            stdev=run_mean_stdev(redblue_summary),
        ), redblue_summary
    if suite == "refusal":
        from atomics.eval.refusal import run_refusal
        from atomics.eval.refusal.fixtures import REFUSAL_FIXTURES

        ref_fixtures = (
            _subset_by_ids(REFUSAL_FIXTURES, fixture_ids) if fixture_ids is not None else None
        )
        refusal_summary = await run_refusal(
            provider,
            judge_provider=judge_provider,
            model=model,
            judge_model=judge_model,
            fixtures=ref_fixtures,
            runs=runs,
            thinking=thinking,
            thinking_budget=thinking_budget,
            temperature=temperature,
            effort=effort,
            reasoning_mode=reasoning_mode,
        )
        return _suite_job(
            model=model,
            suite=suite,
            score=refusal_summary.calibration_score,
            summary=refusal_summary,
            stdev=run_mean_stdev(refusal_summary),
        ), refusal_summary
    if suite == "toolcall":
        from atomics.eval.toolcall.fixtures import ALL_FIXTURES as TC_ALL
        from atomics.eval.toolcall.fixtures import fixtures_for_category
        from atomics.eval.toolcall.runner import run_toolcall_suite

        tc_base = fixtures_for_category(cat_parts) if cat_parts else TC_ALL
        tc_fixtures = (
            tuple(_subset_by_ids(tc_base, fixture_ids)) if fixture_ids is not None else tc_base
        )
        toolcall_summary = await run_toolcall_suite(
            provider=provider,
            model=model,
            judge_provider=judge_provider,
            judge_model=judge_model,
            fixtures=tc_fixtures,
            runs=runs,
            thinking=thinking,
            thinking_budget=thinking_budget,
            temperature=temperature,
            effort=effort,
            reasoning_mode=reasoning_mode,
        )
        probe_error = getattr(toolcall_summary, "probe_error", None)
        if probe_error:
            return SuiteJobResult(
                model=model, suite=suite, ok=False, error=probe_error, exit_code=1
            ), None
        if not toolcall_summary.tool_capable:
            return SuiteJobResult(
                model=model,
                suite=suite,
                ok=skip_incapable,
                tool_capable=False,
                error="model did not emit a tool call",
                exit_code=0 if skip_incapable else 1,
            ), None
        tc_integrity = getattr(toolcall_summary, "integrity", None)
        if tc_integrity is not None and tc_integrity.should_exit_nonzero:
            tc_error = (
                "provider unreachable"
                if tc_integrity.status is RunStatus.INFRASTRUCTURE_INVALID
                else (
                    f"{tc_integrity.attempts_scored}/{tc_integrity.attempts_total} attempts scored "
                    f"({tc_integrity.generation_failures} unanswered, "
                    f"{tc_integrity.judge_failures} unjudged)"
                )
            )
            return SuiteJobResult(
                model=model,
                suite=suite,
                ok=False,
                tool_capable=True,
                error=tc_error,
                exit_code=1,
            ), toolcall_summary
        return SuiteJobResult(
            model=model,
            suite=suite,
            ok=True,
            tool_capable=True,
            headline=toolcall_summary.dangerous_call_rate,
            any_run=(
                toolcall_summary.dangerous_any_run_rate if toolcall_summary.runs > 1 else None
            ),
            fixture_scores=_fixture_scores(
                toolcall_summary,
                "any_run_fixture_scores" if toolcall_summary.runs > 1 else "fixture_scores",
            ),
        ), toolcall_summary
    if suite == "codereview":
        from atomics.eval.codereview import run_codereview
        from atomics.eval.codereview.fixtures import SECURE_CODE_FIXTURES

        cr_fixtures = (
            _subset_by_ids(SECURE_CODE_FIXTURES, fixture_ids) if fixture_ids is not None else None
        )
        review_summary = await run_codereview(
            provider,
            judge_provider=judge_provider,
            model=model,
            judge_model=judge_model,
            fixtures=cr_fixtures,
            runs=runs,
            thinking=thinking,
            thinking_budget=thinking_budget,
            temperature=temperature,
            effort=effort,
            reasoning_mode=reasoning_mode,
        )
        return _suite_job(
            model=model,
            suite=suite,
            score=review_summary.review_score,
            summary=review_summary,
            stdev=review_summary.review_score_stdev,
        ), review_summary
    raise ValueError(f"unknown suite: {suite}")


async def run_model_sweep(
    *,
    provider_factory: Callable[[str], BaseProvider],
    judge_provider: BaseProvider,
    models: list[str],
    fixture_ids: list[str] | None = None,
    judge_model: str | None = None,
    thinking: bool | None = None,
    thinking_budget: int | None = None,
    temperature: float | None = None,
    effort: str | None = None,
    reasoning_mode: str | None = None,
    on_model_done: Callable[[ModelSweepResult], None] | None = None,
    on_fixture_done: Callable[..., object] | None = None,
) -> list[ModelSweepResult]:
    """Run eval fixtures across a list of models via the unified gauntlet dispatcher."""
    results: list[ModelSweepResult] = []
    for model in models:
        provider = provider_factory(model)
        provider_name = str(getattr(provider, "name", "") or "")
        try:
            job, raw_summary = await _dispatch_suite(
                suite="eval",
                provider=provider,
                judge_provider=judge_provider,
                model=model,
                judge_model=judge_model,
                runs=1,
                thinking=thinking,
                thinking_budget=thinking_budget,
                fixture_ids=fixture_ids,
                skip_incapable=False,
                temperature=temperature,
                effort=effort,
                reasoning_mode=reasoning_mode,
                on_fixture_done=on_fixture_done,
            )
            assert isinstance(raw_summary, EvalRunSummary)
            result = ModelSweepResult(
                model=model,
                fixtures_run=len(raw_summary.fixture_results),
                overall_quality=raw_summary.overall_accuracy,
                avg_latency_ms=raw_summary.avg_latency_ms,
                total_tokens=raw_summary.total_tokens,
                total_cost_usd=raw_summary.total_cost_usd,
                value_score=raw_summary.value_score,
                eval_summary=raw_summary,
                error=job.error,
                provider=provider_name,
            )
        except Exception as exc:
            err = sanitize_error(exc)[:200]
            logger.warning("[gauntlet] Model %s failed: %s", model, err)
            result = ModelSweepResult(
                model=model,
                fixtures_run=0,
                overall_quality=None,
                avg_latency_ms=0.0,
                total_tokens=0,
                total_cost_usd=0.0,
                value_score=None,
                eval_summary=None,
                error=err,
                provider=provider_name,
            )
        results.append(result)
        if on_model_done is not None:
            on_model_done(result)
    return results
