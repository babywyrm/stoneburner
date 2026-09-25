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
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atomics.providers.base import BaseProvider
from atomics.validation import sanitize_error

logger = logging.getLogger("atomics.gauntlet")

KNOWN_SUITES = ("eval", "redblue", "refusal", "toolcall", "codereview")

RunSuite = Callable[..., Awaitable["SuiteJobResult"]]


@dataclass
class SuiteJobResult:
    model: str
    suite: str
    ok: bool
    headline: float | None = None
    error: str | None = None
    tool_capable: bool | None = None
    exit_code: int = 0


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
    return f"{base} {metric}={result.headline:.3f}"


def format_headline_cell(result: SuiteJobResult) -> str:
    """Sweep table cell. Toolcall is named so a high rate is not a high score."""
    if result.headline is None:
        return "—"
    pct = f"{result.headline * 100:.1f}%"
    if result.suite == "toolcall":
        return f"dangerous {pct}"
    return pct


def write_status(path: Path, progress: GauntletProgress) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress.to_dict(), indent=2) + "\n", encoding="utf-8")


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

    for model in models:
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
    fixture_ids: list[str] | None = None,
    effort: str | None = None,
    reasoning_mode: str | None = None,
) -> RunSuite:
    """Build the in-process suite callback the CLI hands to `run_gauntlet`."""

    async def run_suite(
        *,
        model: str,
        suite: str,
        skip_incapable: bool,
    ) -> SuiteJobResult:
        provider = provider_factory(model)
        try:
            return await _dispatch_suite(
                suite=suite,
                provider=provider,
                judge_provider=judge_provider,
                model=model,
                judge_model=judge_model,
                runs=runs,
                thinking=thinking,
                thinking_budget=thinking_budget,
                fixture_ids=fixture_ids,
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

    return run_suite


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
    effort: str | None = None,
    reasoning_mode: str | None = None,
) -> SuiteJobResult:
    if suite == "eval":
        from atomics.benchmark.sweep import _filter_fixtures
        from atomics.eval.runner import run_eval

        eval_summary = await run_eval(
            provider,
            judge_provider=judge_provider,
            model=model,
            judge_model=judge_model,
            thinking=thinking,
            thinking_budget=thinking_budget,
            effort=effort,
            reasoning_mode=reasoning_mode,
            fixtures=_filter_fixtures(fixture_ids),
        )
        return SuiteJobResult(
            model=model,
            suite=suite,
            ok=eval_summary.overall_accuracy is not None,
            headline=eval_summary.overall_accuracy,
        )
    if suite == "redblue":
        from atomics.eval.redblue.runner import run_redblue

        redblue_summary = await run_redblue(
            provider,
            judge_provider=judge_provider,
            model=model,
            judge_model=judge_model,
            runs=runs,
            thinking=thinking,
            thinking_budget=thinking_budget,
            effort=effort,
            reasoning_mode=reasoning_mode,
        )
        return SuiteJobResult(
            model=model,
            suite=suite,
            ok=redblue_summary.overall_quality is not None,
            headline=redblue_summary.overall_quality,
        )
    if suite == "refusal":
        from atomics.eval.refusal import run_refusal

        refusal_summary = await run_refusal(
            provider,
            judge_provider=judge_provider,
            model=model,
            judge_model=judge_model,
            thinking=thinking,
            thinking_budget=thinking_budget,
            effort=effort,
            reasoning_mode=reasoning_mode,
        )
        return SuiteJobResult(
            model=model,
            suite=suite,
            ok=refusal_summary.calibration_score is not None,
            headline=refusal_summary.calibration_score,
        )
    if suite == "toolcall":
        from atomics.eval.toolcall.fixtures import ALL_FIXTURES
        from atomics.eval.toolcall.runner import run_toolcall_suite

        toolcall_summary = await run_toolcall_suite(
            provider=provider,
            model=model,
            judge_provider=judge_provider,
            judge_model=judge_model,
            fixtures=ALL_FIXTURES,
            runs=runs,
            thinking=thinking,
            thinking_budget=thinking_budget,
            effort=effort,
            reasoning_mode=reasoning_mode,
        )
        if not toolcall_summary.tool_capable:
            return SuiteJobResult(
                model=model,
                suite=suite,
                ok=skip_incapable,
                tool_capable=False,
                error="model did not emit a tool call",
                exit_code=0 if skip_incapable else 1,
            )
        return SuiteJobResult(
            model=model,
            suite=suite,
            ok=True,
            tool_capable=True,
            headline=toolcall_summary.dangerous_call_rate,
        )
    if suite == "codereview":
        from atomics.eval.codereview import run_codereview

        review_summary = await run_codereview(
            provider,
            judge_provider=judge_provider,
            model=model,
            judge_model=judge_model,
            thinking=thinking,
            thinking_budget=thinking_budget,
            effort=effort,
            reasoning_mode=reasoning_mode,
        )
        return SuiteJobResult(
            model=model,
            suite=suite,
            ok=review_summary.review_score is not None,
            headline=review_summary.review_score,
        )
    raise ValueError(f"unknown suite: {suite}")
