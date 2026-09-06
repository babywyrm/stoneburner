"""Quiet wait view: one line per phase/fixture, a short completed headline.

The job JSON stays on GET /jobs. This is a REPL skin only. ANSI color is opt-in
so tests and redirected stdout stay plain.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_RESET = "\033[0m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_DIM = "\033[2m"


def _paint(text: str, code: str, *, color: bool) -> str:
    if not color:
        return text
    return f"{code}{text}{_RESET}"


def _score_color(score: float | None, *, failed: bool) -> str:
    if failed or score is None:
        return _RED
    if score < 0.5:
        return _RED
    if score < 0.8:
        return _YELLOW
    return _GREEN


def _seconds_label(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "?"
    if number == int(number):
        return f"{int(number)}s"
    return f"{number}s"


def _tps_label(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if number == int(number):
        return str(int(number))
    return f"{number:.1f}"


def format_in_flight(in_flight: dict[str, Any] | None, *, color: bool = False) -> str:
    if not in_flight:
        return ""
    if in_flight.get("task") and not in_flight.get("fixture_id"):
        task = str(in_flight.get("task") or "?")
        model = str(in_flight.get("model") or "")
        return f"  {task}  {model}".rstrip()
    if in_flight.get("suite") and not in_flight.get("fixture_id"):
        model = str(in_flight.get("model") or "?")
        suite = str(in_flight.get("suite") or "?")
        suite_txt = _paint(f"{suite:<8}", _DIM, color=color)
        return f"  {model}  {suite_txt}".rstrip()
    if "elapsed_seconds" in in_flight and not in_flight.get("fixture_id"):
        elapsed = _seconds_label(in_flight.get("elapsed_seconds"))
        conc = in_flight.get("concurrency")
        return f"  {elapsed}  c={conc}"
    if "concurrency" in in_flight and not in_flight.get("fixture_id"):
        conc = in_flight.get("concurrency")
        return f"  c={conc}  {_seconds_label(in_flight.get('phase_seconds'))}"
    fixture = str(in_flight.get("fixture_id") or "?")
    phase = str(in_flight.get("phase") or "?")
    model = str(in_flight.get("model") or "")
    phase_txt = _paint(f"{phase:<8}", _DIM, color=color)
    return f"  {fixture}  {phase_txt}  {model}".rstrip()


def format_task_row(row: dict[str, Any], *, color: bool = False, verbose: bool = False) -> str:
    name = str(row.get("id") or "?")
    status = str(row.get("status") or "")
    failed = status == "failed"
    tokens = int(row.get("tokens") or 0)
    status_txt = _paint(status, _RED if failed else _GREEN, color=color)
    line = f"  {name}  {status_txt}  {tokens} tok"
    if not verbose:
        return line
    latency = row.get("latency_ms")
    if latency is not None:
        try:
            line += f"  {int(round(float(latency)))}ms"
        except (TypeError, ValueError):
            pass
    error = row.get("error")
    if error:
        line += f"\n    error: {error}"
    return line


def format_phase_row(row: dict[str, Any], *, color: bool = False) -> str:
    conc = row.get("concurrency")
    tps = _tps_label(row.get("aggregate_tps"))
    reqs = row.get("requests") or 0
    return f"  c={conc}  {tps} tps  {reqs} req"


def format_sample_row(row: dict[str, Any], *, color: bool = False) -> str:
    elapsed = _seconds_label(row.get("elapsed_seconds"))
    tps = _tps_label(row.get("aggregate_tps"))
    reqs = row.get("requests") or 0
    return f"  {elapsed}  {tps} tps  {reqs} req"


def format_fixture_row(
    row: dict[str, Any], *, color: bool = False, verbose: bool = False
) -> str:
    fixture = str(row.get("id") or "?")
    status = str(row.get("status") or "")
    failed = status == "failed"
    raw = row.get("score")
    score: float | None
    try:
        score = None if raw is None else float(raw)
    except (TypeError, ValueError):
        score = None
    score_txt = "  - " if score is None else f"{score:.2f}"
    score_txt = _paint(score_txt, _score_color(score, failed=failed), color=color)
    tokens = int(row.get("tokens") or 0)
    line = f"  {fixture}  {score_txt}  {status}  {tokens} tok"
    if not verbose:
        return line
    latency = row.get("latency_ms")
    if latency is not None:
        try:
            line += f"  {int(round(float(latency)))}ms"
        except (TypeError, ValueError):
            pass
    extras: list[str] = []
    error = row.get("error")
    if error:
        extras.append(f"    error: {error}")
    response = row.get("response")
    if response:
        extras.extend(f"    {part}" for part in str(response).splitlines() or [""])
    if extras:
        return line + "\n" + "\n".join(extras)
    return line


def _join_list(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _money(value: Any) -> str:
    try:
        return f"${float(value):.2f}"
    except (TypeError, ValueError):
        return "$?"


def format_jobs_list(body: dict[str, Any]) -> str:
    jobs = body.get("jobs") or []
    if not jobs:
        return "no jobs\n"
    lines: list[str] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        kind = str(job.get("kind") or "job")
        status = str(job.get("status") or "?")
        job_id = str(job.get("job_id") or "")[:8]
        parts = [kind, status, job_id]
        request = job.get("request") or {}
        if isinstance(request, dict):
            for key in ("suite", "model", "host"):
                if request.get(key):
                    parts.append(str(request[key]))
            if request.get("models"):
                parts.append(_join_list(request["models"]))
            if request.get("suites"):
                parts.append(_join_list(request["suites"]))
        lines.append("  ".join(parts))
    return "\n".join(lines) + "\n"


def format_models_list(body: dict[str, Any]) -> str:
    provider = str(body.get("provider") or "?")
    models = body.get("models") or []
    names: list[str] = []
    for item in models:
        if isinstance(item, dict):
            names.append(str(item.get("name") or item.get("id") or "?"))
        else:
            names.append(str(item))
    header = f"{provider}  {len(names)}"
    if not names:
        return header + "\n"
    return header + "\n" + "\n".join(names) + "\n"


def format_recent_runs(body: dict[str, Any]) -> str:
    runs = body.get("runs") or []
    if not runs:
        return "no runs\n"
    lines: list[str] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        run_id = str(run.get("run_id") or "")
        provider = str(run.get("provider") or "-")
        model = str(run.get("model") or "-")
        tier = str(run.get("tier") or "-")
        ok = int(run.get("successful_tasks") or 0)
        fail = int(run.get("failed_tasks") or 0)
        tokens = int(run.get("total_tokens") or 0)
        lines.append(
            f"{run_id}  {provider}  {model}  {tier}  {ok} ok  {fail} fail  "
            f"{tokens} tok  {_money(run.get('total_cost_usd'))}"
        )
    return "\n".join(lines) + "\n" if lines else "no runs\n"


def format_get_run(body: dict[str, Any]) -> str:
    raw_run = body.get("run")
    run: dict[str, Any] = raw_run if isinstance(raw_run, dict) else {}
    run_id = str(run.get("run_id") or "")
    provider = str(run.get("provider") or "-")
    model = str(run.get("model") or "-")
    tier = str(run.get("tier") or "-")
    lines = [f"{run_id}  {provider}  {model}  {tier}"]
    for row in body.get("fixtures") or []:
        if not isinstance(row, dict):
            continue
        ident = str(row.get("id") or "?")
        status = str(row.get("status") or "")
        raw = row.get("score")
        try:
            score_txt = "-" if raw is None else f"{float(raw):.2f}"
        except (TypeError, ValueError):
            score_txt = "-"
        lines.append(f"  {ident}  {score_txt}  {status}".rstrip())
    return "\n".join(lines) + "\n"


def format_compare(body: dict[str, Any]) -> str:
    rows = body.get("rows") or []
    by = str(body.get("by") or "provider")
    lines = [f"{by}  {len(rows)}"]
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("group_key") or "-")
        raw = row.get("avg_accuracy_score")
        try:
            score_txt = "-" if raw is None else f"{float(raw):.2f}"
        except (TypeError, ValueError):
            score_txt = "-"
        tasks = int(row.get("task_count") or 0)
        lines.append(f"{key}  {score_txt}  {tasks} tasks  {_money(row.get('total_cost'))}")
    return "\n".join(lines) + "\n"


def format_trends(body: dict[str, Any]) -> str:
    rows = body.get("rows") or []
    hours = body.get("hours") or 0
    lines = [f"{hours}h  {len(rows)}"]
    for row in rows:
        if not isinstance(row, dict):
            continue
        hour = str(row.get("hour") or "-")
        tasks = int(row.get("task_count") or 0)
        tokens = int(row.get("total_tokens") or 0)
        lines.append(f"{hour}  {tasks}  {tokens} tok  {_money(row.get('cost'))}")
    return "\n".join(lines) + "\n"


def format_provider_test(body: dict[str, Any]) -> str:
    model = str(body.get("model") or "-")
    if body.get("ok"):
        latency = body.get("latency_ms")
        try:
            ms = "?ms" if latency is None else f"{int(round(float(latency)))}ms"
        except (TypeError, ValueError):
            ms = "?ms"
        reply = str(body.get("response") or "").replace("\n", " ")
        thinking_tokens = body.get("thinking_tokens") or 0
        try:
            think_n = int(thinking_tokens)
        except (TypeError, ValueError):
            think_n = 0
        think = bool(body.get("think")) or (not reply.strip() and think_n > 0)
        if think:
            return f"THINK  {model}  {ms}  {think_n} thinking tokens\n"
        return f"ok  {model}  {ms}  {reply}\n"
    error = body.get("error") or "failed"
    return f"fail  {model}\n{error}\n"


def format_submitted(body: dict[str, Any]) -> str:
    request = body.get("request")
    request = request if isinstance(request, dict) else {}
    parts: list[str] = [str(body.get("kind") or "job")]
    suite = request.get("suite")
    if suite:
        parts.append(str(suite))
    elif request.get("suites"):
        parts.append(_join_list(request["suites"]))
    model = request.get("model")
    if model:
        parts.append(str(model))
    elif request.get("models"):
        parts.append(_join_list(request["models"]))
    host = request.get("host")
    if host:
        parts.append(str(host))
    job_id = str(body.get("job_id") or "")
    status = str(body.get("status") or "pending")
    return f"{'  '.join(parts)}\n{job_id}  {status}\n"


def format_failed(body: dict[str, Any], *, color: bool = False) -> str:
    kind = str(body.get("kind") or "job")
    error = body.get("error") or {}
    if isinstance(error, dict):
        message = error.get("message") or "failed"
    else:
        message = str(error) or "failed"
    line = f"{kind}  failed\n{message}\n"
    return _paint(line, _RED, color=color)


def format_completed(body: dict[str, Any], *, color: bool = False) -> str:
    request = body.get("request") or {}
    result = body.get("result") or {}
    progress = body.get("progress") or {}
    suite = request.get("suite") or result.get("suite") or body.get("kind") or "job"
    model = request.get("model") or result.get("model") or "-"
    host = request.get("host") or result.get("host") or ""
    headline = result.get("overall_accuracy")
    if headline is None:
        headline = result.get("overall_score")
    current = progress.get("current")
    total = progress.get("total")
    if current is None:
        current = result.get("fixtures_run")
    if total is None:
        total = result.get("fixtures_run")
    tokens = result.get("total_tokens") or 0
    cost = result.get("total_cost_usd")
    try:
        score = None if headline is None else float(headline)
    except (TypeError, ValueError):
        score = None
    jobs = result.get("jobs")
    if isinstance(jobs, list) and (result.get("ok") is not None or result.get("fail") is not None):
        ok = int(result.get("ok") or 0)
        fail = int(result.get("fail") or 0)
        models = request.get("models") or result.get("models") or []
        suites = request.get("suites") or result.get("suites") or []
        if isinstance(models, list):
            model_txt = ",".join(str(item) for item in models)
        else:
            model_txt = str(models)
        if isinstance(suites, list):
            suite_txt = ",".join(str(item) for item in suites)
        else:
            suite_txt = str(suites)
        return f"sweep  {model_txt}  {suite_txt}\n{ok} ok  {fail} fail  {len(jobs)} jobs\n"
    host_txt = f"  {host}" if host else ""
    phases = result.get("phases")
    if isinstance(phases, list) and phases:
        peak = result.get("peak_tps")
        sat = result.get("saturation_concurrency")
        peak_txt = _tps_label(peak)
        return (
            f"stress  {model}{host_txt}\n"
            f"{peak_txt} tps  sat={sat}  {len(phases)} phases\n"
        )
    samples = result.get("samples")
    verdict = result.get("verdict")
    if isinstance(samples, list) and samples and verdict:
        drift = result.get("throughput_drift_pct")
        latency = result.get("latency_drift_pct")
        extras: list[str] = []
        if drift is not None:
            extras.append(f"drift {drift}% tps")
        if latency is not None:
            extras.append(f"{latency}% p95")
        drift_txt = f"  {' / '.join(extras)}" if extras else ""
        return (
            f"soak  {model}{host_txt}\n"
            f"{verdict}{drift_txt}  {len(samples)} samples\n"
        )
    if body.get("kind") == "run" or isinstance(result.get("task_rows"), list):
        ok = int(result.get("success") or 0)
        fail = int(result.get("failed") or 0)
        cost_txt = "" if cost is None else f"  ${float(cost):.2f}"
        return (
            f"run  {model}{host_txt}\n"
            f"{ok} ok  {fail} fail  {tokens} tok{cost_txt}\n"
        )
    score_txt = "-" if score is None else f"{score:.3f}"
    score_txt = _paint(score_txt, _score_color(score, failed=False), color=color)
    count = f"{current}/{total}" if total is not None else str(current or 0)
    cost_txt = "" if cost is None else f"  ${float(cost):.2f}"
    host_txt = f"  {host}" if host else ""
    return (
        f"{suite}  {model}{host_txt}\n"
        f"{score_txt}  {count}  {tokens} tok{cost_txt}\n"
    )


def format_still_running(body: dict[str, Any], *, color: bool = False) -> str:
    progress = body.get("progress") or {}
    current = progress.get("current") or 0
    total = progress.get("total") or "?"
    inflight = progress.get("in_flight") or {}
    fixture = inflight.get("fixture_id") or inflight.get("task") or ""
    phase = inflight.get("phase") or "running"
    line = f"still running  {current}/{total}  {fixture}  {phase}"
    return _paint(line, _YELLOW, color=color)


class QuietWait:
    """Turn job polls into one-liners. Remembers what was already printed."""

    def __init__(
        self,
        emit: Callable[[str], object],
        *,
        color: bool = False,
        verbose: bool = False,
    ) -> None:
        self._emit = emit
        self.color = color
        self.verbose = verbose
        self._seen = 0
        self._trail_seen = 0
        self._inflight: tuple[Any, ...] | None = None
        self._summarized = False

    def update(self, body: Any) -> None:
        if not isinstance(body, dict):
            return
        progress = body.get("progress") or {}
        trail = progress.get("trail") or []
        if isinstance(trail, list) and trail:
            for entry in trail[self._trail_seen :]:
                if isinstance(entry, dict):
                    line = format_in_flight(entry, color=self.color)
                    if line:
                        self._emit(line + "\n")
            self._trail_seen = len(trail)
            self._inflight = _inflight_sig(progress.get("in_flight"))
        else:
            inflight = progress.get("in_flight")
            sig = _inflight_sig(inflight)
            if inflight and sig != self._inflight:
                self._inflight = sig
                line = format_in_flight(inflight, color=self.color)
                if line:
                    self._emit(line + "\n")
        rows, render = _live_rows(body)
        for row in rows[self._seen :]:
            if isinstance(row, dict):
                self._emit(render(row, color=self.color, verbose=self.verbose) + "\n")
        self._seen = len(rows)
        if body.get("status") == "completed" and not self._summarized:
            self._summarized = True
            self._emit(format_completed(body, color=self.color))
        elif body.get("status") == "failed" and not self._summarized:
            self._summarized = True
            self._emit(format_failed(body, color=self.color))

    def finish(self, body: Any) -> None:
        if not isinstance(body, dict):
            return
        if body.get("status") in {"completed", "failed"}:
            self.update(body)
            return
        if not self._summarized:
            self._emit(format_still_running(body, color=self.color) + "\n")


def _live_rows(
    body: dict[str, Any],
) -> tuple[list[Any], Callable[..., str]]:
    result = body.get("result") or {}
    fixtures = result.get("fixtures") or []
    if fixtures:
        return fixtures, lambda row, **kwargs: format_fixture_row(row, **kwargs)
    phases = result.get("phases") or []
    if phases:
        return phases, lambda row, **kwargs: format_phase_row(
            row, color=kwargs.get("color", False)
        )
    samples = result.get("samples") or []
    if samples:
        return samples, lambda row, **kwargs: format_sample_row(
            row, color=kwargs.get("color", False)
        )
    task_rows = result.get("task_rows") or []
    if task_rows:
        return task_rows, lambda row, **kwargs: format_task_row(row, **kwargs)
    return [], lambda row, **kwargs: ""


def _inflight_sig(in_flight: Any) -> tuple[Any, ...] | None:
    if not isinstance(in_flight, dict):
        return None
    return (
        in_flight.get("fixture_id"),
        in_flight.get("phase"),
        in_flight.get("model"),
        in_flight.get("suite"),
        in_flight.get("concurrency"),
        in_flight.get("elapsed_seconds"),
        in_flight.get("task"),
    )
