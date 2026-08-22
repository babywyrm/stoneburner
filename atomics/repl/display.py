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


def format_in_flight(in_flight: dict[str, Any] | None, *, color: bool = False) -> str:
    if not in_flight:
        return ""
    fixture = str(in_flight.get("fixture_id") or "?")
    phase = str(in_flight.get("phase") or "?")
    model = str(in_flight.get("model") or "")
    phase_txt = _paint(f"{phase:<8}", _DIM, color=color)
    return f"  {fixture}  {phase_txt}  {model}".rstrip()


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
    fixture = inflight.get("fixture_id") or ""
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
        self._inflight: tuple[Any, ...] | None = None
        self._summarized = False

    def update(self, body: Any) -> None:
        if not isinstance(body, dict):
            return
        progress = body.get("progress") or {}
        inflight = progress.get("in_flight")
        sig = _inflight_sig(inflight)
        if inflight and sig != self._inflight:
            self._inflight = sig
            line = format_in_flight(inflight, color=self.color)
            if line:
                self._emit(line + "\n")
        fixtures = ((body.get("result") or {}).get("fixtures")) or []
        for row in fixtures[self._seen :]:
            if isinstance(row, dict):
                self._emit(
                    format_fixture_row(row, color=self.color, verbose=self.verbose)
                    + "\n"
                )
        self._seen = len(fixtures)
        if body.get("status") == "completed" and not self._summarized:
            self._summarized = True
            self._emit(format_completed(body, color=self.color))

    def finish(self, body: Any) -> None:
        if not isinstance(body, dict):
            return
        if body.get("status") == "completed":
            self.update(body)
            return
        if not self._summarized:
            self._emit(format_still_running(body, color=self.color) + "\n")


def _inflight_sig(in_flight: Any) -> tuple[Any, ...] | None:
    if not isinstance(in_flight, dict):
        return None
    return (in_flight.get("fixture_id"), in_flight.get("phase"), in_flight.get("model"))
