"""Run one REPL line against the session and the API client."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from atomics.mcp.client import AtomicsApiClient, AtomicsApiError
from atomics.repl.parse import ParsedLine, ParseError, parse_line
from atomics.repl.session import SESSION_KEYS, Session, SessionError
from atomics.repl.wait import wait_for_job

HELP_TEXT = """\
Session: set, show, help, exit
Read:    health, list_models, list_jobs, get_job, get_run, compare, recent_runs, trends
Spend:   provider_test, submit_run, submit_eval, submit_sweep, submit_stress, submit_soak
Poll:    wait [JOB_ID]

set KEY [VALUE]   set or clear a session field
show              print the session as JSON
Verbs match MCP tools. See docs/MCP_SERVER.md and docs/REPL.md.
"""

_BOOL = frozenset({"thinking", "save"})
_INT = frozenset(
    {
        "iterations",
        "interval",
        "runs",
        "limit",
        "hours",
        "duration_seconds",
        "concurrency",
        "sample_interval",
        "max_concurrency",
    }
)
_FLOAT = frozenset({"budget_usd", "since_hours", "phase_seconds"})
_LIST = frozenset({"models", "suites", "fixtures"})
_SESSION_FIELDS = ("provider", "model", "effort", "reasoning_mode")

# verb -> (client method name, allowed kwargs, optional positional kwarg)
_VERBS: dict[str, tuple[str, frozenset[str], str | None]] = {
    "health": ("health", frozenset(), None),
    "list_models": ("list_models", frozenset({"provider", "host"}), None),
    "list_jobs": ("list_jobs", frozenset(), None),
    "get_job": ("get_job", frozenset(), "job_id"),
    "get_run": ("get_run", frozenset(), "run_id"),
    "compare": ("compare", frozenset({"by", "since_hours", "tier", "category"}), None),
    "recent_runs": ("recent_runs", frozenset({"limit"}), None),
    "trends": ("trends", frozenset({"hours"}), None),
    "provider_test": (
        "provider_test",
        frozenset({"provider", "model", "host", "thinking", "effort", "reasoning_mode"}),
        None,
    ),
    "submit_run": (
        "submit_run",
        frozenset(
            {
                "provider",
                "model",
                "tier",
                "iterations",
                "interval",
                "save",
                "thinking",
                "effort",
                "reasoning_mode",
            }
        ),
        None,
    ),
    "submit_eval": (
        "submit_eval",
        frozenset(
            {
                "suite",
                "provider",
                "model",
                "judge_model",
                "fixtures",
                "save",
                "budget_usd",
                "thinking",
                "effort",
                "reasoning_mode",
            }
        ),
        None,
    ),
    "submit_sweep": (
        "submit_sweep",
        frozenset(
            {
                "provider",
                "models",
                "suites",
                "budget_usd",
                "judge_model",
                "runs",
                "thinking",
                "effort",
                "reasoning_mode",
            }
        ),
        None,
    ),
    "submit_stress": (
        "submit_stress",
        frozenset({"provider", "model", "budget_usd", "max_concurrency", "phase_seconds"}),
        None,
    ),
    "submit_soak": (
        "submit_soak",
        frozenset(
            {
                "provider",
                "model",
                "budget_usd",
                "duration_seconds",
                "concurrency",
                "sample_interval",
            }
        ),
        None,
    ),
}


@dataclass
class HandleResult:
    stdout: str = ""
    stderr: str = ""
    exit_loop: bool = False
    exit_code: int = 0


def handle_line(line: str, *, session: Session, client: AtomicsApiClient) -> HandleResult:
    try:
        parsed = parse_line(line)
    except ParseError as exc:
        return HandleResult(stderr=f"{exc}\n{HELP_TEXT}")
    if parsed is None:
        return HandleResult()
    if parsed.verb == "help":
        return HandleResult(stdout=HELP_TEXT)
    if parsed.verb == "exit":
        return HandleResult(exit_loop=True)
    if parsed.verb == "show":
        return HandleResult(stdout=json.dumps(session.as_dict(), indent=2) + "\n")
    if parsed.verb == "set":
        return _set(session, parsed.args)
    if parsed.verb == "wait":
        return _wait(parsed, session=session, client=client)
    if parsed.verb in _VERBS:
        return _call_api(parsed, session=session, client=client)
    return HandleResult(stderr=f"unknown verb {parsed.verb!r}\n{HELP_TEXT}")


def _set(session: Session, args: tuple[str, ...]) -> HandleResult:
    if not args:
        return HandleResult(stderr=f"set KEY [VALUE]\nkeys: {', '.join(SESSION_KEYS)}\n")
    key, value = args[0], (args[1] if len(args) > 1 else None)
    if len(args) > 2:
        return HandleResult(stderr="set takes KEY and optional VALUE\n")
    try:
        session.set(key, value)
    except SessionError as exc:
        return HandleResult(stderr=f"{exc}\n")
    return HandleResult()


def _wait(
    parsed: ParsedLine, *, session: Session, client: AtomicsApiClient
) -> HandleResult:
    job_id = parsed.args[0] if parsed.args else session.last_job_id
    if not job_id:
        return HandleResult(
            stderr="wait needs a job id; submit something first, or pass one\n"
        )
    if len(parsed.args) > 1 or parsed.flags:
        return HandleResult(stderr="wait takes an optional JOB_ID only\n")
    try:
        body = wait_for_job(client, job_id, sleep=time.sleep)
    except AtomicsApiError as exc:
        return HandleResult(stderr=f"{exc}\n")
    return HandleResult(stdout=json.dumps(body, indent=2) + "\n")


def _call_api(
    parsed: ParsedLine, *, session: Session, client: AtomicsApiClient
) -> HandleResult:
    method_name, allowed, positional = _VERBS[parsed.verb]
    unknown = sorted(set(parsed.flags) - allowed)
    if unknown:
        return HandleResult(stderr=f"unknown flag --{unknown[0]}\n{HELP_TEXT}")
    kwargs: dict[str, Any] = {}
    try:
        for name, raw in parsed.flags.items():
            kwargs[name] = _coerce(name, raw)
    except ValueError as exc:
        return HandleResult(stderr=f"{exc}\n")
    if positional is not None:
        if not parsed.args:
            return HandleResult(stderr=f"{parsed.verb} needs a {positional}\n")
        if len(parsed.args) > 1:
            return HandleResult(stderr=f"{parsed.verb} takes one {positional}\n")
        kwargs[positional] = parsed.args[0]
    elif parsed.args:
        return HandleResult(stderr=f"{parsed.verb} does not take positional arguments\n")
    for field in _SESSION_FIELDS:
        if field in allowed and field not in kwargs:
            value = getattr(session, field)
            if value is not None:
                kwargs[field] = value
    if parsed.verb == "submit_sweep" and "models" not in kwargs and session.model is not None:
        kwargs["models"] = [session.model]
    try:
        body = getattr(client, method_name)(**kwargs)
    except AtomicsApiError as exc:
        return HandleResult(stderr=f"{exc}\n")
    except TypeError as exc:
        return HandleResult(stderr=f"{exc}\n")
    if isinstance(body, dict) and body.get("job_id") and parsed.verb.startswith("submit_"):
        session.last_job_id = str(body["job_id"])
    return HandleResult(stdout=json.dumps(body, indent=2) + "\n")


def _coerce(name: str, raw: str) -> Any:
    if name in _BOOL:
        lowered = raw.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        raise ValueError(f"--{name} expects true or false")
    if name in _INT:
        return int(raw)
    if name in _FLOAT:
        return float(raw)
    if name in _LIST:
        return [part for part in (item.strip() for item in raw.split(",")) if part]
    return raw
