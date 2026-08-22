"""Run one REPL line against the session and, later, the API client."""

from __future__ import annotations

import json
from dataclasses import dataclass

from atomics.mcp.client import AtomicsApiClient
from atomics.repl.parse import ParseError, parse_line
from atomics.repl.session import SESSION_KEYS, Session, SessionError

HELP_TEXT = """\
Session: set, show, help, exit
Read:    health, list_models, list_jobs, get_job, get_run, compare, recent_runs, trends
Spend:   provider_test, submit_run, submit_eval, submit_sweep, submit_stress, submit_soak
Poll:    wait [JOB_ID]

set KEY [VALUE]   set or clear a session field
show              print the session as JSON
Verbs match MCP tools. See docs/MCP_SERVER.md and docs/REPL.md.
"""


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
