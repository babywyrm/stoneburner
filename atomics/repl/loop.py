"""Prompt loop. Requires a reachable API; does not spawn a server."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TextIO

from atomics.mcp.client import AtomicsApiClient, AtomicsApiError
from atomics.repl.dispatch import handle_line
from atomics.repl.session import Session

PROMPT = "atomics> "


def run_repl(
    client: AtomicsApiClient,
    *,
    input_fn: Callable[[str], str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    read = input_fn or input
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    try:
        client.health()
    except AtomicsApiError as exc:
        print(str(exc), file=err)
        return 1
    session = Session()
    while True:
        try:
            line = read(PROMPT)
        except EOFError:
            print(file=out)
            return 0
        except KeyboardInterrupt:
            print(file=out)
            continue
        result = handle_line(line, session=session, client=client)
        if result.stdout:
            out.write(result.stdout)
            if not result.stdout.endswith("\n"):
                out.write("\n")
        if result.stderr:
            err.write(result.stderr)
            if not result.stderr.endswith("\n"):
                err.write("\n")
        if result.exit_loop:
            return result.exit_code
