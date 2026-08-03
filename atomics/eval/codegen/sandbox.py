"""Isolated execution of model-generated code for the codegen eval suite.

Generated code is untrusted input. It used to run via `exec()` inside the
evaluating process, which meant it shared that process's memory, filesystem
handles, environment (including every provider API key), and lifetime — and the
codegen suite is reachable over HTTP through `POST /api/v1/evals`.

Each snippet now runs in a child interpreter that gets a scrubbed environment, a
scratch working directory, address-space and CPU limits, and a wall-clock kill.
This is a meaningful boundary, not a jail: it assumes a model producing wrong
code rather than an attacker with a kernel exploit. Treat it accordingly.
"""

from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass

DEFAULT_MEMORY_BYTES = 512 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 5.0

# Runs in the child with no atomics imports, so it stays independent of how the
# parent was installed. Results go out as a repr so tuples and sets survive the
# round trip that JSON would flatten.
_CHILD_PROGRAM = r'''
import ast, json, sys

# Generated code may print. Hand it stderr so stdout carries only our result.
_out = sys.stdout
sys.stdout = sys.stderr

def _emit(obj):
    _out.write(json.dumps(obj))
    _out.flush()
    raise SystemExit(0)

payload = json.loads(sys.stdin.read())

try:
    import resource

    def _limit(name, soft):
        res = getattr(resource, name, None)
        if res is None:
            return
        try:
            hard = resource.getrlimit(res)[1]
            if hard != resource.RLIM_INFINITY:
                soft = min(soft, hard)
            resource.setrlimit(res, (soft, hard))
        except (ValueError, OSError):
            pass

    _limit("RLIMIT_AS", payload["memory_bytes"])
    _limit("RLIMIT_CPU", payload["cpu_seconds"])
    _limit("RLIMIT_FSIZE", 1 << 20)
except ImportError:
    pass

# Blocks casual exfiltration. A determined escape can reach the syscall another
# way; the wall-clock kill and scrubbed environment are the real containment.
try:
    import socket

    def _no_network(*args, **kwargs):
        raise OSError("network access is disabled in the codegen sandbox")

    socket.socket = _no_network
    socket.create_connection = _no_network
except ImportError:
    pass

namespace = {}
try:
    exec(compile(payload["code"], "<generated>", "exec"), namespace)
except BaseException as exc:
    _emit({"status": "compile_error", "detail": type(exc).__name__ + ": " + str(exc)})

func = namespace.get(payload["function_name"])
if func is None:
    _emit({"status": "missing_function"})

try:
    result = func(*ast.literal_eval(payload["args_repr"]))
except BaseException as exc:
    _emit({"status": "runtime_error", "detail": type(exc).__name__ + ": " + str(exc)})

try:
    encoded = repr(result)
except BaseException as exc:
    _emit({"status": "unrepresentable", "detail": type(exc).__name__ + ": " + str(exc)})

_emit({"status": "ok", "result_repr": encoded})
'''


@dataclass(frozen=True)
class SandboxOutcome:
    """What the child reported, normalized for the caller."""

    status: str
    result: object = None
    detail: str = ""


def _child_env() -> dict[str, str]:
    """A minimal environment. Notably absent: every provider credential."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONIOENCODING": "utf-8"}
    # Keeps the child using the same interpreter's stdlib under a venv.
    for passthrough in ("SYSTEMROOT", "LD_LIBRARY_PATH"):
        if passthrough in os.environ:
            env[passthrough] = os.environ[passthrough]
    return env


def execute(
    code: str,
    function_name: str,
    args: tuple | list,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    memory_bytes: int = DEFAULT_MEMORY_BYTES,
) -> SandboxOutcome:
    """Run `function_name(*args)` from `code` in a child interpreter."""
    payload = json.dumps(
        {
            "code": code,
            "function_name": function_name,
            "args_repr": repr(tuple(args)),
            "memory_bytes": memory_bytes,
            # A CPU limit below the wall clock would fire first and mask real
            # timeouts, so give it headroom and let the kill below be decisive.
            "cpu_seconds": max(1, int(timeout_seconds) + 1),
        }
    )

    with tempfile.TemporaryDirectory(prefix="atomics-codegen-") as scratch:
        try:
            proc = subprocess.Popen(  # noqa: S603
                [sys.executable, "-I", "-c", _CHILD_PROGRAM],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_child_env(),
                cwd=scratch,
                text=True,
                # Its own process group, so a snippet that spawns children
                # cannot outlive the kill below.
                start_new_session=True,
            )
        except OSError as exc:
            return SandboxOutcome("spawn_error", detail=str(exc))

        try:
            stdout, _ = proc.communicate(payload, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _terminate_group(proc)
            return SandboxOutcome("timeout")

    if not stdout.strip():
        return SandboxOutcome(
            "crashed", detail=f"child exited with code {proc.returncode}"
        )

    try:
        reported = json.loads(stdout)
    except json.JSONDecodeError:
        return SandboxOutcome("crashed", detail="child produced unreadable output")

    if reported.get("status") != "ok":
        return SandboxOutcome(
            str(reported.get("status", "crashed")),
            detail=str(reported.get("detail", "")),
        )

    try:
        value = ast.literal_eval(reported["result_repr"])
    except (ValueError, SyntaxError):
        return SandboxOutcome(
            "unrepresentable", detail=f"returned {reported['result_repr']}"
        )
    return SandboxOutcome("ok", result=value)


def _terminate_group(proc: subprocess.Popen) -> None:
    """Kill the child's whole process group, then reap it."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()
    try:
        proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        pass
