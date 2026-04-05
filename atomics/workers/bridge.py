"""Bridge contract for npm/external workers.

Workers communicate via JSON over stdin/stdout with a simple protocol:
  Input:  {"task": "...", "prompt": "...", "timeout_ms": 30000}
  Output: {"status": "ok"|"error", "result": "...", "tokens": {...}, "error": "..."}

This module is scaffolded for Phase 3 — npm worker integration.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("atomics.workers")


@dataclass
class WorkerResult:
    status: str
    result: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""


async def invoke_worker(
    worker_cmd: list[str],
    task_name: str,
    prompt: str,
    timeout_ms: int = 30000,
    cwd: Path | None = None,
) -> WorkerResult:
    """Send a task to an external worker process and collect the result."""
    payload = json.dumps({"task": task_name, "prompt": prompt, "timeout_ms": timeout_ms})

    try:
        proc = await asyncio.create_subprocess_exec(
            *worker_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd) if cwd else None,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(payload.encode()),
            timeout=timeout_ms / 1000 + 5,
        )

        if proc.returncode != 0:
            return WorkerResult(status="error", error=stderr.decode()[:500])

        data = json.loads(stdout.decode())
        return WorkerResult(
            status=data.get("status", "error"),
            result=data.get("result", ""),
            input_tokens=data.get("tokens", {}).get("input", 0),
            output_tokens=data.get("tokens", {}).get("output", 0),
            error=data.get("error", ""),
        )
    except asyncio.TimeoutError:
        return WorkerResult(status="error", error="worker timed out")
    except Exception as exc:
        return WorkerResult(status="error", error=str(exc)[:500])
