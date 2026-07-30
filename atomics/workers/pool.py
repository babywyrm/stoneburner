"""Pool manager for running multiple npm workers in parallel."""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger("atomics.workers.pool")


ProcessFactory = Callable[[], Coroutine[Any, Any, asyncio.subprocess.Process]]


class WorkerPool:
    """Spawn and manage a fixed number of identical worker processes."""

    def __init__(self, size: int, factory: ProcessFactory) -> None:
        if size < 1:
            raise ValueError("pool size must be at least 1")
        self.size = size
        self._factory = factory
        self._procs: list[asyncio.subprocess.Process] = []
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start every worker in the pool."""
        for _ in range(self.size):
            proc = await self._factory()
            self._procs.append(proc)
        logger.info("Started pool of %d workers", self.size)

    async def run(self) -> int:
        """Keep the pool alive until a shutdown signal is received.

        Returns the first non-zero exit code from any worker, or 0 if all
        workers exited cleanly.
        """
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown_event.set)

        try:
            await self._shutdown_event.wait()
        finally:
            await self.stop()

        first_nonzero = 0
        for proc in self._procs:
            if proc.returncode is not None and proc.returncode != 0:
                first_nonzero = proc.returncode
                break
        return first_nonzero

    async def stop(self, *, timeout: float = 10.0) -> None:
        """Signal all workers to terminate and wait for them to exit."""
        for proc in self._procs:
            if proc.returncode is None:
                proc.terminate()
        _, pending = await asyncio.wait(
            [asyncio.create_task(self._wait_one(proc)) for proc in self._procs],
            timeout=timeout,
        )
        for proc in self._procs:
            if proc.returncode is None:
                proc.kill()
        if pending:
            await asyncio.wait(
                [asyncio.create_task(self._wait_one(proc)) for proc in self._procs],
                timeout=timeout,
            )
        logger.info("Pool stopped")

    async def _wait_one(self, proc: asyncio.subprocess.Process) -> None:
        await proc.wait()
