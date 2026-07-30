"""Tests for the npm worker pool manager."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from atomics.workers.pool import WorkerPool


@pytest.mark.asyncio
async def test_pool_starts_requested_number_of_workers():
    factory = AsyncMock()
    factory.return_value = MagicMock(returncode=None, terminate=MagicMock())
    pool = WorkerPool(3, factory)
    await pool.start()
    assert factory.await_count == 3


@pytest.mark.asyncio
async def test_pool_run_stops_on_signal():
    proc = MagicMock()
    proc.returncode = None
    proc.terminate = MagicMock()
    proc.wait = AsyncMock()

    factory = AsyncMock(return_value=proc)
    pool = WorkerPool(2, factory)
    await pool.start()

    async def stop_after_short_delay():
        await asyncio.sleep(0.05)
        pool._shutdown_event.set()

    await asyncio.gather(pool.run(), stop_after_short_delay())
    assert proc.terminate.call_count == 2


@pytest.mark.asyncio
async def test_pool_stop_terminates_running_workers():
    proc = MagicMock(returncode=None)
    proc.terminate = MagicMock()
    proc.wait = AsyncMock(return_value=None)

    factory = AsyncMock(return_value=proc)
    pool = WorkerPool(2, factory)
    await pool.start()
    await pool.stop()
    assert proc.terminate.call_count == 2


def test_pool_rejects_zero_size():
    with pytest.raises(ValueError):
        WorkerPool(0, AsyncMock())
