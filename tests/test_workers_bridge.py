"""Tests for worker bridge runtime behavior without monkeypatch."""

from __future__ import annotations

import pytest

from atomics.workers.bridge import invoke_worker


class FakeProc:
    def __init__(self, returncode: int, stdout: bytes, stderr: bytes):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self, _payload: bytes):
        return self._stdout, self._stderr


@pytest.mark.asyncio
async def test_invoke_worker_success():
    async def fake_create(*_args, **_kwargs):
        return FakeProc(
            0,
            b'{"status":"ok","result":"done","tokens":{"input":7,"output":11}}',
            b"",
        )

    result = await invoke_worker(
        ["node", "worker.js"],
        "task",
        "prompt",
        timeout_ms=10,
        create_subprocess_exec=fake_create,
    )
    assert result.status == "ok"
    assert result.result == "done"
    assert result.input_tokens == 7
    assert result.output_tokens == 11


@pytest.mark.asyncio
async def test_invoke_worker_nonzero():
    async def fake_create(*_args, **_kwargs):
        return FakeProc(1, b"", b"boom")

    result = await invoke_worker(
        ["node", "worker.js"],
        "task",
        "prompt",
        timeout_ms=10,
        create_subprocess_exec=fake_create,
    )
    assert result.status == "error"
    assert "boom" in result.error


@pytest.mark.asyncio
async def test_invoke_worker_timeout():
    async def fake_create(*_args, **_kwargs):
        return FakeProc(0, b"{}", b"")

    async def fake_wait_for(coro, **_kwargs):
        # Prevent un-awaited coroutine warnings in the timeout path.
        coro.close()
        raise TimeoutError

    result = await invoke_worker(
        ["node", "worker.js"],
        "task",
        "prompt",
        timeout_ms=10,
        create_subprocess_exec=fake_create,
        wait_for=fake_wait_for,
    )
    assert result.status == "error"
    assert "timed out" in result.error


@pytest.mark.asyncio
async def test_invoke_worker_bad_json():
    async def fake_create(*_args, **_kwargs):
        return FakeProc(0, b"not-json", b"")

    result = await invoke_worker(
        ["node", "worker.js"],
        "task",
        "prompt",
        timeout_ms=10,
        create_subprocess_exec=fake_create,
    )
    assert result.status == "error"
    assert "Expecting value" in result.error
