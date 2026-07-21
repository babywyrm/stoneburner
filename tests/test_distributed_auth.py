from fastapi import Request

from atomics.distributed.auth import WorkerAuth


async def test_worker_auth_valid():
    auth = WorkerAuth({"worker-key"})
    req = Request({"type": "http", "headers": [(b"x-api-key", b"worker-key")]})
    assert await auth.authenticate(req) is True


async def test_worker_auth_invalid():
    auth = WorkerAuth({"worker-key"})
    req = Request({"type": "http", "headers": [(b"x-api-key", b"bad")]})
    assert await auth.authenticate(req) is False


async def test_worker_auth_missing():
    auth = WorkerAuth({"worker-key"})
    req = Request({"type": "http", "headers": []})
    assert await auth.authenticate(req) is False
