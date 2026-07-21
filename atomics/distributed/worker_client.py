"""Async worker client that polls a coordinator for task assignments."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from atomics.distributed.models import TaskAssignment, WorkerRegisterRequest
from atomics.distributed.worker_runner import execute_assignment

logger = logging.getLogger("atomics.distributed.worker_client")

Executor = Callable[..., Awaitable[dict[str, Any]]]


class WorkerClient:
    def __init__(
        self,
        coordinator_url: str,
        api_key: str,
        labels: dict[str, str] | None = None,
        endpoint: str | None = None,
        heartbeat_interval: int = 30,
        executor: Executor | None = None,
        *,
        provider_name: str = "ollama",
        model: str | None = None,
        host: str | None = None,
    ) -> None:
        self.coordinator_url = coordinator_url.rstrip("/")
        self.api_key = api_key
        self.labels = labels or {}
        self.endpoint = endpoint
        self.heartbeat_interval = heartbeat_interval
        self._executor: Executor = executor or execute_assignment
        self._provider_name = provider_name
        self._model = model
        self._host = host
        self._worker_id: str | None = None
        self._shutdown = asyncio.Event()
        self._client = httpx.AsyncClient(
            headers={"X-API-Key": api_key},
            timeout=httpx.Timeout(30.0),
        )

    async def register(self) -> None:
        payload = WorkerRegisterRequest(labels=self.labels, endpoint=self.endpoint)
        resp = await self._client.post(
            f"{self.coordinator_url}/api/v1/workers/register",
            json=payload.model_dump(mode="json"),
        )
        resp.raise_for_status()
        self._worker_id = resp.json()["worker_id"]
        logger.info("Registered worker %s", self._worker_id)

    async def heartbeat(self) -> None:
        if not self._worker_id:
            return
        await self._client.post(
            f"{self.coordinator_url}/api/v1/workers/{self._worker_id}/heartbeat"
        )

    async def poll_and_execute(self) -> bool:
        if not self._worker_id:
            return False
        resp = await self._client.get(
            f"{self.coordinator_url}/api/v1/workers/{self._worker_id}/jobs/next"
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return False
        assignment = TaskAssignment(**data)
        try:
            result = await self._executor(
                assignment,
                provider_name=self._provider_name,
                model=self._model,
                host=self._host,
            )
            await self._client.post(
                f"{self.coordinator_url}/api/v1/workers/{self._worker_id}/jobs/{assignment.assignment_id}/result",
                json={"status": "completed", "result_json": json.dumps(result)},
            )
        except Exception as exc:
            logger.exception("Assignment %s failed", assignment.assignment_id)
            await self._client.post(
                f"{self.coordinator_url}/api/v1/workers/{self._worker_id}/jobs/{assignment.assignment_id}/result",
                json={"status": "failed", "error": str(exc)},
            )
        return True

    async def run(self) -> None:
        await self.register()
        while not self._shutdown.is_set():
            await self.heartbeat()
            worked = await self.poll_and_execute()
            if not worked:
                try:
                    await asyncio.wait_for(
                        self._shutdown.wait(), timeout=self.heartbeat_interval
                    )
                except TimeoutError:
                    pass

    def shutdown(self) -> None:
        self._shutdown.set()

    async def close(self) -> None:
        await self._client.aclose()
