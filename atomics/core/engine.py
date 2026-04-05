"""Main loop engine — orchestrates task selection, execution, and metrics recording."""

from __future__ import annotations

import asyncio
import logging
import random
import signal
import uuid
from datetime import datetime, timezone

from atomics.config import AtomicsSettings
from atomics.core.guard import GuardConfig, RateBudgetGuard
from atomics.core.runner import execute_task
from atomics.models import TaskStatus
from atomics.providers.base import BaseProvider
from atomics.storage.repository import MetricsRepository
from atomics.tasks import get_weighted_task

logger = logging.getLogger("atomics.engine")


class LoopEngine:
    """Continuous benchmarking loop with rate/budget controls."""

    def __init__(
        self,
        provider: BaseProvider,
        repo: MetricsRepository,
        settings: AtomicsSettings,
    ) -> None:
        self._provider = provider
        self._repo = repo
        self._settings = settings
        self._guard = RateBudgetGuard(
            GuardConfig(
                max_tokens_per_hour=settings.max_tokens_per_hour,
                max_requests_per_minute=settings.max_requests_per_minute,
                budget_limit_usd=settings.budget_limit_usd,
                circuit_breaker_threshold=settings.circuit_breaker_threshold,
            )
        )
        self._shutdown = asyncio.Event()
        self._run_id: str = ""

    async def run(self, max_iterations: int | None = None) -> None:
        """Start the benchmarking loop. Runs until shutdown signal or iteration cap."""
        self._install_signal_handlers()
        self._run_id = uuid.uuid4().hex[:12]
        self._repo.create_run(self._run_id)

        logger.info(
            "Atomics engine started — run_id=%s provider=%s model=%s",
            self._run_id,
            self._provider.name,
            self._settings.default_model,
        )

        iteration = 0
        while not self._shutdown.is_set():
            if max_iterations is not None and iteration >= max_iterations:
                logger.info("Reached max iterations (%d), stopping.", max_iterations)
                break

            allowed, reason = self._guard.can_proceed()
            if not allowed:
                if self._guard.circuit_open:
                    logger.error("Circuit breaker open: %s — stopping run.", reason)
                    break
                wait = max(self._guard.seconds_until_allowed(), 5.0)
                logger.info("Guard blocked: %s — waiting %.1fs", reason, wait)
                try:
                    await asyncio.wait_for(self._shutdown.wait(), timeout=wait)
                    break
                except asyncio.TimeoutError:
                    continue

            task_def, topic = get_weighted_task()
            logger.info(
                "[iter %d] Running %s/%s — topic: %s",
                iteration,
                task_def.category.value,
                task_def.name,
                topic[:60],
            )

            result = await execute_task(
                task_def,
                topic,
                provider=self._provider,
                run_id=self._run_id,
                model=self._settings.default_model,
            )

            self._repo.save_task_result(result)
            self._guard.record_request(
                result.total_tokens,
                result.estimated_cost_usd,
                result.status == TaskStatus.SUCCESS,
            )

            self._log_result(iteration, result)
            iteration += 1

            interval = self._settings.loop_interval_seconds + random.uniform(
                -self._settings.loop_jitter_seconds,
                self._settings.loop_jitter_seconds,
            )
            interval = max(1.0, interval)
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=interval)
                break
            except asyncio.TimeoutError:
                pass

        summary = self._repo.complete_run(self._run_id)
        logger.info(
            "Run complete — tasks=%d success=%d failed=%d tokens=%d cost=$%.4f",
            summary.total_tasks,
            summary.successful_tasks,
            summary.failed_tasks,
            summary.total_tokens,
            summary.total_cost_usd,
        )

    def stop(self) -> None:
        self._shutdown.set()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.stop)

    def _log_result(self, iteration: int, result) -> None:
        if result.status == TaskStatus.SUCCESS:
            logger.info(
                "[iter %d] OK — tokens=%d (in=%d out=%d) latency=%.0fms cost=$%.6f",
                iteration,
                result.total_tokens,
                result.input_tokens,
                result.output_tokens,
                result.latency_ms,
                result.estimated_cost_usd,
            )
        else:
            logger.warning(
                "[iter %d] FAIL — %s: %s",
                iteration,
                result.error_class,
                result.error_message[:100],
            )
