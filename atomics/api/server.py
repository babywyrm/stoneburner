"""FastAPI app factory for the atomics API server."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI

from atomics import __version__
from atomics.api.auth import ApiKeyAuth, NoAuth
from atomics.api.config import ServerSettings
from atomics.api.dashboard import router as dashboard_router
from atomics.api.headers import security_headers_middleware
from atomics.api.jobs import JobManager
from atomics.api.routes import router
from atomics.distributed import routes as distributed_routes
from atomics.distributed.auth import WorkerAuth
from atomics.distributed.coordinator import Coordinator
from atomics.storage.schema import init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: ServerSettings = app.state.settings
    app.state.job_manager = JobManager(
        max_active=settings.max_active_jobs,
        max_retained=settings.max_retained_jobs,
        max_active_per_caller=settings.max_active_jobs_per_caller,
    )
    if settings.no_auth:
        logger.warning(
            "Running with --no-auth: every request is the same anonymous caller, "
            "so per-caller job quotas cannot partition capacity."
        )
    if settings.no_auth:
        app.state.worker_auth = NoAuth()
    else:
        if not settings.worker_api_keys and settings.api_keys:
            logger.warning(
                "No --worker-api-key set: workers share the submitter keys, so a "
                "worker credential also authorizes run and eval submission."
            )
        app.state.worker_auth = WorkerAuth(set(settings.effective_worker_keys))
    app.state.coordinator = Coordinator(
        init_db(settings.db_path),
        worker_absent_after_seconds=settings.worker_absent_after_seconds,
    )
    yield
    # Shutdown: cancel any running jobs gracefully
    manager: JobManager = app.state.job_manager
    for job in manager.jobs.values():
        if job._task and not job._task.done():
            job._task.cancel()
            try:
                await job._task
            except asyncio.CancelledError:
                pass
    app.state.coordinator._conn.close()


def create_app(
    settings: ServerSettings | None = None,
    *,
    no_auth: bool | None = None,
    db_path: Path | None = None,
) -> FastAPI:
    settings = settings or ServerSettings()
    if no_auth is not None:
        settings = replace(settings, no_auth=no_auth)
    if db_path is not None:
        settings = replace(settings, db_path=db_path)
    app = FastAPI(
        title="atomics API",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.auth = NoAuth() if settings.no_auth else ApiKeyAuth(settings.api_keys)
    app.middleware("http")(security_headers_middleware)
    app.include_router(router)
    app.include_router(distributed_routes.router)
    if settings.with_dashboard:
        app.include_router(dashboard_router)
    return app
