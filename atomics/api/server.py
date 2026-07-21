"""FastAPI app factory for the atomics API server."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI

from atomics.api.auth import ApiKeyAuth, NoAuth
from atomics.api.config import ServerSettings
from atomics.api.jobs import JobManager
from atomics.api.routes import router
from atomics.distributed import routes as distributed_routes
from atomics.distributed.auth import WorkerAuth
from atomics.distributed.coordinator import Coordinator
from atomics.storage.schema import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: ServerSettings = app.state.settings
    app.state.job_manager = JobManager()
    if settings.no_auth:
        app.state.worker_auth = NoAuth()
    else:
        app.state.worker_auth = WorkerAuth(set(settings.api_keys))
    app.state.coordinator = Coordinator(init_db(settings.db_path))
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
        version="0.11.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.auth = NoAuth() if settings.no_auth else ApiKeyAuth(settings.api_keys)
    app.include_router(router)
    app.include_router(distributed_routes.router)
    return app
