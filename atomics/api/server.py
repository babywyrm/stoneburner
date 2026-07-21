"""FastAPI app factory for the atomics API server."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from atomics.api.auth import ApiKeyAuth, NoAuth
from atomics.api.config import ServerSettings
from atomics.api.jobs import JobManager
from atomics.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.job_manager = JobManager()
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


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    settings = settings or ServerSettings()
    app = FastAPI(
        title="atomics API",
        version="0.11.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.auth = NoAuth() if settings.no_auth else ApiKeyAuth(settings.api_keys)
    app.include_router(router)
    return app
