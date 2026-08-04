"""Shared FastAPI dependencies for the API server.

Both the main API router and the distributed coordinator router authenticate
requests. The dependencies live here rather than in either routes module so that
neither has to import the other.

The two backends are distinguished by audience rather than by key material:
``app.state.auth`` guards submitter-facing endpoints and ``app.state.worker_auth``
guards the worker lifecycle. They are built from the same key set today, so
keeping them separate is what lets a deployment diverge them later without
touching any route definition.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from atomics.api.auth import AuthBackend


def get_auth(request: Request) -> AuthBackend:
    return request.app.state.auth


async def require_auth(request: Request, auth: AuthBackend = Depends(get_auth)) -> str:
    """Authenticate a submitter-facing request and identify the caller.

    Returns the caller identifier so routes can attribute work without
    repeating the lookup, and so quota accounting and request logs agree on who
    the caller was. The identifier is a digest, never the key.
    """
    if not await auth.authenticate(request):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    caller = auth.identify(request)
    # The logging middleware runs outside the dependency graph, so it reads the
    # caller from request state rather than from this return value.
    request.state.caller_id = caller
    return caller


def get_worker_auth(request: Request) -> AuthBackend:
    return request.app.state.worker_auth


async def require_worker_auth(
    request: Request, auth: AuthBackend = Depends(get_worker_auth)
) -> str:
    """Authenticate a request from a worker process."""
    if not await auth.authenticate(request):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid worker API key",
        )
    caller = auth.identify(request)
    request.state.caller_id = caller
    return caller
