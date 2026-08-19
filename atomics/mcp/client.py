"""HTTP client the MCP server uses to reach a running atomics API server.

The MCP server is a pure proxy, and this module is the whole of its reach into
atomics: every tool call becomes one authenticated HTTP request against
`atomics server`. Nothing here builds a provider, opens the metrics database, or
decides what a caller may spend.

That is the point. An MCP client is an LLM agent — a remote, automated caller,
not a local operator spending their own money deliberately. Routing through the
API means the agent inherits the guardrails the API already enforces: API-key
authentication, the per-eval dollar ceiling, and the bounds on iterations and
fixtures in `atomics.api.models`. Re-implementing those here would mean two
copies of a security decision, and the copy an agent talks to would be the one
nobody audits.
"""

from __future__ import annotations

import os
from types import TracebackType
from typing import Any

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8000"
API_URL_ENV = "ATOMICS_API_URL"
API_KEY_ENV = "ATOMICS_API_KEY"
API_PREFIX = "/api/v1"

# Long enough to outlast a slow report query, short enough that an agent gets an
# error instead of hanging. Runs and evals are submitted as jobs and answered
# immediately with an id, so no tool call here waits on actual model work.
DEFAULT_TIMEOUT_SECONDS = 30.0


class AtomicsApiError(RuntimeError):
    """A request to the atomics API failed.

    Carries the HTTP status and the server's own `detail` string so a tool
    reports why it failed — unknown provider, budget exceeded, no such job —
    rather than a generic transport error. The message survives into the
    `ToolError` the MCP runtime raises, which is what the agent ultimately sees.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AtomicsApiClient:
    """Authenticated HTTP client for one atomics API server."""

    def __init__(
        self,
        base_url: str = DEFAULT_API_URL,
        api_key: str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        headers = {"X-API-Key": api_key} if api_key else {}
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> AtomicsApiClient:
        """Build a client from `ATOMICS_API_URL` and `ATOMICS_API_KEY`.

        `ATOMICS_API_KEY` is the variable the CLI reference and `docs/API_SERVER.md`
        already tell operators to export for API calls, so an MCP client
        configured the same way needs no new credential.
        """
        base_url = os.environ.get(API_URL_ENV) or DEFAULT_API_URL
        api_key = os.environ.get(API_KEY_ENV) or None
        return cls(base_url, api_key, **kwargs)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> AtomicsApiClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{API_PREFIX}{path}"
        try:
            response = self._client.request(method, url, json=json, params=params)
        except httpx.ConnectError as exc:
            # By far the most likely failure, and the one whose default message
            # ("All connection attempts failed") tells an agent nothing about
            # the fix. Name the missing piece instead.
            raise AtomicsApiError(
                f"Cannot reach the atomics API at {self.base_url}. "
                "Start one with `atomics server --api-key ...`, or set ATOMICS_API_URL "
                "to point at a running server."
            ) from exc
        except httpx.HTTPError as exc:
            raise AtomicsApiError(f"Request to {self.base_url}{url} failed: {exc}") from exc
        return self._parse(response)

    @staticmethod
    def _parse(response: httpx.Response) -> Any:
        if response.is_success:
            if not response.content:
                return {}
            try:
                return response.json()
            except ValueError:
                return {"raw": response.text}

        detail: str | None = None
        try:
            body = response.json()
        except ValueError:
            detail = response.text.strip() or None
        else:
            if isinstance(body, dict):
                raw = body.get("detail")
                detail = raw if isinstance(raw, str) else None
                if detail is None and raw is not None:
                    # FastAPI validation errors put a list of field errors here.
                    detail = str(raw)
        raise AtomicsApiError(
            detail or f"HTTP {response.status_code} from the atomics API",
            status_code=response.status_code,
        )

    def health(self) -> Any:
        """Liveness of the API server. The one endpoint that needs no key."""
        return self._request("GET", "/health")

    def submit_run(
        self,
        *,
        provider: str,
        model: str | None = None,
        tier: str = "ez",
        iterations: int = 3,
        interval: int = 5,
        save: bool = True,
    ) -> Any:
        payload: dict[str, Any] = {
            "provider": provider,
            "tier": tier,
            "iterations": iterations,
            "interval": interval,
            "save": save,
        }
        if model is not None:
            payload["model"] = model
        return self._request("POST", "/runs", json=payload)

    def submit_eval(
        self,
        *,
        suite: str,
        provider: str,
        model: str | None = None,
        judge_model: str | None = None,
        fixtures: list[str] | None = None,
        save: bool = True,
        budget_usd: float | None = None,
    ) -> Any:
        payload: dict[str, Any] = {"suite": suite, "provider": provider, "save": save}
        if model is not None:
            payload["model"] = model
        if judge_model is not None:
            payload["judge_model"] = judge_model
        if fixtures is not None:
            payload["fixtures"] = fixtures
        # Omitted rather than defaulted here so the server's DEFAULT_EVAL_BUDGET_USD
        # stays the single definition of what an unspecified budget means.
        if budget_usd is not None:
            payload["budget_usd"] = budget_usd
        return self._request("POST", "/evals", json=payload)

    def get_job(self, job_id: str) -> Any:
        return self._request("GET", f"/jobs/{job_id}")

    def list_jobs(self) -> Any:
        """In-memory API jobs. The list omits `result`; poll `get_job` for that."""
        return self._request("GET", "/jobs")

    def get_run(self, run_id: str) -> Any:
        """One persisted run and its fixtures. Prompts and raw JSON are omitted."""
        return self._request("GET", f"/runs/{run_id}")

    def trends(self, *, hours: int = 24) -> Any:
        return self._request("GET", "/reports/trends", params={"hours": hours})

    def compare(
        self,
        *,
        by: str = "provider",
        since_hours: float | None = None,
        tier: str | None = None,
        category: str | None = None,
    ) -> Any:
        params: dict[str, Any] = {"by": by}
        if since_hours is not None:
            params["since_hours"] = since_hours
        if tier is not None:
            params["tier"] = tier
        if category is not None:
            params["category"] = category
        return self._request("GET", "/compare", params=params)

    def recent_runs(self, *, limit: int = 10) -> Any:
        return self._request("GET", "/reports/recent-runs", params={"limit": limit})

    def list_models(self, *, provider: str = "ollama", host: str | None = None) -> Any:
        params: dict[str, Any] = {"provider": provider}
        if host is not None:
            params["host"] = host
        return self._request("GET", "/models", params=params)

    def provider_test(
        self,
        *,
        provider: str,
        model: str | None = None,
        host: str | None = None,
        thinking: bool | None = None,
    ) -> Any:
        payload: dict[str, Any] = {"provider": provider}
        if model is not None:
            payload["model"] = model
        if host is not None:
            payload["host"] = host
        if thinking is not None:
            payload["thinking"] = thinking
        return self._request("POST", "/provider-test", json=payload)
