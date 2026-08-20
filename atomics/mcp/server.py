"""MCP server exposing a running atomics API as tools.

Every tool here is one authenticated HTTP call to `atomics server` (see
`atomics.mcp.client`). The API owns authentication, the per-eval budget ceiling,
and async job scheduling, so an agent driving this server gets exactly the
guardrails a remote HTTP caller gets — no more, and no separate copy of them.

The tool surface is deliberately bounded by what the API already exposes.
`sweep`, `stress`, and `soak` are endpoints with required budgets and hard
caps. `probe` stays CLI-only.

Runs, evals, sweeps, and load jobs are asynchronous: submitting returns a
job id immediately and the agent polls `get_job`. `list_models` and
`provider_test` are the two short synchronous probes. Nothing else blocks
on model work.
"""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from atomics.mcp.client import AtomicsApiClient

INSTRUCTIONS = """Evaluate and benchmark language models through a running atomics API server.

Runs, evals, sweeps, and load jobs are asynchronous. `submit_run`,
`submit_eval`, `submit_sweep`, `submit_stress`, and `submit_soak` return a
job id; poll `get_job` until its status is `completed`.

Read-only tools (`health`, `list_models`, `list_jobs`, `get_job`, `get_run`,
`compare`, `recent_runs`, `trends`) are safe to call freely. `provider_test`
spends a few tokens on a fixed probe.
`submit_run`, `submit_eval`, `submit_sweep`, `submit_stress`, and
`submit_soak` spend real provider tokens and money, so treat them as costly.
`submit_sweep`, `submit_stress`, and `submit_soak` require an explicit
`budget_usd`. Stress caps concurrency at 8 and each phase at 15s. Soak
duration is seconds, 30–300. The CLI still owns hours-long soaks.
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False)
# Submitting spends tokens and money against a real provider account. It creates
# a job rather than mutating existing data, so it is not destructive, but it is
# emphatically not read-only either.
SPENDS = ToolAnnotations(read_only_hint=False, destructive_hint=False)


def build_server(client: AtomicsApiClient | None = None) -> MCPServer:
    """Build the MCP server, optionally against a supplied API client."""
    api = client if client is not None else AtomicsApiClient.from_env()
    server: MCPServer = MCPServer(
        name="atomics",
        title="Atomics model evaluation",
        instructions=INSTRUCTIONS,
        version=_version(),
    )

    @server.tool(annotations=READ_ONLY)
    def list_models(provider: str = "ollama", host: str | None = None) -> Any:
        """List models loaded on an Ollama or vLLM instance.

        `provider` is `ollama` or `vllm`. Pass `host` to override the server's
        configured endpoint. Does not generate and does not spend tokens.
        """
        return api.list_models(provider=provider, host=host)

    @server.tool(annotations=SPENDS)
    def provider_test(
        provider: str,
        model: str | None = None,
        host: str | None = None,
        thinking: bool | None = None,
        effort: str | None = None,
        reasoning_mode: str | None = None,
    ) -> Any:
        """Health-check a provider and generate a fixed 2+2 probe.

        Spends a few tokens. The prompt is fixed server-side. Use this after
        `list_models` to confirm a tag answers before submitting an eval.
        `effort` is the shared reasoning dial (none/low/medium/high/xhigh/max).
        `reasoning_mode` is OpenAI-only (`standard`/`pro`).
        """
        return api.provider_test(
            provider=provider,
            model=model,
            host=host,
            thinking=thinking,
            effort=effort,
            reasoning_mode=reasoning_mode,
        )

    @server.tool(annotations=READ_ONLY)
    def health() -> Any:
        """Check that the atomics API server is reachable and serving.

        Call this first when another tool reports a connection problem.
        """
        return api.health()

    @server.tool(annotations=SPENDS)
    def submit_run(
        provider: str,
        model: str | None = None,
        tier: str = "ez",
        iterations: int = 3,
        interval: int = 5,
        save: bool = True,
    ) -> Any:
        """Start a benchmark run and return its job id immediately.

        Spends provider tokens. `provider` is a provider name such as `claude`,
        `openai`, or `ollama`; omit `model` to use that provider's default.
        `tier` selects the task difficulty profile. Poll `get_job` for the result.
        """
        return api.submit_run(
            provider=provider,
            model=model,
            tier=tier,
            iterations=iterations,
            interval=interval,
            save=save,
        )

    @server.tool(annotations=SPENDS)
    def submit_eval(
        suite: str,
        provider: str,
        model: str | None = None,
        judge_model: str | None = None,
        fixtures: list[str] | None = None,
        save: bool = True,
        budget_usd: float | None = None,
        thinking: bool | None = None,
        effort: str | None = None,
        reasoning_mode: str | None = None,
    ) -> Any:
        """Start an eval suite and return its job id immediately.

        Spends provider tokens, for the judge as well as the model under test.
        `suite` is one of `accuracy`, `rag`, `multiturn`, `adversarial`,
        `codegen`, `refusal`, `redblue`, `toolcall`, or `codereview`.
        `budget_usd` caps the combined spend; leave it unset to accept
        the server's default ceiling. Poll `get_job` for the result.
        """
        return api.submit_eval(
            suite=suite,
            provider=provider,
            model=model,
            judge_model=judge_model,
            fixtures=fixtures,
            save=save,
            budget_usd=budget_usd,
            thinking=thinking,
            effort=effort,
            reasoning_mode=reasoning_mode,
        )

    @server.tool(annotations=SPENDS)
    def submit_sweep(
        provider: str,
        models: list[str],
        suites: list[str],
        budget_usd: float,
        judge_model: str | None = None,
        runs: int = 1,
        thinking: bool | None = None,
        effort: str | None = None,
        reasoning_mode: str | None = None,
    ) -> Any:
        """Start a multi-model, multi-suite campaign and return a job id.

        Spends tokens. `budget_usd` is required. Name the models — there is no
        discover-everything flag; call `list_models` first. `suites` is a list
        from `eval`, `redblue`, `refusal`, `toolcall`, `codereview`. `runs` is
        1–3. Poll `get_job` until `status` is `completed`.
        """
        return api.submit_sweep(
            provider=provider,
            models=models,
            suites=suites,
            budget_usd=budget_usd,
            judge_model=judge_model,
            runs=runs,
            thinking=thinking,
            effort=effort,
            reasoning_mode=reasoning_mode,
        )

    @server.tool(annotations=SPENDS)
    def submit_stress(
        provider: str,
        model: str,
        budget_usd: float,
        max_concurrency: int = 4,
        phase_seconds: float = 10.0,
    ) -> Any:
        """Ramp concurrency to find saturation and return a job id.

        Spends tokens. `budget_usd` is required. One named model. Concurrency
        is 1–8; each phase is at most 15 seconds. Not a hours-long CLI soak.
        Poll `get_job` until `status` is `completed`.
        """
        return api.submit_stress(
            provider=provider,
            model=model,
            budget_usd=budget_usd,
            max_concurrency=max_concurrency,
            phase_seconds=phase_seconds,
        )

    @server.tool(annotations=SPENDS)
    def submit_soak(
        provider: str,
        model: str,
        budget_usd: float,
        duration_seconds: int = 60,
        concurrency: int = 2,
        sample_interval: int = 15,
    ) -> Any:
        """Hold concurrency, classify drift, and return a job id.

        Spends tokens. `budget_usd` is required. `duration_seconds` is 30–300
        (not hours). Concurrency is 1–4. Verdict is STABLE / DEGRADED /
        UNSTABLE. Poll `get_job` until `status` is `completed`.
        """
        return api.submit_soak(
            provider=provider,
            model=model,
            budget_usd=budget_usd,
            duration_seconds=duration_seconds,
            concurrency=concurrency,
            sample_interval=sample_interval,
        )

    @server.tool(annotations=READ_ONLY)
    def get_job(job_id: str) -> Any:
        """Fetch a submitted job's status and, once `completed`, its result."""
        return api.get_job(job_id)

    @server.tool(annotations=READ_ONLY)
    def list_jobs() -> Any:
        """List in-memory API jobs. Results are omitted; poll `get_job` for those."""
        return api.list_jobs()

    @server.tool(annotations=READ_ONLY)
    def get_run(run_id: str) -> Any:
        """Fetch one persisted run and its fixtures.

        Prompts and raw JSON are omitted by the API. This is a recorded run,
        not an in-memory job — use `get_job` for a submission still in flight.
        """
        return api.get_run(run_id)

    @server.tool(annotations=READ_ONLY)
    def trends(hours: int = 24) -> Any:
        """Hourly token and cost series for the last `hours` (1–168)."""
        return api.trends(hours=hours)

    @server.tool(annotations=READ_ONLY)
    def compare(
        by: str = "provider",
        since_hours: float | None = None,
        tier: str | None = None,
        category: str | None = None,
    ) -> Any:
        """Compare recorded results, grouped by `provider` or `model`.

        Reads already-stored results, so it costs nothing and spends nothing.
        """
        return api.compare(by=by, since_hours=since_hours, tier=tier, category=category)

    @server.tool(annotations=READ_ONLY)
    def recent_runs(limit: int = 10) -> Any:
        """List the most recent recorded runs."""
        return api.recent_runs(limit=limit)

    return server


def _version() -> str:
    """The installed atomics version, or a placeholder when unavailable.

    Reported to MCP clients as the server version. An editable checkout without
    installed metadata should still start a server rather than fail here.
    """
    try:
        from atomics import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - defensive; metadata is normally present
        return "0.0.0"


def main() -> None:
    """Run the server on stdio. Entry point for `atomics mcp`."""
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
