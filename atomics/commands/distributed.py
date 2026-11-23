"""Distributed run CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from atomics.commands.common import PROVIDER_CHOICES
from atomics.mcp.client import AtomicsApiClient, AtomicsApiError
from atomics.validation import validate_endpoint_url


@click.group()
def distributed() -> None:
    """Distributed benchmark runs across multiple workers."""


@distributed.command()
@click.option("--coordinator", default="http://127.0.0.1:8000", show_default=True)
@click.option("--api-key", envvar="ATOMICS_API_KEY", help="Client API key")
@click.option(
    "--mode",
    default="split",
    show_default=True,
    type=click.Choice(["split", "fleet", "full"]),
    help="split: divide the tasks across workers. fleet: run every task on every "
    "matching worker, for comparing hosts. full: delegate the entire run to one "
    "worker so it executes the full LoopEngine locally.",
)
@click.option(
    "--provider",
    "-p",
    type=PROVIDER_CHOICES,
    help="Pin every task to this provider. Default: each worker's own provider.",
)
@click.option("--tier", "-t", default="baseline", show_default=True)
@click.option("--model", "-m", help="Model override for the executing provider")
@click.option("-n", "iterations", default=1, show_default=True, help="Number of tasks")
@click.option(
    "--label",
    "labels",
    multiple=True,
    help="Worker selector key=value, repeatable. Fleet and full mode only; a worker "
    "must match every pair. Omit to broadcast to all online workers.",
)
@click.option(
    "--runtime",
    default="python",
    show_default=True,
    help="Required worker capability tag for generated tasks.",
)
@click.option(
    "--suite",
    type=click.Choice(
        [
            "accuracy",
            "rag",
            "multiturn",
            "adversarial",
            "codegen",
            "refusal",
            "redblue",
            "toolcall",
            "codereview",
        ],
        case_sensitive=False,
    ),
    default=None,
    help="Dispatch an eval suite (`eval_request`) instead of a burn-tier benchmark "
    "(fleet or full mode only).",
)
@click.option("--host", default=None, help="Target inference host URL override for --suite.")
@click.option("--judge-model", default=None, help="Judge model override for --suite.")
@click.option("--judge-host", default=None, help="Judge host URL override for --suite.")
@click.option(
    "--runs",
    type=click.IntRange(min=1),
    default=None,
    help="Passes per fixture for --suite.",
)
@click.option(
    "--budget",
    "budget_usd",
    type=float,
    default=None,
    help="Per-worker USD ceiling for --suite.",
)
@click.option(
    "--fixtures",
    "fixtures_filter",
    type=str,
    default=None,
    help="Comma-separated fixture IDs for --suite.",
)
@click.option(
    "--channel",
    type=click.Choice(["both", "tools", "prose"], case_sensitive=False),
    default=None,
    help="Channel for --suite toolcall.",
)
@click.option("--temperature", type=float, default=None, help="Sampling temperature for --suite.")
def run(
    coordinator: str,
    api_key: str,
    mode: str,
    provider: str | None,
    tier: str,
    model: str | None,
    iterations: int,
    labels: tuple[str, ...],
    runtime: str,
    suite: str | None,
    host: str | None,
    judge_model: str | None,
    judge_host: str | None,
    runs: int | None,
    budget_usd: float | None,
    fixtures_filter: str | None,
    channel: str | None,
    temperature: float | None,
) -> None:
    """Submit a distributed run to the coordinator."""
    if not api_key:
        raise click.UsageError("--api-key is required (or set ATOMICS_API_KEY)")
    try:
        coordinator = validate_endpoint_url(coordinator, label="--coordinator")
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--coordinator") from exc
    if suite is not None and mode not in ("fleet", "full"):
        raise click.UsageError(
            "--suite requires --mode full or --mode fleet; split mode only dispatches "
            "burn-tier tasks."
        )
    if labels and mode not in ("fleet", "full"):
        raise click.UsageError(
            "--label only applies to fleet and full mode: split assigns each task to the "
            "next available worker, so a selector would be silently ignored. "
            "Use --mode fleet or --mode full to target workers by label."
        )
    selector: dict[str, str] = {}
    for label in labels:
        if "=" not in label:
            raise click.BadParameter(f"Label must be key=value: {label!r}")
        key, value = label.split("=", 1)
        selector[key] = value
    eval_request: dict[str, object] | None = None
    run_request: dict[str, object] | None = None
    if suite is not None:
        eval_request = {"suite": suite.lower()}
        if provider:
            eval_request["provider"] = provider
        if model:
            eval_request["model"] = model
        if host:
            eval_request["host"] = host
        if judge_model:
            eval_request["judge_model"] = judge_model
        if judge_host:
            eval_request["judge_host"] = judge_host
        if runs is not None:
            eval_request["runs"] = runs
        if budget_usd is not None:
            eval_request["budget_usd"] = budget_usd
        if fixtures_filter is not None:
            parsed_ids = [part.strip() for part in fixtures_filter.split(",") if part.strip()]
            if not parsed_ids:
                raise click.BadParameter("no fixture IDs", param_hint="--fixtures")
            eval_request["fixtures"] = parsed_ids
        if channel is not None:
            eval_request["channel"] = channel.lower()
        if temperature is not None:
            eval_request["temperature"] = temperature
    else:
        run_request = {"tier": tier, "iterations": iterations, "runtime": runtime}
        if provider:
            run_request["provider"] = provider
        if model:
            run_request["model"] = model
    try:
        with AtomicsApiClient(coordinator, api_key) as client:
            data = client.submit_distributed_run(
                mode=mode,
                run_request=run_request,
                eval_request=eval_request,
                worker_selector=selector or None,
            )
    except AtomicsApiError as exc:
        raise click.ClickException(str(exc)) from exc
    Console().print(f"Submitted distributed run: {data['job_id']}")


def _as_float(val: object) -> float:
    return float(val) if isinstance(val, (int, float)) else 0.0


def _render_fleet_table(job: dict[str, object], summary: dict[str, object]) -> None:
    """Print one row per host so the comparison is readable at a glance.

    Nine columns do not fit an 80-column terminal: Rich shrinks them until a
    worker id reads `5b5fc…` and a label reads `box=a…`, which cannot answer
    which host was faster — the only question fleet mode exists to answer. So
    identifiers are allowed to wrap instead of truncate, and the model moves to
    the caption when every host ran the same one, which is the usual case for a
    comparison and buys a whole column back.
    """
    raw_workers = summary.get("workers")
    workers = (
        [w for w in raw_workers if isinstance(w, dict)] if isinstance(raw_workers, list) else []
    )
    # A host that completed nothing has no model to report, and must not be the
    # reason the column reappears: the run where one host died is exactly when
    # the remaining numbers most need the room.
    reported_models = sorted({str(w.get("model")) for w in workers if w.get("model")})
    show_model_column = len(reported_models) > 1
    uniform_model = reported_models[0] if len(reported_models) == 1 else None
    is_eval = any(
        w.get("suite") or w.get("overall_score") is not None or w.get("integrity") is not None
        for w in workers
    )

    job_label = escape(str(job.get("job_id", "")))
    status_label = escape(str(job.get("status", "")))
    table = Table(
        title=f"Fleet run {job_label} — {status_label}",
        caption=f"model: {escape(uniform_model)}" if uniform_model else None,
    )
    # A 12-hex worker id has to arrive in one piece; wrapped after 11 characters
    # it is no more usable than the truncation this replaced.
    table.add_column("Worker", overflow="fold", min_width=12)
    table.add_column("Labels", overflow="fold")
    if show_model_column:
        table.add_column("Model", overflow="fold")
    table.add_column("Done", justify="right")
    table.add_column("Fail", justify="right")
    if is_eval:
        table.add_column("Score", justify="right")
        table.add_column("Integrity")
    table.add_column("Mean ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("Tok/s", justify="right")
    table.add_column("Cost", justify="right")

    for worker in workers:
        raw_labels = worker.get("labels")
        labels_dict = raw_labels if isinstance(raw_labels, dict) else {}
        labels = " ".join(f"{k}={v}" for k, v in labels_dict.items())
        failed = worker.get("failed", 0)
        row = [
            escape(str(worker.get("worker_id", ""))),
            escape(labels) if labels else "-",
        ]
        if show_model_column:
            row.append(escape(str(worker.get("model") or "-")))
        row += [
            str(worker.get("completed", 0)),
            f"[red]{failed}[/red]" if failed else "0",
        ]
        if is_eval:
            raw_score = worker.get("overall_score")
            score_txt = (
                f"{float(raw_score):.3f}"
                if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool)
                else "-"
            )
            row += [score_txt, escape(str(worker.get("integrity") or "-"))]
        row += [
            f"{_as_float(worker.get('mean_latency_ms')):.1f}",
            f"{_as_float(worker.get('p95_latency_ms')):.1f}",
            # No decimals: the fractional part of a tokens-per-second figure is
            # noise, and the two characters it costs pushed the table past 80
            # columns, which clipped the cost column off the right edge.
            f"{_as_float(worker.get('mean_tokens_per_second')):.0f}",
            f"{_as_float(worker.get('estimated_cost_usd')):.4f}",
        ]
        table.add_row(*row)

    console = Console()
    console.print(table)
    raw_integrity = summary.get("integrity")
    integrity_suffix = ""
    if isinstance(raw_integrity, dict) and raw_integrity.get("status"):
        integrity_suffix = f", integrity: {escape(str(raw_integrity['status']))}"
    console.print(
        f"Total: {summary.get('completed', 0)} completed, "
        f"{summary.get('failed', 0)} failed, "
        f"{summary.get('total_output_tokens', 0)} output tokens, "
        f"${summary.get('estimated_cost_usd', 0.0)}{integrity_suffix}"
    )


def _render_full_table(job: dict[str, object], summary: dict[str, object]) -> None:
    """Print a compact summary of a single full-run delegation."""
    nested = summary.get("summary")
    run_summary = nested if isinstance(nested, dict) else summary
    job_label = escape(str(job.get("job_id", "")))
    status_label = escape(str(job.get("status", "")))
    table = Table(title=f"Full run {job_label} — {status_label}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    if "suite" in run_summary or "overall_score" in run_summary:
        raw_integrity = run_summary.get("integrity")
        int_status = (
            str(raw_integrity.get("status", "-"))
            if isinstance(raw_integrity, dict)
            else str(raw_integrity or "-")
        )
        raw_score = run_summary.get("overall_score")
        score_str = (
            f"{float(raw_score):.3f}"
            if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool)
            else "n/a"
        )
        table.add_row("Suite", escape(str(run_summary.get("suite", "-"))))
        table.add_row("Score", score_str)
        table.add_row("Integrity", escape(int_status))
        table.add_row("Fixtures", str(run_summary.get("fixtures_run", 0)))
        table.add_row("Tokens", str(run_summary.get("total_tokens", 0)))
        table.add_row("Cost", f"${run_summary.get('total_cost_usd', 0.0)}")
        Console().print(table)
        return
    table.add_row("Tasks", str(run_summary.get("total_tasks", 0)))
    table.add_row("Success", str(run_summary.get("successful_tasks", 0)))
    table.add_row("Failed", str(run_summary.get("failed_tasks", 0)))
    table.add_row("Tokens", str(run_summary.get("total_tokens", 0)))
    table.add_row("Cost", f"${run_summary.get('total_cost_usd', 0.0)}")
    table.add_row("Avg ms", f"{_as_float(run_summary.get('avg_latency_ms')):.1f}")
    Console().print(table)


@distributed.command()
@click.option("--coordinator", default="http://127.0.0.1:8000", show_default=True)
@click.option("--api-key", envvar="ATOMICS_API_KEY", help="Client API key")
@click.option(
    "-o",
    "--json-out",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Write the job and its per-worker rollup to this file.",
)
@click.argument("job_id")
def status(coordinator: str, api_key: str, job_id: str, json_out: Path | None) -> None:
    """Check status of a distributed run."""
    if not api_key:
        raise click.UsageError("--api-key is required (or set ATOMICS_API_KEY)")
    try:
        coordinator = validate_endpoint_url(coordinator, label="--coordinator")
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--coordinator") from exc
    try:
        with AtomicsApiClient(coordinator, api_key) as client:
            job = client.get_distributed_run(job_id)
    except AtomicsApiError as exc:
        raise click.ClickException(str(exc)) from exc

    summary: dict[str, object] = {}
    raw_summary = job.get("summary_json")
    if isinstance(raw_summary, str) and raw_summary:
        try:
            loaded = json.loads(raw_summary)
            if isinstance(loaded, dict):
                summary = loaded
        except json.JSONDecodeError:
            summary = {}

    if json_out:
        from atomics.paths import write_private_text

        write_private_text(json_out, json.dumps({**job, "summary": summary}, indent=2) + "\n")
        Console().print(f"Wrote {json_out}")
        return

    # A fleet run's whole purpose is the per-host comparison, which does not read
    # well as nested JSON. Split runs keep the machine-readable output they had.
    if job.get("mode") == "fleet" and summary.get("workers"):
        _render_fleet_table(job, summary)
        return
    if job.get("mode") == "full" and summary:
        _render_full_table(job, summary)
        return
    click.echo(json.dumps(job, indent=2))
