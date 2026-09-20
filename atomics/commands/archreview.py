"""Security-architecture review benchmark CLI command."""

from __future__ import annotations

import click
from rich.console import Console
from rich.markup import escape as _rich_escape
from rich.table import Table

from atomics.commands.common import (
    PROVIDER_CHOICES,
    _make_provider,
    budget_option,
    eval_budget_from,
    extra_judges_option,
    parse_extra_judges,
    run_async,
    setup_logging,
    write_summary_json,
)
from atomics.commands.suite_run import finalize_archreview_run, suite_run
from atomics.config import load_settings
from atomics.eval.budget import BudgetMeter
from atomics.providers.base import aclose_providers


@click.command()
@click.option(
    "--repo", "repo_name", required=True, help="Repo spec under atomics/archreview/repos/"
)
@click.option("--models", "models_csv", required=True, help="Comma-separated models under test")
@click.option(
    "--provider", "provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True
)
@click.option("--ollama-host", type=str, default=None)
@click.option("--vllm-host", "vllm_host", type=str, default=None)
@click.option("--region", type=str, default="us-east-1", help="AWS region for Bedrock")
@click.option(
    "--judge-provider",
    "judge_provider_name",
    type=PROVIDER_CHOICES,
    default="ollama",
    show_default=True,
)
@click.option("--judge-model", type=str, default=None)
@click.option("--judge-host", type=str, default=None)
@click.option(
    "--tier",
    type=click.Choice(["floor", "local", "wide", "expanded"]),
    default="floor",
    show_default=True,
)
@click.option(
    "--rounds",
    "--runs",
    "rounds",
    type=int,
    default=1,
    show_default=True,
    help="Number of analysis passes per model (--runs is an alias for cross-suite consistency).",
)
@click.option(
    "--max-output-tokens",
    type=click.IntRange(min=128),
    default=2048,
    show_default=True,
    help="Maximum generated tokens for each model-under-test analysis",
)
@click.option(
    "--inference-timeout",
    type=float,
    default=None,
    help="Per-request provider timeout in seconds (useful for slow local Ollama/vLLM runs)",
)
@click.option(
    "--judge-only",
    is_flag=True,
    default=False,
    help="Skip objective scoring (no answer key needed)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Stream per-model/per-round progress: findings and scores as they complete",
)
@click.option("--save/--no-save", "save_results", default=True)
@click.option(
    "--json-out",
    "json_out",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Write the full run (per-round findings, scores, cost) as JSON to this file.",
)
@extra_judges_option
@budget_option
def archreview(
    repo_name,
    models_csv,
    provider_name,
    ollama_host,
    vllm_host,
    region,
    judge_provider_name,
    judge_model,
    judge_host,
    tier,
    rounds,
    max_output_tokens,
    inference_timeout,
    judge_only,
    verbose,
    save_results,
    json_out,
    extra_judges,
    budget_usd,
):
    """Benchmark models on a security-architecture review of a repo."""
    import os
    from pathlib import Path

    from atomics.archreview.keygen import load_repo_spec
    from atomics.archreview.pack import build_pack
    from atomics.archreview.runner import run_archreview
    from atomics.archreview.scorer import compute_robustness
    from atomics.eval.judge import detect_self_judge

    settings = load_settings()
    setup_logging(settings.log_level)
    console = Console()

    # Like sweep, this builds a provider per model inside the run loop, so the
    # ceiling is held by the meter rather than by any one provider.
    meter = BudgetMeter(eval_budget_from(budget_usd))

    def _build_provider(
        name: str,
        mdl: str | None,
        host: str | None,
        context_tokens: int | None = None,
    ):
        return meter.wrap(
            _make_provider(
                name,
                mdl,
                host,
                settings,
                vllm_host=vllm_host,
                region=region,
                context_tokens=context_tokens,
                inference_timeout=inference_timeout,
            )
        )

    repos_dir = Path(__file__).resolve().parent.parent / "archreview" / "repos"
    if "/" in repo_name or "\\" in repo_name or ".." in repo_name:
        raise click.ClickException(
            f"Invalid repo name: {repo_name!r}. Must be a simple name (no path separators)."
        )
    spec_path = repos_dir / f"{repo_name}.yaml"
    if not spec_path.resolve().is_relative_to(repos_dir.resolve()):
        raise click.ClickException(
            f"Invalid repo name: {repo_name!r}. Path escapes the repos directory."
        )
    if not spec_path.exists():
        available = [p.stem for p in repos_dir.glob("*.yaml")]
        raise click.ClickException(
            f"Unknown repo spec: {repo_name!r}. Available: {', '.join(sorted(available))}"
        )
    spec = load_repo_spec(spec_path)

    repo_dir = os.environ.get(spec.path_env)
    if not repo_dir or not Path(repo_dir).is_dir():
        raise click.ClickException(f"Set {spec.path_env} to the local {spec.name} checkout.")

    tier_config = spec.tier(tier)
    archreview_max_output_tokens = max_output_tokens
    archreview_prompt_overhead_tokens = 4096
    archreview_context_tokens = (
        tier_config.budget_tokens + archreview_prompt_overhead_tokens + archreview_max_output_tokens
    )
    import uuid as _uuid_mod

    archreview_run_id = _uuid_mod.uuid4().hex[:12]

    pack = build_pack(Path(repo_dir), tier_config)
    console.print(
        f"[bold]archreview[/bold] repo=[cyan]{spec.name}[/cyan] tier={tier} "
        f"pack={pack.file_count} files hash={pack.content_hash[:12]} "
        f"context={archreview_context_tokens} reserve={archreview_max_output_tokens} "
        f"overhead={archreview_prompt_overhead_tokens} "
        f"run_id={archreview_run_id} "
        f"{'(truncated)' if pack.truncated else ''}"
    )

    judge_provider = _build_provider(
        judge_provider_name,
        judge_model,
        judge_host or ollama_host or settings.ollama_host,
        context_tokens=8192 if judge_provider_name == "ollama" else None,
    )
    extra_judge_pairs = parse_extra_judges(
        extra_judges,
        build=lambda name, mdl, host: _build_provider(
            name,
            mdl,
            host,
            context_tokens=8192 if name == "ollama" else None,
        ),
        default_host=judge_host or ollama_host or settings.ollama_host,
    )

    judge_label = (
        f"{judge_provider_name}:{judge_model or judge_provider.default_model or 'default'}"
    )

    table = Table(title=f"archreview — {spec.name} ({tier})", show_lines=True)
    table.add_column("Model", no_wrap=True)
    for col in ("Recall", "Prec", "Obj-F", "Judge"):
        table.add_column(col)
    table.add_column("Judge Model", no_wrap=True)
    for col in ("Stability", "Findings"):
        table.add_column(col)

    models = [m.strip() for m in models_csv.split(",") if m.strip()]

    # Drive every model in a SINGLE event loop. The judge provider is built once
    # and its async HTTP client binds to whatever loop first uses it; a per-model
    # asyncio.run() would close that loop after model 1 and break the judge on
    # later models ("Event loop is closed").
    all_results = []

    with suite_run(
        suite="archreview",
        db_path=settings.db_path,
        save=save_results,
        finalize=finalize_archreview_run,
        failure_prefix="Architecture review failed",
    ) as run:
        # Parent run row so archreview runs are listable/queryable like other
        # suites; per-round rows land in archreview_results.
        run.begin(
            archreview_run_id,
            provider=provider_name,
            model=models_csv,
        )
        repo = run.repository

        async def _run_all() -> None:
            created: list = []
            try:
                for mdl in models:
                    test_provider = _build_provider(
                        provider_name,
                        mdl,
                        ollama_host if provider_name == "ollama" else vllm_host,
                        context_tokens=archreview_context_tokens
                        if provider_name == "ollama"
                        else None,
                    )
                    created.append(test_provider)
                    collisions = detect_self_judge(
                        test_provider, mdl, [(judge_provider, judge_model), *extra_judge_pairs]
                    )
                    if collisions:
                        console.print(
                            f"[yellow]warning:[/yellow] judge collides with "
                            f"model under test: {collisions}"
                        )

                    if verbose:
                        console.print(
                            f"\n[bold]→ analyzing with [cyan]{mdl}[/cyan][/bold] "
                            f"({provider_name}, {rounds} round{'s' if rounds != 1 else ''})…"
                        )

                    results = await run_archreview(
                        spec=spec,
                        tier=tier,
                        pack=pack,
                        under_test=test_provider,
                        under_test_model=mdl,
                        judge=judge_provider,
                        judge_model=judge_model,
                        extra_judges=extra_judge_pairs,
                        rounds=rounds,
                        objective=not judge_only,
                        max_output_tokens=max_output_tokens,
                        run_id=archreview_run_id,
                    )
                    all_results.extend(results)
                    if repo:
                        for r in results:
                            repo.save_archreview_result(r)

                    if verbose:
                        for r in results:
                            if r.error_message:
                                console.print(
                                    f"  [red]round {r.round}: "
                                    f"{_rich_escape(r.error_class or '')}: "
                                    f"{_rich_escape(r.error_message or '')}[/red]"
                                )
                                continue
                            judge_str = f"{r.judge_score:.2f}" if r.judge_score is not None else "—"
                            flag = " [yellow](parse failed)[/yellow]" if r.parse_failed else ""
                            console.print(
                                f"  [dim]round {r.round}:[/dim] "
                                f"recall=[green]{r.objective_recall:.2f}[/green] "
                                f"prec={r.objective_precision:.2f} obj-f={r.objective_f:.2f} "
                                f"judge=[magenta]{judge_str}[/magenta] findings={len(r.findings)}"
                                f" matched={r.matched_categories or '—'}{flag}"
                            )
                            for f in r.findings:
                                console.print(
                                    f"      [dim]•[/dim] {f.category} · {f.location} · {f.severity}"
                                )

                    cat_sets = [{f.category for f in r.findings} for r in results]
                    recalls = [r.objective_recall for r in results]
                    stability, _sd = compute_robustness(cat_sets, recalls)
                    avg = lambda xs: round(sum(xs) / len(xs), 3) if xs else 0.0  # noqa: E731
                    judge_vals = [r.judge_score for r in results if r.judge_score is not None]
                    table.add_row(
                        mdl,
                        str(avg(recalls)),
                        str(avg([r.objective_precision for r in results])),
                        str(avg([r.objective_f for r in results])),
                        str(avg(judge_vals) if judge_vals else "—"),
                        judge_label,
                        str(stability),
                        str(round(sum(len(r.findings) for r in results) / len(results), 1)),
                    )

            finally:
                await aclose_providers(*created)

        run_async(
            _run_all(),
            judge_provider,
            *(p for p, _ in extra_judge_pairs),
        )

        console.print(table)

        if json_out:
            from atomics.archreview.models import ArchReviewSummary

            summary = ArchReviewSummary(repo=spec.name, tier=tier, results=all_results)
            write_summary_json(summary, Path(json_out))
            console.print(f"[dim]Wrote JSON results to {json_out}[/dim]")
