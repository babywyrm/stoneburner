"""Security-focused CLI commands."""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.markup import escape
from rich.markup import escape as _rich_escape
from rich.table import Table

from atomics.commands.common import (
    PROVIDER_CHOICES,
    FixtureProgress,
    _attribution_model,
    _make_provider,
    effective_model,
    evaluation_record_from_fixture,
    integrity_exit_code,
    setup_logging,
    write_summary_json,
)
from atomics.config import load_settings
from atomics.storage import MetricsRepository
from atomics.validation import sanitize_error

if TYPE_CHECKING:
    from atomics.eval.codereview.fixtures import SecureCodeFixture
    from atomics.eval.codereview.runner import CodeReviewResult, CodeReviewSummary
    from atomics.eval.refusal.fixtures import RefusalFixture
    from atomics.eval.refusal.runner import RefusalResult, RefusalSummary

_KNOWN_PROVIDERS = {"claude", "bedrock", "openai", "ollama", "vllm", "brain-gateway"}

def _parse_model_spec(spec: str, default_provider: str) -> tuple[str, str, str | None]:
    """Parse a `model`, `provider:model`, or `provider:model@host` spec.

    Ollama model names contain colons (e.g. ``qwen2.5:7b``), so a leading
    ``prefix:`` is only treated as a provider when the prefix is a known
    provider name; otherwise the whole spec (minus any ``@host``) is the model.
    Returns (provider_name, model, host_or_None).
    """
    host: str | None = None
    spec = spec.strip()
    if "@" in spec:
        spec, host = spec.rsplit("@", 1)
    if ":" in spec and spec.split(":", 1)[0].lower() in _KNOWN_PROVIDERS:
        provider_name, model = spec.split(":", 1)
        return provider_name, model, host
    return default_provider, spec, host

@click.command("adversarial")
@click.option("--provider", "-p", "provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--model", "-m", type=str, default=None, help="Model override for the provider under test.")
@click.option("--ollama-host", type=str, default=None, help="Ollama base URL for the model under test.")
@click.option("--vllm-host", "vllm_host", type=str, default=None, help="vLLM/OpenAI-compatible base URL for the model under test.")
@click.option("--judge-provider", "judge_provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--judge-model", type=str, default=None, help="Primary judge model override.")
@click.option("--judge-host", type=str, default=None, help="Ollama base URL for the primary judge.")
@click.option("--extra-judges", type=str, default=None,
              help="Comma-separated extra judges for consensus scoring. "
                   "Format: provider:model or provider:model@host. "
                   "Example: ollama:deepseek-r1:14b,claude:claude-sonnet-4-6")
@click.option("--runs", type=click.IntRange(min=1), default=1, show_default=True,
              help="Run each fixture N times and report mean ± stddev (use 3+ for variance analysis).")
@click.option("--category", type=str, default=None,
              help="Comma-separated categories or group aliases to run (default: all). "
                   "Group aliases: zerotrust, agentic, mcp, tool_safety, multiturn, "
                   "rag_poisoning, tool_desc_injection. "
                   "Base categories: prompt_injection, role_confusion, context_escape, "
                   "instruction_override, social_engineering, data_exfil_attempt.")
@click.option("--thinking/--no-thinking", "thinking_flag", default=None,
              help="Force thinking mode on or off (default: auto-detect).")
@click.option("--thinking-budget", type=int, default=8000, show_default=True)
@click.option("--save/--no-save", "save_results", default=True, show_default=True)
@click.option("--json-out", "json_out", type=click.Path(dir_okay=False, writable=True), default=None,
              help="Write the full run (per-fixture scores, rationales, latency, cost) as JSON to this file.")
@click.option("--compare", "compare_model", type=str, default=None,
              help="Run a second model on the same fixtures and print a per-fixture diff. "
                   "Format: model, provider:model, or provider:model@host.")
@click.option("--fail-on-resilience", "fail_on_resilience", type=float, default=None,
              help="Exit non-zero if overall severity-weighted resilience %% is below this threshold (CI gate).")
@click.option("--allow-partial", is_flag=True,
              help="Allow partial/invalid execution to exit zero after diagnostics.")
@click.option("--verbose", "-v", is_flag=True, help="Show full prompt, model response, and judge reasoning for each fixture.")
def adversarial(
    provider_name: str,
    model: str | None,
    ollama_host: str | None,
    vllm_host: str | None,
    judge_provider_name: str,
    judge_model: str | None,
    judge_host: str | None,
    extra_judges: str | None,
    runs: int,
    category: str | None,
    thinking_flag: bool | None,
    thinking_budget: int,
    save_results: bool,
    json_out: str | None,
    compare_model: str | None,
    fail_on_resilience: float | None,
    allow_partial: bool,
    verbose: bool,
) -> None:
    """Run adversarial LLM resilience eval — measures resistance to manipulation.

    Use --runs 3 for variance-aware scoring. Use --extra-judges for consensus.

    \b
    Examples:
      atomics adversarial --provider ollama -m qwen3:14b --runs 3
      atomics adversarial --judge-model deepseek-r1:14b --extra-judges "claude:claude-sonnet-4-6"
      atomics adversarial --runs 3 --extra-judges "ollama:deepseek-r1:14b@http://ollama-host:11434"
    """
    from atomics.eval.adversarial import select_fixtures
    from atomics.eval.adversarial.runner import run_adversarial

    console = Console()
    settings = load_settings()
    provider = _make_provider(provider_name, model, ollama_host, settings, vllm_host=vllm_host)
    judge = _make_provider(judge_provider_name, judge_model, judge_host or ollama_host, settings, vllm_host=vllm_host)
    actual_provider_name = provider.name
    effective_model = _attribution_model(provider, model)
    actual_judge_name = judge.name
    effective_judge_model = _attribution_model(judge, judge_model)
    categories = [c.strip() for c in category.split(",")] if category else None
    selected_count = len(select_fixtures(categories))

    # Parse --extra-judges "ollama:deepseek-r1:14b@http://host:port,claude:model"
    extra_judge_pairs: list[tuple] = []
    if extra_judges:
        for spec in extra_judges.split(","):
            spec = spec.strip()
            host_override = None
            if "@" in spec:
                spec, host_override = spec.rsplit("@", 1)
            parts = spec.split(":", 1)
            ej_provider_name = parts[0]
            ej_model = parts[1] if len(parts) > 1 else None
            ej_provider = _make_provider(ej_provider_name, ej_model, host_override or judge_host or ollama_host, settings, vllm_host=vllm_host)
            extra_judge_pairs.append((ej_provider, ej_model))

    judge_label = effective_judge_model
    if extra_judge_pairs:
        judge_label += f" + {len(extra_judge_pairs)} extra"

    console.print(
        f"\n[bold]Adversarial eval[/bold] — model under test: [cyan]{actual_provider_name}[/cyan] "
        f"({effective_model})\n"
        f"Judge: [cyan]{actual_judge_name}[/cyan] ({judge_label}) | "
        f"Runs per fixture: [bold]{runs}[/bold]\n"
        f"Fixtures: [bold]{selected_count}[/bold] | "
        f"Categories: [bold]{category or 'all'}[/bold]\n"
    )

    ctx = click.get_current_context()
    created_run_ids: list[str] = []
    repo = None
    if save_results:
        from atomics.storage.repository import MetricsRepository
        from atomics.validation import sanitize_error

        repository = MetricsRepository(settings.db_path)
        repo = repository
        repository_cleaned = False

        def finalize_repository(*, fail_closed: bool = False) -> None:
            nonlocal repository_cleaned
            if repository_cleaned:
                return
            repository_cleaned = True
            failures: list[str] = []
            for created_run_id in created_run_ids:
                try:
                    repository.complete_adversarial_run(created_run_id)
                except Exception as exc:
                    failures.append(
                        f"Failed to finalize adversarial run {created_run_id}: "
                        f"{sanitize_error(exc)}"
                    )
            try:
                repository.close()
            except Exception as exc:
                failures.append(
                    "Failed to close adversarial repository: "
                    f"{sanitize_error(exc)}"
                )
            if failures:
                message = "; ".join(failures)
                if fail_closed:
                    raise click.ClickException(message)
                logging.getLogger("atomics.cli").error(message)

        ctx.call_on_close(finalize_repository)

    run_id = __import__("uuid").uuid4().hex[:12]
    if repo:
        # Parent run row so the adversarial run is listable/queryable alongside
        # other suites; the results themselves land in adversarial_results.
        repo.create_run(
            run_id,
            tier="adversarial",
            provider=actual_provider_name,
            model=effective_model,
            trigger="manual",
        )
        created_run_ids.append(run_id)

    # Actual fixture count (may be filtered by --category) via the single source
    # of truth so header/progress/run all agree.
    from atomics.eval.adversarial import select_fixtures
    adv_fixture_count = len(select_fixtures(categories))

    show_progress = ctx.obj.get("progress", True) if ctx.obj else True
    progress = FixtureProgress(adv_fixture_count, console, label="adversarial") if show_progress else None

    def on_start(idx, fixture):
        if progress:
            progress.on_start(idx, fixture.id, fixture.category)

    def on_done(fr):
        res = fr.resistance
        if progress:
            progress.on_done(0)
        if res:
            color = "green" if res.label == "resisted" else ("yellow" if res.label == "partial" else "red")
            icon = "✓" if res.label == "resisted" else ("~" if res.label == "partial" else "✗")
            run_tag = f" ×{runs}" if runs > 1 else ""
            judge_tag = f" [{len(fr.run_scores)} scores]" if fr.run_scores else ""
            # Score line (always shown)
            console.print(
                f" [{icon}] [bold]{fr.fixture.id}[/bold] ({fr.fixture.category}){run_tag} "
                f"[{color}]{res.label}[/] ({res.score:.2f}){judge_tag}",
            )
            # Rationale: full in verbose, first sentence otherwise
            if verbose:
                console.print(f"     [dim]{_rich_escape(res.rationale)}[/dim]", soft_wrap=True)
            else:
                first_sentence = res.rationale.split(". ")[0].strip()
                if first_sentence:
                    console.print(f"     [dim]{_rich_escape(first_sentence)}.[/dim]")
            console.print()
        if repo:
            repo.save_adversarial_result(
                run_id,
                fr,
                thinking_enabled=thinking_flag is True,
                provider=actual_provider_name,
                model=effective_model,
            )

    summary = asyncio.run(run_adversarial(
        provider,
        judge_provider=judge,
        model=model,
        judge_model=judge_model,
        extra_judges=extra_judge_pairs,
        categories=categories,
        runs=runs,
        run_id=run_id,
        thinking=thinking_flag,
        thinking_budget=thinking_budget,
        on_fixture_start=on_start,
        on_fixture_done=on_done,
        verbose=verbose,
    ))

    title = f"Adversarial Resilience Summary (runs={summary.runs}, judges={len(summary.judges)})"
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Provider", actual_provider_name)
    table.add_row("Model", effective_model)
    table.add_row("Judge", f"{actual_judge_name} / {effective_judge_model}")
    resilience_str = (
        f"{summary.overall_resilience * 100:.1f}%"
        if summary.overall_resilience is not None
        else "N/A"
    )
    if summary.overall_resilience is not None and summary.resilience_stddev is not None:
        resilience_str += f"  ±{summary.resilience_stddev * 100:.1f}%"
    table.add_row("Overall Resilience", resilience_str)
    table.add_row("Runs per fixture", str(summary.runs))
    table.add_row("Judges", ", ".join(summary.judges))
    table.add_row("Fixtures Run", str(summary.total_fixtures))
    integrity = summary.integrity
    table.add_row("Integrity Status", integrity.status.value)
    table.add_row(
        "Fixture Coverage",
        f"{integrity.fixtures_scored}/{integrity.fixtures_total} "
        f"({integrity.fixture_coverage * 100:.1f}%)",
    )
    table.add_row(
        "Attempt Coverage",
        f"{integrity.attempts_scored}/{integrity.attempts_total} "
        f"({integrity.attempt_coverage * 100:.1f}%)",
    )
    table.add_row("Generation Failures", str(integrity.generation_failures))
    table.add_row("Infrastructure Failures", str(integrity.infrastructure_failures))
    table.add_row("Judge Failures", str(integrity.judge_failures))
    table.add_row(
        "Total / Scored Attempts",
        f"{integrity.attempts_total} / {integrity.attempts_scored}",
    )
    table.add_row("Critical Failures", str(len(summary.critical_failures)))
    # Cost + token summary
    total_cost = sum(fr.estimated_cost_usd for fr in summary.fixture_results)
    total_latency = sum(fr.latency_ms for fr in summary.fixture_results)
    avg_latency = total_latency / max(len(summary.fixture_results), 1)
    table.add_row("Total Cost", f"${total_cost:.6f}")
    table.add_row("Avg Latency", f"{avg_latency:.0f}ms")
    table.add_row("Total Latency", f"{total_latency / 1000:.1f}s")
    for cat, score in sorted(summary.category_scores.items()):
        table.add_row(f"  {cat}", f"{score * 100:.1f}%")
    console.print(table)

    if summary.critical_failures:
        console.print(
            f"\n[bold red]⚠ {len(summary.critical_failures)} CRITICAL/HIGH fixture(s) where model complied:[/bold red]"
        )
        for fr in summary.critical_failures:
            console.print(f"  • {fr.fixture.id} [{fr.fixture.severity}] {fr.fixture.category}")

    # ── --compare: run a second model on the same fixtures and diff ──────────
    compare_summary = None
    if compare_model:
        cmp_provider_name, cmp_requested_model, cmp_host = _parse_model_spec(
            compare_model, provider_name
        )
        cmp_provider = _make_provider(
            cmp_provider_name,
            cmp_requested_model,
            cmp_host or ollama_host,
            settings,
            vllm_host=vllm_host,
        )
        actual_cmp_provider_name = cmp_provider.name
        effective_cmp_model = _attribution_model(cmp_provider, cmp_requested_model)
        console.print(
            f"\n[bold]Compare run[/bold] — model B: "
            f"[cyan]{actual_cmp_provider_name}[/cyan] ({effective_cmp_model})\n"
        )
        cmp_run_id = __import__("uuid").uuid4().hex[:12]
        if repo:
            repo.create_run(
                cmp_run_id,
                tier="adversarial",
                provider=actual_cmp_provider_name,
                model=effective_cmp_model,
                trigger="manual",
            )
            created_run_ids.append(cmp_run_id)
        def on_compare_done(fr):
            if repo:
                repo.save_adversarial_result(
                    cmp_run_id,
                    fr,
                    thinking_enabled=thinking_flag is True,
                    provider=actual_cmp_provider_name,
                    model=effective_cmp_model,
                )

        compare_summary = asyncio.run(run_adversarial(
            cmp_provider,
            judge_provider=judge,
            model=cmp_requested_model,
            judge_model=judge_model,
            extra_judges=extra_judge_pairs,
            categories=categories,
            runs=runs,
            run_id=cmp_run_id,
            thinking=thinking_flag,
            thinking_budget=thinking_budget,
            on_fixture_done=on_compare_done,
        ))

        a_label = effective_model
        b_label = effective_cmp_model
        diff = Table(title=f"Per-fixture comparison: A={a_label}  vs  B={b_label}")
        diff.add_column("Fixture")
        diff.add_column("Sev")
        diff.add_column(f"A ({a_label})", justify="right")
        diff.add_column(f"B ({b_label})", justify="right")
        diff.add_column("Δ (B−A)", justify="right")
        b_by_id = {fr.fixture.id: fr for fr in compare_summary.fixture_results}
        for fr in summary.fixture_results:
            b = b_by_id.get(fr.fixture.id)
            a_score = fr.resistance.score if fr.resistance else None
            b_score = b.resistance.score if (b and b.resistance) else None
            a_txt = f"{a_score:.2f}" if a_score is not None else "—"
            b_txt = f"{b_score:.2f}" if b_score is not None else "—"
            if a_score is not None and b_score is not None:
                delta = b_score - a_score
                color = "green" if delta > 0.05 else ("red" if delta < -0.05 else "dim")
                d_txt = f"[{color}]{delta:+.2f}[/{color}]"
            else:
                d_txt = "—"
            diff.add_row(fr.fixture.id, fr.fixture.severity, a_txt, b_txt, d_txt)
        console.print(diff)
        a_overall = summary.overall_resilience
        b_overall = compare_summary.overall_resilience
        a_overall_text = f"{a_overall * 100:.1f}%" if a_overall is not None else "N/A"
        b_overall_text = f"{b_overall * 100:.1f}%" if b_overall is not None else "N/A"
        delta_text = (
            f"{(b_overall - a_overall) * 100:+.1f}%"
            if a_overall is not None and b_overall is not None
            else "N/A"
        )
        console.print(
            f"\nOverall resilience — A: [bold]{a_overall_text}[/bold]  "
            f"B: [bold]{b_overall_text}[/bold]  "
            f"Δ: [bold]{delta_text}[/bold]"
        )

    # ── --json-out: machine-readable export ─────────────────────────────────
    if json_out:
        import json as _json
        payload = {"model_a": summary.to_dict()}
        if compare_summary is not None:
            payload["model_b"] = compare_summary.to_dict()
        with open(json_out, "w", encoding="utf-8") as fh:
            _json.dump(payload, fh, indent=2)
        console.print(f"\n[dim]Wrote JSON results to {json_out}[/dim]")

    exit_code = 0

    # ── --fail-on-resilience: CI gate ───────────────────────────────────────
    if fail_on_resilience is not None:
        if summary.overall_resilience is None:
            console.print(
                "\n[bold red]FAIL[/bold red] — resilience is indeterminate; "
                f"cannot evaluate threshold {fail_on_resilience:.1f}%"
            )
            exit_code = 1
        else:
            actual = summary.overall_resilience * 100
            if actual < fail_on_resilience:
                console.print(
                    f"\n[bold red]FAIL[/bold red] — resilience {actual:.1f}% is below "
                    f"threshold {fail_on_resilience:.1f}%"
                )
                exit_code = 1
            else:
                console.print(
                    f"\n[bold green]PASS[/bold green] — resilience {actual:.1f}% meets "
                    f"threshold {fail_on_resilience:.1f}%"
                )

    integrity_summaries = [("Model A", summary)]
    if compare_summary is not None:
        integrity_summaries.append(("Model B", compare_summary))
    for label, model_summary in integrity_summaries:
        model_integrity = model_summary.integrity
        if model_integrity.should_exit_nonzero:
            override = " (--allow-partial override)" if allow_partial else ""
            console.print(
                f"\n[bold yellow]WARNING[/bold yellow] — {label} integrity status: "
                f"[bold]{model_integrity.status.value}[/bold]{override}"
            )
            if not allow_partial:
                exit_code = 1

    if repo:
        finalize_repository(fail_closed=True)

    if exit_code:
        ctx.exit(exit_code)

# ── atomics redblue ───────────────────────────────────────────────────────────

@click.command("redblue")
@click.option("--provider", "-p", "provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--model", "-m", type=str, default=None)
@click.option("--ollama-host", type=str, default=None)
@click.option("--vllm-host", "vllm_host", type=str, default=None, help="vLLM/OpenAI-compatible base URL.")
@click.option("--judge-provider", "judge_provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--judge-model", type=str, default=None)
@click.option("--judge-host", type=str, default=None)
@click.option("--mode", type=click.Choice(["red", "blue", "all"]), default="all", show_default=True,
              help="Which fixture set to run.")
@click.option("--runs", type=int, default=1, show_default=True,
              help="Run each fixture N times and report mean ± stddev (use 3+ for variance analysis).")
@click.option("--thinking/--no-thinking", "thinking_flag", default=None)
@click.option("--thinking-budget", type=int, default=8000, show_default=True)
@click.option("--save/--no-save", "save_results", default=True, show_default=True)
@click.option("--json-out", "json_out", type=click.Path(dir_okay=False, writable=True), default=None,
              help="Write the full run (per-fixture scores, rationales, latency, cost) as JSON to this file.")
def redblue(
    provider_name: str,
    model: str | None,
    ollama_host: str | None,
    vllm_host: str | None,
    judge_provider_name: str,
    judge_model: str | None,
    judge_host: str | None,
    mode: str,
    runs: int,
    thinking_flag: bool | None,
    thinking_budget: int,
    save_results: bool,
    json_out: str | None,
) -> None:
    """Run red/blue team LLM capability eval — offensive and defensive security tasks.

    Use --runs 3 for variance-aware scoring (mean ± stddev across passes).
    """
    from atomics.eval.redblue.fixtures import ALL_FIXTURES, BLUE_FIXTURES, RED_FIXTURES
    from atomics.eval.redblue.runner import run_redblue

    console = Console()
    fixture_count = {"red": len(RED_FIXTURES), "blue": len(BLUE_FIXTURES), "all": len(ALL_FIXTURES)}[mode]
    settings = load_settings()
    provider = _make_provider(provider_name, model, ollama_host, settings, vllm_host=vllm_host)
    judge = _make_provider(judge_provider_name, judge_model, judge_host or ollama_host, settings, vllm_host=vllm_host)

    console.print(
        f"\n[bold]Red/Blue eval[/bold] — model: [cyan]{provider_name}[/cyan] ({model or 'default'})\n"
        f"Judge: [cyan]{judge_provider_name}[/cyan] | Mode: [bold]{mode}[/bold] | "
        f"Fixtures: [bold]{fixture_count}[/bold] | Runs per fixture: [bold]{runs}[/bold]\n"
    )

    run_id = __import__("uuid").uuid4().hex[:12]
    repo = None
    if save_results:
        from atomics.storage.repository import MetricsRepository
        repo = MetricsRepository(settings.db_path)
        # Red/blue fixture rows are stored in task_results, which require a
        # parent row in runs. Create it before on_done persists fixture rows.
        repo.create_run(
            run_id,
            tier=f"redblue-{mode}",
            provider=provider_name,
            model=model or "default",
            trigger="manual",
        )

    ctx = click.get_current_context()
    show_progress = ctx.obj.get("progress", True) if ctx.obj else True
    verbose = ctx.obj.get("verbose", False) if ctx.obj else False
    progress = FixtureProgress(fixture_count, console, label="redblue") if show_progress else None

    def on_start(idx, fixture):
        if progress:
            progress.on_start(idx, fixture.id, fixture.category)

    def on_done(fr):
        j = fr.judge
        if progress:
            progress.on_done(0)
        if j:
            pct = int(j.score * 100)
            color = "green" if pct >= 80 else ("yellow" if pct >= 60 else "red")
            console.print(
                f"       [{fr.fixture.team.upper()}] [bold]{fr.fixture.id}[/bold] "
                f"[{color}]{pct}%[/] ({fr.fixture.category}) — {_rich_escape(j.rationale[:80])}"
            )
            if verbose:
                console.print(f"       [dim]Response ({fr.task_result.output_tokens} tokens, "
                              f"{fr.task_result.latency_ms:.0f}ms):[/dim]")
                console.print(f"       [dim]{_rich_escape((fr.task_result.response or '')[:200])}...[/dim]")
        if repo:
            repo.save_task_result(fr.task_result, suite=f"redblue-{fr.fixture.team}")

    summary = asyncio.run(run_redblue(
        provider,
        judge_provider=judge,
        mode=mode,
        model=model,
        judge_model=judge_model,
        runs=runs,
        run_id=run_id,
        thinking=thinking_flag,
        thinking_budget=thinking_budget,
        on_fixture_start=on_start,
        on_fixture_done=on_done,
    ))

    table = Table(title=f"Red/Blue Eval Summary ({mode})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Provider", provider_name)
    table.add_row("Model", model or "default")
    table.add_row("Judge", f"{judge_provider_name} / {judge_model or 'default'}")
    table.add_row("Mode", mode)
    table.add_row("Runs per fixture", str(summary.runs))
    quality_str = f"{(summary.overall_quality or 0) * 100:.1f}%"
    if summary.quality_stddev is not None:
        quality_str += f"  ±{summary.quality_stddev * 100:.1f}%"
    table.add_row("Overall Quality", quality_str)
    table.add_row("Fixtures Run", str(summary.total_fixtures))
    table.add_row("Avg Latency", f"{summary.avg_latency_ms:.0f}ms")
    table.add_row("Total Cost", f"${summary.total_cost_usd:.6f}")
    for cat, score in sorted(summary.category_scores.items()):
        table.add_row(f"  {cat}", f"{score * 100:.1f}%")
    console.print(table)

    if repo:
        repo.complete_run(run_id)
        repo.close()

    if json_out:
        import json as _json
        with open(json_out, "w", encoding="utf-8") as fh:
            _json.dump(summary.to_dict(), fh, indent=2)
        console.print(f"\n[dim]Wrote JSON results to {json_out}[/dim]")

# ── atomics probe ─────────────────────────────────────────────────────────────

@click.command("multiturn")
@click.option("--provider", "-p", "provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--model", "-m", type=str, default=None, help="Model override.")
@click.option("--ollama-host", type=str, default=None, help="Ollama base URL.")
@click.option("--vllm-host", "vllm_host", type=str, default=None, help="vLLM base URL.")
@click.option("--region", type=str, default="us-east-1", help="AWS region for Bedrock.")
@click.option("--judge-provider", "judge_provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--judge-model", type=str, default=None, help="Model for the judge.")
@click.option("--judge-host", type=str, default=None, help="Ollama host for the judge.")
@click.option("--fixtures", "fixtures_filter", type=str, default=None,
              help="Comma-separated fixture IDs (e.g. mt-eval-01,mt-eval-05).")
@click.option("--save/--no-save", "save_results", default=True, help="Persist results.")
@click.option("--json-out", "json_out", type=click.Path(dir_okay=False, writable=True), default=None)
@click.option("--thinking/--no-thinking", "thinking_flag", default=None)
@click.option("--thinking-budget", type=int, default=None)
def multiturn(
    provider_name: str,
    model: str | None,
    ollama_host: str | None,
    vllm_host: str | None,
    region: str,
    judge_provider_name: str,
    judge_model: str | None,
    judge_host: str | None,
    fixtures_filter: str | None,
    save_results: bool,
    json_out: str | None,
    thinking_flag: bool | None,
    thinking_budget: int | None,
) -> None:
    """Multi-turn conversation evaluation — context retention, coherence, instruction following."""
    settings = load_settings()
    setup_logging(settings.log_level)
    console = Console()

    from atomics.eval.multiturn.fixtures import ALL_MULTITURN_FIXTURES
    from atomics.eval.multiturn.runner import ConversationResult, run_multiturn

    test_provider = _make_provider(
        provider_name, model, ollama_host, settings,
        vllm_host=vllm_host, region=region,
    )
    judge_provider = _make_provider(
        judge_provider_name, judge_model, judge_host or ollama_host, settings,
        vllm_host=vllm_host, region=region,
    )

    selected_fixtures = ALL_MULTITURN_FIXTURES
    if fixtures_filter:
        ids = [f.strip() for f in fixtures_filter.split(",")]
        fixture_map = {f.id: f for f in ALL_MULTITURN_FIXTURES}
        missing = [i for i in ids if i not in fixture_map]
        if missing:
            console.print(f"[red]Unknown fixture IDs: {', '.join(missing)}[/red]")
            sys.exit(1)
        selected_fixtures = [fixture_map[i] for i in ids]

    total_turns = sum(len(f.turns) for f in selected_fixtures)
    console.print(
        f"\n[bold]Multi-Turn Eval[/bold] — provider: [cyan]{provider_name}[/cyan] | "
        f"model: [cyan]{model or 'default'}[/cyan] | "
        f"judge: [cyan]{judge_provider_name}:{judge_model or 'default'}[/cyan]\n"
        f"Conversations: [bold]{len(selected_fixtures)}[/bold] | "
        f"Total turns: [bold]{total_turns}[/bold]\n"
    )

    repo = None
    if save_results:
        from atomics.storage.repository import MetricsRepository
        repo = MetricsRepository(settings.db_path)

    import uuid as _uuid
    mt_run_id = _uuid.uuid4().hex[:12]
    if provider_name == "ollama":
        effective_model = model or settings.ollama_model
    elif provider_name == "vllm":
        effective_model = model or settings.vllm_model
    else:
        effective_model = model or settings.default_model
    if repo:
        repo.create_run(
            mt_run_id, tier="multiturn", provider=provider_name,
            model=effective_model, trigger="eval",
        )

    result_table = Table(title="Multi-Turn Eval Results", show_lines=True)
    result_table.add_column("ID", style="dim")
    result_table.add_column("Turns", justify="right")
    result_table.add_column("Turn Avg", justify="right", style="green")
    result_table.add_column("Retain", justify="right")
    result_table.add_column("Consist", justify="right")
    result_table.add_column("Instruct", justify="right")
    result_table.add_column("Overall", justify="right", style="green bold")
    result_table.add_column("Latency", justify="right")
    result_table.add_column("Cost", justify="right", style="yellow")

    def on_done(cr: ConversationResult) -> None:
        turn_scores = [t.judge.score for t in cr.turn_results if t.judge and not t.judge.parse_failed]
        turn_avg = f"{sum(turn_scores)/len(turn_scores)*100:.0f}%" if turn_scores else "—"
        cj = cr.conversation_judge
        if cj and not cj.parse_failed:
            ret = str(cj.retention)
            con = str(cj.consistency)
            ins = str(cj.instruction)
        else:
            ret = con = ins = "—"
        overall = f"{cr.overall_score*100:.0f}%" if cr.overall_score is not None else "—"

        result_table.add_row(
            cr.fixture.id, str(len(cr.turn_results)),
            turn_avg, ret, con, ins, overall,
            f"{cr.task_result.latency_ms:.0f}ms",
            f"${cr.task_result.estimated_cost_usd:.6f}",
        )
        if repo:
            repo.save_task_result(cr.task_result, suite="multiturn")

    eff_thinking = thinking_flag
    if eff_thinking is None and model:
        from atomics.model_classes import supports_thinking
        if supports_thinking(model):
            eff_thinking = True

    summary = asyncio.run(run_multiturn(
        test_provider,
        judge_provider=judge_provider,
        model=model,
        judge_model=judge_model,
        run_id=mt_run_id,
        on_conversation_done=on_done,
        thinking=eff_thinking,
        thinking_budget=thinking_budget,
        fixtures=selected_fixtures,
    ))

    console.print(result_table)

    summary_table = Table(title="Multi-Turn Summary", show_lines=True)
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value", style="bold")
    summary_table.add_row("Provider", provider_name)
    summary_table.add_row("Model", model or "default")
    ts = summary.avg_turn_score
    summary_table.add_row("Avg Turn Score", f"[green]{ts*100:.1f}%[/green]" if ts else "—")
    cs = summary.avg_conversation_score
    summary_table.add_row("Avg Conversation Score", f"[green]{cs*100:.1f}%[/green]" if cs else "—")
    ret = summary.avg_retention
    summary_table.add_row("Avg Retention", f"{ret*100:.1f}%" if ret else "—")
    con = summary.avg_consistency
    summary_table.add_row("Avg Consistency", f"{con*100:.1f}%" if con else "—")
    summary_table.add_row("Total Turns", str(summary.total_turns))
    summary_table.add_row("Total Tokens", f"{summary.total_tokens:,}")
    summary_table.add_row("Total Cost", f"${summary.total_cost_usd:.6f}")
    console.print(summary_table)

    if repo:
        repo.complete_run(mt_run_id)
        repo.close()

    if json_out:
        import json as _json
        with open(json_out, "w", encoding="utf-8") as fh:
            _json.dump(summary.to_dict(), fh, indent=2)
        console.print(f"[dim]Wrote JSON results to {json_out}[/dim]")

# ── atomics rag ───────────────────────────────────────────────────────────────

@click.command("refusal")
@click.option(
    "--provider",
    "-p",
    "provider_name",
    type=PROVIDER_CHOICES,
    default="ollama",
    show_default=True,
)
@click.option("--model", "-m", type=str, default=None)
@click.option("--ollama-host", type=str, default=None)
@click.option("--vllm-host", "vllm_host", type=str, default=None)
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
    "-o",
    "--json-out",
    "json_out",
    type=click.Path(path_type=Path),
    default=None,
)
@click.option("--save/--no-save", default=True, show_default=True)
@click.option(
    "--allow-partial",
    is_flag=True,
    help="Return success for a partial run while preserving integrity details.",
)
@click.pass_context
def refusal(
    ctx: click.Context,
    provider_name: str,
    model: str | None,
    ollama_host: str | None,
    vllm_host: str | None,
    judge_provider_name: str,
    judge_model: str | None,
    judge_host: str | None,
    json_out: Path | None,
    save: bool,
    allow_partial: bool,
) -> None:
    """Refusal calibration — measure over- and under-refusal."""
    from atomics.eval.refusal import REFUSAL_FIXTURES, run_refusal

    console = Console()
    try:
        settings = load_settings()
        provider = _make_provider(
            provider_name,
            model,
            ollama_host,
            settings,
            vllm_host=vllm_host,
        )
        judge = _make_provider(
            judge_provider_name,
            judge_model,
            judge_host or ollama_host,
            settings,
            vllm_host=vllm_host,
        )
        attributed_model = effective_model(model, provider)
        attributed_judge = effective_model(judge_model, judge)
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(
            f"Refusal evaluation setup failed: {sanitize_error(exc)}"
        ) from exc
    console.print(
        f"\n[bold]Refusal calibration[/bold] — model: [cyan]"
        f"{escape(provider_name)}[/cyan] ({escape(attributed_model)})\n"
        f"Judge: [cyan]{escape(judge_provider_name)}[/cyan] "
        f"({escape(attributed_judge)})\n"
    )

    show_progress = bool((ctx.obj or {}).get("progress", True))
    progress = (
        FixtureProgress(len(REFUSAL_FIXTURES), console, label="refusal")
        if show_progress
        else None
    )
    current_index = -1
    run_id = uuid.uuid4().hex[:12]
    repository: MetricsRepository | None = None
    parent_created = False

    def on_start(fixture: RefusalFixture) -> None:
        nonlocal current_index
        current_index += 1
        if progress is not None:
            progress.on_start(
                current_index,
                escape(fixture.id),
                escape(fixture.category),
            )

    def on_done(result: RefusalResult) -> None:
        if progress is not None:
            progress.on_done(max(current_index, 0))
        payload = result.to_dict()
        if payload["score"] is None:
            mark = "[yellow]ERROR[/yellow]"
        else:
            mark = "[green]OK[/green]" if result.correct else "[red]MISS[/red]"
        flag = ""
        if result.over_refusal:
            flag = " [yellow](over-refusal)[/yellow]"
        elif result.under_refusal:
            flag = " [red](under-refusal)[/red]"
        console.print(
            f"  {mark} [bold]{escape(result.fixture.id)}[/bold] "
            f"expected={escape(result.fixture.expected)} "
            f"got={escape(result.classification)}{flag}"
        )
        if repository is not None:
            repository.save_evaluation_result(
                evaluation_record_from_fixture(
                    run_id=run_id,
                    suite="refusal",
                    provider=provider_name,
                    model=attributed_model,
                    payload=payload,
                )
            )

    failure: Exception | None = None
    try:
        if save:
            repository = MetricsRepository(settings.db_path)
            repository.create_run(
                run_id,
                tier="refusal",
                provider=provider_name,
                model=attributed_model,
            )
            parent_created = True
        summary = asyncio.run(
            run_refusal(
                provider,
                judge_provider=judge,
                model=model,
                judge_model=judge_model,
                run_id=run_id,
                on_fixture_start=on_start,
                on_fixture_done=on_done,
            )
        )
        _render_refusal_summary(
            console,
            summary,
            provider_name=provider_name,
            model=attributed_model,
            judge_provider=judge_provider_name,
            judge_model=attributed_judge,
        )

        if json_out is not None:
            write_summary_json(summary, json_out)
            console.print(
                f"\n[dim]Wrote JSON results to {escape(str(json_out))}[/dim]"
            )

        if integrity_exit_code(summary.integrity, allow_partial=allow_partial):
            raise click.exceptions.Exit(1)
    except Exception as exc:
        failure = exc
    finally:
        if repository is not None:
            if parent_created:
                try:
                    repository.complete_evaluation_run(run_id)
                except Exception as exc:
                    if failure is None:
                        failure = exc
            try:
                repository.close()
            except Exception as exc:
                if failure is None:
                    failure = exc

    if failure is not None:
        if isinstance(failure, (click.ClickException, click.exceptions.Exit)):
            raise failure
        raise click.ClickException(
            f"Refusal evaluation failed: {sanitize_error(failure)}"
        ) from failure

def _render_refusal_summary(
    console: Console,
    summary: RefusalSummary,
    *,
    provider_name: str,
    model: str,
    judge_provider: str,
    judge_model: str,
) -> None:
    table = Table(title="Refusal Calibration Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Provider", provider_name)
    table.add_row("Model", model)
    table.add_row("Judge", f"{judge_provider} / {judge_model}")
    table.add_row("Run ID", summary.run_id)
    calibration = summary.calibration_score
    over_refusal = summary.over_refusal_rate
    under_refusal = summary.under_refusal_rate
    table.add_row(
        "Calibration score",
        f"{calibration * 100:.1f}%" if calibration is not None else "n/a",
    )
    table.add_row(
        "Over-refusal rate",
        f"{over_refusal * 100:.1f}%" if over_refusal is not None else "n/a",
    )
    table.add_row(
        "Under-refusal rate",
        f"{under_refusal * 100:.1f}%" if under_refusal is not None else "n/a",
    )
    table.add_row("Fixtures", str(len(summary.results)))
    table.add_row("Integrity", summary.integrity.status.value)
    table.add_row(
        "Generation failures",
        str(summary.integrity.generation_failures),
    )
    table.add_row("Judge failures", str(summary.integrity.judge_failures))
    table.add_row(
        "Fixture coverage",
        f"{summary.integrity.fixture_coverage * 100:.1f}%",
    )
    console.print(table)

@click.command("codereview")
@click.option(
    "--provider",
    "-p",
    "provider_name",
    type=PROVIDER_CHOICES,
    default="ollama",
    show_default=True,
)
@click.option("--model", "-m", type=str, default=None)
@click.option("--ollama-host", type=str, default=None)
@click.option("--vllm-host", "vllm_host", type=str, default=None)
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
    "-o",
    "--json-out",
    "json_out",
    type=click.Path(path_type=Path),
    default=None,
)
@click.option("--save/--no-save", default=True, show_default=True)
@click.option(
    "--allow-partial",
    is_flag=True,
    help="Return success for a partial run while preserving integrity details.",
)
@click.pass_context
def codereview(
    ctx: click.Context,
    provider_name: str,
    model: str | None,
    ollama_host: str | None,
    vllm_host: str | None,
    judge_provider_name: str,
    judge_model: str | None,
    judge_host: str | None,
    json_out: Path | None,
    save: bool,
    allow_partial: bool,
) -> None:
    """Secure code review — measure planted-vulnerability detection."""
    from atomics.eval.codereview import SECURE_CODE_FIXTURES, run_codereview

    console = Console()
    try:
        settings = load_settings()
        provider = _make_provider(
            provider_name,
            model,
            ollama_host,
            settings,
            vllm_host=vllm_host,
        )
        judge = _make_provider(
            judge_provider_name,
            judge_model,
            judge_host or ollama_host,
            settings,
            vllm_host=vllm_host,
        )
        attributed_model = effective_model(model, provider)
        attributed_judge = effective_model(judge_model, judge)
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(
            f"Code review evaluation setup failed: {sanitize_error(exc)}"
        ) from exc
    console.print(
        f"\n[bold]Secure code review[/bold] — model: [cyan]"
        f"{escape(provider_name)}[/cyan] ({escape(attributed_model)})\n"
        f"Judge: [cyan]{escape(judge_provider_name)}[/cyan] "
        f"({escape(attributed_judge)})\n"
    )

    show_progress = bool((ctx.obj or {}).get("progress", True))
    progress = (
        FixtureProgress(len(SECURE_CODE_FIXTURES), console, label="codereview")
        if show_progress
        else None
    )
    current_index = -1
    run_id = uuid.uuid4().hex[:12]
    repository: MetricsRepository | None = None
    parent_created = False

    def on_start(fixture: SecureCodeFixture) -> None:
        nonlocal current_index
        current_index += 1
        if progress is not None:
            progress.on_start(
                current_index,
                escape(fixture.id),
                escape(fixture.mode),
            )

    def on_done(result: CodeReviewResult) -> None:
        if progress is not None:
            progress.on_done(max(current_index, 0))
        fixture = result.fixture
        expected_label = fixture.cwe if fixture.is_vulnerable else "clean code"
        payload = result.to_dict()
        if payload["score"] is None:
            mark = "[yellow]ERROR[/yellow]"
        else:
            mark = "[green]OK[/green]" if result.passed else "[red]MISS[/red]"
        console.print(
            f"  {mark} [bold]{escape(fixture.id)}[/bold] "
            f"({escape(fixture.mode)}) verdict={escape(result.verdict)} — "
            f"{escape(expected_label)}"
        )
        if repository is not None:
            repository.save_evaluation_result(
                evaluation_record_from_fixture(
                    run_id=run_id,
                    suite="codereview",
                    provider=provider_name,
                    model=attributed_model,
                    payload=payload,
                )
            )

    failure: Exception | None = None
    try:
        if save:
            repository = MetricsRepository(settings.db_path)
            repository.create_run(
                run_id,
                tier="codereview",
                provider=provider_name,
                model=attributed_model,
            )
            parent_created = True
        summary = asyncio.run(
            run_codereview(
                provider,
                judge_provider=judge,
                model=model,
                judge_model=judge_model,
                run_id=run_id,
                on_fixture_start=on_start,
                on_fixture_done=on_done,
            )
        )
        _render_codereview_summary(
            console,
            summary,
            provider_name=provider_name,
            model=attributed_model,
            judge_provider=judge_provider_name,
            judge_model=attributed_judge,
        )

        if json_out is not None:
            write_summary_json(summary, json_out)
            console.print(
                f"\n[dim]Wrote JSON results to {escape(str(json_out))}[/dim]"
            )

        if integrity_exit_code(summary.integrity, allow_partial=allow_partial):
            raise click.exceptions.Exit(1)
    except Exception as exc:
        failure = exc
    finally:
        if repository is not None:
            if parent_created:
                try:
                    repository.complete_evaluation_run(run_id)
                except Exception as exc:
                    if failure is None:
                        failure = exc
            try:
                repository.close()
            except Exception as exc:
                if failure is None:
                    failure = exc

    if failure is not None:
        if isinstance(failure, (click.ClickException, click.exceptions.Exit)):
            raise failure
        raise click.ClickException(
            f"Code review evaluation failed: {sanitize_error(failure)}"
        ) from failure

def _render_codereview_summary(
    console: Console,
    summary: CodeReviewSummary,
    *,
    provider_name: str,
    model: str,
    judge_provider: str,
    judge_model: str,
) -> None:
    table = Table(title="Secure Code Review Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Provider", provider_name)
    table.add_row("Model", model)
    table.add_row("Judge", f"{judge_provider} / {judge_model}")
    table.add_row("Run ID", summary.run_id)
    detection = summary.detection_rate
    false_positive = summary.false_positive_rate
    review = summary.review_score
    table.add_row(
        "Detection rate",
        f"{detection * 100:.1f}%" if detection is not None else "n/a",
    )
    table.add_row(
        "False-positive rate",
        f"{false_positive * 100:.1f}%" if false_positive is not None else "n/a",
    )
    table.add_row(
        "Review score (F1)",
        f"{review * 100:.1f}%" if review is not None else "n/a",
    )
    table.add_row("Fixtures", str(len(summary.results)))
    table.add_row("Integrity", summary.integrity.status.value)
    table.add_row(
        "Generation failures",
        str(summary.integrity.generation_failures),
    )
    table.add_row("Judge failures", str(summary.integrity.judge_failures))
    table.add_row(
        "Fixture coverage",
        f"{summary.integrity.fixture_coverage * 100:.1f}%",
    )
    console.print(table)
