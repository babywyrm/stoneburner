"""RAG pipeline, index, and retrieval CLI commands."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from atomics.commands.common import (
    PROVIDER_CHOICES,
    _make_provider,
    budget_option,
    effort_options,
    eval_budget_from,
    extra_judges_option,
    integrity_exit_code,
    parse_extra_judges,
    run_async,
    setup_logging,
    write_summary_json,
)
from atomics.commands.suite_run import finalize_task_run, suite_run
from atomics.config import load_settings
from atomics.eval.budget import share_budget


@click.command("rag")
@click.option(
    "--provider", "-p", "provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True
)
@click.option(
    "--model", "-m", type=str, default=None, help="Model override for the provider under test."
)
@click.option("--ollama-host", type=str, default=None, help="Ollama base URL.")
@click.option(
    "--vllm-host", "vllm_host", type=str, default=None, help="vLLM/OpenAI-compatible base URL."
)
@click.option("--region", type=str, default="us-east-1", help="AWS region for Bedrock.")
@click.option(
    "--judge-provider",
    "judge_provider_name",
    type=PROVIDER_CHOICES,
    default="ollama",
    show_default=True,
    help="Provider for the RAG judge.",
)
@click.option("--judge-model", type=str, default=None, help="Model for the RAG judge.")
@click.option("--judge-host", type=str, default=None, help="Ollama host for the judge model.")
@click.option(
    "--fixtures",
    "fixtures_filter",
    type=str,
    default=None,
    help="Comma-separated fixture IDs (e.g. rag-05 or rag-01,rag-10).",
)
@click.option(
    "--index",
    "index_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to a sqlite-vec RAG index built by rag-index.",
)
@click.option(
    "--top-k",
    type=int,
    default=5,
    show_default=True,
    help="Number of chunks to retrieve when --index is provided.",
)
@click.option(
    "--save/--no-save", "save_results", default=True, help="Persist results to the database."
)
@click.option(
    "--json-out",
    "json_out",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Write the full run as JSON to this file.",
)
@click.option(
    "--thinking/--no-thinking", "thinking_flag", default=None, help="Enable/disable thinking."
)
@click.option("--thinking-budget", type=int, default=None, help="Max thinking tokens.")
@effort_options
@extra_judges_option
@click.option(
    "--allow-partial",
    is_flag=True,
    help="Return success for a partial run while preserving integrity details.",
)
@budget_option
def rag(
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
    effort: str | None,
    reasoning_mode: str | None,
    index_path: Path | None,
    top_k: int,
    extra_judges: str | None,
    allow_partial: bool,
    budget_usd: float | None,
) -> None:
    """RAG pipeline evaluation — grounding, faithfulness, and abstention scoring."""
    settings = load_settings()
    setup_logging(settings.log_level)
    console = Console()

    from atomics.eval.rag.fixtures import ALL_RAG_FIXTURES
    from atomics.eval.rag.runner import RAGFixtureResult, run_rag

    test_provider = _make_provider(
        provider_name,
        model,
        ollama_host,
        settings,
        vllm_host=vllm_host,
        region=region,
    )
    judge_provider = _make_provider(
        judge_provider_name,
        judge_model,
        judge_host or ollama_host,
        settings,
        vllm_host=vllm_host,
        region=region,
    )
    extra_judge_pairs = parse_extra_judges(
        extra_judges,
        build=lambda name, mdl, host: _make_provider(
            name,
            mdl,
            host,
            settings,
            vllm_host=vllm_host,
            region=region,
        ),
        default_host=judge_host or ollama_host,
    )
    budget = eval_budget_from(budget_usd)
    if budget is not None:
        guarded = share_budget(
            budget, test_provider, judge_provider, *(p for p, _ in extra_judge_pairs)
        )
        test_provider, judge_provider = guarded[0], guarded[1]
        extra_judge_pairs = [(guarded[2 + i], mdl) for i, (_, mdl) in enumerate(extra_judge_pairs)]

    selected_fixtures = ALL_RAG_FIXTURES
    if fixtures_filter:
        ids = [f.strip() for f in fixtures_filter.split(",")]
        fixture_map = {f.id: f for f in ALL_RAG_FIXTURES}
        missing = [i for i in ids if i not in fixture_map]
        if missing:
            console.print(f"[red]Unknown fixture IDs: {', '.join(missing)}[/red]")
            sys.exit(1)
        selected_fixtures = [fixture_map[i] for i in ids]

    fixture_count = len(selected_fixtures)
    console.print(
        f"\n[bold]RAG Evaluation[/bold] — provider: [cyan]{provider_name}[/cyan] | "
        f"model: [cyan]{model or 'default'}[/cyan] | "
        f"judge: [cyan]{judge_provider_name}:{judge_model or 'default'}[/cyan]\n"
        f"Fixtures: [bold]{fixture_count}[/bold] | "
        f"Results saved: [bold]{'yes' if save_results else 'no'}[/bold]\n"
    )

    rag_run_id = uuid.uuid4().hex[:12]
    if provider_name == "ollama":
        effective_model = model or settings.ollama_model
    elif provider_name == "vllm":
        effective_model = model or settings.vllm_model
    else:
        effective_model = model or settings.default_model

    result_table = Table(title="RAG Eval Results", show_lines=True)
    result_table.add_column("ID", style="dim")
    result_table.add_column("Type", style="cyan")
    result_table.add_column("Ground", justify="right")
    result_table.add_column("Faith", justify="right")
    result_table.add_column("Abst", justify="right")
    result_table.add_column("Score", justify="right", style="green bold")
    result_table.add_column("Latency", justify="right")
    result_table.add_column("Tokens", justify="right")
    result_table.add_column("Cost", justify="right", style="yellow")
    result_table.add_column("Rationale", no_wrap=False, max_width=35, style="dim")

    with suite_run(
        suite="rag",
        db_path=settings.db_path,
        save=save_results,
        finalize=finalize_task_run,
        failure_prefix="RAG eval failed",
    ) as run:
        run.begin(
            rag_run_id,
            provider=provider_name,
            model=effective_model,
            trigger="eval",
        )
        repo = run.repository

        def on_done(fr: RAGFixtureResult) -> None:
            tr = fr.task_result
            j = fr.judge
            if tr.status.value == "failed":
                score_str = "[red]FAIL[/red]"
                rationale = tr.error_message[:60]
                g_str = f_str = a_str = "—"
            elif j and not j.parse_failed:
                score_str = f"{j.score * 100:.0f}%"
                rationale = j.rationale[:60]
                g_str = str(j.grounding)
                f_str = str(j.faithfulness)
                a_str = str(j.abstention)
            else:
                score_str = "[yellow]?[/yellow]"
                rationale = "judge parse failed"
                g_str = f_str = a_str = "?"

            ctx_type = "answer" if fr.fixture.context_contains_answer else "abstain"
            result_table.add_row(
                fr.fixture.id,
                ctx_type,
                g_str,
                f_str,
                a_str,
                score_str,
                f"{tr.latency_ms:.0f}ms",
                str(tr.total_tokens),
                f"${tr.estimated_cost_usd:.6f}",
                rationale,
            )
            if repo:
                repo.save_task_result(tr, suite="rag")

        eff_thinking = thinking_flag
        if eff_thinking is None and model:
            from atomics.benchmark.model_classes import supports_thinking

            if supports_thinking(model):
                eff_thinking = True

        index = None
        if index_path is not None:
            try:
                import sentence_transformers  # noqa: F401
                import sqlite_vec  # noqa: F401
            except ImportError as exc:
                console.print(
                    "[red]RAG indexing requires the [rag] extra:[/red] "
                    'uv pip install "stoneburner-atomics[rag]"'
                )
                raise SystemExit(1) from exc
            from atomics.eval.rag.retrieval import (
                LocalSentenceTransformerEmbedder,
                MockEmbedder,
                RAGIndex,
            )

            index_meta = RAGIndex(index_path, embedder=MockEmbedder()).info()
            embedding_model = index_meta.get("embedding_model") or "all-MiniLM-L6-v2"
            embedder = LocalSentenceTransformerEmbedder(embedding_model)
            index = RAGIndex(index_path, embedder=embedder)

        summary = run_async(
            run_rag(
                test_provider,
                judge_provider=judge_provider,
                model=model,
                judge_model=judge_model,
                extra_judges=extra_judge_pairs,
                run_id=rag_run_id,
                on_fixture_done=on_done,
                thinking=eff_thinking,
                thinking_budget=thinking_budget,
                effort=effort,
                reasoning_mode=reasoning_mode,
                fixtures=selected_fixtures,
                index=index,
                top_k=top_k,
            ),
            test_provider,
            judge_provider,
            *(p for p, _ in extra_judge_pairs),
        )

        console.print(result_table)

        summary_table = Table(title="RAG Eval Summary", show_lines=True)
        summary_table.add_column("Metric", style="dim")
        summary_table.add_column("Value", style="bold")
        summary_table.add_row("Provider", provider_name)
        summary_table.add_row("Model", model or "default")

        rag_score = summary.overall_rag_score
        summary_table.add_row(
            "Overall RAG Score",
            f"[green]{rag_score * 100:.1f}%[/green]" if rag_score is not None else "—",
        )
        gs = summary.grounding_score
        summary_table.add_row("Grounding", f"{gs * 100:.1f}%" if gs is not None else "—")
        fs = summary.faithfulness_score
        summary_table.add_row("Faithfulness", f"{fs * 100:.1f}%" if fs is not None else "—")
        aa = summary.abstention_accuracy
        summary_table.add_row("Abstention Accuracy", f"{aa * 100:.1f}%" if aa is not None else "—")
        hr = summary.hallucination_rate
        hr_style = (
            "green"
            if hr is not None and hr < 0.1
            else "yellow"
            if hr is not None and hr < 0.3
            else "red"
        )
        summary_table.add_row(
            "Hallucination Rate",
            f"[{hr_style}]{hr * 100:.1f}%[/{hr_style}]" if hr is not None else "—",
        )
        summary_table.add_row("Avg Latency", f"{summary.avg_latency_ms:.0f}ms")
        summary_table.add_row("Total Tokens", f"{summary.total_tokens:,}")
        summary_table.add_row("Total Cost", f"${summary.total_cost_usd:.6f}")
        summary_table.add_row("Fixtures Run", str(len(summary.fixture_results)))
        pf = summary.parse_failure_rate
        pf_style = "green" if pf == 0 else "yellow" if pf < 0.1 else "red"
        summary_table.add_row("Judge Parse Failures", f"[{pf_style}]{pf * 100:.1f}%[/{pf_style}]")
        console.print(summary_table)

        if json_out:
            write_summary_json(summary, Path(json_out))
            console.print(f"[dim]Wrote JSON results to {json_out}[/dim]")

        if integrity_exit_code(summary.integrity, allow_partial=allow_partial):
            raise click.exceptions.Exit(1)


# ── atomics rag-index ─────────────────────────────────────────────────────────


@click.command("rag-index")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--db",
    "db_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output sqlite-vec database file.",
)
@click.option(
    "--chunk-size",
    type=int,
    default=512,
    show_default=True,
    help="Target chunk size in characters.",
)
@click.option(
    "--overlap",
    type=int,
    default=50,
    show_default=True,
    help="Overlap between chunks in characters.",
)
@click.option(
    "--embedding-model",
    type=str,
    default="all-MiniLM-L6-v2",
    show_default=True,
    help="sentence-transformers model name.",
)
@click.option("--force/--no-force", default=False, help="Rebuild the index from scratch.")
def rag_index(
    path: Path,
    db_path: Path | None,
    chunk_size: int,
    overlap: int,
    embedding_model: str,
    force: bool,
) -> None:
    """Build a sqlite-vec RAG index from documents in PATH."""
    settings = load_settings()
    setup_logging(settings.log_level)
    console = Console()

    try:
        import sentence_transformers  # noqa: F401
        import sqlite_vec  # noqa: F401
    except ImportError as exc:
        console.print(
            "[red]RAG indexing requires the [rag] extra:[/red] "
            'uv pip install "stoneburner-atomics[rag]"'
        )
        raise SystemExit(1) from exc

    from atomics.eval.rag.retrieval import (
        LocalSentenceTransformerEmbedder,
        RAGIndex,
        load_documents,
    )

    if db_path is None:
        data_dir = Path.home() / ".local" / "share" / "atomics" / "rag"
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = data_dir / "default.vec"

    if db_path.exists() and force:
        db_path.unlink()

    embedder = LocalSentenceTransformerEmbedder(embedding_model)
    index = RAGIndex(db_path, embedder=embedder)
    documents = load_documents(path)
    chunk_count = index.build(documents, chunk_size=chunk_size, overlap=overlap)

    console.print(
        f"Loaded {len(documents)} files, created {chunk_count} chunks, stored in {db_path}"
    )


# ── atomics rag-retrieval ─────────────────────────────────────────────────────


@click.command("rag-retrieval")
@click.option(
    "--index",
    "index_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to a sqlite-vec RAG index.",
)
@click.option(
    "--gold",
    "gold_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="JSON file with query IDs mapped to relevant source IDs and scores.",
)
@click.option(
    "--queries",
    "queries_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="JSON file with query ID to query text mapping.",
)
@click.option(
    "--top-k",
    type=int,
    default=5,
    show_default=True,
    help="Number of chunks to retrieve per query.",
)
@click.option(
    "--json-out",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Write the retrieval report as JSON.",
)
def rag_retrieval(
    index_path: Path,
    gold_path: Path,
    queries_path: Path | None,
    top_k: int,
    json_out: str | None,
) -> None:
    """Evaluate retrieval quality from a RAG index against a gold relevance set."""
    import json

    settings = load_settings()
    setup_logging(settings.log_level)
    console = Console()

    try:
        import sentence_transformers  # noqa: F401
        import sqlite_vec  # noqa: F401
    except ImportError as exc:
        console.print(
            "[red]RAG retrieval requires the [rag] extra:[/red] "
            'uv pip install "stoneburner-atomics[rag]"'
        )
        raise SystemExit(1) from exc

    from atomics.eval.rag.metrics import (
        mean_reciprocal_rank,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
    )
    from atomics.eval.rag.retrieval import (
        LocalSentenceTransformerEmbedder,
        MockEmbedder,
        RAGIndex,
    )

    with open(gold_path, encoding="utf-8") as fh:
        gold = json.load(fh)
    queries: dict[str, str] = {}
    if queries_path:
        with open(queries_path, encoding="utf-8") as fh:
            queries = json.load(fh)

    index_meta = RAGIndex(index_path, embedder=MockEmbedder()).info()
    embedding_model = index_meta.get("embedding_model") or "all-MiniLM-L6-v2"
    embedder = LocalSentenceTransformerEmbedder(embedding_model)
    index = RAGIndex(index_path, embedder=embedder)

    per_query: list[dict] = []
    relevant_sets: list[set[str]] = []
    retrieved_lists: list[list[str]] = []
    for query_id, entry in gold.items():
        query_text = queries.get(query_id, query_id)
        results = index.search(query_text, top_k=top_k)
        retrieved_sources = [r.source for r in results]
        relevant = set(entry.get("relevant", []))
        scores = entry.get("scores", {})
        per_query.append(
            {
                "query_id": query_id,
                "recall@k": recall_at_k(relevant, retrieved_sources, top_k),
                "precision@k": precision_at_k(relevant, retrieved_sources, top_k),
                "ndcg@k": ndcg_at_k(scores, retrieved_sources, top_k),
                "retrieved": retrieved_sources,
            }
        )
        relevant_sets.append(relevant)
        retrieved_lists.append(retrieved_sources)

    avg_recall = sum(q["recall@k"] for q in per_query) / len(per_query) if per_query else 0.0
    avg_precision = sum(q["precision@k"] for q in per_query) / len(per_query) if per_query else 0.0
    avg_ndcg = sum(q["ndcg@k"] for q in per_query) / len(per_query) if per_query else 0.0
    if not per_query:
        console.print("[yellow]No queries in gold file; metrics are zero.[/yellow]")
        mrr = 0.0
    else:
        mrr = mean_reciprocal_rank(relevant_sets, retrieved_lists)

    report = {
        "index": str(index_path),
        "top_k": top_k,
        "queries": len(per_query),
        "avg_recall_at_k": avg_recall,
        "avg_precision_at_k": avg_precision,
        "avg_ndcg_at_k": avg_ndcg,
        "mrr": mrr,
        "per_query": per_query,
    }

    console.print(f"[bold]Retrieval metrics[/bold] (top_k={top_k})")
    console.print(f"Recall@k: {avg_recall:.3f}")
    console.print(f"Precision@k: {avg_precision:.3f}")
    console.print(f"nDCG@k: {avg_ndcg:.3f}")
    console.print(f"MRR: {mrr:.3f}")

    if json_out:
        with open(json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        console.print(f"[dim]Wrote report to {json_out}[/dim]")
