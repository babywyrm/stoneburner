"""RAG and related evaluation CLI commands."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markup import escape as _rich_escape
from rich.table import Table

from atomics.commands.common import (
    PROVIDER_CHOICES,
    _make_provider,
    setup_logging,
)
from atomics.config import load_settings


@click.command("rag")
@click.option("--provider", "-p", "provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--model", "-m", type=str, default=None, help="Model override for the provider under test.")
@click.option("--ollama-host", type=str, default=None, help="Ollama base URL.")
@click.option("--vllm-host", "vllm_host", type=str, default=None, help="vLLM/OpenAI-compatible base URL.")
@click.option("--region", type=str, default="us-east-1", help="AWS region for Bedrock.")
@click.option("--judge-provider", "judge_provider_name", type=PROVIDER_CHOICES, default="ollama",
              show_default=True, help="Provider for the RAG judge.")
@click.option("--judge-model", type=str, default=None, help="Model for the RAG judge.")
@click.option("--judge-host", type=str, default=None, help="Ollama host for the judge model.")
@click.option("--fixtures", "fixtures_filter", type=str, default=None,
              help="Comma-separated fixture IDs (e.g. rag-05 or rag-01,rag-10).")
@click.option("--index", "index_path", type=click.Path(exists=True, path_type=Path), default=None,
              help="Path to a sqlite-vec RAG index built by rag-index.")
@click.option("--top-k", type=int, default=5, show_default=True,
              help="Number of chunks to retrieve when --index is provided.")
@click.option("--save/--no-save", "save_results", default=True, help="Persist results to the database.")
@click.option("--json-out", "json_out", type=click.Path(dir_okay=False, writable=True), default=None,
              help="Write the full run as JSON to this file.")
@click.option("--thinking/--no-thinking", "thinking_flag", default=None, help="Enable/disable thinking.")
@click.option("--thinking-budget", type=int, default=None, help="Max thinking tokens.")
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
    index_path: Path | None,
    top_k: int,
) -> None:
    """RAG pipeline evaluation — grounding, faithfulness, and abstention scoring."""
    settings = load_settings()
    setup_logging(settings.log_level)
    console = Console()

    from atomics.eval.rag.fixtures import ALL_RAG_FIXTURES
    from atomics.eval.rag.runner import RAGFixtureResult, run_rag

    test_provider = _make_provider(
        provider_name, model, ollama_host, settings,
        vllm_host=vllm_host, region=region,
    )
    judge_provider = _make_provider(
        judge_provider_name, judge_model, judge_host or ollama_host, settings,
        vllm_host=vllm_host, region=region,
    )

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

    repo = None
    if save_results:
        from atomics.storage.repository import MetricsRepository
        repo = MetricsRepository(settings.db_path)

    import uuid as _uuid
    rag_run_id = _uuid.uuid4().hex[:12]
    if provider_name == "ollama":
        effective_model = model or settings.ollama_model
    elif provider_name == "vllm":
        effective_model = model or settings.vllm_model
    else:
        effective_model = model or settings.default_model
    if repo:
        repo.create_run(
            rag_run_id, tier="rag", provider=provider_name,
            model=effective_model, trigger="eval",
        )

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
            fr.fixture.id, ctx_type, g_str, f_str, a_str, score_str,
            f"{tr.latency_ms:.0f}ms", str(tr.total_tokens),
            f"${tr.estimated_cost_usd:.6f}", rationale,
        )
        if repo:
            repo.save_task_result(tr, suite="rag")

    eff_thinking = thinking_flag
    if eff_thinking is None and model:
        from atomics.model_classes import supports_thinking
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
                'uv pip install "atomics[rag]"'
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

    summary = asyncio.run(run_rag(
        test_provider,
        judge_provider=judge_provider,
        model=model,
        judge_model=judge_model,
        run_id=rag_run_id,
        on_fixture_done=on_done,
        thinking=eff_thinking,
        thinking_budget=thinking_budget,
        fixtures=selected_fixtures,
        index=index,
        top_k=top_k,
    ))

    console.print(result_table)

    summary_table = Table(title="RAG Eval Summary", show_lines=True)
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value", style="bold")
    summary_table.add_row("Provider", provider_name)
    summary_table.add_row("Model", model or "default")

    rag_score = summary.overall_rag_score
    summary_table.add_row("Overall RAG Score",
                          f"[green]{rag_score * 100:.1f}%[/green]" if rag_score is not None else "—")
    gs = summary.grounding_score
    summary_table.add_row("Grounding",
                          f"{gs * 100:.1f}%" if gs is not None else "—")
    fs = summary.faithfulness_score
    summary_table.add_row("Faithfulness",
                          f"{fs * 100:.1f}%" if fs is not None else "—")
    aa = summary.abstention_accuracy
    summary_table.add_row("Abstention Accuracy",
                          f"{aa * 100:.1f}%" if aa is not None else "—")
    hr = summary.hallucination_rate
    hr_style = "green" if hr is not None and hr < 0.1 else "yellow" if hr is not None and hr < 0.3 else "red"
    summary_table.add_row("Hallucination Rate",
                          f"[{hr_style}]{hr * 100:.1f}%[/{hr_style}]" if hr is not None else "—")
    summary_table.add_row("Avg Latency", f"{summary.avg_latency_ms:.0f}ms")
    summary_table.add_row("Total Tokens", f"{summary.total_tokens:,}")
    summary_table.add_row("Total Cost", f"${summary.total_cost_usd:.6f}")
    summary_table.add_row("Fixtures Run", str(len(summary.fixture_results)))
    pf = summary.parse_failure_rate
    pf_style = "green" if pf == 0 else "yellow" if pf < 0.1 else "red"
    summary_table.add_row("Judge Parse Failures", f"[{pf_style}]{pf * 100:.1f}%[/{pf_style}]")
    console.print(summary_table)

    if repo:
        repo.complete_run(rag_run_id)
        repo.close()

    if json_out:
        import json as _json
        with open(json_out, "w", encoding="utf-8") as fh:
            _json.dump(summary.to_dict(), fh, indent=2)
        console.print(f"[dim]Wrote JSON results to {json_out}[/dim]")

# ── atomics rag-index ─────────────────────────────────────────────────────────

@click.command("rag-index")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None,
              help="Output sqlite-vec database file.")
@click.option("--chunk-size", type=int, default=512, show_default=True,
              help="Target chunk size in characters.")
@click.option("--overlap", type=int, default=50, show_default=True,
              help="Overlap between chunks in characters.")
@click.option("--embedding-model", type=str, default="all-MiniLM-L6-v2", show_default=True,
              help="sentence-transformers model name.")
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
            'uv pip install "atomics[rag]"'
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
@click.option("--index", "index_path", type=click.Path(exists=True, path_type=Path),
              required=True, help="Path to a sqlite-vec RAG index.")
@click.option("--gold", "gold_path", type=click.Path(exists=True, path_type=Path),
              required=True,
              help="JSON file with query IDs mapped to relevant source IDs and scores.")
@click.option("--queries", "queries_path",
              type=click.Path(exists=True, path_type=Path), default=None,
              help="JSON file with query ID to query text mapping.")
@click.option("--top-k", type=int, default=5, show_default=True,
              help="Number of chunks to retrieve per query.")
@click.option("--json-out", type=click.Path(dir_okay=False, writable=True), default=None,
              help="Write the retrieval report as JSON.")
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
            'uv pip install "atomics[rag]"'
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
        per_query.append({
            "query_id": query_id,
            "recall@k": recall_at_k(relevant, retrieved_sources, top_k),
            "precision@k": precision_at_k(relevant, retrieved_sources, top_k),
            "ndcg@k": ndcg_at_k(scores, retrieved_sources, top_k),
            "retrieved": retrieved_sources,
        })
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

# ── atomics adversarial ───────────────────────────────────────────────────────

@click.command("codegen")
@click.option("--provider", "-p", "provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--model", "-m", type=str, default=None, help="Model override.")
@click.option("--ollama-host", type=str, default=None, help="Ollama base URL.")
@click.option("--vllm-host", "vllm_host", type=str, default=None, help="vLLM base URL.")
@click.option("--region", type=str, default="us-east-1", help="AWS region for Bedrock.")
@click.option("--fixtures", "fixtures_filter", type=str, default=None,
              help="Comma-separated fixture IDs (e.g. cg-01,cg-05).")
@click.option("--save/--no-save", "save_results", default=True, help="Persist results.")
@click.option("--json-out", "json_out", type=click.Path(dir_okay=False, writable=True), default=None)
@click.option("--thinking/--no-thinking", "thinking_flag", default=None)
@click.option("--thinking-budget", type=int, default=None)
def codegen(
    provider_name: str,
    model: str | None,
    ollama_host: str | None,
    vllm_host: str | None,
    region: str,
    fixtures_filter: str | None,
    save_results: bool,
    json_out: str | None,
    thinking_flag: bool | None,
    thinking_budget: int | None,
) -> None:
    """Code generation evaluation — functional correctness via test execution."""
    settings = load_settings()
    setup_logging(settings.log_level)
    console = Console()

    from atomics.eval.codegen.fixtures import ALL_CODEGEN_FIXTURES
    from atomics.eval.codegen.runner import CodegenFixtureResult, run_codegen

    test_provider = _make_provider(
        provider_name, model, ollama_host, settings,
        vllm_host=vllm_host, region=region,
    )

    selected_fixtures = ALL_CODEGEN_FIXTURES
    if fixtures_filter:
        ids = [f.strip() for f in fixtures_filter.split(",")]
        fixture_map = {f.id: f for f in ALL_CODEGEN_FIXTURES}
        missing = [i for i in ids if i not in fixture_map]
        if missing:
            console.print(f"[red]Unknown fixture IDs: {', '.join(missing)}[/red]")
            sys.exit(1)
        selected_fixtures = [fixture_map[i] for i in ids]

    if provider_name == "ollama":
        effective_model = model or settings.ollama_model
    elif provider_name == "vllm":
        effective_model = model or settings.vllm_model
    else:
        effective_model = model or settings.default_model

    console.print(
        f"\n[bold]Code Generation Eval[/bold] — provider: [cyan]{provider_name}[/cyan] | "
        f"model: [cyan]{effective_model}[/cyan]\n"
        f"Fixtures: [bold]{len(selected_fixtures)}[/bold]\n"
    )

    repo = None
    if save_results:
        from atomics.storage.repository import MetricsRepository
        repo = MetricsRepository(settings.db_path)

    import uuid as _uuid
    cg_run_id = _uuid.uuid4().hex[:12]
    if repo:
        repo.create_run(
            cg_run_id, tier="codegen", provider=provider_name,
            model=effective_model, trigger="eval",
        )

    result_table = Table(title="Code Generation Results", show_lines=True)
    result_table.add_column("ID", style="dim")
    result_table.add_column("Function", style="cyan")
    result_table.add_column("Tests", justify="right")
    result_table.add_column("Pass", justify="right", style="green bold")
    result_table.add_column("Rate", justify="right")
    result_table.add_column("Latency", justify="right")
    result_table.add_column("Tokens", justify="right")
    result_table.add_column("Cost", justify="right", style="yellow")

    def on_done(fr: CodegenFixtureResult) -> None:
        rate_style = "green" if fr.pass_rate == 1.0 else "yellow" if fr.pass_rate > 0 else "red"
        result_table.add_row(
            fr.fixture.id,
            fr.fixture.function_name,
            str(fr.tests_total),
            str(fr.tests_passed),
            f"[{rate_style}]{fr.pass_rate*100:.0f}%[/{rate_style}]",
            f"{fr.task_result.latency_ms:.0f}ms",
            str(fr.task_result.total_tokens),
            f"${fr.task_result.estimated_cost_usd:.6f}",
        )
        if repo:
            repo.save_task_result(fr.task_result, suite="codegen")

    eff_thinking = thinking_flag
    if eff_thinking is None and model:
        from atomics.model_classes import supports_thinking
        if supports_thinking(model):
            eff_thinking = True

    summary = asyncio.run(run_codegen(
        test_provider,
        model=model,
        run_id=cg_run_id,
        on_fixture_done=on_done,
        thinking=eff_thinking,
        thinking_budget=thinking_budget,
        fixtures=selected_fixtures,
    ))

    console.print(result_table)

    summary_table = Table(title="Code Generation Summary", show_lines=True)
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value", style="bold")
    summary_table.add_row("Provider", provider_name)
    summary_table.add_row("Model", effective_model)
    pr = summary.overall_pass_rate
    pr_style = "green" if pr and pr >= 0.8 else "yellow" if pr and pr >= 0.5 else "red"
    summary_table.add_row("Overall Pass Rate",
                          f"[{pr_style}]{pr*100:.1f}%[/{pr_style}]" if pr is not None else "\u2014")
    summary_table.add_row("Fully Correct", f"{summary.fixtures_fully_correct}/{len(summary.fixture_results)}")
    summary_table.add_row("Total Tokens", f"{summary.total_tokens:,}")
    summary_table.add_row("Total Cost", f"${summary.total_cost_usd:.6f}")
    console.print(summary_table)

    if repo:
        repo.complete_run(cg_run_id)
        repo.close()

    if json_out:
        import json as _json
        with open(json_out, "w", encoding="utf-8") as fh:
            _json.dump(summary.to_dict(), fh, indent=2)
        console.print(f"[dim]Wrote JSON results to {json_out}[/dim]")

# ── atomics multiturn ──────────────────────────────────────────────────────────

@click.command("probe")
@click.option("--provider", "-p", "provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--model", "-m", type=str, default=None)
@click.option("--ollama-host", type=str, default=None)
@click.option("--vllm-host", "vllm_host", type=str, default=None, help="vLLM/OpenAI-compatible base URL.")
@click.option("--judge-provider", "judge_provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--judge-model", type=str, default=None)
@click.option("--judge-host", type=str, default=None)
@click.option("--probes-file", type=click.Path(exists=True), default=None,
              help="Path to probes.yaml config file.")
@click.option("--artifact", type=click.Choice([
    "json-security-report", "inference-api", "access-log",
    "k8s-audit-log", "config-file", "api-response",
]), default=None, help="Artifact type for single-file mode.")
@click.option("--file", "artifact_file", type=click.Path(exists=True), default=None,
              help="Artifact file path for single-file mode (use with --artifact).")
@click.option("--thinking/--no-thinking", "thinking_flag", default=None)
@click.option("--thinking-budget", type=int, default=8000, show_default=True)
@click.option("--alert-on-regression/--no-alert-on-regression", default=False,
              help="Warn if any check score drops >10% from last run.")
@click.option("--save/--no-save", "save_results", default=True, show_default=True)
@click.option("--json-out", "json_out", type=click.Path(dir_okay=False, writable=True), default=None,
              help="Write the full run (per-target scores, rationales, regressions) as JSON to this file.")
def probe(
    provider_name: str,
    model: str | None,
    ollama_host: str | None,
    vllm_host: str | None,
    judge_provider_name: str,
    judge_model: str | None,
    judge_host: str | None,
    probes_file: str | None,
    artifact: str | None,
    artifact_file: str | None,
    thinking_flag: bool | None,
    thinking_budget: int,
    alert_on_regression: bool,
    save_results: bool,
    json_out: str | None,
) -> None:
    """Run LLM-evaluated live ecosystem health probes against configured artifact targets."""
    from pathlib import Path

    from atomics.probe.config import ProbeTarget, load_probe_config
    from atomics.probe.runner import run_probe

    console = Console()
    settings = load_settings()
    provider = _make_provider(provider_name, model, ollama_host, settings, vllm_host=vllm_host)
    judge = _make_provider(judge_provider_name, judge_model, judge_host or ollama_host, settings, vllm_host=vllm_host)

    targets = []
    if probes_file:
        targets = load_probe_config(Path(probes_file))
    elif artifact and artifact_file:
        targets = [ProbeTarget(
            name=Path(artifact_file).name,
            artifact_type=artifact,
            source="file",
            path=artifact_file,
        )]
    else:
        console.print("[red]Provide --probes-file or both --artifact and --file.[/red]")
        raise SystemExit(2)

    console.print(
        f"\n[bold]Ecosystem probe[/bold] — model: [cyan]{provider_name}[/cyan] ({model or 'default'})\n"
        f"Judge: [cyan]{judge_provider_name}[/cyan] | Targets: [bold]{len(targets)}[/bold]\n"
    )

    repo = None
    run_id = __import__("uuid").uuid4().hex[:12]
    if save_results:
        from atomics.storage.repository import MetricsRepository
        repo = MetricsRepository(settings.db_path)
        # Parent run row so probe runs are listable/queryable like other suites.
        repo.create_run(
            run_id, tier="probe", provider=provider_name,
            model=model or "default", trigger="manual",
        )

    def on_result(r):
        color = "green" if (r.score or 0) >= 0.8 else ("yellow" if (r.score or 0) >= 0.6 else "red")
        reg_tag = " [bold red][REGRESSION][/bold red]" if r.regressed else ""
        console.print(
            f" [bold]{r.target_name}[/bold] ({r.artifact_type}) "
            f"[{color}]{(r.score or 0) * 100:.1f}%[/]{reg_tag} — {r.judge_rationale[:80]}"
        )
        if repo:
            repo.save_probe_result(run_id, r)

    summary = asyncio.run(run_probe(
        provider,
        judge_provider=judge,
        targets=targets,
        model=model,
        judge_model=judge_model,
        thinking=thinking_flag,
        thinking_budget=thinking_budget,
        regression_threshold=0.10,
        on_result=on_result,
    ))

    table = Table(title="Probe Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Provider", provider_name)
    table.add_row("Targets", str(len(summary.results)))
    table.add_row("Overall Score", f"{(summary.overall_score or 0) * 100:.1f}%")
    if summary.regressions:
        table.add_row("[red]Regressions[/red]", str(len(summary.regressions)))
    console.print(table)

    if repo:
        repo.complete_probe_run(run_id)
        repo.close()

    if json_out:
        import json as _json
        with open(json_out, "w", encoding="utf-8") as fh:
            _json.dump(summary.to_dict(), fh, indent=2)
        console.print(f"[dim]Wrote JSON results to {json_out}[/dim]")

    if alert_on_regression and summary.regressions:
        console.print(
            f"\n[bold red]⚠ {len(summary.regressions)} probe(s) regressed >10% from last run[/bold red]"
        )
        for r in summary.regressions:
            console.print(f"  • {r.target_name}: {(r.prev_score or 0)*100:.1f}% → {(r.score or 0)*100:.1f}%")

# ── atomics qa ────────────────────────────────────────────────────────────────

@click.command()
@click.option("--repo", "repo_name", required=True, help="Repo spec under atomics/archreview/repos/")
@click.option("--models", "models_csv", required=True, help="Comma-separated models under test")
@click.option("--provider", "provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--ollama-host", type=str, default=None)
@click.option("--vllm-host", "vllm_host", type=str, default=None)
@click.option("--region", type=str, default="us-east-1", help="AWS region for Bedrock")
@click.option("--judge-provider", "judge_provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--judge-model", type=str, default=None)
@click.option("--judge-host", type=str, default=None)
@click.option("--tier", type=click.Choice(["floor", "local", "wide", "expanded"]), default="floor", show_default=True)
@click.option("--rounds", "--runs", "rounds", type=int, default=1, show_default=True,
              help="Number of analysis passes per model (--runs is an alias for cross-suite consistency).")
@click.option("--max-output-tokens", type=click.IntRange(min=128), default=2048, show_default=True,
              help="Maximum generated tokens for each model-under-test analysis")
@click.option("--inference-timeout", type=float, default=None,
              help="Per-request provider timeout in seconds (useful for slow local Ollama/vLLM runs)")
@click.option("--judge-only", is_flag=True, default=False, help="Skip objective scoring (no answer key needed)")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Stream per-model/per-round progress: findings and scores as they complete")
@click.option("--save/--no-save", "save_results", default=True)
@click.option("--json-out", "json_out", type=click.Path(dir_okay=False, writable=True), default=None,
              help="Write the full run (per-round findings, scores, cost) as JSON to this file.")
def archreview(repo_name, models_csv, provider_name, ollama_host, vllm_host,
               region, judge_provider_name, judge_model, judge_host, tier, rounds,
               max_output_tokens, inference_timeout, judge_only, verbose, save_results,
               json_out):
    """Benchmark models on a security-architecture review of a repo."""
    import asyncio
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

    def _build_provider(
        name: str,
        mdl: str | None,
        host: str | None,
        context_tokens: int | None = None,
    ):
        return _make_provider(
            name, mdl, host, settings,
            vllm_host=vllm_host, region=region,
            context_tokens=context_tokens, inference_timeout=inference_timeout,
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
        raise click.ClickException(
            f"Set {spec.path_env} to the local {spec.name} checkout."
        )

    tier_config = spec.tier(tier)
    archreview_max_output_tokens = max_output_tokens
    archreview_prompt_overhead_tokens = 4096
    archreview_context_tokens = (
        tier_config.budget_tokens
        + archreview_prompt_overhead_tokens
        + archreview_max_output_tokens
    )
    import uuid as _uuid_mod
    archreview_run_id = _uuid_mod.uuid4().hex[:12]

    pack = build_pack(Path(repo_dir), tier_config)
    console.print(f"[bold]archreview[/bold] repo=[cyan]{spec.name}[/cyan] tier={tier} "
                  f"pack={pack.file_count} files hash={pack.content_hash[:12]} "
                  f"context={archreview_context_tokens} reserve={archreview_max_output_tokens} "
                  f"overhead={archreview_prompt_overhead_tokens} "
                  f"run_id={archreview_run_id} "
                  f"{'(truncated)' if pack.truncated else ''}")

    judge_provider = _build_provider(
        judge_provider_name,
        judge_model,
        judge_host or ollama_host or settings.ollama_host,
        context_tokens=8192 if judge_provider_name == "ollama" else None,
    )

    repo = None
    if save_results:
        from atomics.storage.repository import MetricsRepository
        repo = MetricsRepository(settings.db_path)
        # Parent run row so archreview runs are listable/queryable like other
        # suites; per-round rows land in archreview_results.
        repo.create_run(
            archreview_run_id, tier="archreview", provider=provider_name,
            model=models_csv, trigger="manual",
        )

    judge_label = f"{judge_provider_name}:{judge_model or judge_provider.default_model or 'default'}"

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

    async def _run_all() -> None:
        for mdl in models:
            test_provider = _build_provider(provider_name, mdl,
                                            ollama_host if provider_name == "ollama" else vllm_host,
                                            context_tokens=archreview_context_tokens
                                            if provider_name == "ollama" else None)
            collisions = detect_self_judge(test_provider, mdl, [(judge_provider, judge_model)])
            if collisions:
                console.print(f"[yellow]warning:[/yellow] judge collides with model under test: {collisions}")

            if verbose:
                console.print(f"\n[bold]→ analyzing with [cyan]{mdl}[/cyan][/bold] "
                              f"({provider_name}, {rounds} round{'s' if rounds != 1 else ''})…")

            results = await run_archreview(
                spec=spec, tier=tier, pack=pack,
                under_test=test_provider, under_test_model=mdl,
                judge=judge_provider, judge_model=judge_model,
                rounds=rounds, objective=not judge_only,
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
                        console.print(f"  [red]round {r.round}: {_rich_escape(r.error_class or '')}: {_rich_escape(r.error_message or '')}[/red]")
                        continue
                    judge_str = f"{r.judge_score:.2f}" if r.judge_score is not None else "—"
                    flag = " [yellow](parse failed)[/yellow]" if r.parse_failed else ""
                    console.print(
                        f"  [dim]round {r.round}:[/dim] recall=[green]{r.objective_recall:.2f}[/green] "
                        f"prec={r.objective_precision:.2f} obj-f={r.objective_f:.2f} "
                        f"judge=[magenta]{judge_str}[/magenta] findings={len(r.findings)}"
                        f" matched={r.matched_categories or '—'}{flag}"
                    )
                    for f in r.findings:
                        console.print(f"      [dim]•[/dim] {f.category} · {f.location} · {f.severity}")

            cat_sets = [{f.category for f in r.findings} for r in results]
            recalls = [r.objective_recall for r in results]
            stability, _sd = compute_robustness(cat_sets, recalls)
            avg = lambda xs: round(sum(xs) / len(xs), 3) if xs else 0.0  # noqa: E731
            judge_vals = [r.judge_score for r in results if r.judge_score is not None]
            table.add_row(
                mdl, str(avg(recalls)), str(avg([r.objective_precision for r in results])),
                str(avg([r.objective_f for r in results])),
                str(avg(judge_vals) if judge_vals else "—"),
                judge_label, str(stability), str(round(sum(len(r.findings) for r in results) / len(results), 1)),
            )

    asyncio.run(_run_all())

    console.print(table)
    if repo:
        repo.complete_archreview_run(archreview_run_id)
        repo.close()

    if json_out:
        import json as _json

        from atomics.archreview.models import ArchReviewSummary
        summary = ArchReviewSummary(repo=spec.name, tier=tier, results=all_results)
        with open(json_out, "w", encoding="utf-8") as fh:
            _json.dump(summary.to_dict(), fh, indent=2)
        console.print(f"[dim]Wrote JSON results to {json_out}[/dim]")

# ── refusal / codereview (folded from legacy modules) ─────────────────────

@click.command("qa")
@click.option("--file", "-f", "qa_file", type=click.Path(exists=True), required=True,
              help="QA fixture YAML file (prompts + pass/fail patterns — no secrets).")
@click.option("--profile", "-p", "profile_path", type=click.Path(exists=True), default=None,
              help="Target profile YAML for app-level gates (gitignored, replaces --model/--ollama-host).")
@click.option("--model", "-m", type=str, default=None,
              help="Override model from fixture file (raw Ollama mode).")
@click.option("--ollama-host", type=str, default=None,
              help="Override Ollama host from fixture file (raw Ollama mode).")
@click.option("--num-predict", type=int, default=1024, show_default=True,
              help="Max output tokens per fixture prompt (raw Ollama mode only).")
@click.option("--fail-fast", is_flag=True, default=False,
              help="Stop after the first FAIL or ERROR.")
def qa(
    qa_file: str,
    profile_path: str | None,
    model: str | None,
    ollama_host: str | None,
    num_predict: int,
    fail_fast: bool,
) -> None:
    """QA validation — fire fixture prompts and check pass/fail patterns.

    Two modes:

    \b
    RAW OLLAMA (default): talks directly to an Ollama model.
      atomics qa --file qa/examples/ctf-solvability.yaml --model gemma4:26b

    \b
    PROFILE MODE: routes requests through an app-level HTTP target.
    The profile lives in profiles/local/ (gitignored — keeps your real
    box IPs and credentials out of the repo). The fixture file is safe
    to commit; it only contains prompts and patterns.
      atomics qa --file qa/examples/app-gate-guardrails.yaml \\
                 --profile profiles/local/my-gate.yaml

    \b
    Other examples:
      atomics qa --file qa/examples/ai-gate-regression.yaml --fail-fast
      atomics qa --file qa/examples/app-gate-guardrails.yaml \\
                 --profile profiles/local/my-policy.yaml
    """
    import asyncio as _asyncio
    import logging as _logging

    from atomics.qa_runner import QAResult, load_qa_suite, run_qa_suite

    _logging.getLogger("httpx").setLevel(_logging.WARNING)
    _logging.getLogger("httpcore").setLevel(_logging.WARNING)

    console = Console()

    file_model, file_host, fixtures = load_qa_suite(qa_file)

    # Load profile if given — it handles all transport details
    loaded_profile = None
    target_label: str
    if profile_path:
        from atomics.profiles import load_profile
        loaded_profile = load_profile(profile_path)
        target_label = f"profile:[bold cyan]{loaded_profile.name}[/bold cyan] ({loaded_profile.type})"
    else:
        effective_model = model or file_model
        effective_host = ollama_host or file_host
        if not effective_model:
            console.print("[red]No model specified. Set 'model' in the YAML or use --model.[/red]")
            raise SystemExit(1)
        target_label = f"[cyan]{effective_model}[/cyan]  Host: {effective_host}"

    console.print(
        f"[bold]QA Suite[/bold] — {len(fixtures)} fixture(s)\n"
        f"Target: {target_label}\n"
    )

    stopped_early = False
    results: list[QAResult] = []

    def _on_result(r: QAResult) -> None:
        icon = {"PASS": "[green]✓[/green]", "FAIL": "[red]✗[/red]", "ERROR": "[yellow]![/yellow]"}.get(r.status, "?")
        console.print(f"  {icon} [{r.status}] {r.fixture.id}  ({r.latency_ms/1000:.1f}s)")
        results.append(r)
        if fail_fast and r.status in ("FAIL", "ERROR"):
            raise KeyboardInterrupt("fail-fast triggered")

    try:
        suite = _asyncio.run(run_qa_suite(
            model=effective_model if not loaded_profile else "",
            host=effective_host if not loaded_profile else "",
            fixtures=fixtures,
            num_predict=num_predict,
            on_result=_on_result,
            profile=loaded_profile,
        ))
    except KeyboardInterrupt:
        stopped_early = True
        from atomics.qa_runner import QASuiteResult
        suite = QASuiteResult(model=effective_model, host=effective_host, results=results)

    console.print()
    rtable = Table(title="QA Results", show_lines=True)
    rtable.add_column("ID", style="cyan")
    rtable.add_column("Status", justify="center")
    rtable.add_column("Matched pass patterns")
    rtable.add_column("Matched fail patterns")
    rtable.add_column("Latency", justify="right")

    status_style_map = {"PASS": "[green]PASS[/green]", "FAIL": "[red]FAIL[/red]", "ERROR": "[yellow]ERROR[/yellow]"}
    for r in suite.results:
        rtable.add_row(
            r.fixture.id,
            status_style_map.get(r.status, r.status),
            ", ".join(r.matched_pass) or "-",
            ", ".join(r.matched_fail) or "-",
            f"{r.latency_ms/1000:.1f}s" if r.latency_ms else "-",
        )

    console.print(rtable)

    pass_color = "green" if suite.pass_rate == 1.0 else ("yellow" if suite.pass_rate >= 0.5 else "red")
    console.print(
        f"\n[bold]Pass rate:[/bold] [{pass_color}]{suite.passed}/{suite.total}[/{pass_color}]"
        + (" [dim](stopped early)[/dim]" if stopped_early else "")
    )
