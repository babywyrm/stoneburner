# RAG Pipeline Evaluation

## Problem

Single-prompt evals don't measure how well models use retrieved context.
Production deployments rely on retrieval-augmented generation, where a
model must ground answers in provided documents, synthesize across
chunks, resist distractors, and abstain when context doesn't contain the
answer. None of the existing suites test this.

## Design

### RAGFixture model

```python
@dataclass(frozen=True)
class RAGFixture:
    id: str                          # e.g. "rag-01"
    complexity: TaskComplexity       # LIGHT / MODERATE / HEAVY
    question: str                    # user question
    context_chunks: list[RAGChunk]   # retrieved documents
    gold_criteria: list[str]         # key concepts a correct answer must cover
    context_contains_answer: bool    # True = answer is in chunks; False = should abstain
    max_output_tokens: int           # generation budget

@dataclass(frozen=True)
class RAGChunk:
    content: str
    label: str          # "relevant" | "distractor"
    source: str         # e.g. "CVE-2026-1234.md", "api-docs/auth.md"
```

### Fixtures (20 total)

**Security (10):** CVE advisory retrieval, multi-chunk synthesis from
incident reports, policy compliance with contradictory versions, threat
intel from noisy feeds, abstention on missing CVEs, credential exposure
detection in audit logs, SBOM dependency analysis, security runbook
extraction, alert triage from mixed logs, patch guidance synthesis.

**General technical (10):** API doc synthesis spanning endpoints,
architecture decision retrieval with outdated distractors, runbook
procedure extraction, config troubleshooting from mixed logs, abstention
on undocumented features, error message diagnosis, migration guide
synthesis, capacity planning from metrics docs, API versioning guidance,
deployment checklist from scattered docs.

### Judge rubric (RAG-specific)

| Dimension | Range | Measures |
|-----------|-------|----------|
| Grounding | 0–4 | References/uses the provided context |
| Faithfulness | 0–3 | Stays within what context says (no hallucination) |
| Abstention | 0–3 | Correctly declines when context lacks the answer |

Normalized: `(grounding + faithfulness + abstention) / 10.0` → 0.0–1.0.

For `context_contains_answer=True`, abstention scores NOT abstaining.
For `context_contains_answer=False`, abstention scores correctly declining.

### Metrics

- `grounding_score` — avg grounding across fixtures
- `faithfulness_score` — avg faithfulness
- `abstention_accuracy` — % correct abstain/answer decisions
- `hallucination_rate` — % of answers with facts not in context
- `overall_rag_score` — weighted composite (0.0–1.0)

### CLI

```bash
atomics rag --provider ollama -m qwen3:14b --judge-model qwen2.5:14b
atomics rag --fixtures rag-05,rag-12
atomics rag --json-out rag.json
atomics rag --runs 3
```

Follows existing eval patterns: `--save/--no-save`, `--allow-partial`,
`--json-out`, provider/judge construction via `commands.common`.

### Persistence

Results to `task_results` with `suite='rag'`. Composite score on
`accuracy_score`. Per-dimension RAG scores in `evaluation_results`
`result_json`. Reuses existing schema v20 — no migration needed.

### File layout

```
atomics/eval/rag/
├── __init__.py       # RAGFixture, RAGChunk, ALL_FIXTURES
├── fixtures.py       # 20 fixtures (security + general)
├── judge.py          # RAG rubric, score_rag_response()
└── runner.py         # run_rag() async orchestrator
```

### Commit plan

1. RAG eval framework — models, fixtures, judge, runner, CLI command
2. Tests, roadmap update, docs
