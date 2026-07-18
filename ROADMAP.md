# Roadmap

Current priorities and future directions for Stoneburner / Atomics.

## Recently Shipped

- **RAG Pipeline Evaluation** — `atomics rag` with 20 fixtures (security + general technical), grounding/faithfulness/abstention judge rubric, hallucination detection
- **README refactor** — 805 → 205 lines with TOC, anchors, and 5 linked focused docs
- **Compare improvements** — P50/P95 latency percentiles, $/1K tokens, model class taxonomy with mixed-class warnings
- **Schedule status** — `atomics schedule-status` with OS health checks, install/uninstall registry
- **Schema v20** — run metadata (tier, provider, model, trigger), schedules table, evaluation results ledger

## In Progress

### Eval Depth
- [x] **Multi-turn conversation benchmarks** — test context retention, coherence drift, and instruction following across multiple exchanges. Requires extending the `generate()` contract or building a turn-accumulating runner.
- [x] **Cost optimization advisor** — `atomics advisor` that analyzes historical runs and recommends cheaper models meeting a quality threshold. Pure SQL aggregation on existing data, no new API calls.

## Planned

### Infrastructure
- [ ] Dashboard / web UI for results visualization
- [ ] Webhook/Slack notifications on scheduled run regression
- [x] GitHub Actions workflow template for eval CI gates
- [ ] Distributed runs across multiple hosts with results aggregation
- [ ] API server mode (run atomics as a service, query via REST)

### Eval Quality
- [ ] RAG pipeline with real retrieval (vector DB integration, not just fixture chunks)
- [ ] Multi-turn conversation eval fixtures (context retention, contradiction detection)
- [ ] Code generation benchmarks (functional correctness, not just quality judging)
- [ ] Multilingual evaluation fixtures

### Provider Coverage
- [x] Google Gemini provider
- [x] Groq provider (fast inference)
- [x] Together AI provider
- [ ] Local llama.cpp direct (without Ollama wrapper)

### Phase 3 (npm workers)
- [ ] `atomics/workers/bridge.py` — Node.js worker integration for browser-based benchmarks
- [ ] npm worker pool for parallel fixture execution

## Design Principles

- **No breaking changes** to existing CLI commands or persistence
- **Additive schema migrations** with fresh-start policy pre-1.0
- **Every eval suite** gets: fixtures, judge rubric, runner, CLI command, `--json-out`, `--save/--no-save`, tests
- **Security by default** — sanitize errors, validate URLs, no self-judging, secrets in keychain
