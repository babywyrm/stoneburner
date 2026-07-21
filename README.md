# Stoneburner

> **Atomics** — Agentic token usage benchmarking platform

A continuous, cron-schedulable benchmarking harness that runs realistic everyday tasks against LLM providers to measure token consumption, cost, throughput, and performance trends over time. Supports tiered usage profiles, multiple providers (including local Ollama), and a full security evaluation suite.

> **New here?** Start with [**QUICKSTART.md**](QUICKSTART.md) — copy-pasteable commands grouped by goal.
>
> **Contributing?** Read [**ARCHITECTURE.md**](ARCHITECTURE.md) — layer map, primitives, how to add an eval suite.

## Table of Contents

- [Quick Start](#quick-start)
- [Providers](#providers)
- [Burn Tiers](#burn-tiers)
- [Key Commands](#key-commands)
- [Security Suites](#security-suites)
- [Load Testing](#load-testing)
- [Thinking Mode](#thinking-mode)
- [Configuration](#configuration)
- [Secrets Management](#secrets-management)
- [Architecture](#architecture)
- [Running Tests](#running-tests)
- [Further Reading](#further-reading)

## Quick Start

```bash
uv sync
export ANTHROPIC_API_KEY=sk-ant-...

uv run atomics provider-test          # verify connection
uv run atomics run -n 5               # run 5 benchmark tasks
uv run atomics report                 # view results

# other providers
uv run atomics run --provider openai -n 5
uv run atomics run --provider bedrock --region us-east-1 -n 5
uv run atomics run --provider ollama -n 5
```

## Providers

| Provider | Flag | Install |
|----------|------|---------|
| **Claude** (Anthropic) | `--provider claude` (default) | `uv sync` |
| **Bedrock** (AWS) | `--provider bedrock --region us-east-1` | `uv sync --extra bedrock` |
| **OpenAI / Codex** | `--provider openai` | `uv sync --extra openai` |
| **Ollama** (local) | `--provider ollama` | `uv sync` (uses httpx) |
| **brain-gateway** | `--provider brain-gateway` | `uv sync` (uses httpx) |
| **Groq** (cloud) | `--provider groq` | `uv sync` (uses httpx) |
| **Together AI** (cloud) | `--provider together` | `uv sync` (uses httpx) |
| **Google Gemini** | `--provider gemini` | `uv sync` (uses httpx) |
| **llama.cpp** (local) | `--provider llamacpp` | `uv sync` (uses httpx) |
| **vLLM** (OpenAI-compat) | `--provider vllm` | `uv sync` (uses httpx) |

Compare providers after running benchmarks: `uv run atomics compare` — see [docs/COMPARING.md](docs/COMPARING.md) for model classes, metrics fidelity, and judge accuracy.

> **Optional extras:** Real RAG retrieval (`atomics rag-index`, `atomics rag-retrieval`, `atomics rag --index`) requires `uv sync --extra rag` to install `sqlite-vec` and `sentence-transformers`. Bedrock and OpenAI providers need `--extra bedrock` and `--extra openai` respectively.

The API server mode requires `uv sync --extra api` to install FastAPI and uvicorn.

## Burn Tiers

| Tier | Tasks | Model | Interval | Budget | Tokens/hr |
|------|-------|-------|----------|--------|-----------|
| **ez** | Light only | Haiku 4.5 | 300s | $5 | 15k |
| **baseline** | Light + Moderate | Sonnet 4.6 | 120s | $50 | 100k |
| **mega** | All (incl. Heavy) | Sonnet 4.6 | 30s | $250 | 500k |

```bash
uv run atomics run --tier ez -n 3 -i 5
uv run atomics tiers                   # show all tier profiles
```

## Key Commands

| Command | Description |
|---------|-------------|
| `atomics run` | Start benchmarking loop |
| `atomics compare` | Provider/model side-by-side comparison |
| `atomics report` | Usage reports and trends |
| `atomics eval` | Quality evaluation suite |
| `atomics adversarial` | Adversarial resilience eval (72 fixtures) |
| `atomics redblue` | Red/blue security capability eval |
| `atomics stress` | GPU saturation testing |
| `atomics soak` | Long-duration stability test |
| `atomics rag` | RAG pipeline evaluation (grounding, faithfulness, abstention) — also supports real retrieval from an indexed corpus |
| `atomics rag-index` | Build a sqlite-vec index from local documents for real RAG retrieval |
| `atomics rag-retrieval` | Measure retrieval quality (recall@k, precision@k, MRR, nDCG@k) from an index |
| `atomics multiturn` | Multi-turn conversation eval (context retention, coherence) |
| `atomics advisor` | Cost optimization recommendations from historical data |
| `atomics codegen` | Code generation eval (functional correctness via test execution) |
| `atomics sweep` | Multi-model eval sweep |
| `atomics doctor` | Installation health check |
| `atomics server` | Run atomics as an HTTP API server |

Full reference: [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md)

## Security Suites

Five eval suites for LLM security assessment:

| Suite | What it measures | Fixtures |
|-------|-----------------|----------|
| **adversarial** | Resistance to manipulation (prompt injection, jailbreaks, MCP attacks) | 72 |
| **redblue** | Offensive/defensive security capability (OSINT, vuln analysis, IR) | 10 |
| **refusal** | Over-refusal vs under-refusal calibration | per-suite |
| **codereview** | Vulnerability detection in code snippets and diffs | per-suite |
| **archreview** | Security architecture reasoning against whole repos | per-repo |

Plus **probe** (live infrastructure analysis) and **sweep** (multi-model ranked comparison).

Full documentation: [docs/SECURITY_SUITES.md](docs/SECURITY_SUITES.md) ·
Leaderboards: [adversarial](docs/LEADERBOARD.md) · [red/blue](docs/LEADERBOARD-REDBLUE.md)

## Load Testing

| Command | Purpose |
|---------|---------|
| `atomics stress` | Ramp concurrency to find GPU saturation point |
| `atomics soak` | Long-duration stability with drift analysis |
| `atomics scenario` | Mixed-workload simulation with SLA scoring |
| `atomics capacity` | User load projection from stress data |
| `atomics labcompare` | Two-host throughput + quality bench-off |

Full documentation: [docs/LOAD_TESTING.md](docs/LOAD_TESTING.md)

## Thinking Mode

Auto-detects reasoning-capable models (Claude extended thinking, OpenAI o-series, Ollama qwen3) and tracks thinking tokens separately.

```bash
uv run atomics run --provider ollama -m qwen3:14b -n 5   # auto-detected
uv run atomics run --provider claude --thinking -n 5      # explicit
uv run atomics run --provider openai -m o3 --no-thinking  # forced off for A/B
```

Full documentation: [docs/THINKING.md](docs/THINKING.md)

## Configuration

Set via environment variables (prefix `ATOMICS_`) or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Claude provider |
| `OPENAI_API_KEY` | — | OpenAI provider |
| `ATOMICS_DEFAULT_MODEL` | `claude-sonnet-4-6` | Default model |
| `ATOMICS_OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `ATOMICS_OLLAMA_MODEL` | `qwen2.5:7b` | Default Ollama model |
| `ATOMICS_OLLAMA_TIMEOUT` | `300` | Per-request timeout (s) |
| `ATOMICS_DB_PATH` | (platform) | SQLite location |
| `ATOMICS_BUDGET_LIMIT_USD` | `50.00` | Cost cap per run |

**Database defaults:** macOS: `data/atomics.db` · Linux: `~/.local/share/atomics/atomics.db` (XDG)

CLI flags (`--tier`, `--budget`, `--interval`) override these at runtime.

## Secrets Management

Layered resolution: environment variable → `.env` file → OS keychain (macOS Keychain / Linux secret-service).

```bash
atomics secrets set ANTHROPIC_API_KEY   # store securely (hidden input)
atomics secrets list                    # verify
atomics secrets delete ANTHROPIC_API_KEY
```

## Architecture

```
stoneburner/
├── atomics/              # Core Python package
│   ├── commands/         # Extracted Click commands and shared CLI policy
│   ├── core/             # Loop engine, task runner, rate/budget guard
│   ├── eval/             # Evaluation framework (eval, adversarial, redblue)
│   ├── probe/            # Live ecosystem probe suite
│   ├── archreview/       # Security-architecture repo benchmark
│   ├── providers/        # LLM adapters (Claude, Bedrock, OpenAI, Ollama, vLLM, brain-gateway)
│   ├── storage/          # SQLite metrics persistence (schema v20)
│   ├── scheduler/        # Cron/systemd/launchd generation and installation
│   └── cli.py            # Click CLI entry point
├── profiles/             # Custom target profiles (local/ gitignored)
├── qa/                   # QA fixture suites (local/ gitignored)
├── tests/                # 1593+ tests at 85% coverage
└── docs/                 # Detailed documentation
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full layer map and contributor guide.

## Running Tests

```bash
uv sync --extra dev
uv run python -m pytest -v
uv run python -m pytest --cov=atomics --cov-report=term-missing
```

## Further Reading

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | Recipe-first guide grouped by goal |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layer map, primitives, contributor guide |
| [SECURITY.md](SECURITY.md) | Operational security considerations |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [ROADMAP.md](ROADMAP.md) | Priorities and future directions |
| [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md) | Full CLI command reference |
| [docs/SECURITY_SUITES.md](docs/SECURITY_SUITES.md) | Security evaluation suites |
| [docs/LOAD_TESTING.md](docs/LOAD_TESTING.md) | Stress, soak, scenario, capacity testing |
| [docs/COMPARING.md](docs/COMPARING.md) | Provider comparison, model classes, judge accuracy |
| [docs/THINKING.md](docs/THINKING.md) | Thinking/reasoning mode internals |
| [docs/LEADERBOARD.md](docs/LEADERBOARD.md) | Adversarial resistance leaderboard |
| [docs/LEADERBOARD-REDBLUE.md](docs/LEADERBOARD-REDBLUE.md) | Red/blue capability leaderboard |
| [docs/FRONTIER_COMPARISON.md](docs/FRONTIER_COMPARISON.md) | Local vs frontier model comparison |
| [docs/INFERENCE_ENV.md](docs/INFERENCE_ENV.md) | Vendor-neutral inference control file spec |

## License

MIT
