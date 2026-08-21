# Stoneburner

[![PyPI](https://img.shields.io/pypi/v/stoneburner-atomics.svg)](https://pypi.org/project/stoneburner-atomics/)
[![GitHub release](https://img.shields.io/github/v/release/babywyrm/stoneburner)](https://github.com/babywyrm/stoneburner/releases/latest)
[![Python](https://img.shields.io/pypi/pyversions/stoneburner-atomics.svg)](https://pypi.org/project/stoneburner-atomics/)
[![License: MIT](https://img.shields.io/pypi/l/stoneburner-atomics.svg)](https://github.com/babywyrm/stoneburner/blob/main/LICENSE)
[![CI](https://github.com/babywyrm/stoneburner/actions/workflows/ci.yml/badge.svg)](https://github.com/babywyrm/stoneburner/actions/workflows/ci.yml)

Local-first LLM evaluation: token cost, quality, and security suites.
The same commands cover a laptop Ollama box and a cloud API.

Install **[stoneburner-atomics](https://pypi.org/project/stoneburner-atomics/)**.
The CLI and the import stay **`atomics`**. (`atomics` on PyPI is a different
package; `stoneburner` is too similar to an existing `stone-burner`.)

This is a desk tool, not a research harness and not an unsupervised agent.
It records cost, quality, and security-suite results in SQLite. A
finished-looking percentage on a partial run is the failure mode it is
built to avoid: incomplete coverage prints `n/a (scored/total scored)`
and JSON nulls the headline.

`atomics doctor` ends with one `Next:` command when the check is healthy.
Typical first-run output (Ollama on localhost, no cloud key):

```text
$ atomics doctor
Python 3.13.11 OK
Platform: Darwin (arm64)
Database path: data/atomics.db
SQLite database OK (readable / creatable)
ANTHROPIC_API_KEY not set (optional; needed for Claude)
OPENAI_API_KEY not set (optional; needed for OpenAI)
inference.env: not found (optional; $INFERENCE_ENV or /etc/agentic/inference.env)
Ollama endpoint: http://localhost:11434
Ollama reachable — 3 model(s): qwen2.5:7b, gemma3:4b, llama3.2:3b

Next: atomics provider-test --provider ollama --no-thinking
      Ollama is reachable.
```

```text
$ atomics toolcall --provider ollama --channel tools --runs 3 --no-thinking

Summary
  tool-capable: yes
  outcomes: safe call=6  no call=14
  channel divergence (resisted in prose, complied with tools): not measured (no qualifying fixtures)
  response divergence (dangerous call, refusing text): not measured (no qualifying fixtures)
  cost: $0.0000
```

A tools-only first run is valid. Channel divergence needs a second model
as judge. Thinking models that spend the token budget on hidden reasoning
are recorded as `thinking_budget`, not as a mystery generation failure.

> **New here?** [QUICKSTART](https://github.com/babywyrm/stoneburner/blob/main/QUICKSTART.md)
> — copy-pasteable commands grouped by goal.
>
> **Contributing?** [ARCHITECTURE](https://github.com/babywyrm/stoneburner/blob/main/ARCHITECTURE.md)
> — layer map, primitives, how to add an eval suite.

## Table of Contents

- [Install](#install)
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

## Install

Ollama on `http://localhost:11434` is the one-box path. No cloud key required.

```bash
uv tool install stoneburner-atomics
atomics doctor
atomics provider-test --provider ollama --no-thinking
atomics toolcall --provider ollama --channel tools --runs 3 --no-thinking
```

`--no-thinking` keeps reasoning models from spending the whole token budget
on hidden chain-of-thought. A tools-only `toolcall` run is a valid first
run; channel divergence needs a second model as judge.

API, MCP, or RAG extras:

```bash
uv tool install 'stoneburner-atomics[api,mcp]'
uv add 'stoneburner-atomics[rag]'          # from another project
```

From a clone, `uv sync --all-extras`. Bare `uv sync` drops the API, MCP,
RAG, and test extras.

Cloud providers work the same way once a key is set. `--effort` is the
shared reasoning dial (`none` / `minimal` / `low` / `medium` / `high` /
`xhigh` / `max`; aliases `xl`, `ultra`):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
atomics provider-test --effort high
atomics run -n 5 --effort medium
atomics report

atomics provider-test --provider openai -m gpt-5.6-sol --effort high
atomics eval --provider openai -m gpt-5.6-luna --effort low --verbose \
  --judge-provider claude --judge-model claude-haiku-4-5 --fixtures ev-01,ev-02
atomics run --provider bedrock --region us-east-1 --effort high -n 5
atomics provider-test --provider groq --effort medium
atomics provider-test --provider gemini --effort high
atomics provider-test --provider together --effort medium
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

Compare providers after running benchmarks: `atomics compare` — see
[COMPARING](https://github.com/babywyrm/stoneburner/blob/main/docs/COMPARING.md)
for model classes, metrics fidelity, and judge accuracy.

> **Optional extras:** from PyPI, `uv add 'stoneburner-atomics[rag]'` (or
> `[bedrock]`, `[openai]`, `[api]`, `[mcp]`). From a clone,
> `uv sync --all-extras`. RAG is `atomics rag-index` / `rag-retrieval` /
> `rag --index`. `atomics server --with-dashboard` serves the read-only UI
> at `/dashboard`. `atomics mcp` proxies a running API server and inherits
> that server's authentication and spend ceilings. See
> [MCP_SERVER](https://github.com/babywyrm/stoneburner/blob/main/docs/MCP_SERVER.md).

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
| `atomics eval` | Quality evaluation suite (`--effort` / `--verbose` for full transcripts) |
| `atomics adversarial` | Adversarial resilience eval (72 fixtures) |
| `atomics toolcall` | Tool-call divergence: refuses in prose, complies via function call (20 fixtures) |
| `atomics redblue` | Red/blue security capability eval (10 fixtures) |
| `atomics refusal` | Over- vs under-refusal calibration (12 fixtures) |
| `atomics codereview` | Planted-vuln detection in snippets and diffs (8 fixtures) |
| `atomics judge-agreement` | Same generation, N judges; pairwise agreement and majority-flip rate |
| `atomics labcompare` | Two-host throughput + quality bench-off |
| `atomics stress` | GPU saturation testing |
| `atomics soak` | Long-duration stability test |
| `atomics rag` | RAG pipeline evaluation (grounding, faithfulness, abstention) — also supports real retrieval from an indexed corpus |
| `atomics rag-index` | Build a sqlite-vec index from local documents for real RAG retrieval |
| `atomics rag-retrieval` | Measure retrieval quality (recall@k, precision@k, MRR, nDCG@k) from an index |
| `atomics multiturn` | Multi-turn conversation eval (context retention, coherence) |
| `atomics advisor` | Cost optimization recommendations from historical data |
| `atomics codegen` | Code generation eval (`--effort` on the generated function) |
| `atomics sweep` | Overnight multi-suite driver (`--suites`, `--runs 3`, status file + detachable log) |
| `atomics doctor` | Installation health check |
| `atomics server` | Run atomics as an HTTP API server |
| `atomics mcp` | Expose atomics to LLM agents over MCP (proxies the API server) |

Full reference: [CLI_REFERENCE](https://github.com/babywyrm/stoneburner/blob/main/docs/CLI_REFERENCE.md)

## Security Suites

Six eval suites for LLM security assessment:

| Suite | What it measures | Fixtures |
|-------|-----------------|----------|
| **adversarial** | Resistance to manipulation (prompt injection, jailbreaks, MCP attacks) | 72 |
| **toolcall** | Whether a prose refusal survives contact with a function call | 20 |
| **redblue** | Offensive/defensive security capability (OSINT, vuln analysis, IR) | 10 |
| **refusal** | Over-refusal vs under-refusal calibration | 12 |
| **codereview** | Vulnerability detection in code snippets and diffs | 8 |
| **archreview** | Security architecture reasoning against whole repos | per-repo |

Plus **probe** (live infrastructure analysis) and **sweep** (multi-model ranked comparison).

Full documentation: [SECURITY_SUITES](https://github.com/babywyrm/stoneburner/blob/main/docs/SECURITY_SUITES.md) ·
Leaderboards: [adversarial](https://github.com/babywyrm/stoneburner/blob/main/docs/LEADERBOARD.md) ·
[red/blue](https://github.com/babywyrm/stoneburner/blob/main/docs/LEADERBOARD-REDBLUE.md)

## Load Testing

| Command | Purpose |
|---------|---------|
| `atomics stress` | Ramp concurrency to find GPU saturation point |
| `atomics soak` | Long-duration stability with drift analysis |
| `atomics scenario` | Mixed-workload simulation with SLA scoring |
| `atomics capacity` | User load projection from stress data |
| `atomics labcompare` | Two-host throughput + quality bench-off |

Full documentation: [LOAD_TESTING](https://github.com/babywyrm/stoneburner/blob/main/docs/LOAD_TESTING.md)

## Thinking Mode

Auto-detects reasoning-capable models (Claude extended thinking, OpenAI o-series, Ollama qwen3) and tracks thinking tokens separately.

```bash
uv run atomics run --provider ollama -m qwen3:14b -n 5   # auto-detected
uv run atomics run --provider claude --thinking -n 5      # explicit
uv run atomics run --provider openai -m o3 --no-thinking  # forced off for A/B

# Shared --effort dial (mapped per provider). OpenAI pro uses Responses.
uv run atomics eval --provider openai -m gpt-5.6-sol --effort high
uv run atomics eval --provider openai -m gpt-5.6-sol --effort max --reasoning-mode pro
uv run atomics eval --provider claude -m claude-opus-4-6 --effort high
uv run atomics provider-test --provider bedrock --region us-east-1 --effort high
uv run atomics eval --provider openai -m gpt-5.6-luna --effort low --verbose \
  --judge-provider claude --judge-model claude-haiku-4-5 --fixtures ev-01,ev-02
```

Full documentation: [THINKING](https://github.com/babywyrm/stoneburner/blob/main/docs/THINKING.md)

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
│   ├── api/              # HTTP API server (FastAPI) — runs, evals, reports, jobs, dashboard
│   ├── commands/         # Click command modules (auth, admin, benchmark, eval, security, load, api, worker, distributed)
│   ├── distributed/      # Coordinator + worker for split and fleet runs
│   │   ├── coordinator.py
│   │   ├── models.py
│   │   ├── worker_client.py
│   │   ├── worker_runner.py
│   │   ├── routes.py
│   │   ├── rollup.py     # Per-worker aggregation of fleet results
│   │   └── auth.py
│   ├── load/             # Stress, soak, scenario, capacity
│   ├── benchmark/        # Sweep, qa_runner, labcompare
│   ├── reporting/        # Reports, exporters, webhooks
│   ├── core/             # Loop engine, task runner, rate/budget guard
│   ├── eval/             # Evaluation framework (eval, adversarial, redblue)
│   ├── probe/            # Live ecosystem probe suite
│   ├── archreview/       # Security-architecture repo benchmark
│   ├── providers/        # LLM adapters (Claude, Bedrock, OpenAI, Ollama, vLLM, brain-gateway)
│   ├── storage/          # SQLite metrics persistence (schema v21)
│   ├── scheduler/        # Cron/systemd/launchd generation and installation
│   └── cli.py            # Thin Click root — registers commands from commands/
├── profiles/             # Custom target profiles (local/ gitignored)
├── qa/                   # QA fixture suites (local/ gitignored)
├── tests/                # 2500+ tests
└── docs/                 # Detailed documentation
```

See [ARCHITECTURE](https://github.com/babywyrm/stoneburner/blob/main/ARCHITECTURE.md)
for the full layer map and contributor guide.

## Running Tests

The `api` extra is required to run the suite: the API and distributed test
modules import FastAPI at module scope, so without it pytest errors during
collection instead of skipping. The `mcp` extra is not required for collection
— those tests skip if the SDK is missing — but CI installs it so the MCP
surface is actually tested. Sync it locally too.

```bash
uv sync --all-extras
uv run pytest -q
uv run pytest -q --cov=atomics --cov-report=term-missing --cov-fail-under=85
```

The suite drives FastAPI's `TestClient`, an in-process shim, so it proves the
logic but not that the pieces work as separate processes. For that:

```bash
uv run python scripts/smoke_fleet.py
```

This starts a real coordinator and real worker processes, runs a two-host fleet
job against a stubbed OpenAI-compatible endpoint, then kills a worker mid-run to
confirm the job resolves to `partial` instead of waiting on a dead host. It needs
no credentials and no model, and touches no real database.

## Further Reading

Links are absolute so they work on PyPI as well as GitHub.

| Document | Description |
|----------|-------------|
| [QUICKSTART](https://github.com/babywyrm/stoneburner/blob/main/QUICKSTART.md) | Recipe-first guide grouped by goal |
| [CONTRIBUTING](https://github.com/babywyrm/stoneburner/blob/main/CONTRIBUTING.md) | Setup, the checks CI runs, and project conventions |
| [ARCHITECTURE](https://github.com/babywyrm/stoneburner/blob/main/ARCHITECTURE.md) | Layer map, primitives, contributor guide |
| [SECURITY](https://github.com/babywyrm/stoneburner/blob/main/SECURITY.md) | Vulnerability reporting and operational security considerations |
| [CHANGELOG](https://github.com/babywyrm/stoneburner/blob/main/CHANGELOG.md) | Version history |
| [RELEASING](https://github.com/babywyrm/stoneburner/blob/main/RELEASING.md) | Release process, versioning and tag conventions |
| [ROADMAP](https://github.com/babywyrm/stoneburner/blob/main/ROADMAP.md) | Priorities and future directions |
| [CLI_REFERENCE](https://github.com/babywyrm/stoneburner/blob/main/docs/CLI_REFERENCE.md) | Full CLI command reference |
| [API_SERVER](https://github.com/babywyrm/stoneburner/blob/main/docs/API_SERVER.md) | HTTP API server, authentication, distributed runs, dashboard |
| [MCP_SERVER](https://github.com/babywyrm/stoneburner/blob/main/docs/MCP_SERVER.md) | MCP server for LLM agents, tool surface, trust model |
| [SECURITY_SUITES](https://github.com/babywyrm/stoneburner/blob/main/docs/SECURITY_SUITES.md) | Security evaluation suites |
| [ADVERSARIAL_SUITES](https://github.com/babywyrm/stoneburner/blob/main/docs/ADVERSARIAL_SUITES.md) | Adversarial fixture flow, scoring, and categories |
| [LOAD_TESTING](https://github.com/babywyrm/stoneburner/blob/main/docs/LOAD_TESTING.md) | Stress, soak, scenario, capacity testing |
| [COMPARING](https://github.com/babywyrm/stoneburner/blob/main/docs/COMPARING.md) | Provider comparison, model classes, judge accuracy |
| [THINKING](https://github.com/babywyrm/stoneburner/blob/main/docs/THINKING.md) | Thinking/reasoning mode internals |
| [LEADERBOARD](https://github.com/babywyrm/stoneburner/blob/main/docs/LEADERBOARD.md) | Adversarial resistance leaderboard |
| [LEADERBOARD-REDBLUE](https://github.com/babywyrm/stoneburner/blob/main/docs/LEADERBOARD-REDBLUE.md) | Red/blue capability leaderboard |
| [FRONTIER_COMPARISON](https://github.com/babywyrm/stoneburner/blob/main/docs/FRONTIER_COMPARISON.md) | Local vs frontier model comparison |
| [INFERENCE_ENV](https://github.com/babywyrm/stoneburner/blob/main/docs/INFERENCE_ENV.md) | Vendor-neutral inference control file spec |

## License

MIT — see [LICENSE](https://github.com/babywyrm/stoneburner/blob/main/LICENSE).
