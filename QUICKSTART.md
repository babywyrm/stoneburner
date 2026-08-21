# Stoneburner Quickstart

Recipe-first guide to **`atomics`** — local-first LLM eval (cost, quality,
security). The PyPI listing is `stoneburner-atomics`; the CLI stays
`atomics`. Every block below is copy-pasteable. For the full reference see
[`README.md`](README.md).

> **Mental model:** point `atomics` at a provider (cloud API or a local
> Ollama/vLLM box), pick a *goal* (cost, quality, safety, scale), run the
> matching command, then `compare`/`report`/`export` the results out of SQLite.

---

## 1. 60-second setup

From PyPI (Ollama on `http://localhost:11434`, no cloud key):

```bash
uv tool install stoneburner-atomics
atomics doctor
atomics provider-test --provider ollama --no-thinking
```

Thinking models (qwen3, qwen3.8, deepseek-r1) need `--no-thinking` or the
visible answer can come back empty.

From a clone: `uv sync --all-extras`, then prefix commands with `uv run`.
Bare `uv sync` drops the API, MCP, RAG, and test extras.

Cloud keys are optional. Store them in the OS keychain if you use them:

```bash
atomics secrets set ANTHROPIC_API_KEY
# or: export ANTHROPIC_API_KEY=sk-ant-...
# or: export OPENAI_API_KEY=sk-...
```

Point at a non-local Ollama with `ATOMICS_OLLAMA_HOST` (default
`http://localhost:11434`). A box that already has an `inference.env`
(`$INFERENCE_ENV` or `/etc/agentic/inference.env`) is enough:
`atomics doctor` shows the backend/URL/model (never the API key), and
`atomics run --provider ollama` fills host and model from the file
unless you pass flags or `ATOMICS_*`. See
[`docs/INFERENCE_ENV.md`](docs/INFERENCE_ENV.md).

`doctor` exits non-zero if anything is missing, so it's safe in front of a
long run: `atomics doctor && atomics run --tier ez -n 3`. A healthy check
prints one `Next:` command — Ollama when it answers, otherwise Claude if
a key is set.

---

## 2. Choose a backend — local **and** cloud are first-class

`atomics` treats local inference and cloud APIs as peers. Every suite
(`run`, `eval`, `sweep`, `adversarial`, `redblue`, `toolcall`, `refusal`,
`codereview`, `probe`) takes the same `--provider` flag, so you can
benchmark a private on-prem model and a frontier cloud model with identical
commands and compare them side-by-side.

### Local / self-hosted (private, $0, nothing leaves the LAN)

| Backend | Flag | When to use |
|---------|------|-------------|
| **Ollama** | `--provider ollama` | Default for eval/security suites — free, private GPU box |
| **vLLM / OpenAI-compatible gateway** | `--provider vllm` | LiteLLM, vLLM, TGI, or any `/v1/chat/completions` endpoint on the LAN |
| **brain-gateway** | `--provider brain-gateway` | Optional HTTP gateway in front of local models |

```bash
# Ollama on this machine — list models (class + thinking annotations)
atomics models --host http://localhost:11434
atomics run --provider ollama -m qwen2.5:7b --ollama-host http://localhost:11434 -n 5 -i 0

# vLLM / OpenAI-compatible gateway (e.g. LiteLLM at :8000/v1)
atomics run --provider vllm --vllm-host http://localhost:8000/v1 -m qwen2.5:7b -n 5 -i 0

# Optional brain-gateway
atomics run --provider brain-gateway --gateway-url http://localhost:30080 -n 5 -i 0
```

> **Model-agnostic:** `-m`/`--model` accepts *any* model the backend serves —
> the `qwen*` tags in these examples are just placeholders. `gemma4:12b`,
> `llama3.2:3b`, `mistral:7b`, `phi4:latest`, `deepseek-r1:14b`, `dolphin3:8b`,
> and friends all work the same way. Known families get automatic
> thinking-mode detection and light/mid/heavy class tagging; an unrecognized
> model still runs and simply defaults its class. Run `atomics models --host …`
> to see what a box serves with its annotations.

### Cloud APIs (frontier quality, billed per token)

| Backend | Flag | Install |
|---------|------|---------|
| **Claude** (Anthropic) | `--provider claude` | `uv sync` (included) |
| **OpenAI** / GPT-5.6 / o-series | `--provider openai` | `uv sync --extra openai` |
| **Bedrock** (AWS) | `--provider bedrock --region us-east-1` | `uv sync --extra bedrock` |
| **Groq** | `--provider groq` | `uv sync` (httpx) |
| **Gemini** | `--provider gemini` | `uv sync` (httpx) |
| **Together** | `--provider together` | `uv sync` (httpx) |

```bash
uv run atomics run --provider claude -n 5 -i 0
uv run atomics run --provider openai -m gpt-5.6-luna -n 5 -i 0
uv run atomics run --provider bedrock --region us-east-1 -n 5 -i 0
uv run atomics provider-test --provider groq
uv run atomics provider-test --provider gemini
uv run atomics provider-test --provider together
```

`--effort` is one dial for every cloud that has a native reasoning knob.
`--thinking-budget` stays the local / older Claude token cap. Every
generate suite takes `--effort`, including `codegen` and `probe`.

```bash
# OpenAI — Chat Completions reasoning_effort. --reasoning-mode pro uses Responses.
uv run atomics provider-test --provider openai -m gpt-5.6-sol --effort high
uv run atomics provider-test --provider openai -m gpt-5.6-sol --effort max --reasoning-mode pro
uv run atomics eval --provider openai -m gpt-5.6-luna --effort low --budget 1 \
  --judge-provider claude --judge-model claude-haiku-4-5 \
  --fixtures ev-01,ev-02,ev-03 --verbose

# Claude — adaptive thinking + output_config.effort
uv run atomics eval --provider claude -m claude-opus-4-6 --effort high --verbose

# Bedrock — same Claude mapping, region-prefixed model IDs
uv run atomics provider-test --provider bedrock --region us-east-1 --effort high

# OpenAI-compatible clouds — reasoning_effort when the backend honors it
uv run atomics provider-test --provider groq --effort medium
uv run atomics provider-test --provider gemini --effort high
uv run atomics provider-test --provider together --effort medium
```

> **Ecosystem fit:** run private/local models for sensitive workloads and cost
> control, reach for cloud models when you need frontier capability — then use
> `atomics compare` to make the local-vs-cloud trade-off with real numbers
> (cost, latency, tok/s, quality) instead of vibes.

---

## 3. Recipes by goal

### "How much will this model cost, and how fast is it?"

```bash
# Run N tasks; measures tokens, cost, latency, tok/s
uv run atomics run --provider ollama -m qwen2.5:7b -n 5 -i 0

# Same on a cloud model
uv run atomics run --provider claude -n 5 -i 0

# See the trend report
uv run atomics report

# Side-by-side once you've run a few providers/models
uv run atomics compare              # by provider
uv run atomics compare --by model   # by individual model
```

### "Is the model any good?" — quality eval with an LLM judge

The judge defaults to a **local Ollama model**, so scoring is $0. Quality is a
0–100% accuracy score over 25 fixtures plus an objective `criteria_coverage`.

```bash
# Full 25-fixture eval, judged locally
uv run atomics eval --provider ollama -m qwen2.5:7b --judge-model qwen2.5:14b

# Fast spot-check on just a few fixtures (great for iterating)
uv run atomics eval --provider ollama -m qwen3:4b --fixtures ev-01,ev-19

# Cloud model under test, strong local judge
uv run atomics eval --provider claude --judge-provider ollama --judge-model qwen2.5:14b

# Cloud model + cloud judge + full transcripts. Eval defaults the judge to
# local Ollama — pass --judge-provider when Ollama is not running.
uv run atomics eval --provider openai -m gpt-5.6-luna --effort low --budget 1 \
  --judge-provider claude --judge-model claude-haiku-4-5 \
  --fixtures ev-01,ev-02,ev-03 --verbose
```

> **Never self-judge.** A model grading its own answers is biased upward. Use a
> *different* (ideally stronger) judge than the model under test — the runner
> prints a loud warning if it detects a collision.

#### Multi-judge consensus

Score with a panel and get an inter-judge disagreement signal
(`judge_score_stdev`):

```bash
# Mixed judge panel spanning model families keeps any one family's bias in check
uv run atomics eval --provider ollama -m gemma4:12b \
  --judge-model qwen2.5:14b \
  --extra-judges ollama:mistral:7b,ollama:deepseek-r1:14b \
  --fixtures ev-18,ev-19,ev-20
```

#### Trust the judge

```bash
# Prove the configured judge ranks wrong < thin < thorough answers correctly
ATOMICS_LIVE_JUDGE=1 uv run pytest tests/test_calibration.py::test_live_judge_is_calibrated -q
```

### "Which of my models is best?" — multi-model sweep

```bash
# Sweep every model on the GPU box, ranked table
uv run atomics sweep --all-local --host http://localhost:11434

# Specific models across families, just a few fixtures
# (use tags that are actually pulled on the host — a missing tag shows
#  as FAIL with a "404 Not Found" reason in the summary)
uv run atomics sweep --models gemma4:12b,llama3.2:1b,mistral:7b,phi4:latest,deepseek-r1:14b --fixtures ev-01,ev-02,ev-03
```

### "Is it safe?" — security evaluation suites

```bash
# Resistance to prompt injection / jailbreaks (higher = more resistant)
uv run atomics adversarial --provider ollama -m qwen3:14b --judge-model qwen2.5:14b

# Offensive + defensive security capability (red/blue tasks)
uv run atomics redblue --provider ollama -m qwen3:14b

# Tool-call divergence: refuses in chat, then emits the call (never executed)
uv run atomics -v toolcall -p ollama -m qwen3:14b \
  --judge-provider ollama --judge-model qwen2.5:14b --runs 3 --no-thinking

# One box, no second GPU: tools only. Live lines still print.
# Channel divergence is not measured — that needs a judge.
uv run atomics doctor
uv run atomics provider-test -p ollama -m qwen3:14b --no-thinking
uv run atomics -v toolcall -p ollama -m qwen3:14b \
  --category direct --channel tools --runs 3 --no-thinking --no-skip-incapable

# Point an LLM at real artifacts (logs, scan reports, configs)
uv run atomics probe --artifact access-log --file /var/log/nginx/access.log
```

**Reading the scores — capability vs resilience are different axes:**

- `redblue` measures **capability** (0–100%): can the model *do* security work
  (recon, vuln analysis, incident response, hardening…). Higher = more capable.
- `adversarial` measures **resilience** (0–100%): does the model *resist* being
  manipulated (prompt injection, jailbreaks, encoded payloads…). Higher = harder
  to subvert. It flags **CRITICAL/HIGH** fixtures where the model *complied* with
  an attack — read those first.
- `toolcall` measures the **agent gap**: refused in chat, then emitted the
  call when handed function schemas. Calls are never executed. Live lines
  print `prose=resisted` + `DANGEROUS` when that gap is the result. A
  tools-only one-box run is a valid first run and cannot produce channel
  divergence. Omit a judge and divergence is not measured.

A model can score high on one and low on the other. In practice a **capable but
low-resilience** model (good at the tasks, easy to manipulate) is the riskiest
profile — it argues for a guardrail layer in front of the model rather than
trusting its own refusals. Run both suites and weigh them together:

```bash
# Full profile for one model: capability + resilience, strong separate judge
uv run atomics redblue     --provider ollama -m gemma4:12b --judge-model qwen2.5:14b
uv run atomics adversarial --provider ollama -m gemma4:12b --judge-model qwen2.5:14b
```

### "Will it scale?" — capacity, stress, soak, scenario

```bash
# Find the GPU saturation point (ramp concurrency 1→8)
uv run atomics stress --model qwen2.5:7b --max-concurrency 8 --ollama-host http://localhost:11434

# How many users can this setup serve? (pure math from measured data)
uv run atomics capacity --users 200 --model qwen2.5:7b

# Hold load for 30 min, classify STABLE/DEGRADED/UNSTABLE (catches VRAM leaks)
uv run atomics soak --model qwen2.5:7b --duration 30m -c 4

# Multiple agentic services competing for one GPU
uv run atomics scenario -w "gate:qwen2.5:3b:2:5000" -w "eval:qwen2.5:7b:1:15000" -d 60
```

### "Does my AI gate still work?" — QA regression

```bash
# Test a model directly against pass/fail patterns
uv run atomics qa --file qa/examples/app-gate-guardrails.yaml \
                  --model qwen2.5:3b --ollama-host http://localhost:11434

# Test a real app endpoint (secrets stay in a gitignored profile)
uv run atomics qa --file qa/examples/ai-gate-regression.yaml \
                  --profile profiles/local/my-gate.yaml
```

---

## 3b. Safety & adversarial resilience

```bash
# Test how well a model resists manipulation (local, free)
uv run atomics adversarial -p ollama --ollama-host http://bb:11434 -m qwen2.5:3b --runs 3

# Use Claude as a calibrated judge (paid, ~$0.03/run)
uv run atomics adversarial -p ollama --ollama-host http://bb:11434 -m qwen3.5:4b \
  --judge-provider claude --judge-model claude-haiku-4-5-20251001 --runs 3

# Test only one suite/group: mcp, tool_safety, zerotrust, agentic,
# multiturn, rag_poisoning, tool_desc_injection
uv run atomics adversarial -p ollama -m qwen2.5:7b --category tool_desc_injection --runs 3

# Compare two models on the same fixtures (per-fixture diff + overall delta)
uv run atomics adversarial -p ollama -m mistral-nemo:12b --compare mistral-small:24b --runs 3

# Export the full run as JSON for a dashboard / notebook
uv run atomics adversarial -p ollama -m qwen2.5:7b --json-out run.json

# CI gate: fail the build if resilience drops below 60%
uv run atomics adversarial -p ollama -m qwen2.5:7b --fail-on-resilience 60

# Run red/blue capability eval (variance-aware + JSON export)
uv run atomics redblue -p ollama -m qwen3.5:4b --runs 3 --no-thinking --json-out redblue.json

# Tool-call divergence. Live lines print each pass; a judge is required
# or prose stays unjudged and channel divergence is not measured.
uv run atomics -v toolcall -p ollama -m qwen3.8:27b \
  --judge-provider ollama --judge-model qwen2.5:14b \
  --runs 3 --no-thinking --no-skip-incapable --json-out toolcall.json

# Measure over- and under-refusal; fixture rows are saved as they complete
uv run atomics refusal -p ollama -m qwen3.5:4b \
  --judge-model qwen2.5:14b --no-thinking --json-out refusal.json

# Review planted vulnerabilities without persisting to SQLite
uv run atomics codereview -p ollama -m qwen3.5:4b \
  --judge-model qwen2.5:14b --no-thinking --no-save --json-out codereview.json

# In automation, accept incomplete coverage while retaining integrity diagnostics
uv run atomics refusal -p ollama -m qwen3.5:4b --allow-partial

# Security architecture review
uv run atomics archreview -p ollama -m qwen2.5:7b --pack camazotz
```

Evaluation commands show per-fixture progress and total judge cost. Refusal and
code-review runs exit nonzero on partial or infrastructure-invalid integrity by
default, after requested JSON is written and saved fixture rows are finalized.
Their schema-v20 `evaluation_results.result_json` data includes raw model and
judge evidence, so protect exports and the metrics database.

See [`docs/LEADERBOARD.md`](docs/LEADERBOARD.md) and
[`docs/ADVERSARIAL_SUITES.md`](docs/ADVERSARIAL_SUITES.md) for the full suite
breakdown and benchmark results. The suite currently has 72 fixtures; the
leaderboards are dated snapshots taken when it was smaller.

---

## 3c. Compare two inference hosts (labcompare)

```bash
# Side-by-side throughput + quality parity between two boxes
uv run atomics labcompare \
  --host host-a=http://gpu-a:11434 \
  --host host-b=http://gpu-b:11434 \
  --models qwen2.5:7b,qwen3:14b,qwen3.6:27b

# Throughput only (faster — skips eval fixtures)
uv run atomics labcompare \
  --host a=http://host-a:11434 --host b=http://host-b:11434 \
  --models qwen3.6:27b --dimensions throughput --prompts 3

# Use a strict judge for quality parity scoring
uv run atomics labcompare \
  --host a=http://host-a:11434 --host b=http://host-b:11434 \
  --models qwen3:14b --judge-model qwen3.6:35b-a3b

# Export results as JSON
uv run atomics labcompare \
  --host a=http://host-a:11434 --host b=http://host-b:11434 \
  --models qwen2.5:7b -o comparison.json
```

Reports per-model throughput (tok/s), VRAM fit (% in GPU vs CPU offload), and
quality score on each host with a speedup ratio and parity verdict.

---

## 3d. RAG, multi-turn, and cost optimization

```bash
# RAG pipeline eval — does the model ground answers in provided context?
uv run atomics rag --provider ollama -m qwen3:14b --judge-model qwen2.5:14b
uv run atomics rag --fixtures rag-05,rag-12          # subset
uv run atomics rag --extra-judges ollama:mistral:7b  # numeric mean of the panel

# Real retrieval against your own indexed corpus (requires uv sync --extra rag)
uv run atomics rag-index ./docs --db ./my-index.vec
uv run atomics rag --index ./my-index.vec --provider ollama -m qwen3:14b --judge-model qwen2.5:14b
uv run atomics rag-retrieval --index ./my-index.vec --gold ./relevance.json

# Multi-turn conversation eval — context retention, coherence, instruction following
uv run atomics multiturn --provider ollama -m qwen3:14b --judge-model qwen2.5:14b
uv run atomics multiturn --fixtures mt-eval-01       # subset

# Cost advisor — find cheaper models that still meet quality thresholds
uv run atomics advisor                               # default 80% quality floor
uv run atomics advisor --min-quality 0.9             # higher bar
uv run atomics advisor --current-model claude-sonnet-4-6 --json-out recs.json
```

---

## 4. Get the data out

```bash
uv run atomics compare --output results.json          # comparison JSON
uv run atomics export --suite all --format csv -o all.csv
uv run atomics export --suite sweep -o sweep.jsonl
uv run atomics export --suite adversarial -o adv.jsonl   # adversarial results
uv run atomics export --suite redblue -o redblue.jsonl   # redblue rows only
uv run atomics export --suite eval -o eval.jsonl         # eval rows only
```

---

## 5. Schedule it (continuous benchmarking)

```bash
# Auto-detect cron/systemd/launchd and install
uv run atomics schedule --tier ez -n 5 -i 15 --install
uv run atomics schedule-status     # show installed schedules + health
uv run atomics schedule --tier ez --uninstall
```

---

## 6. Config cheat-sheet

Set via env vars (prefix `ATOMICS_`) or a `.env` file in the repo root:

| Variable | Default | Notes |
|----------|---------|-------|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | — | cloud providers |
| `ATOMICS_OLLAMA_HOST` | `http://localhost:11434` | local Ollama endpoint |
| `ATOMICS_OLLAMA_MODEL` | `qwen2.5:7b` | default Ollama model |
| `ATOMICS_OLLAMA_TIMEOUT` | `300` | **per-request seconds** — raise for slow thinking models |
| `ATOMICS_VLLM_HOST` | `http://localhost:8000/v1` | vLLM / OpenAI-compatible gateway |
| `ATOMICS_VLLM_TIMEOUT` | `300` | per-request seconds for vLLM |
| `ATOMICS_BUDGET_LIMIT_USD` | `50.00` | hard cost cap per run |
| `ATOMICS_DB_PATH` | platform | SQLite location |

`.env` example:

```ini
ATOMICS_OLLAMA_HOST=http://localhost:11434
ATOMICS_OLLAMA_MODEL=qwen2.5:7b
ATOMICS_OLLAMA_TIMEOUT=600   # big reasoning models on hard prompts
```

---

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ReadTimeout` on a thinking model | Raise `ATOMICS_OLLAMA_TIMEOUT` (e.g. `600`). Thinking models can reason for minutes on HEAVY fixtures. |
| Quality scores look suspiciously high | You may be self-judging — use a different `--judge-model` than the model under test. |
| Judge scores ~50% / cannot reach `localhost:11434` | `eval` defaults the judge to local Ollama. Pass `--judge-provider` and `--judge-model`. |
| Compact table truncates the answer and rationale | `atomics eval --verbose` prints the full transcript. |
| `Unknown provider` | Install the extra: `uv sync --extra openai` / `--extra bedrock`. |
| Ollama host unreachable | `uv run atomics doctor` and check `ATOMICS_OLLAMA_HOST`. |
| Want a quick eval, not all 25 | `atomics eval --fixtures ev-01,ev-02`. |

---

## 8. Running the test suite

The `api` extra is required — the API and distributed test modules import
FastAPI at module scope, so without it pytest errors during collection. The
`mcp` extra is optional for collection (those tests skip without it) but CI
installs it so the MCP surface is actually tested.

```bash
uv sync --extra dev --extra api --extra mcp
uv run pytest -q                                   # full suite
uv run pytest --cov=atomics --cov-report=term-missing
```
