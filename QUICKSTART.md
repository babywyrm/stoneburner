# Stoneburner Quickstart

Recipe-first guide to **`atomics`** — the agentic token-usage benchmarking and
LLM-evaluation harness. Every block below is copy‑pasteable. For the full
reference see [`README.md`](README.md).

> **Mental model:** point `atomics` at a provider (cloud API or a local
> Ollama/vLLM box), pick a *goal* (cost, quality, safety, scale), run the
> matching command, then `compare`/`report`/`export` the results out of SQLite.

---

## 1. 60‑second setup

```bash
# 1. Install (uv manages the venv)
uv sync

# 2. Pick a backend — set whichever you'll use
export ANTHROPIC_API_KEY=sk-ant-...        # Claude (default provider)
# Or store securely in the OS keychain (no plaintext file needed):
# atomics secrets set ANTHROPIC_API_KEY
export OPENAI_API_KEY=sk-...               # OpenAI / o-series
export ATOMICS_OLLAMA_HOST=http://gpu:11434  # local Ollama (zero cost)

# 3. Pre-flight: confirms keys, hosts, DB are all wired
uv run atomics doctor

# 4. Smoke test the provider you plan to use
uv run atomics provider-test -p ollama -m qwen2.5:7b
```

`doctor` exits non‑zero if anything is missing, so it's safe in front of a long
run: `uv run atomics doctor && uv run atomics run --tier ez -n 3`.

---

## 2. Choose a backend — local **and** cloud are first-class

`atomics` treats local inference and cloud APIs as peers. Every suite
(`run`, `eval`, `sweep`, `adversarial`, `redblue`, `probe`) takes the same
`--provider` flag, so you can benchmark a private on-prem model and a frontier
cloud model with identical commands and compare them side-by-side.

### Local / self-hosted (private, $0, nothing leaves the LAN)

| Backend | Flag | When to use |
|---------|------|-------------|
| **Ollama** | `--provider ollama` | Default for eval/security suites — free, private GPU box |
| **vLLM / OpenAI-compatible gateway** | `--provider vllm` | LiteLLM, vLLM, TGI, or any `/v1/chat/completions` endpoint on the LAN |
| **brain-gateway** (camazotz) | `--provider brain-gateway` | Internal agentic gateway in the ecosystem |

```bash
# Ollama on a GPU box — list models (class + thinking annotations)
uv run atomics models --host http://gpu:11434
uv run atomics run --provider ollama -m qwen2.5:7b --ollama-host http://gpu:11434 -n 5 -i 0

# vLLM / OpenAI-compatible gateway (e.g. a LiteLLM gateway at :8000/v1)
uv run atomics run --provider vllm --vllm-host http://gpu:8000/v1 -m qwen2.5:7b -n 5 -i 0

# Internal brain-gateway
uv run atomics run --provider brain-gateway --gateway-url http://nuc:30080 -n 5 -i 0
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
| **OpenAI** / o-series | `--provider openai` | `uv sync --extra openai` |
| **Bedrock** (AWS) | `--provider bedrock --region us-east-1` | `uv sync --extra bedrock` |

```bash
uv run atomics run --provider claude -n 5 -i 0
uv run atomics run --provider openai -m gpt-4o -n 5 -i 0
uv run atomics run --provider bedrock --region us-east-1 -n 5 -i 0
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
uv run atomics sweep --all-local --host http://gpu:11434

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
uv run atomics stress --model qwen2.5:7b --max-concurrency 8 --ollama-host http://gpu:11434

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
                  --model qwen2.5:3b --ollama-host http://gpu:11434

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
uv run atomics redblue -p ollama -m qwen3.5:4b --runs 3 --json-out redblue.json

# Security architecture review
uv run atomics archreview -p ollama -m qwen2.5:7b --pack camazotz
```

Results show per-fixture resistance verdicts + total cost (if using a paid judge).
See [`docs/LEADERBOARD.md`](docs/LEADERBOARD.md) and
[`docs/ADVERSARIAL_SUITES.md`](docs/ADVERSARIAL_SUITES.md) for the full 64-fixture
suite breakdown and benchmark results.

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
ATOMICS_OLLAMA_HOST=http://gpu-host:11434
ATOMICS_OLLAMA_MODEL=qwen2.5:7b
ATOMICS_OLLAMA_TIMEOUT=600   # big reasoning models on hard prompts
```

---

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ReadTimeout` on a thinking model | Raise `ATOMICS_OLLAMA_TIMEOUT` (e.g. `600`). Thinking models can reason for minutes on HEAVY fixtures. |
| Quality scores look suspiciously high | You may be self-judging — use a different `--judge-model` than the model under test. |
| `Unknown provider` | Install the extra: `uv sync --extra openai` / `--extra bedrock`. |
| Ollama host unreachable | `uv run atomics doctor` and check `ATOMICS_OLLAMA_HOST`. |
| Want a quick eval, not all 25 | `atomics eval --fixtures ev-01,ev-02`. |

---

## 8. Running the test suite

```bash
uv sync --extra dev
uv run pytest -q                                   # full suite
uv run pytest --cov=atomics --cov-report=term-missing
```
