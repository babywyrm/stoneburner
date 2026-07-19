# Security Evaluation Suites

Stoneburner includes purpose-built security testing suites that complement infrastructure scanning tools (like `mcpnuke`) by focusing on *LLM reasoning quality and resilience*.

> **Two independent axes — score them separately.** `redblue` measures **capability** (can the model *do* security work) and `adversarial` measures **resilience** (does the model *resist* being manipulated). These do not correlate: in live runs a non-thinking 12B scored 93% capability / 76% resilience, while a *thinking* 2B scored 54% / 91%. The riskiest profile is **high capability + low resilience** — argues for a guardrail layer in front of the model rather than trusting its own refusals. Run both suites and weigh them together.

## Fixture Coverage (72 adversarial + eval suites)

**Categories:** prompt injection, role confusion, context escape, instruction override, social engineering, data exfil, tool manipulation, breakglass extraction, credential exfil, behavior persistence, blocklist evasion, MCP/agentic resistance, tool-use safety, encoding obfuscation, many-shot context stuffing, multi-turn manipulation, RAG poisoning, tool-description injection.

## `atomics adversarial` — Resilience Eval

> **Leaderboard:** [`docs/LEADERBOARD.md`](LEADERBOARD.md)

Tests whether a model resists adversarial manipulation. Uses inverted scoring — higher = better resistance.

```bash
# All 72 fixtures, judge with local Ollama
uv run atomics adversarial --provider ollama -m qwen3:14b --judge-model qwen2.5:14b

# Target specific categories
uv run atomics adversarial --provider claude --category prompt_injection,role_confusion

# With thinking enabled
uv run atomics adversarial --provider ollama -m qwen3:14b --thinking

# Variance-aware scoring
uv run atomics adversarial --runs 3

# Compare two models
uv run atomics adversarial --compare mistral-small:24b

# CI gate
uv run atomics adversarial --fail-on-resilience 60
```

**Categories:** `prompt_injection` · `role_confusion` · `context_escape` · `instruction_override` · `social_engineering` · `data_exfil_attempt` · `multiturn` · `rag_poisoning` · `tool_desc_injection` · `encoding` · `many_shot` · `zerotrust` · `agentic`

## `atomics redblue` — Offensive/Defensive Capability Eval

Benchmarks LLM performance on real security domain tasks — OSINT, vulnerability analysis, privilege escalation, incident response, hardening, threat modelling, and detection engineering.

```bash
# All 10 fixtures (5 red + 5 blue)
uv run atomics redblue --provider ollama -m qwen3:14b

# Red team only / Blue team only
uv run atomics redblue --provider claude --mode red
uv run atomics redblue --provider openai -m gpt-4o --mode blue

# Persist results + variance scoring
uv run atomics redblue --provider ollama -m qwen3:14b --save --runs 3
```

## `atomics refusal` — Refusal Calibration

Measures **both** safety failure modes: over-refusal (blocking legitimate security work) and under-refusal (complying with harmful requests). Reports `over_refusal_rate`, `under_refusal_rate`, and `calibration_score`.

```bash
uv run atomics refusal -p ollama -m qwen3:14b \
  --judge-model qwen2.5:14b --json-out refusal.json
```

## `atomics codereview` — Secure Code Review

Tests vulnerability detection on code snippets and unified diffs. Vulnerable fixtures carry known CWEs (SQLi, command injection, path traversal, etc.); clean fixtures measure false positives. Reports `detection_rate`, `false_positive_rate`, and `review_score`.

```bash
uv run atomics codereview -p ollama -m qwen3:14b \
  --judge-model qwen2.5:14b --json-out codereview.json
```

## `atomics probe` — Live Ecosystem Probe

Fetches real artifacts from infrastructure (logs, API responses, scan reports) and uses an LLM to analyse them. Targets defined in `probes.yaml`.

```bash
uv run atomics probe --probes-file /path/to/probes.yaml
uv run atomics probe --artifact access-log --file /var/log/nginx/access.log
uv run atomics probe --probes-file probes.yaml --alert-on-regression
```

<details>
<summary>probes.yaml example</summary>

```yaml
targets:
  - name: nginx-access-logs
    artifact_type: access-log
    source: file
    path: /var/log/nginx/access.log

  - name: ollama-api
    artifact_type: inference-api
    source: http
    url: http://ollama-host:11434/api/tags

  - name: k8s-cluster-audit
    artifact_type: k8s-audit-log
    source: file
    path: /var/log/kubernetes/audit.log
```

**Supported artifact types:** `json-security-report` · `inference-api` · `access-log` · `k8s-audit-log` · `config-file` · `api-response`
</details>

## `atomics archreview` — Security Architecture Benchmark

Benchmarks how well models reason about the security architecture of a whole repository. The model receives a deterministic evidence pack and emits structured findings, scored two ways:

- **Objective score** — difficulty-weighted OWASP-category recall and precision against a per-repo answer key. Deterministic.
- **Reasoning score** — a separate judge rates architectural reasoning quality 0–10. Self-judge-guarded.

**Tiered context:** `floor` (16k) · `local` (32k) · `wide` (48k) · `expanded` (128k) — packs are byte-identical and content-hashed for reproducibility.

```bash
JUICE_SHOP_PATH=~/juice-shop atomics archreview --repo juice-shop \
  --models "qwen2.5:14b,qwen3.5:4b" --provider ollama \
  --judge-provider claude --judge-model claude-opus-4-7 \
  --tier floor --rounds 3
```

Answer keys are pluggable per repo (`atomics/archreview/repos/<name>.yaml`). See the README's archreview section for parser tolerance, taxonomy, and tier details.


## `atomics rag` — RAG Pipeline Evaluation

Tests how well models use retrieved context: grounding (does it reference the docs?), faithfulness (does it stay within what the docs say?), and abstention (does it correctly decline when the answer isn't in the context?).

20 fixtures: 10 security (CVE retrieval, incident analysis, threat intel, SBOM, alert triage) + 10 general technical (API docs, ADRs, K8s runbooks, capacity planning).

```bash
uv run atomics rag --provider ollama -m qwen3:14b --judge-model qwen2.5:14b
uv run atomics rag --fixtures rag-05,rag-12
uv run atomics rag --json-out rag.json
```

Metrics: `grounding_score`, `faithfulness_score`, `abstention_accuracy`, `hallucination_rate`, `overall_rag_score`.

### Real retrieval mode

You can also evaluate RAG against your own indexed corpus. Real retrieval requires the optional extra:

```bash
uv pip install "atomics[rag]"
uv run atomics rag-index ./docs --db ./my-index.vec
uv run atomics rag --index ./my-index.vec --provider ollama -m qwen3:14b
uv run atomics rag-retrieval --index ./my-index.vec --gold relevance.json
```

## `atomics multiturn` — Multi-Turn Conversation Evaluation

Tests context retention, coherence, and instruction following across scripted multi-turn conversations. Now includes 35 fixtures covering context retention, instruction following, contradiction detection, persona drift/stability, long-context retention (8+ turns), simulated multi-turn tool-use chaining, and security-focused scenarios (social engineering refusal, least-privilege access control, and credential non-echo).

```bash
uv run atomics multiturn --provider ollama -m qwen3:14b --judge-model qwen2.5:14b
uv run atomics multiturn --fixtures mt-eval-33,mt-eval-34,mt-eval-35
uv run atomics multiturn --json-out multiturn.json
```

Metrics: per-turn accuracy/context-use/coherence, conversation-level retention/consistency/instruction, and overall score.

## `atomics sweep` — Multi-Model Comparison

Sweep the eval fixture set across multiple models and produce a ranked comparison table.

```bash
uv run atomics sweep --all-local --host http://gpu-host:11434
uv run atomics sweep --models qwen2.5:1.5b,qwen2.5:3b,mistral:7b
uv run atomics sweep --provider claude --models claude-sonnet-4-6,claude-haiku-4-5-20251001
```

## Integrity & Persistence

`refusal` and `codereview` report typed run integrity and save each fixture immediately. Partial or infrastructure-invalid runs still write JSON and finalize stored results, then exit nonzero. Use `--allow-partial` when incomplete coverage is acceptable.

Saved fixture rows live in the schema-v20 `evaluation_results` ledger. Its `result_json` contains raw model and judge evidence — treat the database and JSON exports as potentially sensitive.
