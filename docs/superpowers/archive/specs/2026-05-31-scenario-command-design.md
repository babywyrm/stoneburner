# Design: `atomics scenario` — Mixed-Workload Inference Simulation

**Date:** 2026-05-31
**Status:** Approved

## Problem

In production agentic architectures, multiple services share a single LLM
inference backend: admission controllers, security gates, code review agents,
content filters, evaluation pipelines. The existing `atomics stress` command
tests uniform load with identical long prompts. It cannot answer the critical
question: **when heterogeneous agentic workloads compete for the same GPU, how
does each workload's latency degrade?**

## Solution

A new `atomics scenario` command that runs multiple concurrent workload
profiles against a shared Ollama host, measures per-workload latency and SLA
compliance, and computes cross-workload interference scores.

## Architecture

### New Files

| File | Purpose |
|------|---------|
| `atomics/scenario.py` | Runner: orchestrates mixed concurrent workloads, computes interference |
| `atomics/scenario_prompts.py` | Built-in prompt fixtures for gate and eval archetypes |
| `atomics/scenario_models.py` | Dataclasses: `WorkloadSpec`, `WorkloadResult`, `ScenarioResult` |

### Modified Files

| File | Change |
|------|--------|
| `atomics/cli.py` | New `atomics scenario` command |
| `atomics/stress.py` | No changes needed — `_single_request` and `_single_request_provider` are already importable |

### Approach

Thin orchestrator (Approach C) that imports the low-level request primitives
from `stress.py` and adds its own prompt fixtures, results model, and
interference scoring. No duplication of HTTP/provider plumbing.

## Workload Archetypes

Two built-in types with distinct prompt characteristics:

### gate

Short structured prompts simulating gatekeeping decisions. Short input
(~100-200 tokens), very short output.

- Default `num_predict`: 32
- Typical use: admission controllers, deployment gates, content filters,
  approval workflows
- 6-8 built-in prompts, each asking the model to respond with a structured
  verdict (ALLOW/DENY, APPROVED/DENIED, SAFE/BLOCKED)

Example prompts:
- "You are an admission controller. Evaluate this pod spec and respond ALLOW
  or DENY with a one-line reason: {structured JSON}"
- "You are a deployment gate. Check if this hostname is on the allowlist and
  respond APPROVED or DENIED: {hostname}"
- "You are a content filter. Classify this message as SAFE or BLOCKED:
  {message}"

### eval

Medium-length prompts simulating evaluation/analysis tasks. Medium input
(~200-400 tokens), medium output.

- Default `num_predict`: 256
- Typical use: code review, compliance checking, log analysis, security
  assessment
- 6-8 built-in prompts, each asking the model to produce a structured
  analysis

Example prompts:
- "Review this code change for security issues. Summarize findings: {diff}"
- "Evaluate this configuration for compliance violations: {config snippet}"
- "Analyze this log entry for anomalous behavior: {log lines}"

### Custom prompts

Users can provide a custom prompts file (plain text, one prompt per `---`
separator) to test domain-specific workloads.

## Data Model

```python
@dataclass
class WorkloadSpec:
    name: str
    type: str                    # "gate" or "eval"
    model: str
    concurrency: int
    sla_ms: float | None = None  # optional latency SLA threshold
    num_predict: int = 32        # auto-set from type if not specified
    prompts: list[str]           # resolved from built-in or custom file

@dataclass
class WorkloadResult:
    spec: WorkloadSpec
    requests: int
    failed: int
    p50_ms: float
    p95_ms: float
    avg_tps: float
    sla_violations: int          # requests exceeding sla_ms
    sla_compliance_pct: float    # % of requests within SLA

@dataclass
class ScenarioResult:
    duration_seconds: float
    workloads: list[WorkloadResult]
    interference: dict[str, float]  # workload_name → degradation factor
```

## Interference Scoring

The unique value proposition of `scenario` over `stress`.

1. **Solo baseline phase** (automatic, ~5s per workload): run each workload
   alone to establish its isolated P50 latency.
2. **Mixed phase** (user-specified duration): run all workloads concurrently.
3. **Interference factor** = `mixed_p50 / solo_p50` per workload.
   - 1.0 = no interference
   - 2.5 = workload is 2.5x slower when competing
   - Displayed in a summary interference table

The baseline phase can be skipped with `--skip-baseline` for faster runs
(no interference score computed).

## CLI Interface

```bash
# ── YAML mode ──
atomics scenario --file scenario.yaml \
  --ollama-host http://your-gpu-host:11434 --duration 60

# ── CLI mode (repeatable --workload flag) ──
atomics scenario \
  --workload "gate:qwen2.5:3b:2:5000" \
  --workload "eval:qwen2.5:7b:1:15000" \
  --ollama-host http://your-gpu-host:11434 \
  --duration 60

# ── Minimal (single workload, quick test) ──
atomics scenario --workload "gate:qwen2.5:3b:3" --duration 30

# Workload format: type:model:concurrency[:sla_ms]
```

### Flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--file` | `-f` | — | YAML scenario file |
| `--workload` | `-w` | — | Repeatable CLI shorthand |
| `--ollama-host` | — | env/config | Target Ollama endpoint |
| `--duration` | `-d` | 60 | Total test duration in seconds |
| `--skip-baseline` | — | false | Skip solo baseline (no interference score) |
| `--save/--no-save` | — | save | Persist results to DB |

`--file` and `--workload` are mutually exclusive.

### YAML Format

```yaml
workloads:
  - name: admission-controller
    type: gate
    model: qwen2.5:3b
    concurrency: 2
    sla_ms: 5000

  - name: code-review
    type: eval
    model: qwen2.5:7b
    concurrency: 1
    sla_ms: 15000

  - name: content-filter
    type: gate
    model: qwen2.5:3b
    concurrency: 3
    sla_ms: 3000
```

## Output

Rich table per workload:

| Workload | Type | Model | Conc. | P50 | P95 | tok/s | SLA | Compliance |
|----------|------|-------|-------|-----|-----|-------|-----|------------|
| admission-controller | gate | qwen2.5:3b | 2 | 1.2s | 2.8s | 42.1 | 5000ms | 100% |
| code-review | eval | qwen2.5:7b | 1 | 8.3s | 12.1s | 31.2 | 15000ms | 95% |

Interference summary table (when baseline is run):

| Workload | Solo P50 | Mixed P50 | Interference |
|----------|----------|-----------|--------------|
| admission-controller | 0.8s | 1.2s | 1.5x |
| code-review | 6.1s | 8.3s | 1.4x |

## Runner Flow

1. Parse workload specs from YAML or CLI flags
2. Resolve prompts: built-in for gate/eval, or load from custom file
3. (Unless `--skip-baseline`) Run solo baseline: each workload alone for ~5s
4. Run mixed phase: all workloads concurrently for `--duration` seconds
   - One async worker pool per workload
   - Each pool has `concurrency` workers
   - Workers cycle through the workload's prompt list
   - Each request uses `stress._single_request` (Ollama raw) with the
     workload's model and `num_predict`
5. Compute per-workload stats (P50, P95, tok/s, SLA compliance)
6. Compute interference factors (mixed P50 / solo P50)
7. Print results and optionally save to DB

## Testing

- `tests/test_scenario.py` with mocked Ollama responses
- Test: single workload produces valid results
- Test: multiple workloads run concurrently
- Test: SLA violation counting is correct
- Test: interference factor computation
- Test: YAML parsing and CLI flag parsing
- Test: custom prompts file loading

## Scope Boundaries

- No model accuracy/quality scoring (use `atomics eval` for that)
- No raw throughput ramp-to-saturation (use `atomics stress` for that)
- No adversarial/security testing (use `atomics adversarial` for that)
- Ollama-only for v1 (provider abstraction can be added later if needed)
- Two archetypes for v1 (chat/stream can be added later)
