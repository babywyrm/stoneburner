# Target Profiles

Target profiles let you soak/stress/scenario-test application-level AI gates
and Ollama endpoints with custom system prompts.

## Two Modes

| Mode     | What it hits | Use case |
|----------|-------------|----------|
| `ollama` | Ollama `/api/generate` with system prompt, temperature, num_predict | Test model behavior under your app's exact prompt |
| `http`   | Any HTTP endpoint (Flask, Spring, MCP JSON-RPC) | Test the full app stack end-to-end |

## Directory Layout

```
profiles/
  examples/           # sanitized examples (committed)
  local/              # your real profiles (gitignored)
```

Put real profiles with IPs, API keys, and spoiler system prompts in `local/`.

## Usage

```bash
# Soak test against a profile
atomics soak --profile profiles/local/my-gate.yaml -d 30m

# Stress test (ramp concurrency)
atomics stress --profile profiles/local/my-gate.yaml

# Scenario with profile-based workloads (add profile: field to workload YAML)
atomics scenario --file scenario.yaml
```

## YAML Format

See `ollama-with-system-prompt.yaml` and `http-flask-endpoint.yaml` for
annotated examples.

### Template Variables

Body templates support `{{ prompt }}`, `{{ model }}`, and `{{ num_predict }}`
via simple string replacement (no Jinja dependency).

### Classification

When `classify` is defined, results include a breakdown (e.g., "412 APPROVED,
3 BLOCKED, 1 ERROR") alongside standard throughput/latency metrics. This detects
model drift under load — a gate that starts approving dangerous requests after
30 minutes of sustained traffic.
