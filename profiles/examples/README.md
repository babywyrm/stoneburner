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
# Scenario: a gate and an eval on one box
atomics scenario --file profiles/examples/scenario-gate-and-eval.yaml -d 60

# Soak test against a profile
atomics soak --profile profiles/local/my-gate.yaml -d 30m

# Stress test (ramp concurrency)
atomics stress --profile profiles/local/my-gate.yaml
```

## YAML Format

See `scenario-gate-and-eval.yaml` for two workloads on one box,
`ollama-with-system-prompt.yaml` for a soak profile, and
`http-flask-endpoint.yaml` for an HTTP gate.

### Template Variables

Body templates support `{{ prompt }}`, `{{ model }}`, and `{{ num_predict }}`
via simple string replacement (no Jinja dependency).

### Classification

When `classify` is defined, results include a breakdown (e.g., "412 APPROVED,
3 BLOCKED, 1 ERROR") alongside standard throughput/latency metrics. This detects
model drift under load — a gate that starts approving dangerous requests after
30 minutes of sustained traffic.
