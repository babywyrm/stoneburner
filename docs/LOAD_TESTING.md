# Load Testing

Tools for measuring inference throughput, stability, and capacity under load.

## `atomics stress` — GPU Saturation Testing

Ramp concurrent requests from 1 to N against an Ollama host to find the throughput saturation point. Reports per-phase TPS, latency percentiles, VRAM usage, and throttling detection.

```bash
uv run atomics stress --model qwen2.5:7b --max-concurrency 8
uv run atomics stress --ollama-host http://gpu:11434 -c 16 -s 30
uv run atomics stress --profile profiles/local/gatekeeper.yaml
uv run atomics stress --no-save
```

### Multi-Model VRAM Contention

Run two or more models simultaneously to measure how shared GPU memory affects each one:

```bash
uv run atomics stress --models qwen2.5:3b,qwen2.5:7b --ollama-host http://gpu:11434
# Reports contention factor per model (<1.0 = degraded by sharing)
# Color coded: green ≥0.9x  yellow ≥0.7x  red <0.7x
```

## `atomics soak` — Long-Duration Stability Test

Hold fixed concurrency for minutes or hours. Samples throughput and latency at regular intervals and computes linear-regression drift to classify the run as **STABLE**, **DEGRADED**, or **UNSTABLE**. Detects slow VRAM leaks, thermal throttling, and gradual latency creep.

```bash
uv run atomics soak --model qwen2.5:7b --duration 30m
uv run atomics soak --model qwen2.5:7b -d 2h -c 8 --ollama-host http://gpu:11434
uv run atomics soak --model qwen2.5:7b -d 30m -c 4 --think-time 5
uv run atomics soak --provider openai --model gpt-4o-mini -d 15m -c 2
uv run atomics soak --profile profiles/local/gatekeeper.yaml -d 30m
```

### Baseline Regression

```bash
# Save a named baseline
uv run atomics soak --model qwen2.5:3b -d 30m --save-baseline gpu-host-3b

# Compare against it later
uv run atomics soak --model qwen2.5:3b -d 30m --compare-baseline gpu-host-3b

# List all saved baselines
uv run atomics baselines
```

### Verdict Thresholds

| Metric | STABLE | DEGRADED | UNSTABLE |
|--------|--------|----------|----------|
| Throughput drift | > -5% | -5% to -15% | ≤ -15% |
| Latency drift | < +10% | +10% to +25% | ≥ +25% |
| Error rate | < 0.5% | 0.5% to 5% | ≥ 5% |

## `atomics scenario` — Mixed-Workload Simulation

Simulate multiple agentic services competing for one GPU. Runs heterogeneous workload profiles concurrently, measures per-workload latency and SLA compliance, and computes interference scores.

Built-in archetypes: **gate** (~32 output tokens) and **eval** (~256 output tokens). Custom prompt files supported.

```bash
uv run atomics scenario -w "gate:qwen2.5:3b:2:5000" -w "eval:qwen2.5:7b:1:15000" -d 60
uv run atomics scenario --file scenario.yaml --ollama-host http://gpu-host:11434
uv run atomics scenario -w "gate:qwen2.5:3b:4" -d 60 --ramp 10
uv run atomics scenario --skip-baseline
```

## `atomics capacity` — User Load Projector

Projects how many users your setup can handle using queueing theory and real stress test data. No live requests needed.

```bash
uv run atomics capacity --users 200 --model qwen2.5:7b
uv run atomics capacity --users 100 --peak-tps 50 --single-latency 3000
uv run atomics capacity --users 200 --think-time 600 --model qwen2.5:7b --burst 0.3
```

## `atomics labcompare` — Two-Host Bench-off

Compare two lab boxes on the same models — throughput and quality side by side.

```bash
uv run atomics labcompare \
  --host host-a=http://gpu-a:11434 \
  --host host-b=http://gpu-b:11434 \
  --models qwen2.5:7b,qwen3:14b,qwen3.6:27b \
  --judge-host http://gpu-b:11434 --judge-model qwen3.6:35b-a3b
```

- **Throughput:** single-stream tok/s, latency, prompt-eval rate, VRAM fit from `/api/ps`.
- **Quality parity:** same fixtures, one judge, so identical weights should score identically.
- Run one dimension alone with `--dimensions throughput` or `--dimensions quality`.
