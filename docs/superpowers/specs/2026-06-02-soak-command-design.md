# atomics soak — Long-Duration Stability Test

**Date:** 2026-06-02
**Status:** Approved
**Schema bump:** v9 → v10

---

## Problem

`atomics stress` ramps concurrency in short phases (~15s each) to find peak throughput and saturation points. This is useful for capacity sizing but cannot detect:

- Throughput degradation over hours (tok/s dropping as VRAM fragments or model state drifts)
- Latency creep (P95 climbing as Ollama's KV cache fills or GC pauses accumulate)
- VRAM leaks (memory usage growing steadily without release)
- Model eviction under sustained load (Ollama unloading/reloading when `OLLAMA_MAX_LOADED_MODELS` is exceeded)
- Intermittent failures that only appear after hundreds of requests

A soak test holds **fixed concurrency for an extended period** (minutes to hours) and tracks these signals via periodic metric sampling.

---

## Solution

New CLI command: `atomics soak`

### CLI Interface

```
atomics soak \
  --model qwen2.5:7b \
  --concurrency 4 \
  --duration 60m \
  --ollama-host http://gpu-host:11434 \
  --sample-interval 30 \
  --save
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--provider` | `-p` | choice | `ollama` | Any supported provider |
| `--model` | `-m` | str | env default | Model to soak |
| `--concurrency` | `-c` | int | 4 | Fixed concurrent workers (no ramp) |
| `--duration` | `-d` | str | `30m` | Human-friendly: `15m`, `2h`, `1h30m` |
| `--ollama-host` | | str | env | Ollama endpoint |
| `--sample-interval` | `-s` | int | 30 | Seconds between metric snapshots |
| `--num-predict` | | int | 2048 | Max output tokens per request |
| `--save/--no-save` | | flag | save | Persist results to DB |

### Duration Parsing

Accept human-friendly duration strings with minutes as the base unit:

- `30m` → 1800s
- `2h` → 7200s
- `1h30m` → 5400s
- `90` → 5400s (bare number treated as minutes)

Implementation: simple regex parser, no external dependency.

---

## Architecture

### Module: `atomics/soak.py`

New module, parallel to `stress.py` and `scenario.py`.

#### Data classes

```python
@dataclass
class SoakSample:
    """One metric snapshot taken every sample_interval seconds."""
    elapsed_seconds: float
    requests: int
    failed: int
    total_output_tokens: int
    aggregate_tps: float
    avg_latency_ms: float
    p95_latency_ms: float
    vram_used_mb: float | None

@dataclass
class SoakResult:
    model: str
    host: str
    provider: str
    concurrency: int
    duration_seconds: float
    actual_duration_seconds: float
    sample_interval: int
    total_requests: int
    total_failed: int
    total_tokens: int
    samples: list[SoakSample]

    # Drift metrics (computed from samples via linear regression)
    throughput_drift_pct: float      # negative = degrading
    latency_drift_pct: float         # positive = degrading
    vram_start_mb: float | None
    vram_end_mb: float | None
    vram_drift_mb: float | None

    # Summary
    avg_tps: float
    peak_tps: float
    min_tps: float
    avg_p95_ms: float
    error_rate: float                # total_failed / total_requests
    verdict: str                     # STABLE / DEGRADED / UNSTABLE
    total_cost_usd: float
```

#### Core function

```python
async def run_soak(
    host: str,
    model: str,
    concurrency: int = 4,
    duration_seconds: float = 1800,
    sample_interval: int = 30,
    num_predict: int = 2048,
    on_sample: Callable[[SoakSample], None] | None = None,
) -> SoakResult:
```

There is also a `run_soak_provider` variant for non-Ollama providers, following the same pattern as `run_stress` / `run_stress_provider`.

#### Algorithm

1. Start `concurrency` async workers, each looping: pick prompt from `STRESS_PROMPTS`, call `_single_request`, record result.
2. Every `sample_interval` seconds, snapshot the metrics accumulated since the last sample into a `SoakSample`. Reset per-window accumulators.
3. After `duration_seconds` total, stop all workers.
4. Compute drift via least-squares linear regression on the time-series:
   - `throughput_drift_pct = (slope * duration) / first_sample_tps * 100`
   - `latency_drift_pct = (slope * duration) / first_sample_p95 * 100`
5. Compute verdict from thresholds.

#### Verdict thresholds

| Verdict | Throughput drift | Latency drift | Error rate |
|---------|-----------------|---------------|------------|
| STABLE | < 5% drop | < 10% rise | < 0.5% |
| DEGRADED | 5-15% drop OR | 10-25% rise OR | 0.5-5% |
| UNSTABLE | > 15% drop OR | > 25% rise OR | > 5% |

Worst-case across all three dimensions wins.

#### Drift computation

Simple linear regression (numpy not required — pure Python least-squares):

```python
def _linear_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    return num / den if den != 0 else 0.0
```

---

## Live Output

One line per sample window, overwriting-style with Rich:

```
[00:00:30] 42.1 tok/s  P50 1.2s  P95 2.8s  0 fails  VRAM 8.2/12GB
[00:01:00] 41.8 tok/s  P50 1.3s  P95 2.9s  0 fails  VRAM 8.2/12GB
[00:01:30] 40.2 tok/s  P50 1.4s  P95 3.1s  0 fails  VRAM 8.4/12GB
```

Timestamps formatted as `HH:MM:SS` elapsed.

---

## Final Summary

Rich table after completion:

```
Soak Test Summary
  Model:             qwen2.5:7b
  Provider:          ollama
  Target:            http://gpu-host:11434
  Duration:          60m (actual: 59m 58s)
  Concurrency:       4
  Total requests:    847 (2 failed)
  Total tokens:      412,000
  Throughput:        42.1 → 40.8 tok/s (drift: -3.2%)
  Latency (P95):     2.8s → 3.0s (drift: +8.1%)
  VRAM:              8.2GB → 8.4GB (drift: +200MB)
  Error rate:        0.24%
  Verdict:           STABLE
```

---

## Database Persistence

### New table: `soak_results`

```sql
CREATE TABLE IF NOT EXISTS soak_results (
    result_id    TEXT PRIMARY KEY,
    model        TEXT NOT NULL,
    host         TEXT NOT NULL,
    provider     TEXT NOT NULL DEFAULT 'ollama',
    concurrency  INTEGER NOT NULL,
    duration_seconds    REAL NOT NULL,
    actual_duration_seconds REAL NOT NULL,
    sample_interval     INTEGER NOT NULL,
    total_requests      INTEGER NOT NULL,
    total_failed        INTEGER NOT NULL,
    total_tokens        INTEGER NOT NULL,
    avg_tps             REAL NOT NULL,
    peak_tps            REAL NOT NULL,
    min_tps             REAL NOT NULL,
    throughput_drift_pct REAL NOT NULL,
    latency_drift_pct   REAL NOT NULL,
    avg_p95_ms          REAL NOT NULL,
    vram_start_mb       REAL,
    vram_end_mb         REAL,
    vram_drift_mb       REAL,
    error_rate          REAL NOT NULL,
    verdict             TEXT NOT NULL,
    total_cost_usd      REAL NOT NULL DEFAULT 0.0,
    samples_json        TEXT NOT NULL,
    timestamp           TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_soak_model ON soak_results(model);
CREATE INDEX IF NOT EXISTS idx_soak_verdict ON soak_results(verdict);
```

Schema version: 9 → 10.

### Repository methods

- `save_soak_result(sr: SoakResult) -> None`
- `get_soak_results(*, model: str | None = None) -> list[dict]`

### Export support

Add `soak` to `atomics export --suite` choices (tasks, stress, sweep, scenario, soak, all).

---

## Reused infrastructure

| Component | Source | How used |
|-----------|--------|----------|
| `_single_request` | `stress.py` | Ollama HTTP call |
| `_single_request_provider` | `stress.py` | Cloud provider call |
| `_get_gpu_info` / `_get_vram_used_mb` | `stress.py` | VRAM sampling |
| `STRESS_PROMPTS` | `stress.py` | Prompt rotation |
| `_make_provider` | `cli.py` | Provider factory |
| `_percentile` | `stress.py` | P50/P95 within each window |

No new dependencies. No changes to existing modules beyond:
- `storage/schema.py`: bump version, add table
- `storage/repository.py`: add save/get methods
- `cli.py`: add `soak` command

---

## Testing

### Unit tests: `tests/test_soak.py`

Minimum test coverage:

1. **Duration parsing:** `15m` → 900, `2h` → 7200, `1h30m` → 5400, `90` → 5400, invalid → error
2. **SoakSample defaults:** all fields initialize correctly
3. **SoakResult properties:** verdict computation from drift values
4. **Linear regression:** known slope cases (flat, positive, negative)
5. **Verdict thresholds:** STABLE/DEGRADED/UNSTABLE boundary conditions
6. **Runner (mocked):** single-sample, multi-sample, failure injection
7. **Drift calculation:** stable series → ~0% drift, degrading series → negative throughput drift
8. **VRAM tracking:** present when nvidia-smi available, None otherwise
9. **CLI integration:** `--help` renders, `--duration` parsing, mutual exclusion
10. **DB persistence:** save and retrieve soak results, samples_json round-trip

Target: 25+ tests.

### Regression gate

Full existing suite must pass (`uv run pytest tests/ -v`) before and after.

---

## Non-goals

- No real-time graphing (future work — could pipe samples to a live terminal chart)
- No automatic remediation (just reports the verdict)
- No multi-model soak in a single run (use `scenario` for that; soak is single-model endurance)
- No warm-up phase (start measuring immediately; if the user wants warm-up data, the first few samples show it)
