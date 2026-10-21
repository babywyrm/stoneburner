# Provider Comparison

Run benchmarks with multiple providers, then compare them side-by-side.

## Quick Start

```bash
uv run atomics run --provider claude --tier ez -n 3 -i 5
uv run atomics run --provider bedrock --tier ez -n 3 -i 5
uv run atomics run --provider openai --tier ez -n 3 -i 5
uv run atomics run --provider ollama --tier ez -n 3 -i 5

# Compare by provider (model(s), class, tok/s, P50/P95 latency, $/1K tokens)
uv run atomics compare

# Compare by individual model
uv run atomics compare --by model

# Filter by time window or tier
uv run atomics compare --since-hours 24 --tier ez
```

## Model Classes

Models are tagged by class (light/mid/heavy) so you can spot apples-to-oranges comparisons. A warning is shown when mixed classes are detected.

| Class | Claude | OpenAI | Bedrock | Ollama |
|-------|--------|--------|---------|--------|
| **light** | Haiku 4.5 | gpt-4o-mini, gpt-4.1-nano | Haiku on Bedrock | qwen2.5:1.5b, qwen3:1.7b |
| **mid** | Sonnet 4.6 | gpt-4o, gpt-4.1, o4-mini | Sonnet on Bedrock | qwen2.5:7b, llama3.2:3b |
| **heavy** | Opus 4.6 | o3 | Opus on Bedrock | — |

## Metrics & Fidelity

Stoneburner reports only what a provider can actually observe, so cross-model comparisons stay honest:

- **Cost** — token usage × per-model pricing. For Claude, prompt-caching is priced correctly: cache reads at 0.10× and writes at 1.25× the base input rate.
- **Thinking tokens** — populated only when the provider truly reports a count (OpenAI `reasoning_tokens`, Claude `thinking_tokens`). For Ollama/vLLM, a character-proportional estimate anchored to the real output-token total.
- **Throughput (`tokens_per_second`)** — total output tokens ÷ elapsed time. The `tps_basis` field labels `wall_clock` vs `generation` (Ollama decode time). Compare tok/s across providers with the basis in mind.

## Judge Accuracy

Quality scores come from an LLM-as-judge (`atomics eval` / `redblue`), defaulting to a local Ollama model ($0). Key properties:

- **No self-judging** — warns when judge matches the model under test
- **Deterministic** — `temperature=0.0` for all judge calls
- **Fair completeness** — response truncation scales to fixture's expected output length
- **Gold-criteria coverage** — lexical coverage measure independent of the judge
- **Multi-judge consensus** — `--extra-judges provider:model[@host],…` for panel scoring
- **Calibration guard** — regression test ensures judge ranks wrong → thin → thorough monotonically

```bash
# Validate the configured Ollama judge ranks answers correctly
ATOMICS_LIVE_JUDGE=1 uv run pytest tests/test_calibration.py::test_live_judge_is_calibrated -q
```

All judge fields persist to `task_results` and surface in `atomics compare`.

### Which judge

A judge can rank models correctly and still score all of them high. A blind
human grade of the red/blue answers the local judges disagreed on most
(2026-10-03) measured mean absolute error against the grade: `gemma4:26b`
0.09, `qwen3.8:27b` 0.18, `granite4.2:8b` 0.22. Granite scored about 0.22
high on every item, yet all three ranked the models the same way.

- **Rankings** from any of the three hold.
- **Absolute numbers** need the reference judge, `gemma4:26b`, or a cloud
  judge. Re-check the reference when a new local judge appears.
- Name the judge next to every published score.

### Calibrate a judge

`atomics judge-calibrate` scores a judge against a human-graded set on the
judge's own rubric and reports mean error, bias (judge minus human, so
positive means lenient), and pair order. It passes when mean error is at most
`--max-error` (default 0.15).

```bash
uv run atomics judge-calibrate --judge-model gemma4:26b
uv run atomics judge-calibrate --judge-provider claude --judge-model claude-sonnet-4-6 --budget 1
uv run atomics judge-calibrate --judge-model gemma4:26b --set qa/local/my-graded-set.yaml
```

The shipped set is the six blue answers from the 2026-10-03 grade. The red
answers are model output on offensive tasks and stay out of the repo; keep
sets like that in gitignored `qa/local/`, in the same YAML format
(`atomics/eval/calibration_sets/redblue-blue.yaml`). Replayed from the
study's stored scores, the blue six give the following. Live runs on
2026-10-20 matched to the item:

| Judge | Mean error | Bias | Default verdict |
|---|---|---|---|
| `gemma4:26b` | 0.133 | −0.067 | pass |
| `granite4.2:8b` | 0.200 | +0.200 | fail |
| `qwen3.8:27b` | 0.233 | −0.133 | fail |

Six items is small: each is a sixth of the mean, so treat a result near the
threshold as a tie.

A complete run is stored, keyed by judge provider and model. Every judged
suite (eval, adversarial, redblue, refusal, codereview, probe, multiturn) then
prints the judge's latest result under its header, and API eval and battery
results carry it as `judge_calibration`:

```
Judge calibration: calibrated · error 0.13 · bias -0.07 · redblue-blue-2026-10-03 · 2026-10-20
```

The label is advisory. It never changes a score, integrity, or exit code. A
judge with no stored run reads `uncalibrated`. A run with any unscored item
exits nonzero and is not stored.

### Model temperature

Judges always run at 0. The model under test runs at its own temperature
unless you pass `--temperature` (on `qa`, `toolcall`, `adversarial`,
`redblue`, `refusal`, `codereview`, `eval`, `battery`, `sweep`, and
`labcompare`).

- **Comparing models or builds:** `--temperature 0`, so a fixture that
  flips between runs is not read as a difference between models.
- **Safety and promotion evidence:** leave it off and use `--runs 3` or
  more. Pinning can hide how often a model makes a dangerous call.
- Each suite's JSON records `temperature`; `null` means the model's own.
