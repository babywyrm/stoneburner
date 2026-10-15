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

### Model temperature

Judges always run at 0. The model under test runs at its own temperature
unless you pass `--temperature` (on `qa`, `toolcall`, `adversarial`,
`redblue`, `refusal`, `codereview`, and `battery`).

- **Comparing models or builds:** `--temperature 0`, so a fixture that
  flips between runs is not read as a difference between models.
- **Safety and promotion evidence:** leave it off and use `--runs 3` or
  more. Pinning can hide how often a model makes a dangerous call.
- Each suite's JSON records `temperature`; `null` means the model's own.
