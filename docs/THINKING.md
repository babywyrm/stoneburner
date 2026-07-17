# Thinking Mode

Stoneburner auto-detects models with thinking/reasoning capabilities and handles them transparently. Thinking tokens are tracked separately from visible output so benchmarks measure what users actually see.

## Usage

```bash
# Auto-detect: qwen3 models enable thinking automatically
uv run atomics run --provider ollama -m qwen3:14b -n 5

# Explicit control
uv run atomics run --provider claude -m claude-sonnet-4-6 --thinking -n 5
uv run atomics run --provider openai -m o3 --no-thinking -n 5

# Custom thinking budget (Claude)
uv run atomics run --provider claude --thinking --thinking-budget 20000 -n 5

# Provider test shows thinking token breakdown
uv run atomics provider-test -p ollama -m qwen3:14b --thinking
```

## Supported Models

| Provider | Models | Mechanism |
|----------|--------|-----------|
| **Claude** | Opus 4.x, Sonnet 4.x | Extended thinking API (`budget_tokens`) |
| **OpenAI** | o3, o3-mini, o3-pro, o4-mini, gpt-5.x | Reasoning tokens (`completion_tokens_details`) |
| **Ollama** | qwen3 family | `<think>` tag parsing, auto-stripped from output |

When `--thinking` / `--no-thinking` is omitted, stoneburner checks the model against its capability registry and enables thinking automatically for known models. Use `--no-thinking` to force it off for A/B comparisons.

## How the Engine Handles Thinking Tokens

The core challenge: thinking/reasoning tokens are **real computation** (they consume budget and affect latency) but are **invisible to the user** (stripped from the final answer). Stoneburner tracks them separately so benchmarks reflect what users actually see while still accounting for the full inference cost.

### Per-Provider Mechanism

| Provider | How thinking is requested | How thinking tokens are counted |
|----------|--------------------------|-------------------------------|
| **Ollama** | `body.think = true` (native API field). For older builds: `/no_think` prefix disables it. `num_predict` is inflated by `thinking_budget` so the visible answer isn't starved. | `<think>...</think>` tags are stripped from `response`. Thinking token count is **estimated** by character proportion of the total `eval_count` (Ollama doesn't report thinking tokens separately). |
| **Claude** | `thinking.budget_tokens` in the API request (extended thinking mode). | API returns `thinking_tokens` directly in the response metadata — no estimation needed. |
| **OpenAI** | Reasoning models (o3, o4-mini, gpt-5.x) handle it internally. | `completion_tokens_details.reasoning_tokens` from the API response. |

### Key Behaviors

1. **Auto-detection:** `model_classes.supports_thinking()` checks a registry of known thinking-capable model families. If the model supports it and `--thinking` wasn't explicitly set, thinking is enabled automatically.
2. **Suppression:** when thinking is *disabled* for a model that supports it, the Ollama provider prepends `/no_think` to the prompt AND sets `body.think = false` to prevent Ollama from auto-enabling it (which some models like gemma4 trigger).
3. **Budget management:** `thinking_budget` (default 8000 tokens) is added to `num_predict` so the model has room for both reasoning and the visible answer. Without this, thinking would eat the entire generation budget.
4. **Separation in output:** `ProviderResponse.thinking_tokens` and `ProviderResponse.thinking_text` are always populated separately from `output_tokens` and `text`. The `report` command shows them as distinct columns.

> **Why estimate thinking tokens for Ollama?** Ollama's `/api/generate` returns `eval_count` (total generated tokens including `<think>` content) but no breakdown. Since we have the character lengths of both the thinking and visible spans, we proportion the real token count by character ratio. This is inexact (tokenizers aren't character-linear) but stays anchored to the real token total rather than an unrelated word count.
