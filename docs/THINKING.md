# Thinking Mode

Stoneburner auto-detects models with thinking/reasoning capabilities and handles them transparently. Thinking tokens are tracked separately from visible output so benchmarks measure what users actually see.

## Usage

```bash
# Auto-detect: qwen3 models enable thinking automatically
uv run atomics run --provider ollama -m qwen3:14b -n 5

# Explicit control
uv run atomics run --provider claude -m claude-sonnet-4-6 --thinking -n 5
uv run atomics run --provider openai -m o3 --no-thinking -n 5

# Shared effort dial (mapped per provider). xl → xhigh, ultra → max.
uv run atomics provider-test --provider openai -m gpt-5.6-sol --effort high
uv run atomics provider-test --provider openai -m gpt-5.6-sol --effort max --reasoning-mode pro
uv run atomics eval --provider claude -m claude-opus-4-6 --effort high
uv run atomics provider-test --provider bedrock --region us-east-1 --effort high
uv run atomics provider-test --provider groq --effort medium
uv run atomics provider-test --provider gemini --effort high
uv run atomics provider-test --provider together --effort medium

# Full prompt, model reply, thinking, and judge rationale (no truncated table)
uv run atomics eval --provider openai -m gpt-5.6-luna --effort low --verbose \
  --judge-provider claude --judge-model claude-haiku-4-5 --fixtures ev-01,ev-02,ev-03

# Custom thinking budget (Claude)
uv run atomics run --provider claude --thinking --thinking-budget 20000 -n 5

# Provider test shows thinking token breakdown
uv run atomics provider-test -p ollama -m qwen3:14b --thinking

# Smoke-test a thinking model without burning the visible answer
uv run atomics provider-test -p ollama -m qwen3.8:27b --no-thinking
```

## Supported Models

| Provider | Models | Mechanism |
|----------|--------|-----------|
| **Claude** | Opus 4.x, Sonnet 4.x | Extended thinking API (`budget_tokens`) |
| **OpenAI** | o3, o3-mini, o3-pro, o4-mini, gpt-5.x (including Sol/Terra/Luna) | Reasoning tokens (`completion_tokens_details`) |
| **Ollama** | qwen3 family (including qwen3.8), deepseek-r1, phi4-*-reasoning | Native `thinking` field, plus `<think>` tag fallback |

When `--thinking` / `--no-thinking` is omitted, stoneburner checks the model against its capability registry and enables thinking automatically for known models. Use `--no-thinking` to force it off for A/B comparisons.

The same `--thinking` / `--no-thinking` / `--thinking-budget` grammar is on every suite that calls `generate`: `run`, `eval`, `adversarial`, `redblue`, `refusal`, `toolcall` (prose channel), `codereview`, `multiturn`, `rag`, `codegen`, and `probe`. `--effort` / `--reasoning-mode` are on those same suites. For local thinking models on short fixtures, `--no-thinking` is the difference between a visible answer and an empty generation that spent the whole token budget on hidden reasoning. When that still happens, the attempt is `thinking_budget` (CLI: `THINK`), not a provider crash.

## How the Engine Handles Thinking Tokens

The core challenge: thinking/reasoning tokens are **real computation** (they consume budget and affect latency) but are **invisible to the user** (stripped from the final answer). Stoneburner tracks them separately so benchmarks reflect what users actually see while still accounting for the full inference cost.

### Per-Provider Mechanism

| Provider | How thinking is requested | How thinking tokens are counted |
|----------|--------------------------|-------------------------------|
| **Ollama** | `body.think = true` (native API field). For older builds: `/no_think` prefix disables it. `num_predict` is inflated by `thinking_budget` so the visible answer isn't starved. | Newer Ollama returns a top-level `thinking` string; older builds embed `<think>...</think>` in `response`. Both are captured. Thinking token count is **estimated** by character proportion of the total `eval_count` (Ollama doesn't report thinking tokens separately). |
| **Claude** | `thinking.budget_tokens` in the API request (extended thinking mode). | API returns `thinking_tokens` directly in the response metadata — no estimation needed. |
| **OpenAI** | `--effort` → Chat Completions `reasoning_effort`, or Responses `reasoning.effort`. `--reasoning-mode pro` forces the Responses API and sets `reasoning.mode`. | `completion_tokens_details.reasoning_tokens` from the API response. |
| **Claude (4.6+)** | `--effort` → `thinking: {type: "adaptive"}` plus `output_config.effort`. `--thinking` without `--effort` still uses `budget_tokens`. | Thinking blocks plus usage metadata. |
| **Bedrock** | Same Claude mapping, sent in `additionalModelRequestFields`. Region-prefixed IDs (`us.anthropic.claude-…`) resolve to the Claude family. | Usage metadata from Converse. |

`--effort` values: `none`, `minimal`, `low`, `medium`, `high`, `xhigh` (alias `xl`), `max` (alias `ultra`). Claude 4.6 maps `xhigh` to `max`. OpenAI-compatible clouds (Groq, Together, Gemini, vLLM) receive `reasoning_effort` when the backend honors it. The native payload is recorded on the response as `reasoning_request`.

### Key Behaviors

1. **Auto-detection:** `model_classes.supports_thinking()` checks a registry of known thinking-capable model families. If the model supports it and `--thinking` wasn't explicitly set, thinking is enabled automatically.
2. **Suppression:** when thinking is *disabled* for a model that supports it, the Ollama provider prepends `/no_think` to the prompt AND sets `body.think = false` to prevent Ollama from auto-enabling it (which some models like gemma4 trigger).
3. **Budget management:** `thinking_budget` (default 8000 tokens) is added to `num_predict` so the model has room for both reasoning and the visible answer. Without this, thinking would eat the entire generation budget.
4. **Separation in output:** `ProviderResponse.thinking_tokens` and `ProviderResponse.thinking_text` are always populated separately from `output_tokens` and `text`. The `report` command shows them as distinct columns.

> **Why estimate thinking tokens for Ollama?** Ollama's `/api/generate` returns `eval_count` (total generated tokens including `<think>` content) but no breakdown. Since we have the character lengths of both the thinking and visible spans, we proportion the real token count by character ratio. This is inexact (tokenizers aren't character-linear) but stays anchored to the real token total rather than an unrelated word count.
