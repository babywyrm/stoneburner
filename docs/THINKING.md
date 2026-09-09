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
uv run atomics provider-test --provider vllm -m qwen3.8:27b --effort low --thinking-budget 512
uv run atomics provider-test --provider ollama -m qwen3:14b --effort low

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
| **Ollama** | qwen3 family (including qwen3.8), granite4.2, gemma4, deepseek-r1, gpt-oss, phi4-*-reasoning | Native `think` field: bool, or `low` / `medium` / `high` / `max` from `--effort`. Plus `<think>` tag fallback. Do not send think levels to tags that lack the capability (mistral, gemma3, phi4-mini) — Ollama 400s. |
| **vLLM / SGLang** (`--provider vllm`) | qwen3 family (including qwen3.8) | `chat_template_kwargs.enable_thinking` plus mapped `reasoning_effort`; optional `custom_params.thinking_budget` |

When `--thinking` / `--no-thinking` is omitted, stoneburner checks the model against its capability registry and enables thinking automatically for known models. Use `--no-thinking` to force it off for A/B comparisons.

The same `--thinking` / `--no-thinking` / `--thinking-budget` grammar is on every suite that calls `generate`: `run`, `eval`, `adversarial`, `redblue`, `refusal`, `toolcall` (prose and tool channels), `codereview`, `multiturn`, `rag`, `codegen`, and `probe`. `--effort` / `--reasoning-mode` are on those same suites. The tool channel keeps Claude extended thinking off and still forwards `output_config.effort`. For local thinking models on short fixtures, `--no-thinking` is the difference between a visible answer and an empty generation that spent the whole token budget on hidden reasoning. When that still happens, the attempt is `thinking_budget` (CLI: `THINK`), not a provider crash.

On `--provider vllm`, those Qwen template keys are written by `generate()` and `generate_with_tools`. If SGLang `--tool-call-parser qwen3_coder` still loops with `--effort` on `--channel tools`, use `--no-thinking` on that channel.

## How the Engine Handles Thinking Tokens

The core challenge: thinking/reasoning tokens are **real computation** (they consume budget and affect latency) but are **invisible to the user** (stripped from the final answer). Stoneburner tracks them separately so benchmarks reflect what users actually see while still accounting for the full inference cost.

### Per-Provider Mechanism

| Provider | How thinking is requested | How thinking tokens are counted |
|----------|--------------------------|-------------------------------|
| **Ollama** | `body.think` is a bool, or `low` / `medium` / `high` / `max` from `--effort` (never `none` — that 400s). `--no-thinking` and `--effort none` send `false`. `num_predict` is inflated by `thinking_budget` so the visible answer isn't starved. GPT-OSS ignores bool `think` and wants a level. | Newer Ollama returns a top-level `thinking` string; older builds embed `<think>...</think>` in `response`. Both are captured. Thinking token count is **estimated** by character proportion of the total `eval_count` (Ollama doesn't report thinking tokens separately). |
| **vLLM / SGLang** (`--provider vllm`) | Qwen3: `chat_template_kwargs.enable_thinking` on/off. `--effort` is dual-written: top-level `reasoning_effort` (OpenAI-compat) and `chat_template_kwargs.reasoning_effort` mapped to `low` / `medium` / `xhigh` (Qwen Jinja ignores unknown keys and falls through to xhigh). `--thinking-budget` is SGLang `custom_params.thinking_budget` when thinking is on; a hard cap also needs the server flag `--enable-strict-thinking`. Hermes `/reasoning` does not reach the model. | Prefer `usage.reasoning_tokens` when the gateway reports it; otherwise estimate from `reasoning_content` character share. |
| **llama.cpp** (`--provider llamacpp`) | OpenAI-compat `reasoning_effort` on `generate()` | Gateway-reported usage when present |
| **Claude** | `thinking.budget_tokens` in the API request (extended thinking mode). | API returns `thinking_tokens` directly in the response metadata — no estimation needed. |
| **OpenAI** | `--effort` → Chat Completions `reasoning_effort`, or Responses `reasoning.effort`. `--reasoning-mode pro` forces the Responses API and sets `reasoning.mode`. | `completion_tokens_details.reasoning_tokens` from the API response. |
| **Claude (4.6+)** | `--effort` → `thinking: {type: "adaptive"}` plus `output_config.effort`. `--thinking` without `--effort` still uses `budget_tokens`. | Thinking blocks plus usage metadata. |
| **Bedrock** | Same Claude mapping, sent in `additionalModelRequestFields`. Region-prefixed IDs (`us.anthropic.claude-…`) resolve to the Claude family. | Usage metadata from Converse. |

`--effort` values: `none`, `minimal`, `low`, `medium`, `high`, `xhigh` (alias `xl`), `max` (alias `ultra`). Claude 4.6 maps `xhigh` to `max`. OpenAI-compatible clouds (Groq, Together, Gemini, vLLM, llama.cpp) receive `reasoning_effort` when the backend honors it. Ollama maps the same dial onto native `think` (`low` / `medium` / `high` / `max`). On `--provider vllm` Qwen3 models, that value is also mapped into `chat_template_kwargs` (`high` → `medium`, `max` → `xhigh`) so the template cannot silently ignore it. The native payload is recorded on the response as `reasoning_request`. HTTP / MCP take the same fields on `POST /runs`, `POST /evals`, `POST /sweeps`, and `POST /provider-test` (and the matching `submit_*` / `provider_test` tools). `probe` stays CLI-only.

### Key Behaviors

1. **Auto-detection:** `model_classes.supports_thinking()` checks a registry of known thinking-capable model families. If the model supports it and `--thinking` wasn't explicitly set, thinking is enabled automatically.
2. **Suppression:** when thinking is *disabled* for a model that supports it, the Ollama provider sets `body.think = false`. Do not prefix `/no_think` into the user prompt — current Ollama honors the native field, and the prefix leaks into the answer (`qwen3:4b` narrated the token instead of answering). If the model still dumps chain-of-thought into `response` (paired `<think>` tags or an orphan `</think>`), that span is split into `thinking_text` so the visible answer is the part after the closer. `/api/chat` (`generate_with_tools`) uses the same split on `message.content`.
3. **Budget management:** `thinking_budget` is added to `num_predict` on Ollama so the visible answer isn't starved. On `--provider vllm` Qwen3, a non-`None` budget is `custom_params.thinking_budget` (SGLang enforces a hard cap only with `--enable-strict-thinking`). Security suites that default `--thinking-budget 8000` will send that field; stock vLLM that forbids extra body keys may `422`. `provider-test` / `eval` default the flag to unset. Claude uses `budget_tokens`.
4. **Separation in output:** `ProviderResponse.thinking_tokens` and `ProviderResponse.thinking_text` are always populated separately from `output_tokens` and `text`. The `report` command shows them as distinct columns.

> **Why estimate thinking tokens for Ollama?** Ollama's `/api/generate` returns `eval_count` (total generated tokens including `<think>` content) but no breakdown. Since we have the character lengths of both the thinking and visible spans, we proportion the real token count by character ratio. This is inexact (tokenizers aren't character-linear) but stays anchored to the real token total rather than an unrelated word count.
