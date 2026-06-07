---
name: stoneburner-inference-env
description: Use Stoneburner's inference.env standard and brain utilities to route model evaluation through a selected backend. Use when reading or writing INFERENCE_BACKEND, INFERENCE_URL, INFERENCE_MODEL, INFERENCE_THINK, /opt/agentic/inference.env, atomics.inference, provider_from_target, brain-status, brain-switch, or brain-vllm.
---

# Stoneburner Inference Env

Use this skill when a task involves the vendor-neutral `inference.env` control
file or the `brain/` inference backend utilities.

## Control File Purpose

`inference.env` lets any app, evaluator, or bootstrapper describe the current
inference target without hardcoding provider-specific variables.

Default search order:

1. `$INFERENCE_ENV`
2. `$BRAIN_ENV`
3. `/opt/agentic/inference.env`
4. `/etc/agentic/inference.env`

Canonical fields:

```text
INFERENCE_BACKEND=ollama|vllm|openai|claude|bedrock|brain-gateway
INFERENCE_URL=<endpoint-base-url>
INFERENCE_MODEL=<model-tag>
INFERENCE_THINK=true|false
INFERENCE_API_KEY=<optional-token>
```

Intent/provenance fields such as `INFERENCE_DIFFICULTY`, `INFERENCE_POOL`,
`INFERENCE_RESOLVED_AT`, and `INFERENCE_RESOLVED_BY` are useful for audit but
should not replace the resolved fields.

## Python Usage

Prefer the reference helpers:

```python
from atomics.inference import load_control_file, provider_from_target

target = load_control_file()
if target is not None:
    provider = provider_from_target(target)
```

Do not hand-roll parser logic unless the reference helper cannot be used.

## CLI/Ops Usage

Provider health:

```bash
uv run atomics provider-test --provider ollama --model <model>
uv run atomics provider-test --provider vllm --model <model>
```

Brain utilities:

```bash
./brain/brain-status
./brain/brain-status --json
./brain/brain-switch status
./brain/brain-switch ollama
./brain/brain-switch openai
./brain/brain-vllm status
```

On a single-GPU host, avoid running Ollama and a vLLM fleet in a way that causes
unexpected VRAM contention. Stop or switch backends intentionally before
benchmarking.

## Backend Semantics

- `ollama`: native Ollama API.
- `vllm`: any local OpenAI-compatible gateway such as vLLM, LiteLLM, or
  llama.cpp server.
- `openai`: cloud OpenAI API.
- `brain-gateway`: gateway endpoint used to route through another orchestrator.

Do not conflate local OpenAI-compatible gateways with the OpenAI cloud service.

## Safety Rules

- Do not commit real control files with private endpoints or credentials.
- Use sanitized examples in docs.
- Keep local machine or customer-specific values in `.env`, `$INFERENCE_ENV`, or
  other ignored local files.
- When validating a model switch, record both the resolved target and the
  evidence command output in sanitized form.
