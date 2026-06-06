# brain/ — Inference Backend Utilities

Portable ops scripts for managing the inference layer on any box —
local dev, a GPU host, or cloud nodes. These are standalone shell scripts
with no dependency on the `atomics` Python package.

## Scripts

| Script | Purpose |
|--------|---------|
| `brain-status` | Print running models, VRAM mapping, gateway state |
| `brain-switch` | Toggle a box between Ollama and OpenAI-compatible (vLLM/LiteLLM) |
| `brain-vllm` | Start/stop/restart the vLLM engine + gateway systemd fleet |

## Environment

Scripts auto-detect the backend from a control file or env vars:

```
INFERENCE_API=ollama|openai       # which wire format to use
OPENAI_BASE_URL=http://...        # gateway endpoint (openai mode)
OPENAI_MODEL=qwen2.5:3b           # model name the gateway routes
OPENAI_API_KEY=dummy               # placeholder for local gateways
OLLAMA_URL=http://127.0.0.1:11434 # native Ollama endpoint
OLLAMA_MODEL=qwen3.5:0.8b         # Ollama model tag
OLLAMA_THINK=false                 # thinking mode toggle
```

These are the same variables consumed by downstream agent services —
any app that reads `INFERENCE_API` plus the Ollama/OpenAI vars (directly,
via a small adapter, or via a k8s ConfigMap) self-configures from them.

## Control file

Drop an `inference.env` at `/etc/agentic/inference.env` (or set
`BRAIN_ENV` to point elsewhere). Scripts source it if present.
The same file becomes a k8s ConfigMap via `envFrom` in production.

## Usage

```bash
# On any box with SSH access:
./brain-status                     # what's running right now?
./brain-status --json              # machine-readable output

./brain-switch ollama              # switch to native Ollama
./brain-switch openai              # switch to gateway/vLLM
./brain-switch status              # show current backend

./brain-vllm status                # vLLM fleet + GPU state
./brain-vllm stop                  # stop gateway + all engines (free VRAM)
./brain-vllm start                 # start enabled engines + gateway
./brain-vllm restart               # full fleet bounce
```

## VRAM contention

vLLM and Ollama both claim GPU memory. On a single-GPU box you generally
run one OR the other for a given test. To hand the whole card to a large
Ollama model, stop the vLLM fleet first:

```bash
./brain-vllm stop        # free VRAM held by vLLM engines
./brain-switch ollama --model gemma4:12b
# ... run tests ...
./brain-vllm start       # bring vLLM back when done
./brain-switch openai
```
