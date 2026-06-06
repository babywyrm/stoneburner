# brain/ — Inference Backend Utilities

Portable ops scripts for managing the inference layer on any box —
local dev, gpu-host, or cloud nodes. These are standalone shell scripts
with no dependency on the `atomics` Python package.

## Scripts

| Script | Purpose |
|--------|---------|
| `brain-status` | Print running models, VRAM mapping, gateway state |
| `brain-switch` | Toggle a box between Ollama and OpenAI-compatible (vLLM/LiteLLM) |

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

These are the same variables consumed by the agent services (target-app,
policy-service) and will be consumed by app-gate once it gets the
Spring AI OpenAI-starter swap.

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
```
