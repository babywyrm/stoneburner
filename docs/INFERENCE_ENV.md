# The `inference.env` Standard

A small, vendor-neutral control file that lets **any box describe the LLM
inference target it is wired to**, so that any tool or app on (or pointed at)
that box can self-configure without hardcoding hosts, models, or backends.

It is the single contract shared by:

- **producers** — anything that provisions a box (a bootstrapper, a CTF
  platform, a cloud-init script, an operator by hand) and *writes* the file;
- **consumers** — anything that runs inference or evaluates it (the `atomics`
  providers, the `brain/` ops scripts, a downstream agent service) and *reads*
  it.

The reference reader/resolver for this standard lives in
[`atomics/inference.py`](../atomics/inference.py).

---

## File location

Consumers search these paths in order (first hit wins):

1. `$INFERENCE_ENV` (explicit override)
2. `$BRAIN_ENV` (legacy override, still honored)
3. `/opt/agentic/inference.env`
4. `/etc/agentic/inference.env`

Recommended on-box placement: `/opt/agentic/inference.env`, owned `root:root`,
mode `0640`. In Kubernetes the same file becomes a ConfigMap consumed via
`envFrom`.

## Format

Plain `KEY=VALUE` lines (shell-sourceable, dotenv-compatible). `#` comment
lines and blanks are ignored. Values are **not** quoted by the reference
writer.

---

## Canonical schema

### Resolved fields (what to actually use)

| Key | Meaning | Example |
|-----|---------|---------|
| `INFERENCE_BACKEND` | wire dialect / provider name (see below) | `ollama` |
| `INFERENCE_URL` | endpoint base URL | `http://10.0.0.9:11434` |
| `INFERENCE_MODEL` | model tag the backend routes | `gemma3:4b` |
| `INFERENCE_THINK` | thinking-mode toggle (`true`/`false`) | `false` |
| `INFERENCE_API_KEY` | bearer token; empty for local gateways | `` |

### Intent fields (what was requested — optional)

| Key | Meaning |
|-----|---------|
| `INFERENCE_DIFFICULTY` | requested tier: `easy` \| `medium` \| `hard` |
| `INFERENCE_POOL` | requested endpoint pool name |

### Provenance fields (audit — optional)

| Key | Meaning |
|-----|---------|
| `INFERENCE_RESOLVED_AT` | ISO-8601 UTC timestamp the file was written |
| `INFERENCE_RESOLVED_BY` | who/what resolved it |

### Backend values

Canonical `INFERENCE_BACKEND` values are the `atomics` provider names:

`ollama` · `vllm` · `openai` · `claude` · `bedrock` · `brain-gateway`

`vllm` means **any local OpenAI-compatible gateway** (vLLM, LiteLLM,
llama.cpp server). `openai` means the **cloud** OpenAI API. This distinction
matters: a LAN gateway is `vllm`, not `openai`.

---

## Legacy compatibility

Earlier `brain/` scripts wrote a different set of keys. The reference reader
**normalizes** them into the canonical schema, so old files keep working:

| Legacy key | Maps to |
|------------|---------|
| `INFERENCE_API=ollama` | `INFERENCE_BACKEND=ollama` |
| `INFERENCE_API=openai` | `INFERENCE_BACKEND=vllm` (local gateway semantics) |
| `OLLAMA_URL` | `INFERENCE_URL` (when backend resolves to ollama) |
| `OLLAMA_MODEL` | `INFERENCE_MODEL` |
| `OLLAMA_THINK` | `INFERENCE_THINK` |
| `OPENAI_BASE_URL` | `INFERENCE_URL` (when backend resolves to vllm/openai) |
| `OPENAI_MODEL` | `INFERENCE_MODEL` |
| `OPENAI_API_KEY` | `INFERENCE_API_KEY` |

Canonical `INFERENCE_*` keys always win over legacy keys when both are present.

---

## Example

```sh
# INTENT
INFERENCE_DIFFICULTY=easy
INFERENCE_POOL=brainbox

# RESOLVED
INFERENCE_BACKEND=ollama
INFERENCE_URL=http://10.0.0.9:11434
INFERENCE_MODEL=gemma3:4b
INFERENCE_THINK=false
INFERENCE_API_KEY=

# PROVENANCE
INFERENCE_RESOLVED_AT=2026-06-06T17:00:00Z
INFERENCE_RESOLVED_BY=control-plane-resolver
```

## Consuming it from Python

```python
from atomics.inference import load_control_file, provider_from_target

target = load_control_file()          # searches the standard paths
provider = provider_from_target(target)   # an OllamaProvider/VllmProvider/...
```

If no control file is found, `load_control_file()` returns `None` and the
caller should fall back to its normal configuration (`AtomicsSettings`).
