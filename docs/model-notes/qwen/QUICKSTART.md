# Qwen 3.6 Quickstart — MoE vs Dense, Explained for Practitioners

## What is MoE (Mixture of Experts)?

Traditional "dense" models activate **every parameter** for every token they
generate. A 27B dense model does 27 billion multiplications per token. Slow.

**MoE (Mixture of Experts)** splits the model into hundreds of small "expert"
sub-networks. For each token, a **router** picks just a handful of experts to
activate. The rest sit idle.

```
Dense 27B:         [████████████████████████████] ← all 27B params fire per token
                   Slow (7 tok/s on our hardware), but each token gets full attention

MoE 35B-A3B:      [██░░░░░░░░░░░░░░░░░░░░░░░░░░] ← only 3B active per token
                   Fast (61 tok/s), but draws from a 35B knowledge pool
                   256 experts total, 8 routed + 1 shared per token
```

### Why does this matter for us?

| Property | Dense 27B | MoE 35B-A3B |
|----------|-----------|-------------|
| Total knowledge | 27B params | 35B params (more!) |
| Active compute per token | 27B | **3B** (9x less!) |
| Speed on our RTX 5070 | 7 tok/s | **61 tok/s** |
| Quality (our benchmarks) | Excellent | Slightly below dense |
| VRAM needed | 17GB (offloads) | 24GB (offloads more, but compute is cheap) |
| Best for | Max quality, batch | **Interactive use, judging, rapid iteration** |

### The "A3B" naming convention

```
qwen3.6:35b-a3b
         │    │
         │    └── "Active 3 Billion" — only 3B params compute per token
         └─────── "35 Billion total" — the full weight pool
```

Other MoE examples in the wild:
- Mixtral 8x7B = 46.7B total, ~12B active
- DeepSeek V3 = 671B total, 37B active
- Qwen 3.6 35B-A3B = 35B total, 3B active (extremely efficient)

---

## Getting started on brainbox

### Pull models (already done)

```bash
# Dense (slower, max quality)
ollama pull qwen3.6:27b

# MoE (fast, nearly same quality)
ollama pull qwen3.6:35b-a3b
```

### Quick test — see the speed difference

```bash
# MoE — ~60 tok/s, response in seconds
curl -s http://192.168.1.239:11434/api/generate -d '{
  "model": "qwen3.6:35b-a3b",
  "prompt": "Explain SSRF in 3 sentences.",
  "stream": false,
  "options": {"num_predict": 200},
  "think": false
}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"eval_count\"]} tok in {d[\"eval_duration\"]/1e9:.1f}s = {d[\"eval_count\"]/(d[\"eval_duration\"]/1e9):.0f} tok/s'); print(d['response'])"

# Dense — ~7 tok/s, takes 30 seconds
curl -s http://192.168.1.239:11434/api/generate -d '{
  "model": "qwen3.6:27b",
  "prompt": "Explain SSRF in 3 sentences.",
  "stream": false,
  "options": {"num_predict": 200},
  "think": false
}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"eval_count\"]} tok in {d[\"eval_duration\"]/1e9:.1f}s = {d[\"eval_count\"]/(d[\"eval_duration\"]/1e9):.0f} tok/s'); print(d['response'])"
```

### Use with stoneburner

```bash
# As the model under test (blue-team — where safety alignment helps)
uv run atomics redblue -p ollama --ollama-host "http://192.168.1.239:11434" \
  -m qwen3.6:35b-a3b --judge-model qwen2.5:7b --mode blue

# As the JUDGE (recommended — stricter, better differentiation)
uv run atomics -v redblue -p ollama --ollama-host "http://192.168.1.239:11434" \
  -m qwen3:14b --judge-model qwen3.6:35b-a3b --mode red

# Adversarial resistance test
uv run atomics -v adversarial -p ollama --ollama-host "http://192.168.1.239:11434" \
  -m qwen3.6:27b --judge-model qwen3.6:35b-a3b --category prompt_injection
```

---

## Key discoveries from our testing

### 1. Thinking mode ≠ refusal

When qwen3.6 appears to "refuse" an offensive task, it's often just running out
of token budget in the thinking phase:

```
think: true  + num_predict: 600   → empty response (all tokens spent thinking)
think: true  + num_predict: 2000  → STILL empty (8739 chars of thinking, 0 visible)
think: false + num_predict: 500   → genuine refusal ("I cannot provide...")
```

The model DOES reason about attack chains correctly in the thinking layer — it
just won't surface them as visible output. This is safety alignment at the
output layer, not inability.

### 2. MoE is a better judge than smaller dense models

| Judge model | Score for qwen3:14b | Behavior |
|-------------|-------------------|----------|
| qwen2.5:7b (4.7GB) | 100% | Rubber-stamps everything |
| qwen3.6:35b-a3b (24GB, MoE) | 94% | Finds real gaps, provides actionable rationale |

The MoE's frontier-class reasoning lets it evaluate responses more critically
while still being fast enough (61 tok/s) for practical use.

### 3. CPU offload explained

When a model doesn't fit in VRAM (12GB on our RTX 5070):

```
qwen3.6:27b (17GB model):
  ┌─────────────┐     ┌──────────────┐
  │   GPU VRAM  │ ←──→│   CPU RAM    │
  │   11GB      │PCIe │   6GB        │
  │  (fast)     │bus  │  (slow)      │
  └─────────────┘     └──────────────┘
  Result: 7 tok/s (bottlenecked by PCIe transfers)

qwen3.6:35b-a3b (24GB model, but only 3B active):
  ┌─────────────┐     ┌──────────────┐
  │   GPU VRAM  │ ←──→│   CPU RAM    │
  │   11GB      │PCIe │   13GB       │
  │  (fast)     │bus  │  (less work) │
  └─────────────┘     └──────────────┘
  Result: 61 tok/s (less compute needed per token despite more offload)
```

The MoE wins because even though MORE weight is offloaded, LESS computation
happens per token (only 3B active vs 27B active).

---

## When to use which

| Scenario | Use this | Why |
|----------|----------|-----|
| Overnight sweep (20 models) | qwen3.6:35b-a3b as **judge** | Fast + strict + consistent |
| Red-team model-under-test | qwen3:14b | Will do offensive work, 97% capability |
| Blue-team model-under-test | qwen3.6:35b-a3b | Excellent defensive analysis |
| Testing adversarial resistance | qwen3.6:27b | Resists almost everything |
| Quick iteration / scripting | qwen2.5:3b | 228 tok/s, good enough quality |
| Maximum quality (time no object) | qwen3.6:27b with `think: true` | Frontier-class but 7 tok/s |

---

## Further reading

- [Quesma: Qwen 3.6 27B is the sweet spot](https://quesma.com/blog/qwen-36-is-awesome/)
- [Ollama: qwen3.6 model page](https://ollama.com/library/qwen3.6)
- Our detailed analysis: `docs/model-notes/qwen/README.md`
- Our detailed initial findings: `docs/model-notes/qwen/qwen3.6-analysis.md`
- Adversarial leaderboard: `docs/LEADERBOARD.md`
- Capability leaderboard: `docs/LEADERBOARD-REDBLUE.md`
