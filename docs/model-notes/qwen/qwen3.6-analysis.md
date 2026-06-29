# Qwen 3.6 — Initial Analysis (2026-06-29)

## Models tested

| Variant | Params | Architecture | VRAM (Q4) | tok/s on brainbox | Notes |
|---------|--------|-------------|-----------|-------------------|-------|
| qwen3.6:27b | 27.8B | Dense | 11GB + 6GB CPU | 7 tok/s | CPU offload bottleneck |
| qwen3.6:35b-a3b | 35B total / 3B active | MoE (256 experts) | ~11GB + CPU | **60 tok/s** | 8.5x faster than dense |

## Infrastructure

- **Host:** brainbox (RTX 5070, 12GB VRAM, 48GB RAM, Ollama)
- **Quantization:** Q4_K_M (Ollama default)
- **CPU offload:** Both variants exceed 12GB VRAM; layers spill to system RAM
- **GPU utilization during inference:** ~18% (waiting on CPU)
- **CPU utilization:** 586% (6 cores saturated doing offloaded layers)

## Key findings

### Speed

The MoE variant is dramatically faster because only 3B parameters are active per
token — the "35B" is misleading. Despite total weights being larger (23GB vs 17GB),
the compute per token is lightweight, making CPU offload less painful.

### Safety alignment (too strong for red-team)

Both variants **refuse offensive security tasks**. When asked to write a pentest
report with attack chains and kubectl commands, they pivot to defensive analysis:

> "I cannot provide instructions, specific commands, or an attack path for
> exploiting vulnerable configurations to achieve unauthorized cluster takeover."

This makes them unsuitable as the model-under-test for red-team fixtures.

### Quality comparison (SSRF prompt, 200 tokens)

| Model | Response quality | Instruction following |
|-------|-----------------|---------------------|
| qwen3.6:27b | Excellent — precise, concise, technically correct | Perfect (3 sentences as asked) |
| qwen3:14b | Excellent — very comparable, slightly less precise | Perfect |
| qwen3:4b | Mediocre — verbose, ignores constraints | Poor (rambled) |

On simple focused prompts, qwen3.6:27b and qwen3:14b are nearly identical in
quality. The gap widens on complex multi-step reasoning tasks, but our current
redblue fixture set doesn't surface it strongly.

### Recommended roles in stoneburner

| Role | Best model | Why |
|------|-----------|-----|
| **Judge** | qwen3.6:35b-a3b | 60 tok/s, frontier quality, safety bias helps fair scoring |
| **Red-team model-under-test** | qwen3:14b | Willing to do offensive work, 97% capability |
| **Blue-team model-under-test** | qwen3.6:35b-a3b | Excellent defensive analysis, fast |
| **Adversarial resistance testing** | qwen3.6:27b/35b | Will refuse almost everything (useful to test) |

### Benchmark context (from Quesma blog)

Per Artificial Analysis benchmarks, qwen3.6:27b scores at "mid-2025 frontier"
level (GPT-5 / Claude Sonnet 4.5 tier). The MoE variant is slightly below at
"early 2025" (o3 / Claude 4 Sonnet tier).

## Next steps

- [ ] Test qwen3.6:35b-a3b as judge (replacing qwen2.5:7b) — expected quality uplift
- [ ] Run blue-team only fixtures with qwen3.6 as model-under-test
- [ ] Compare judge quality: qwen2.5:7b vs qwen3.6:35b-a3b on same fixture set
- [ ] Test with `think: true` to see if thinking mode changes refusal behavior
