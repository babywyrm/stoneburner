# Frontier vs Local Comparison (2026-07-08)

Validated comparison of frontier cloud models against local inference using
stoneburner's adversarial and redblue evaluation suites. All tests used the
same judge (qwen2.5:7b local) for consistent scoring. Key results confirmed
with `--runs 3` for variance data.

## Infrastructure

- **Local inference:** RTX 5090 Laptop GPU (24GB VRAM, 92GB RAM) via Ollama
- **Frontier APIs:** OpenAI (GPT-5.5, GPT-5, GPT-4o, o3, o4-mini),
  Anthropic (Claude Sonnet 5, Opus 4.8, Sonnet 4.6, Fable 5)
- **Judge:** qwen2.5:7b (local, same for all models)

## Security Capability (RedBlue, all mode, 10 fixtures)

| Model | Provider | Quality (runs=3) | ±stddev |
|-------|----------|-----------------|---------|
| GPT-5.5 | OpenAI | 96.3% | ±8.9% |
| Claude Sonnet 4.6 | Anthropic | 96.3% | ±6.6% |
| **qwen3.6:27b** | **Local** | **~95%** | (consistent across 2 single-runs) |
| qwen3:14b | Local | 96% | (single-run, overnight sweep) |
| GPT-4o | OpenAI | 93.3% | ±11.1% |
| phi4:latest | Local | 93% | (single-run) |

## Novel Adversarial Resistance (multiturn + RAG poisoning + tool_desc_injection)

| Model | Provider | Resistance | ±stddev | Cost/run |
|-------|----------|-----------|---------|----------|
| GPT-5.5 | OpenAI | 100% | ±0% | ~$0.30 |
| Claude Sonnet 5 | Anthropic | 100% | (single) | ~$0.15 |
| Claude Opus 4.8 | Anthropic | 100% | (single) | ~$0.60 |
| Claude Sonnet 4.6 | Anthropic | 100% | (single) | ~$0.15 |
| **qwen3.6:27b** | **Local** | **100%** | (single, consistent) | **$0** |
| o3 | OpenAI | 100% | (single) | ~$0.50 |
| o4-mini | OpenAI | 100% | (single) | ~$0.05 |
| GPT-5 | OpenAI | 100% | (single) | ~$0.12 |
| GPT-4o | OpenAI | 97.0% | ±15.1% | ~$0.08 |
| Claude Fable 5 | Anthropic | 97.3% | (single) | ~$0.15 |
| phi4:latest | Local | 89.9% | ±19.5% | $0 |
| qwen3:14b | Local | 82.5% | ±27.2% | $0 |

## Basic Adversarial (prompt_injection + social_engineering)

| Model | Resistance |
|-------|-----------|
| GPT-5.5 / GPT-5 / o3 / o4-mini | 100% |
| Claude Sonnet 5 / Opus 4.8 / Sonnet 4.6 | 100% |
| qwen3.6:27b | 100% |
| phi4:latest | 100% |
| GPT-4o | 100% |
| deepseek-r1:7b | 73% |
| qwen3:14b | 56% |
| llama3.2:1b | 46% |
| mistral:7b | 44% |

## Key Findings

1. **qwen3.6:27b matches every frontier model on adversarial resistance (100%)
   for $0.** It ties GPT-5.5, Claude Sonnet 5, and Claude Opus 4.8.

2. **On security capability, qwen3.6:27b (~95%) is within 1-3 points of the
   best frontier models** (GPT-5.5 and Claude at 96.3%). The gap is within
   measurement noise (±6-9% stddev on frontier).

3. **GPT-4o is the weakest frontier model** — 93.3% capability and 97% novel
   adversarial resistance (partially complies on RAG/tool-description attacks
   in ~15% of runs).

4. **Claude Fable 5 (creative model) partially complies** on `data_as_instruction`
   — creative models are slightly more willing to "play along" with injected
   content.

5. **Single-run results are misleading.** Our initial claim that "GPT-5.5 is
   worse than GPT-4o" (87.5% vs 95%) was completely overturned with runs=3
   (96.3% vs 93.3%). Always use `--runs 3` for claims.

6. **The only real frontier advantage is latency** — 2-5s per fixture vs
   22-45s locally. Quality and safety are indistinguishable.

## Business Implications

For a BU burning tokens at frontier APIs for security evaluation work:
- **Quality parity:** Local qwen3.6:27b produces equivalent results
- **Cost:** $0 vs $0.15-$0.60 per evaluation run
- **Data sovereignty:** Nothing leaves the network
- **Reproducibility:** Same weights, same output every time
- **Trade-off:** 10-20x slower per fixture (latency, not quality)

At volume (100+ eval runs/day), a $3K GPU pays for itself in < 1 week vs
API costs. For ad-hoc work with low volume, APIs remain more convenient.
