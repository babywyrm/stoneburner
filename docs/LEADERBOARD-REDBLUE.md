# Red/Blue Security Capability Leaderboard

Measures how well each model performs real security work — both offensive (red)
and defensive (blue) tasks. This complements the adversarial resistance
leaderboard by answering: "Can this model actually do useful security reasoning?"

**Date:** 2026-06-29 (overnight sweep)
**Infrastructure:** local GPU host (NVIDIA RTX 5070, 12GB VRAM, Ollama)
**Judge:** qwen2.5:7b (local, same infrastructure)
**Fixtures:** 10 per model (5 red + 5 blue)
**Mode:** all (red + blue combined)
**Total evaluations:** 200 (20 models × 10 fixtures)
**Runtime:** ~64 minutes

---

## Results

| Rank | Model | Quality | Avg Latency | Tier |
|------|-------|---------|-------------|------|
| 1 | qwen3:14b | **97%** | 63.0s | Elite |
| 2 | qwen2.5:14b | **96%** | 25.4s | Elite |
| 3 | qwen3:4b | **92%** | 27.6s | Strong |
| 4 | gemma4:12b | **92%** | 18.6s | Strong |
| 5 | phi4:latest | **89%** | 28.4s | Strong |
| 6 | ministral-3:3b | **89%** | 11.3s | Strong |
| 7 | gemma4:e4b | 85% | 9.4s | Capable |
| 8 | qwen2.5:7b | 84% | 6.1s | Capable |
| 9 | gemma3:4b | 84% | 7.5s | Capable |
| 10 | qwen3.5:4b | 78% | 33.7s | Capable |
| 11 | qwen2.5:3b | 75% | 8.4s | Moderate |
| 12 | mistral:7b | 70% | 7.9s | Moderate |
| 13 | dolphin3:8b | 70% | 7.2s | Moderate |
| 14 | cogito:3b | 66% | 3.1s | Moderate |
| 15 | qwen3.5:0.8b | 64% | 10.6s | Weak |
| 16 | qwen3.5:2b | 64% | 18.6s | Weak |
| 17 | deepseek-r1:7b | 64% | 14.7s | Weak |
| 18 | phi4-mini:latest | 61% | 3.2s | Weak |
| 19 | qwen2.5:1.5b | 58% | 7.0s | Weak |
| 20 | llama3.2:1b | 42% | 1.5s | Minimal |

---

## Mistral family addendum (2026-07-04)

Same methodology (10 fixtures, mode `all`, qwen2.5:7b judge, single run).

| Model | Quality | Avg Latency | Tier |
|-------|:-------:|:-----------:|------|
| mistral-small:24b | **92%** | 83.8s | Strong |
| mistral-nemo:12b | **85%** | 10.0s | Capable |

- **mistral-small:24b (92%)** lands in the Strong tier — matching qwen3:4b /
  gemma4:12b on capability — but at **83.8s average latency** because 37% of the
  17GB model runs on CPU (only ~12GB fits on the RTX 5070). It is capable but the
  slowest model in the fleet to run locally.
- **mistral-nemo:12b (85%)** is Capable at a far more practical 10s latency —
  the better local pick when you want Mistral-family capability without the
  CPU-offload penalty. Both are a clear step up from `mistral:7b` (70%).

**Anchor check:** re-running `qwen2.5:7b` (already ranked #8 at 84%) with the same
setup produced **92%** this pass. The 8-point swing is single-run variance on a
10-fixture eval (redblue is not yet multi-run averaged; use `--runs 3` for tighter
confidence). Read the addendum cohort as approximate — capability tiers hold, exact
percentages carry ±~8 points of run-to-run noise at this fixture count.

---

## On-card cohort (2026-08-14)

Same methodology (10 fixtures, mode `all`, qwen2.5:7b judge, `--no-thinking`,
single run). All five models fit fully in the RTX 5070 12GB. Integrity: 50/50
fixtures scored, zero generation / infrastructure / judge failures.

| Rank | Model | Quality | Red | Blue | Avg Latency | Size | Tier |
|------|-------|:-------:|:---:|:----:|:-----------:|:----:|------|
| 1 | granite4.1:8b | **97%** | 100% | 94% | 9.4s | 5.3 GB | Elite |
| 2 | ministral-3:8b | **94%** | 94% | 94% | 11.9s | 6.0 GB | Strong |
| 2 | gemma4:12b | **94%** | 96% | 92% | 20.3s | 7.6 GB | Strong |
| 4 | qwen3:8b | **91%** | 94% | 88% | 10.4s | 5.2 GB | Strong |
| 5 | qwen3.5:9b | **85%** | 76% | 94% | 11.0s | 6.6 GB | Capable |

- **`granite4.1:8b` (97%)** matches the June `qwen3:14b` headline at **9.4s**
  instead of 63s, because it stays on the 5070. Only miss: detection-engineering
  at 70%. This is the on-card workhorse.
- **`ministral-3:8b` (94%)** steps up from the June `ministral-3:3b` (89%) and
  ties `gemma4:12b` at about half the latency.
- **`gemma4:12b` (94%)** re-ran within 2 points of the June 92% result. Same
  Strong tier; slower than the new 8B class.
- **`qwen3:8b` (91%)** fills the 4b → 14b gap and stays on-card. Family
  placement is consistent with June (`qwen3:4b` 92%, `qwen3:14b` 97%).
- **`qwen3.5:9b` (85%)** was **91%** on a smoke run an hour earlier. Blue is
  solid; red tradecraft (priv-esc / lateral / web) is the swing. Treat as
  Capable until a `--runs 3` pass.

**Lab default recommendation:** `granite4.1:8b` for on-card work. Keep
`qwen3:8b` when the box needs the Qwen3 thinking family. Leave the 27B tags
for offload comparison only.

---

## Overnight `--runs 3` (2026-08-14/15)

Same host and judge, 32 on-card models, 10 fixtures × 3 runs, `--no-thinking`.
Integrity: 320/320 red/blue fixtures scored. Off-card 15 GB+ tags (qwen3.6
27b/35b, both 24b Mistrals) were skipped.

| Rank | Model | Quality | ± | vs June single-run |
|------|-------|:-------:|:-:|--------------------|
| 1 | granite4.1:8b | **97.3%** | 8.1 | new on-card default |
| 1 | gemma4:e4b | **97.3%** | 8.1 | June 85% |
| 1 | gemma4:12b | **97.3%** | 8.1 | June 92% |
| 4 | ministral-3:8b | 96.0% | 10.2 | new |
| 5 | qwen3:14b | 94.3% | 11.5 | June **97%** |
| 6 | ministral-3:3b | 93.7% | 12.2 | June 89% |
| 7 | mistral-nemo:12b | 93.3% | 13.0 | July addendum 85% |
| 8 | qwen2.5:14b | 93.0% | 12.7 | June 96% |
| 9 | phi4:latest | 92.7% | 12.4 | June 89% |
| 10 | qwen2.5:7b | 92.3% | 12.8 | June 84% |

Promote on `--runs 3`, not a single pass. The June `qwen3:14b` 97% is now
**94.3% ±11.5**. Stdev on this card is still 8–13 points on ten fixtures;
`nemotron-3-nano:4b` hit ±32.7. Single-run leaderboard rows above remain
historical. `functiongemma:latest` scored 8.3% ±12.9 — not a security model.

---

## Key findings

- **`qwen3:14b` tops capability** at 97% — almost perfect security reasoning
  across both offensive and defensive tasks. But it's slow (63s average due to
  CPU offload on 12GB VRAM).

- **`qwen2.5:14b` is the efficiency winner** — 96% quality at 25s latency.
  Nearly matches qwen3:14b in capability while being 2.5x faster.

- **`ministral-3:3b` punches way above its weight** — 89% capability from a 3B
  model at 11s latency. Best quality-per-parameter in the fleet.

- **The qwen3.5 family trades safety for capability.** qwen3.5:4b leads the
  adversarial resistance leaderboard (98%) but drops to 78% on capability. Its
  strong alignment may overconstrain it on offensive security tasks.

- **`gemma3:4b` → `gemma4:e4b` is a big jump.** 84% → 85% numerically similar,
  but gemma4 achieves it with better structured responses and fewer hallucinated
  tool names.

- **Reasoning models (deepseek-r1, qwen3.5) underperform here.** The extended
  thinking overhead doesn't help on these focused security tasks, and the models
  often over-deliberate instead of answering directly.

---

## Category breakdown (top 5 models)

| Model | Recon | Vuln Analysis | Priv Esc | Lateral | Web Exploit | IR | Hardening | Detection |
|-------|-------|---------------|----------|---------|-------------|-----|-----------|-----------|
| qwen3:14b | 100% | 100% | 100% | 90% | 100% | 100% | 90% | 90% |
| qwen2.5:14b | 100% | 100% | 90% | 70% | 100% | 100% | 100% | 100% |
| qwen3:4b | 100% | 100% | 100% | 70% | 100% | 100% | 70% | 90% |
| gemma4:12b | 100% | 100% | 90% | 70% | 100% | 100% | 90% | 70% |
| phi4:latest | 100% | 100% | 70% | 90% | 100% | 100% | 70% | 70% |

---

## Resistance vs. Capability Matrix

The real insight comes from combining both leaderboards. A model should ideally
be both safe (resists manipulation) AND capable (performs useful security work).

```
                    HIGH CAPABILITY (≥85%)
                    │
        IDEAL       │      CAPABLE BUT RISKY
   ─────────────────┼──────────────────────────
   qwen2.5:14b(71/96)  qwen3:14b(75/97)
   gemma4:12b(94/92)   qwen3:4b(67/92)
   phi4(79/89)         ministral-3(69/89)
   gemma4:e4b(83/85)   gemma3:4b(46/84)
                    │
  HIGH RESISTANCE ──┼── LOW RESISTANCE
   (≥70%)           │   (<70%)
                    │
        SAFE BUT    │      AVOID
        LIMITED     │
   qwen3.5:4b(98/78)   dolphin3(48/70)
   qwen3.5:2b(85/64)   deepseek-r1(48/64)
   qwen3.5:0.8b(73/64) qwen2.5:1.5b(42/58)
   phi4-mini(77/61)    llama3.2:1b(58/42)
                    │
                    LOW CAPABILITY (<85%)
```

Format: model(resistance%/capability%)

### Quadrant recommendations

| Quadrant | Models | Use case |
|----------|--------|----------|
| **Ideal** (high resist + high capable) | **qwen3.6:27b**, **phi4**, gemma4:12b, gemma4:e4b, qwen2.5:14b | Production security tooling |
| **Capable but risky** (low resist + high capable) | qwen3:14b, qwen3:4b, ministral-3, gemma3:4b | Supervised red-team work only |
| **Safe but limited** (high resist + low capable) | qwen3.5:4b/2b/0.8b, phi4-mini | User-facing safety filtering |
| **Avoid** (low resist + low capable) | dolphin3, deepseek-r1:7b, llama3.2:1b, mistral:7b | Not recommended |

---

## RTX 5090 Capability Results (2026-07-07/08)

Tested on RTX 5090 Laptop GPU (24GB VRAM). Same eval fixtures, 6 models across
5 makers. Judge: qwen2.5:7b.

| Rank | Model | Maker | Quality | Latency | Notes |
|------|-------|-------|---------|---------|-------|
| 1 | qwen3:14b | Alibaba | **96%** | 38.1s | Best capability, but 56% resistance |
| 2 | qwen3.6:27b | Alibaba | **95%** | ~90s | Near-perfect + 100% resistance = ideal |
| 3 | phi4:latest | Microsoft | **93%** | ~30s | Strong + 100% resistance = ideal |
| 4 | mistral:7b | Mistral | 73% | ~15s | Moderate |
| 5 | deepseek-r1:7b | DeepSeek | 52% | ~20s | Weak — thinking overhead doesn't help |
| 6 | llama3.2:1b | Meta | 43% | ~4s | Too small for security tasks |

---

## Infrastructure notes

- **5070 results (June 2026):** RTX 5070, 12GB VRAM, 48GB RAM, 20 models
- **5090 results (July 2026):** RTX 5090 Laptop, 24GB VRAM, 92GB RAM, 6 models
- Models >12GB offload layers to CPU RAM on the 5070 (slower but functional)
- Models up to 17GB fit in VRAM on the 5090 (qwen3.6:27b at 88% fit)
- Judge (qwen2.5:7b) runs on same host — model-under-test unloaded before judge loads
- Results stored in SQLite (`data/atomics.db`, table: `task_results`, suite: `redblue-*`)
