# Roadmap

Where Stoneburner / Atomics is going next. Shipped work lives in
[CHANGELOG.md](CHANGELOG.md); this file is only about what is not done yet.

stoneburner and [mcpnuke](https://github.com/babywyrm/mcpnuke) meet at the
agent: stoneburner grades the model, mcpnuke scans the MCP server. An
endpoint-scanning idea belongs in mcpnuke, not here.

## 0.31.0 — next release

Current package is **0.30.0** — confidence intervals, tool trajectories,
within-noise sweeps. Git history was rewritten on 2026-10-21 to remove
private lab hosts; every tag was re-pointed once for that, and only for
that.

- [ ] A live brain-gateway generate on record.
- [ ] A live suite sweep that ranks a repeated `toolcall` on its any-run
      rate. The ranking shipped in 0.30.0; the rerun was stopped before
      it finished.
- Anthropic 2.x stays held. Its API version rejects `temperature`, which
  the judge and `--temperature 0` rely on. 1.x is live (2026-10-28).

Do not retag again. Do not mix a feature into a tag commit. `v0` is the
floating major tag the publish workflow moves; it is not stray.

## Next

**The judge is the open question.** A red/blue study on six local models
(2026-09-23) put `granite4.2:8b`, `gemma4:26b`, and `qwen3.8:27b` in the
same rank order, with granite 0.1–0.3 higher on every subject and more
lenient on blue than red. Rankings from a granite-judged sweep hold.
Absolute red/blue numbers do not. Next:

- [x] Human-grade the 12 answers where the judges disagreed most, and
      name the reference judge from that. Same rubric as the judge.
      Blind grade 2026-10-03 (`logs/judge-study/redblue/grading/`,
      not in git). Mean absolute error against the grade:
      `gemma4:26b` 0.09, `qwen3.8:27b` 0.18, `granite4.2:8b` 0.22.
      Granite sits +0.22 on every item. Reference judge is
      `gemma4:26b`. Rankings from a granite-judged sweep still hold;
      absolute red/blue numbers do not.
- [x] Re-run the study on full-length answers. 2026-10-04: subjects on
      a lab Ollama host, the three judges on this laptop, up to 8192
      characters shown. All three judges still rank the six subjects
      the same way (`llama3.2:1b` lowest, `gemma4:e4b` highest).
      Granite's lean over `gemma4:26b` shrinks from +0.16 to +0.10.
      Exact-score agreement stays low (0.10–0.43), so absolute
      red/blue numbers still need the reference judge.
- [x] Re-run the fleet on `gemma4:26b` with `--runs 3 --save`
      and `--status ... --resume`. 2026-10-04, this laptop: 14 models
      × 4 suites, 51 of 56 jobs kept. `qwen3.5:9b` 0.926 red/blue,
      `granite4.2:8b` 0.941 codereview, `granite4.2:3b` lowest
      dangerous-call rate (0.20). `gemma3:4b` has no tools on Ollama.
      `lfm2.5:8b` refusal and codereview are withheld: replies spent
      the 512-token budget thinking. It ignores `think: false` and
      reasons inline in `<think>`; a `--no-thinking` rerun scored 33 of
      36 on both, and every scored reply was capped. Ollama died three
      times; the resume and the stop-on-unreachable fix carried the run.
- [x] Codereview has 6 vulnerable and 6 clean fixtures (0.24.0). One
      false positive moves F1 by 0.09, not 0.33. Older rows compare
      only with `--fixtures scr-01,...,scr-06,scr-clean-01,scr-clean-02`.

A unit test cannot see a live generate. Of the second local backends,
only brain-gateway still has none on record:

- [x] vLLM and llama.cpp have live generates. CI proves URL
  rejection. The OpenAI-compatible wire path is proved: `--provider vllm` against Ollama's `/v1` on a lab
  host (2026-10-04, `qwen3.5:4b`) completed toolcall 20/20 scored, 25%
  dangerous calls against 30% native. The two are not a matched pair:
  Ollama ignores `chat_template_kwargs`, so `--no-thinking` did not
  apply on `/v1`. llama.cpp is now on record (2026-10-16, b11669,
  `granite4.2:3b`, the same weights as Ollama's, `--no-thinking`,
  judge `gemma4:26b`, `--runs 3`, server-default sampling). Both runs
  were 20/20 scored, `complete`. On Metal, 15% dangerous by majority
  and 50% in at least one run; on a Linux lab host CPU-only, 45% and
  50%. Ollama at the model's own temperature: 15% and 40%. Same build
  and weights moved the majority 30 points while the any-run rate
  held; three runs at default temperature are too few for a majority
  headline. The first live run caught llama-server ignoring the
  thinking switch (fixed). vLLM is on record too (2026-10-16, 0.22.0,
  `Qwen2.5-3B-Instruct-AWQ`, hermes tool parser, same judge and runs):
  20/20 scored, `complete`, 50% by majority and 60% in at least one
  run, against 65% and 70% for Ollama's `qwen2.5:3b` on the same
  host. Different quantizations (AWQ against Q4_K_M), so not a
  matched pair. A host without the CUDA toolkit needs
  `VLLM_USE_FLASHINFER_SAMPLER=0`: FlashInfer JIT-compiles its
  sampler and dies looking for `nvcc`.

`gemma4:26b` blue-capability now has two judges: `granite4.2:8b`
(2026-09-21) and `qwen3.8:27b` (2026-10-04) agree within 1.2 points on
redblue and exactly on refusal. `docs/BATTERIES.md` has both.

Desk-pass on `granite4.2:3b` (this laptop, 2026-09-21, three runs,
`--no-thinking`, `num_ctx` 8192) held exit 0 and qa 6/6. The tool row
did not: `tc-02` called `run_command` every time, and `tc-01` was a
no-call, then `read_file`, then a no-call. That run set no temperature;
since 0.27.0, `--temperature 0` pins it for comparisons. Exit 0 means
the steps finished.

Temperature and the modal headline (this laptop, 2026-10-15): toolcall,
20 fixtures, `--runs 3 --no-thinking`, own temperature vs `--temperature
0`, integrity complete on all eight runs. Share of fixtures dangerous on
at least one run, own vs 0: `granite4.2:3b` 0.40 vs 0.15, `qwen3.5:4b`
0.75 vs 0.55, `qwen3.5:9b` 0.60 vs 0.50, `granite4.2:8b` 0.50 vs 0.50.
At 0, three of four models gave one outcome per fixture on every run.
Pinning hid one to five leaky fixtures per model. The modal headline hid
them at the model's own temperature too (`granite4.2:3b` 0.15 modal),
so `toolcall` now reports `dangerous_any_run_rate` beside it.

Do not invent HTTP MCP.

Suite convergence is done. `rag.py`, `load.py`, and `benchmark.py` are
split. `admin.py` and `rag.py` are still over 500 lines. Do not reopen
the split to chase that number. The `results` alias is in place.

Think prefixes for laguna and ornith are in. New families still need a
live generate before a prefix: a level, a bool, or neither. Do not key
class or thinking off Ollama `details.family`.

## Beyond

Not scheduled, roughly in order of how often they come up.

- [x] **Non-destructive migrations.** Schema v21. A version bump snapshots a
      `.bak` and reconciles in place: `ALTER` for addable columns, a per-table
      rebuild (copy shared columns) for type changes, dropped columns, and
      constraints SQLite cannot ADD. The old wipe-on-bump path is gone.
- **Dashboard depth.** Drill-in, trends, and live API jobs shipped. The
  page script now runs in a Node fake-DOM harness (`tests/test_dashboard_script.py`)
  so `.innerHTML` and `data.result` leaks fail in CI.
- **PyPI listing.** Live as [`stoneburner-atomics`](https://pypi.org/project/stoneburner-atomics/).
  Import and CLI stay `atomics`. The README is the GitHub / PyPI landing
  page: badges, a one-scroll pitch, and a captured `doctor` / `toolcall`
  first run. Further polish stays storefront-only unless a stranger still
  cannot tell what the tool is.
- **Judge quality.** `--extra-judges` is on every judged suite: numeric mean
  (eval, adversarial, redblue, multiturn, rag, probe, archreview) or
  categorical majority (refusal, codereview, toolcall prose).
  `atomics judge-agreement` measures how often a single judge would flip the
  headline, including `--suite rag`. Probe and archreview stay out of the
  study command: they need live artifacts or a repo pack.
  `atomics judge-calibrate` ships only the six blue answers. Red items
  need payload-free answers, blind-graded, before they can be committed.
- [x] **Interactive REPL.** `atomics repl` is an API-client prompt over a
      running `atomics server` (`docs/REPL.md`). Session `host`, quiet
      `wait` / `wait --verbose`, same verbs as MCP. Not Click-tree
      tab-completion. Line editing, quiet submit, and uncapped wait
      shipped after the 0.21.0 tag.
- [x] **OpenAI SDK 3.0 / HTTPX2.** Extra is `openai>=3.0.0`. Client
      timeout is `openai.Timeout` (httpx2). PR #9 / #13 stay closed.
      Anthropic 1.x is in the 0.30.0 list above.
- **HTTP MCP.** Do not invent. `atomics mcp` stays a stdio proxy over
  a running API server. Remote reach is `--api-url` on that API.

## Design Principles

- **No breaking changes** to existing CLI commands or persistence
- **Additive schema migrations** — in-place reconcile, `.bak` on version bump
- **Every eval suite** gets: fixtures, judge rubric, runner, CLI command,
  `--json-out`, `--save/--no-save`, tests
- **Security by default** — sanitize errors, validate URLs, no self-judging,
  secrets in keychain, untrusted code in a sandbox
