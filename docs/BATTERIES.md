# Security batteries

Named jobs. They compose suites that already exist. They are not a
13th suite and they are not the full 72-fixture adversarial run.

Two axes stay separate: **capability** (can it do the work) and
**resilience** (does it resist manipulation). Tool calls are a third
channel. Score them separately.

    uv run atomics battery list
    uv run atomics battery show desk-pass -m lfm2.5:8b
    uv run atomics battery run desk-pass -p ollama -m granite4.2:3b --no-save

`show` prints copy-pasteable commands. `run` executes those steps in
order and stops on the first nonzero exit unless `--keep-going`.

Same `-p` as every other suite: `ollama`, `vllm`, `brain-gateway`, `openai`,
`claude`, `bedrock`, `groq`, `together`, `gemini`, `llamacpp`.

    # local
    uv run atomics battery show desk-pass -p ollama -m lfm2.5:8b

    # lab — omit -m so inference.env fills when INFERENCE_BACKEND matches -p
    uv run atomics battery show agent-gate -p vllm --vllm-host http://127.0.0.1:8000/v1
    uv run atomics battery show blue-capability -p brain-gateway --judge-provider ollama --judge-model granite4.2:8b

    # frontier — always pass --budget
    uv run atomics battery show agent-gate -p openai -m gpt-4.1 --budget 5
    uv run atomics battery show blue-capability -p claude -m claude-sonnet-4-6 --judge-provider claude --judge-model claude-sonnet-4-6 --budget 5

Raw `qa --file` is Ollama HTTP. Other providers skip that step unless
`--profile` points at an app-level gate.

Start with `provider-test`. Use `--no-thinking` on short fixtures so
the visible answer is not eaten by CoT. Promotion evidence is `--runs 3`
on the judged suites, not a bigger fixture list.

## desk-pass (cheap, no judge)

Health + app-gate patterns + `tc-01,tc-02` on the tool channel.

Label hint: `FUNCTION_COMPATIBLE` if health and qa hold. Tool-capable
must be honest: a skipped probe is not resistance.

Not a pass: walkthrough, overnight, or a resilience percentage.

## blue-capability (medium, judge required)

`redblue --mode blue`, `codereview`, refusal benign ids `rc-b01`–`rc-b06`.

When: SOC / analyst copilot.

Not a pass: resilience. A high blue score can still be `DANGEROUS` on tools.

## red-capability (medium, judge required)

`redblue --mode red` plus `rc-b05` (authorized scan knowledge).

When: authorized pentest copilot. Conceptual only.

Not a pass: permission to emit operational malware. Harmful operational
prompts stay on `refusal` as should-refuse.

## agent-gate (medium, judge on the prose step)

`adversarial --category mcp,tool_safety,tool_desc_injection` (21 fixtures)
and `toolcall --category direct --channel tools`.

Score prose and tools separately. Probe skip ≠ refusal.

## threat-model (medium, judge required)

`redblue --fixtures rb-b04` (STRIDE on an agent) and
`adversarial --category agentic`. Optional: `archreview --repo juice-shop`
when the repo pack is present.

Not a pass: a complete threat-model practice. `rb-b04` is one prompt.
Deeper STRIDE fixtures are a later increment.

## What not to run by default

- Full `atomics adversarial` (72)
- Full `atomics toolcall` 20 fixtures × `--runs 3` on a 27b overnight
- 125b MLX on a laptop
- Mixing a battery result into a release tag
