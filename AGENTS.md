# Stoneburner Agent Guide

Stoneburner (`atomics`) is a model and inference-system evaluation toolkit. Use
it to benchmark models, validate AI-gated applications, compare providers,
measure capacity, and track regressions.

This guide is intentionally general-purpose. Do not assume a specific lab,
customer, box, cloud, or model family.

## Development Basics

- Package manager: `uv`
- Test command: `uv run pytest`
- Focused tests: `uv run pytest tests/test_<area>.py -q`
- Lint: `uv run ruff check .`
- CLI entry point: `uv run atomics ...`
- Python package: `atomics/`
- Committed examples: `qa/examples/`, `profiles/examples/`
- Private/local inputs: `qa/local/`, `profiles/local/`, `.env`

Never commit real IPs, credentials, customer endpoints, unreleased challenge
spoilers, tokens, raw flags, or private profiles. Put those in gitignored local
files and document only sanitized patterns.

## What To Use When

- `atomics provider-test`: verify a provider/model can answer at all.
- `atomics models`: discover available local or gateway-routed models.
- `atomics qa`: run pass/fail fixtures against a model or app-level AI gate.
- `atomics adversarial`: measure resistance to prompt injection, role confusion,
  social engineering, and data exfiltration attempts.
- `atomics sweep`: compare multiple models across the standard eval set.
- `atomics stress`: find throughput, saturation, and VRAM contention limits.
- `atomics soak`: detect long-duration stability, latency, throughput, and error
  drift.
- `atomics scenario`: simulate mixed concurrent workloads sharing inference
  capacity.
- `atomics probe`: analyze live artifacts such as logs, reports, config files, or
  API responses.

## Compatibility Vocabulary

Use explicit labels when evaluating AI-backed applications or challenges:

- `UNTESTED`: no evidence for the model/backend/target combination.
- `FUNCTION_COMPATIBLE`: health, inference, output contracts, and individual
  AI-mediated function checks pass.
- `WALKTHROUGH_COMPATIBLE`: the full user-facing or operator-facing workflow
  passes repeated rounds.
- `TOO_SAFE_FOR_CHAIN`: the model refuses or redacts an intentionally expected
  behavior needed by the evaluated workflow.
- `UNSAFE_GATE_BEHAVIOR`: the model approves unsafe actions, blocks intended safe
  paths, or emits unsafe gate decisions.
- `BROKEN_RUNTIME`: the model times out, fails to serve, emits unparsable output,
  or breaks orchestration.

Do not collapse these into "works" or "does not work." The distinction is the
point of the evaluation.

## Evaluation Rules

1. Start with provider health before deeper evaluation.
2. Keep committed fixtures generic and sanitized.
3. Put real endpoints, credentials, and environment-specific payloads in
   `profiles/local/` or `qa/local/`.
4. For AI-gated applications, test both positive and negative controls.
5. For promotion evidence, prefer repeated rounds. Three rounds is a practical
   default; five or more is better for nondeterministic behavior.
6. Record sanitized evidence: model, backend, fixture suite, pass/fail counts,
   compatibility label, and redacted snippets when useful.
7. Do not treat adversarial resistance as universal goodness. A safer model can
   be incompatible with workflows that intentionally test permissiveness,
   refusal variance, or vulnerable behavior.

## Project Skills

Use these project skills when available:

- `stoneburner-model-qa`: model QA, compatibility labels, fixture selection,
  promotion evidence.
- `stoneburner-target-profiles`: creating sanitized app-level target profiles.
- `stoneburner-inference-env`: consuming or producing the vendor-neutral
  `inference.env` control file.
