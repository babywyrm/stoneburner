# Stoneburner Agent Guide

Stoneburner (`atomics`) is a model and inference-system evaluation toolkit. Use
it to benchmark models, validate AI-gated applications, compare providers,
measure capacity, and track regressions.

This guide is intentionally general-purpose. Do not assume a specific lab,
customer, box, cloud, or model family.

## Development Basics

- Package manager: `uv` (`uv sync --all-extras` from a clone)
- Test command: `uv run pytest`
- Focused tests: `uv run pytest tests/test_<area>.py -q`
- Lint and format: `uv run ruff check .` and
  `uv run ruff format --check atomics/ tests/ scripts/`
- Typecheck: `uv run mypy atomics/`
- CLI entry point: `uv run atomics ...`
- Python package: `atomics/`
- Committed examples: `qa/examples/`, `profiles/examples/`
- Private/local inputs: `qa/local/`, `profiles/local/`, `.env`

CI runs exactly the checks in `CONTRIBUTING.md`; run them before claiming done.
`ARCHITECTURE.md` is the layer map, the primitives, and the security model.
Read the layer your change belongs to before editing.

Never commit real IPs, hostnames, credentials, customer endpoints, unreleased
challenge spoilers, tokens, raw flags, or private profiles. Put those in
gitignored local files and document only sanitized patterns (`localhost`,
`example`, `<host>`).

## Code Conventions

- Build providers through `providers.factory.make_provider()`; command modules
  use `commands.common._make_provider()`. No new provider-name switch.
- Persist through `MetricsRepository` (`storage/repository/`), never raw
  `sqlite3`. SQL uses bound parameters; YAML uses `safe_load`.
- A schema change bumps `SCHEMA_VERSION` and migrates in place; test it against
  a populated database, including rows other tables reference.
- Judges share `JUDGE_MAX_TOKENS` from `eval/judge.py`. Do not add a per-suite
  cap.
- New fixtures register in their suite's single list (for adversarial,
  `ALL_FIXTURES`); never build a parallel list.
- The API server never imports `commands/`. The MCP server, REPL, and workers
  are HTTP clients of the API; new MCP tools mean new API endpoints first.
- Import implementations, not the compatibility shims at the package root.

## Testing Conventions

- Tests mock providers and HTTP and must not hit the network by default.
- An autouse fixture isolates the database, data directory, and keyring.
  Do not point tests at `data/atomics.db` or the real keychain.
- API tests use the `client` fixture or
  `TestClient(app, base_url="http://127.0.0.1")`. `--no-auth` rejects the
  default `testserver` host.
- Keep waits short: patch sleeps and durations instead of sleeping for real.
  Any test over 60 seconds fails (`pytest-timeout`).
- A bug fix starts with a failing test.

## What To Use When

- `atomics doctor`: check Python, database, keys, and backend reachability.
- `atomics provider-test`: verify a provider/model can answer at all.
- `atomics models`: discover available local or gateway-routed models.
- `atomics battery`: run a named security job that composes the suites below.
  Start here instead of the full adversarial suite.
- `atomics qa`: run pass/fail fixtures against a model or app-level AI gate.
- `atomics adversarial`: measure resistance to prompt injection, role confusion,
  social engineering, data exfiltration, MCP/agentic manipulation (tool-call
  compliance, authority fabrication, breakglass injection, context poisoning,
  agent-loop escape), and tool-use safety attempts. Use `--category mcp` or
  `--category tool_safety` to focus. See `docs/LEADERBOARD.md` for latest results.
- `atomics toolcall`: measure the agent gap — refused in prose, then emitted
  the call when given tools. Calls are never executed.
- `atomics refusal`: measure over- and under-refusal.
- `atomics codereview`: measure planted-vulnerability detection and false
  positives.
- `atomics sweep`: compare multiple models across the standard eval set.
- `atomics stress`: find throughput, saturation, and VRAM contention limits.
- `atomics soak`: detect long-duration stability, latency, throughput, and error
  drift.
- `atomics scenario`: simulate mixed concurrent workloads sharing inference
  capacity.
- `atomics probe`: analyze live artifacts such as logs, reports, config files, or
  API responses.
- `atomics prompts` and `atomics --show-prompt <command>`: show exactly what the
  model and judge are sent.
- `atomics server`, `atomics mcp`, `atomics repl`: the HTTP API, an MCP proxy
  over it, and a human prompt over it. See `docs/API_SERVER.md`.

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

## Run Integrity

Suites report integrity as `complete`, `partial`, `unscored` (nothing scored,
model and judge reachable), or `infrastructure_invalid` (model or judge
outage). Anything but `complete` exits nonzero; `--allow-partial` changes only
the exit code, not the stored integrity. A headline of
`n/a (scored/total scored)` means the score is withheld, not zero. Only
`complete` runs are promotion evidence.

## Evaluation Rules

1. Start with provider health before deeper evaluation.
2. Keep committed fixtures generic and sanitized.
3. Put real endpoints, credentials, and environment-specific payloads in
   `profiles/local/` or `qa/local/`.
4. For AI-gated applications, test both positive and negative controls.
5. For promotion evidence, prefer repeated rounds. Three rounds is a practical
   default; five or more is better for nondeterministic behavior.
6. Use a judge that is not the model under test, and name it next to any
   published score.
7. Record sanitized evidence: model, backend, fixture suite, pass/fail counts,
   integrity, compatibility label, and redacted snippets when useful.
8. Do not treat adversarial resistance as universal goodness. A safer model can
   be incompatible with workflows that intentionally test permissiveness,
   refusal variance, or vulnerable behavior.

## Project Skills

Use these project skills when available:

- `stoneburner-model-qa`: model QA, compatibility labels, run integrity,
  fixture selection, promotion evidence.
- `stoneburner-target-profiles`: creating sanitized app-level target profiles.
- `stoneburner-inference-env`: consuming or producing the vendor-neutral
  `inference.env` control file.
