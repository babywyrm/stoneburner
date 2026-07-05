# Architecture

This document is the map of Stoneburner (the `atomics` package) for contributors.
It describes the layers, the load-bearing primitives you build on, how the
evaluation suites are structured, and the security model. If you are adding a
feature, read the layer your change belongs to plus the "Primitives" section.

> Status: the codebase is cohesive at the infrastructure level (providers,
> storage, config) but the evaluation suites are being converged onto a shared
> shape. See [Known divergences](#known-divergences-being-converged) before
> copying an existing suite.

---

## Layers

Dependencies point downward only. Nothing below the CLI imports the CLI; storage
never imports eval; providers never import storage.

```
CLI / entry          cli.py, __main__.py
Orchestration        sweep, scenario, qa_runner
Burn loop            core/, tasks/, tiers, hooks
Eval / security      eval/, probe/, archreview/
Load testing         stress, soak, contention, capacity, profiles, regression
Providers            providers/, auth/, model_classes
Storage              storage/
Support / infra      config, paths, secrets, doctor, reporting, exporters,
                     scheduler, inference
```

### Layer responsibilities

| Layer | Owns | Key modules |
|-------|------|-------------|
| CLI | Argument parsing, wiring, Rich output. No business logic that can't be reached another way. | `cli.py` |
| Orchestration | Multi-run/multi-model coordination over the lower layers. | `sweep.py`, `scenario.py`, `qa_runner.py` |
| Burn loop | The continuous token-burn benchmark. | `core/engine.py`, `core/runner.py`, `core/guard.py`, `tasks/` |
| Eval / security | LLM quality and security evaluation suites. | `eval/`, `probe/`, `archreview/` |
| Load testing | Throughput/latency/stability under concurrency. | `stress.py`, `soak.py`, `contention.py`, `capacity.py` |
| Providers | One uniform async interface to every LLM backend. | `providers/base.py` + adapters, `auth/` |
| Storage | SQLite persistence and queries. | `storage/repository.py`, `storage/schema.py` |
| Support | Config, secrets, paths, diagnostics, reporting. | `config.py`, `secrets.py`, `paths.py`, ... |

---

## Primitives (the real public API)

These four modules are what everything else is built on. Treat their signatures
as stable; changing them ripples across the whole tree.

### `providers/base.py` — the provider contract

Every backend implements `BaseProvider`:

```python
class BaseProvider(ABC):
    @property
    def name(self) -> str: ...
    @property
    def default_model(self) -> str | None: ...
    async def generate(self, prompt: str, *, system: str = "", model: str | None = None,
                       max_tokens: int = 1024, thinking: bool | None = None,
                       thinking_budget: int | None = None,
                       temperature: float | None = None) -> ProviderResponse: ...
    async def health_check(self) -> bool: ...
```

`ProviderResponse` carries text, token counts (incl. thinking + cache), latency,
cost, and `tps_basis`. Adapters live in `providers/{claude,openai,bedrock,ollama,
vllm,brain_gateway}.py`. Pricing is centralized in `providers/pricing.py`.

Build providers through the single CLI factory `_make_provider()` — do not write a
new provider-name switch.

### `models.py` — shared domain vocabulary

Pydantic models for the burn/eval domain: `TaskResult`, `RunSummary`,
`TaskCategory`, `TaskStatus`, `BurnTier`. Persisted rows map to `TaskResult`.

### `storage/repository.py` — persistence hub

`MetricsRepository` wraps SQLite. The run lifecycle contract is:

```
create_run(run_id, tier=..., provider=..., model=...)   # once, before saving items
save_*(...)                                              # per item
complete_run(run_id)  (or complete_adversarial_run)     # once, at the end
```

All new persistence goes through `MetricsRepository`, never raw `sqlite3`.
Queries return `list[dict]`; prefer adding a typed accessor over ad-hoc SQL in
callers.

### `eval/judge.py` — LLM-as-judge

`score_response()` / `score_consensus()` produce a `JudgeResult`
(score, label, rationale, parse_failed, criteria_coverage, stdev). Helpers:
`detect_self_judge()` (warn when the model under test is also the judge),
`char_budget_for_tokens()`, `compute_criteria_coverage()`. Adversarial resistance
scoring (`eval/adversarial/scorer.py`) is a parallel judge with an inverted rubric.

---

## The evaluation suites

Five suites share the "fixtures → provider → judge → summary → storage" shape:

| Suite | Fixtures | Runner | Judge | Measures |
|-------|----------|--------|-------|----------|
| `eval` | `EvalFixture` | `eval/runner.py` | quality (`judge.py`) | general answer quality |
| `adversarial` | `AdversarialFixture` | `eval/adversarial/runner.py` | resistance (`scorer.py`) | resistance to manipulation |
| `redblue` | `RedBlueFixture` | `eval/redblue/runner.py` | quality (`judge.py`) | offensive/defensive capability |
| `archreview` | `RepoSpec` + evidence pack | `archreview/runner.py` | objective + reasoning | security-architecture review |
| `probe` | `ProbeTarget` | `probe/runner.py` | quality (`judge.py`) | live-artifact regression |

### How to add a new adversarial fixture suite

1. Create `atomics/eval/adversarial/<name>.py` with a `list[AdversarialFixture]`
   (fields: `id`, `category`, `severity`, `prompt`, `attack_goal`,
   `resistance_criteria`, optional `prior_turns`, `max_output_tokens`).
2. Register it in `atomics/eval/adversarial/__init__.py`: add to `ALL_FIXTURES`
   and add a `GROUP_ALIASES` entry so `--category <name>` works.
3. Add tests mirroring `tests/test_adversarial.py` (registration, unique IDs,
   valid severities).
4. `ALL_FIXTURES` is the single source of truth — the runner, CLI header, and
   progress bar all select via `select_fixtures()`. Never build a parallel list.

### How to add a whole new suite

Follow the run lifecycle contract above. Match the conventions the other suites
are converging on (see next section) rather than the oldest one you find:
a `*Summary` dataclass with a `to_dict()`, a `run_id`, `runs` (not `rounds`) for
multi-pass, `fixture_results` for the item list, and create/complete a parent run
row in storage.

---

## Known divergences (being converged)

New code should follow the target column, not copy whichever suite you opened first.

| Concern | Target convention | Status |
|---------|-------------------|--------|
| Item list field | `fixture_results` | redblue/probe now expose it as an alias for `results` |
| Multi-pass arg | `runs` | archreview accepts `--runs` as an alias for `--rounds` |
| JSON export | `Summary.to_dict()` + `--json-out` | done — all five suites |
| Parent run row | `create_run()` + `complete_*_run()` | done — all suites create + finalize a run row |
| Stats helpers | one shared `stats` module | done — `atomics/stats.py` |
| Provider build | `_make_provider()` | done — single factory |

---

## Security model

Stoneburner is a **local, single-operator CLI** and trusts its own config, but it
handles API keys and can send data to LLM providers. Rules for contributors:

- **Secrets never get logged or auto-printed.** `secrets.py` logs key *names*
  only. Any command that surfaces a secret value must require an explicit,
  visible opt-in (e.g. a `--show` flag), never print by default.
- **Resolution order is env → `.env` → OS keychain** (`config.load_settings`).
  Do not add new secret sources without documenting the order.
- **Prompts/responses are persisted** to SQLite and included in `export`. Do not
  put credentials in fixtures. Treat `task_results.prompt/response` as sensitive.
- **YAML is always `safe_load`.** No `pickle`, no `eval`/`exec`, no `yaml.load`.
- **SQL uses bound parameters.** Never string-interpolate user input into SQL.
- **`post_run_hook` runs a shell command from config** — this is a trusted-user
  feature; keep it opt-in and documented.
- **`archreview`/`probe` send local files/repos to LLM providers** by design;
  keep that behavior explicit and visible in output.

---

## Testing & CI

- Tests live in `tests/`, mock providers/httpx, and must not hit the network by
  default. The one live test (`test_calibration.py`) is gated behind
  `ATOMICS_LIVE_JUDGE=1`.
- CI (`.github/workflows/ci.yml`) runs ruff + pytest on Linux/macOS across
  Python 3.11–3.13.
- Run the suite before every commit: `uv run pytest -q`.

---

## Module split candidates (tech debt, tracked)

Large modules that are future split targets: `cli.py` (~3500 lines; per-command
modules), `storage/repository.py` (~780; per-suite persistence mixins). Orphaned
utilities to wire in or remove: `inference.py`, `workers/bridge.py`.
