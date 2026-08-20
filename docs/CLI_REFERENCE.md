# CLI Reference

Full command reference for `atomics`. See also [QUICKSTART.md](../QUICKSTART.md) for recipe-first usage.

## Core Commands

| Command | Description |
|---------|-------------|
| `atomics run` | Start the benchmarking loop (continuous or bounded) |
| `atomics run --tier mega -n 10` | Run 10 mega-tier tasks |
| `atomics run --provider bedrock` | Use AWS Bedrock instead of Claude API |
| `atomics run --provider openai` | Use OpenAI / Codex |
| `atomics run --provider ollama` | Use local Ollama inference |
| `atomics run --provider ollama --ollama-host http://localhost:11434` | Use local Ollama |
| `atomics run --provider brain-gateway` | Use a brain-gateway |
| `atomics run --provider brain-gateway --gateway-url http://localhost:30080` | Use a remote brain-gateway |
| `atomics run --thinking` | Enable thinking/reasoning mode for capable models |
| `atomics run --no-thinking` | Force thinking off (A/B comparison) |
| `atomics run --thinking-budget 20000` | Set max thinking tokens (provider-specific default otherwise) |
| `atomics provider-test -p openai -m gpt-5.6-sol --effort high` | Shared effort dial (none/low/medium/high/xhigh/max; xl and ultra aliases) |
| `atomics provider-test -p openai -m gpt-5.6-sol --effort max --reasoning-mode pro` | OpenAI pro mode via the Responses API |
| `atomics eval --provider claude -m claude-opus-4-6 --effort high` | Claude adaptive thinking + output_config.effort |
| `atomics run -b 5.0` | Run with $5 budget cap |
| `atomics run -i 10` | Override interval to 10 seconds |
| `atomics compare` | Compare providers side-by-side (cost, latency, tokens) |
| `atomics compare --by model` | Compare individual models across providers |
| `atomics compare --output results.json` | Write comparison JSON alongside table |
| `atomics report` | Display usage reports and trends |
| `atomics tiers` | Show available burn tiers and their profiles |

## Provider Management

| Command | Description |
|---------|-------------|
| `atomics provider-test` | Health check the configured provider |
| `atomics provider-test -p bedrock` | Health check Bedrock |
| `atomics provider-test -p openai` | Health check OpenAI |
| `atomics provider-test -p ollama` | Health check Ollama |
| `atomics provider-test -p ollama -m qwen3.8:27b --no-thinking` | Smoke-test a thinking model without burning the visible answer |
| `atomics provider-test -p brain-gateway` | Health check brain-gateway |
| `atomics provider-test -p groq` | Health check Groq |
| `atomics provider-test -p together` | Health check Together AI |
| `atomics provider-test -p gemini` | Health check Gemini |
| `atomics provider-test -p llamacpp` | Health check llama.cpp |
| `atomics models` | List available models on Ollama host with class/thinking annotations |
| `atomics models --provider vllm` | List models on a vLLM-compatible endpoint |
| `atomics doctor` | Check installation health, config, and `inference.env` (never prints the API key). Prints one next command when the check is healthy. |

## Evaluation Suites

| Command | Description |
|---------|-------------|
| `atomics eval` | Run evaluation suite against a provider |
| `atomics eval --verbose` | Print the full prompt, model reply, thinking, and judge rationale |
| `atomics eval --fixtures ev-19` | Run a fixture subset for a fast spot-check |
| `atomics eval --extra-judges ollama:mistral:7b` | Multi-judge consensus scoring |
| `atomics judge-agreement --suite refusal --judges ollama:a,ollama:b` | Generate once, judge N times; report pairwise agreement and majority-flip rate |
| `atomics judge-agreement --suite rag --judges ollama:a,ollama:b` | Same study on the RAG rubric (numeric mean / flip) |
| `atomics adversarial` | Adversarial resilience eval — resistance to manipulation (72 fixtures) |
| `atomics adversarial --category tool_desc_injection` | Run one suite/group |
| `atomics adversarial --runs 3` | Variance-aware scoring (mean ± stddev). Live `run 2/3` lines print each pass so a 1/3 comply is visible |
| `atomics adversarial --compare mistral-small:24b` | Run a second model, print per-fixture diff |
| `atomics adversarial --json-out run.json` | Write full per-fixture results as JSON |
| `atomics adversarial --fail-on-resilience 60` | CI gate — non-zero exit if resilience < 60% |
| `atomics toolcall` | Tool-call divergence — refuses in prose, complies via function call (20 fixtures). Calls are never executed |
| `atomics toolcall --extra-judges ollama:mistral:7b` | Multi-judge consensus on the prose channel only |
| `atomics toolcall --category exfil --verbose` | Run one category and print the arguments the model asked for |
| `atomics toolcall --channel tools` | Tool channel only — deterministic, needs no judge |
| `atomics toolcall --channel tools --runs 3` | One-box first run. No judge. Divergence not measured. |
| `atomics toolcall --runs 3` | Three passes per fixture, modal outcome reported. Live `run 2/3` lines print each pass so a 1/3 leak is visible |
| `atomics -v toolcall --judge-provider ollama --judge-model qwen2.5:14b` | Required for channel divergence. Live lines show `prose=resisted` + `DANGEROUS` when the model refused in chat and still called the tool |
| `atomics toolcall --no-skip-incapable` | Non-zero exit if the model cannot emit tool calls (for sweeps) |
| `atomics toolcall --no-thinking` | Force thinking off on the prose channel (same grammar as red/blue) |
| `atomics refusal` | Refusal-calibration eval — over- vs under-refusal |
| `atomics refusal --extra-judges ollama:mistral:7b` | Majority-vote classification; ties are unresolved |
| `atomics refusal --no-thinking` | Force thinking off so reasoning models do not burn the fixture budget |
| `atomics codereview` | Secure-code-review eval — planted-vuln detection + false positives |
| `atomics codereview --extra-judges ollama:mistral:7b` | Majority-vote verdict; ties are unresolved |
| `atomics codereview --no-thinking` | Force thinking off so the review is visible text, not hidden reasoning |
| `atomics redblue --mode all` | Red/blue security capability eval (offensive + defensive) |
| `atomics redblue --extra-judges ollama:mistral:7b` | Multi-judge mean ± stdev on the quality score |
| `atomics redblue --runs 3 --json-out rb.json` | Variance-aware capability scoring + JSON export. Live `run 2/3` lines print each pass |
| `atomics redblue --no-thinking --runs 3` | Multi-run capability score with thinking forced off |
| `atomics redblue --max-output-tokens 3072` | Floor the visible-answer budget (fixtures default to 2048) |
| `atomics probe --probes-file probes.yaml` | Live ecosystem probe against real artifacts |
| `atomics probe --extra-judges ollama:mistral:7b` | Multi-judge mean ± stdev on the quality score |
| `atomics archreview --repo juice-shop --models qwen3.5:4b` | Security-architecture repo benchmark |
| `atomics archreview --extra-judges ollama:mistral:7b` | Multi-judge mean on the reasoning score |
| `atomics archreview --tier local --max-output-tokens 512` | Practical local-GPU repo review |
| `atomics archreview --tier wide --rounds 3` | Broader evidence pack with stability reporting |
| `atomics archreview --tier expanded --rounds 3` | Largest pack for large-context/cloud backends |
| `atomics multiturn` | Multi-turn conversation eval — context retention, coherence, contradiction detection, persona drift, long-context retention, tool-use chaining, and security scenarios |
| `atomics multiturn --effort high` | Shared reasoning effort on each turn |
| `atomics multiturn --extra-judges ollama:mistral:7b` | Multi-judge consensus on the conversation score only |
| `atomics multiturn --fixtures mt-eval-01,mt-eval-05` | Run a fixture subset |
| `atomics codegen` | Code generation eval (functional correctness) |
| `atomics codegen --fixtures cg-01,cg-05` | Run a fixture subset |
| `atomics advisor` | Cost optimization recommendations from benchmark history |
| `atomics advisor --min-quality 0.9` | Higher quality threshold |
| `atomics advisor --current-model claude-sonnet-4-6` | Optimize from a specific model |
| `atomics eval --fixtures ml-01,ml-05` | Run multilingual fixtures (8 languages) |
| `atomics rag --extra-judges ollama:mistral:7b` | Multi-judge mean ± stdev on the RAG score |
| `atomics rag` | RAG pipeline evaluation — grounding, faithfulness, abstention (uses hand-crafted or retrieved context) |
| `atomics rag --effort high` | Shared reasoning effort on the grounded answer |
| `atomics rag --index ./index.vec` | Run RAG fixtures against chunks retrieved from a sqlite-vec index |
| `atomics rag-index ./docs --db ./index.vec` | Build a sqlite-vec index from a directory of documents |
| `atomics rag-retrieval --index ./index.vec --gold gold.json` | Report recall@k, precision@k, MRR, and nDCG@k |
| `atomics rag --fixtures rag-05,rag-12` | Run a fixture subset |
| `atomics rag --json-out rag.json` | Write results as JSON |
| `atomics sweep` | Multi-model eval sweep with ranked comparison |
| `atomics sweep --verbose` | Print the full prompt, model reply, thinking, and judge rationale per fixture |
| `atomics sweep --suites redblue,refusal,toolcall,codereview --runs 3 --no-thinking --models-from ollama --status sweep.status.json --log sweep.log` | Overnight multi-suite driver: status file + detachable log. Toolcall uses `--no-skip-incapable`. |

Real retrieval (`rag-index`, `rag --index`, `rag-retrieval`) requires the optional extra: `uv pip install "stoneburner-atomics[rag]"` (or `uv sync --extra rag` from a clone).

### Capping eval spend

Every command that runs an eval suite accepts `--budget`, a dollar ceiling for
the whole run — the nine suite commands plus `sweep`, `archreview`, and `probe`:

| Command | Description |
|---------|-------------|
| `atomics adversarial --budget 5.00` | Stop once the run has spent $5 |
| `atomics eval --extra-judges claude:claude-sonnet-4-6 --budget 2.50` | One ceiling covering the model *and* every judge |
| `atomics sweep --models a,b,c --budget 10.00` | $10 for the whole sweep, not $10 per model |

The ceiling is shared, not per-provider: the model under test and each judge
draw from the same $5. This matters most with `--runs`, `--extra-judges`, and
`sweep`/`archreview`, where each addition is another full pass over the
fixtures.

There is no default ceiling on the CLI — omit `--budget` and nothing changes
about how your runs behave today. The API server takes the opposite default and
always meters, since its callers are remote; see
[API_SERVER.md](API_SERVER.md#spend-ceiling).

A run that reaches the ceiling stops and reports what it spent. Hitting a
per-minute rate limit is waited out instead, since that clears on its own.

**`--budget` does nothing for free providers.** Ollama, vLLM, and llama.cpp
report `$0.00` per call, so a dollar ceiling is never reached no matter how long
the run goes — there is nothing to bill. `--budget 0.01` against a local model
gives you a complete run, not an immediate stop. The flag is meaningful for
paid providers (Claude, OpenAI, Bedrock, Gemini, Groq, Together) and for mixed
runs such as a local model judged by a paid one.

## Load Testing

| Command | Description |
|---------|-------------|
| `atomics stress` | Ramp concurrency to find GPU saturation point |
| `atomics stress --models a,b` | Multi-model VRAM contention — solo baseline then simultaneous |
| `atomics soak` | Long-duration stability test with drift analysis |
| `atomics soak --save-baseline NAME` | Save run metrics as named baseline |
| `atomics soak --compare-baseline NAME` | Compare run against baseline |
| `atomics soak --think-time 5` | Simulate realistic user pauses between requests |
| `atomics baselines` | List all saved soak baselines |
| `atomics scenario` | Mixed-workload simulation with SLA and interference scoring |
| `atomics scenario --ramp 10` | Gradual worker start over 10s instead of all at t=0 |
| `atomics capacity` | Project user load capacity from stress data |
| `atomics labcompare --host a=URL --host b=URL --models m` | Compare two inference hosts |
| `atomics labcompare --dimensions throughput --prompts 5` | Throughput-only bench |

## QA & Validation

| Command | Description |
|---------|-------------|
| `atomics qa --file suite.yaml` | Fire fixture prompts, check pass/fail patterns |
| `atomics qa --file suite.yaml --profile profiles/local/gate.yaml` | Test app-level AI gate |
| `atomics qa --fail-fast` | Stop at first FAIL or ERROR |

## Scheduling

| Command | Description |
|---------|-------------|
| `atomics schedule` | Generate scheduler configs |
| `atomics schedule --install` | Install schedule on this system |
| `atomics schedule --uninstall` | Remove installed schedule |
| `atomics schedule-status` | Show installed schedules and OS health |

## Data & Auth

| Command | Description |
|---------|-------------|
| `atomics export` | Export benchmark data (CSV, JSON) for any suite |
| `atomics export --suite stress` | Export stress test history |
| `atomics export --suite sweep -o out.jsonl` | Export sweep results to file |
| `atomics export --suite adversarial` | Export adversarial results |
| `atomics export --suite all --format csv -o all.csv` | Export all suites as CSV |
| `atomics secrets set ANTHROPIC_API_KEY` | Store an API key in the OS keychain |
| `atomics login` | OAuth/OIDC login (browser or device code) |
| `atomics logout` | Clear cached OAuth tokens |
| `atomics whoami` | Show current auth mode and identity |
| `atomics completion` | Generate shell completion scripts |

## API Server

| Command | Description |
|---------|-------------|
| `atomics server` | Run the atomics HTTP API server |
| `atomics server --no-auth` | Disable API key authentication. Refused unless `--host` is loopback |
| `atomics server --api-key KEY` | Allow one submitter API key (repeatable) |
| `atomics server --worker-api-key KEY` | Allow one worker-only API key (repeatable). Without it, workers share `--api-key` |
| `atomics server --host 0.0.0.0 --port 8080` | Bind to all interfaces on port 8080 (requires `--api-key`) |
| `atomics server --log-level debug` | Verbose uvicorn logging |
| `atomics server --worker-absent-after N` | Seconds of worker silence before it is marked offline (default: 120) |
| `atomics server --with-dashboard` | Serve an optional web dashboard at `/dashboard` (default: off) |
| `atomics server --db-path PATH` | SQLite database path (default: atomics state directory) |
| `atomics worker` | Start a distributed worker (polls coordinator for tasks) |
| `atomics worker-npm` | Start a Node.js worker bridge for distributed task execution |
| `atomics distributed run` | Submit a distributed run (split, fleet, or full mode) |
| `atomics distributed status JOB_ID` | Poll distributed run status |

## MCP Server

Exposes atomics to LLM agents over the Model Context Protocol, as a proxy over a
running API server. Requires the `[mcp]` extra. See
[MCP_SERVER.md](MCP_SERVER.md).

| Command | Description |
|---------|-------------|
| `atomics mcp` | Serve atomics over MCP on stdio, proxying `http://127.0.0.1:8000` |
| `atomics mcp --api-url URL` | API server to proxy (or `ATOMICS_API_URL`) |
| `atomics mcp --api-key KEY` | Client API key, sent as `X-API-Key` (or `ATOMICS_API_KEY`) |

Serves on stdio only: the process holds a spend-authorized API key and no
MCP-layer credential would guard a listening port. To reach a remote atomics,
point `--api-url` at its authenticated API server and run this locally.

Tools: `health`, `list_models`, `list_jobs`, `get_job`, `get_run`, `compare`,
`recent_runs`, `trends` (read-only), `provider_test` (fixed 2+2 probe), plus
`submit_run`, `submit_eval`, `submit_sweep`, `submit_stress`, and
`submit_soak`, which spend tokens and return a job id to poll until
`status` is `completed`. `submit_eval` suites: `accuracy`, `rag`,
`multiturn`, `adversarial`, `codegen`, `refusal`, `redblue`, `toolcall`,
`codereview`. `submit_sweep` uses `eval` (not `accuracy`) plus `redblue`,
`refusal`, `toolcall`, `codereview`; budget is required; at most 8 models
and 3 runs. `submit_stress` / `submit_soak` require a budget; stress is
c≤8 and phase ≤15s; soak duration is 30–300 seconds.

## atomics worker

Start a worker process that registers with the coordinator, heartbeats, polls for task assignments, executes them, and submits results.

| Option | Description |
|--------|-------------|
| `--coordinator URL` | Coordinator base URL (default: `http://127.0.0.1:8000`) |
| `--api-key KEY` | Worker API key (or `ATOMICS_WORKER_API_KEY`) |
| `--label KEY=VALUE` | Worker label (repeatable) |
| `--endpoint URL` | Optional push endpoint URL for this worker |
| `--heartbeat-interval N` | Heartbeat interval in seconds (default: 30) |
| `--provider`, `-p` | Provider used by this worker (default: `ollama`) |
| `--model`, `-m` | Model override for this worker |
| `--host`, `-h` | Provider host/URL override (e.g. `http://localhost:30080`) |

`--host` applies to whichever provider the worker selected, including `vllm`.

If you raise `--heartbeat-interval`, raise the coordinator's
`--worker-absent-after` to match: a worker silent for longer than that window is
marked offline and its pinned fleet work fails. The default 120 seconds allows
roughly four missed heartbeats at the default 30-second interval.

Both examples read the key from the environment (`export ATOMICS_WORKER_API_KEY=...`)
rather than inlining it, keeping keys out of shell history.

```bash
uv run atomics worker --label gpu=1
uv run atomics worker --provider brain-gateway --host http://localhost:30080 --model qwen3:4b
uv run atomics worker --provider vllm --host http://localhost:8000/v1 --label gpu=4090
```

## atomics worker-npm

Start a Node.js worker that joins the distributed pool via the JSON-over-stdin bridge protocol. The npm worker registers with the coordinator, heartbeats, polls for assignments, and executes them by spawning the command provided by `--worker-cmd`.

| Option | Description |
|--------|-------------|
| `--coordinator URL` | Coordinator base URL (default: `http://127.0.0.1:8000`) |
| `--api-key KEY` | Worker API key (or `ATOMICS_WORKER_API_KEY`) |
| `--label KEY=VALUE` | Worker label, repeatable |
| `--capability TEXT` | Worker capability, repeatable (default: `node`) |
| `--endpoint URL` | Optional push endpoint URL for this worker |
| `--worker-cmd CMD` | Command used to execute each task (default: `node task-runner.js`) |
| `--heartbeat-interval N` | Heartbeat interval in seconds (default: 30) |
| `--pool-size N` | Number of independent npm workers to run on this host (default: 1) |
| `--npm-dir PATH` | Path to the npm worker package |

Use `--runtime node` on `atomics distributed run` to generate node-runtime tasks that only npm workers can claim.

```bash
# Terminal 1 — start a single npm worker
uv run atomics worker-npm --api-key worker-key --label box=239

# Terminal 1 — start four npm workers on the same host
uv run atomics worker-npm --api-key worker-key --label box=239 --pool-size 4

# Terminal 2 — submit node-runtime work
uv run atomics distributed run --mode split --runtime node -n 10 --api-key client-key
```

The bundled `task-runner.js` is a minimal example that echoes the task name. Real deployments should point `--worker-cmd` at their own Node.js runner that implements the bridge protocol.

## atomics distributed

Submit and inspect distributed benchmark runs. Three modes:

- `split` divides the tasks across registered workers, to finish a run faster.
- `fleet` gives every matching worker the identical task set, to compare hosts.
- `full` delegates an entire run to one worker, which executes the full `LoopEngine` locally.

### `atomics distributed run`

| Option | Description |
|--------|-------------|
| `--coordinator URL` | Coordinator base URL (default: `http://127.0.0.1:8000`) |
| `--api-key KEY` | Client API key (or `ATOMICS_API_KEY`) |
| `--mode [split\|fleet\|full]` | Job mode (default: `split`) |
| `-p, --provider TEXT` | Pin every task to this provider (default: each worker's own) |
| `-t, --tier TEXT` | Burn tier (default: `baseline`) |
| `-m, --model TEXT` | Model override for the executing provider |
| `-n INTEGER` | Tasks per worker in fleet mode; tasks in total in split mode (default: 1) |
| `--label KEY=VALUE` | Worker selector, repeatable. Fleet and full mode only; **rejected for `split`**, which assigns each task to the next available worker |
| `--runtime [python\|node]` | Runtime for generated tasks. `node` routes tasks to npm workers via the bridge (default: `python`) |

Reads the client key from `ATOMICS_API_KEY`; pass `--api-key` to override.

```bash
# Split: divide 4 tasks across whichever workers are free
uv run atomics distributed run -t baseline -n 4

# Split, pinning the whole run to one provider
uv run atomics distributed run -p ollama -t baseline -n 4

# Fleet: run all 20 tasks on every 4090 in the lab, then compare
uv run atomics distributed run --mode fleet --label gpu=4090 --label site=lab -n 20

# Fleet across every online worker
uv run atomics distributed run --mode fleet -n 20

# Full: delegate an entire run to one worker (first matching box=239)
uv run atomics distributed run --mode full --label box=239 -t ez -n 5

# Full without selector: first worker that claims it runs the whole run
uv run atomics distributed run --mode full -t ez -n 5
```

Fleet notes:

- A worker must match **every** `--label` pair. Omitting `--label` broadcasts to
  all online workers.
- The set of workers is snapshotted when the run is submitted, so a worker that
  registers mid-run does not join it.
- A selector matching no online worker is rejected rather than creating a job that
  cannot progress.
- Every host receives the same prompts; that is what makes the results comparable.
- A host that stops heartbeating for 120s has its remaining tasks marked failed.
  They are never re-run on another host, since that would silently blend two
  machines into one result. The job then reports `partial`.

Full mode notes:

- One worker executes the entire `LoopEngine` locally. Use it when a worker is
  faster than the coordinator machine or when you want a single host to own a
  complete run end-to-end.
- The worker uses its own `--provider`, `--model`, and `--host` unless the run
  request pins them via `-p` / `-m`.
- A selector pins the job to the first matching worker, but if no selector is
  given any worker can claim the assignment.
- The full run result is a `summary` plus `task_results`, which `atomics distributed status`
  renders as a compact table.

### `atomics distributed status`

| Argument / Option | Description |
|-------------------|-------------|
| `JOB_ID` | Distributed job id (required) |
| `--coordinator URL` | Coordinator base URL (default: `http://127.0.0.1:8000`) |
| `--api-key KEY` | Client API key (or `ATOMICS_API_KEY`) |
| `--json-out PATH` | Write the job and its per-worker rollup to a file |

Fleet jobs print a row per host — completed, failed, mean and p95 latency,
throughput, and cost. Split jobs print the job as JSON.

```bash
uv run atomics distributed status <job_id>
uv run atomics distributed status <job_id> --json-out fleet.json
```
