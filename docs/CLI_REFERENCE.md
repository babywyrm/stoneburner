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
| `atomics run --provider ollama --ollama-host http://gpu:11434` | Use remote Ollama |
| `atomics run --provider brain-gateway` | Use camazotz brain-gateway |
| `atomics run --provider brain-gateway --gateway-url http://nuc:30080` | Use remote brain-gateway |
| `atomics run --thinking` | Enable thinking/reasoning mode for capable models |
| `atomics run --no-thinking` | Force thinking off (A/B comparison) |
| `atomics run --thinking-budget 20000` | Set max thinking tokens (provider-specific default otherwise) |
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
| `atomics provider-test -p brain-gateway` | Health check brain-gateway |
| `atomics models` | List available models on Ollama host with class/thinking annotations |
| `atomics models --provider vllm` | List models on a vLLM-compatible endpoint |
| `atomics doctor` | Check installation health and config |

## Evaluation Suites

| Command | Description |
|---------|-------------|
| `atomics eval` | Run evaluation suite against a provider |
| `atomics eval --fixtures ev-19` | Run a fixture subset for a fast spot-check |
| `atomics eval --extra-judges ollama:mistral:7b` | Multi-judge consensus scoring |
| `atomics adversarial` | Adversarial resilience eval — resistance to manipulation (72 fixtures) |
| `atomics adversarial --category tool_desc_injection` | Run one suite/group |
| `atomics adversarial --runs 3` | Variance-aware scoring (mean ± stddev) |
| `atomics adversarial --compare mistral-small:24b` | Run a second model, print per-fixture diff |
| `atomics adversarial --json-out run.json` | Write full per-fixture results as JSON |
| `atomics adversarial --fail-on-resilience 60` | CI gate — non-zero exit if resilience < 60% |
| `atomics refusal` | Refusal-calibration eval — over- vs under-refusal |
| `atomics codereview` | Secure-code-review eval — planted-vuln detection + false positives |
| `atomics redblue --mode all` | Red/blue security capability eval (offensive + defensive) |
| `atomics redblue --runs 3 --json-out rb.json` | Variance-aware capability scoring + JSON export |
| `atomics probe --probes-file probes.yaml` | Live ecosystem probe against real artifacts |
| `atomics archreview --repo juice-shop --models qwen3.5:4b` | Security-architecture repo benchmark |
| `atomics archreview --tier local --max-output-tokens 512` | Practical brainbox repo review |
| `atomics archreview --tier wide --rounds 3` | Broader evidence pack with stability reporting |
| `atomics archreview --tier expanded --rounds 3` | Largest pack for large-context/cloud backends |
| `atomics rag` | RAG pipeline evaluation — grounding, faithfulness, abstention |
| `atomics rag --fixtures rag-05,rag-12` | Run a fixture subset |
| `atomics rag --json-out rag.json` | Write results as JSON |
| `atomics sweep` | Multi-model eval sweep with ranked comparison |

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
