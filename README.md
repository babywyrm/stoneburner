# Stoneburner

[![PyPI](https://img.shields.io/pypi/v/stoneburner-atomics.svg)](https://pypi.org/project/stoneburner-atomics/)
[![GitHub release](https://img.shields.io/github/v/release/babywyrm/stoneburner)](https://github.com/babywyrm/stoneburner/releases/latest)
[![Python](https://img.shields.io/pypi/pyversions/stoneburner-atomics.svg)](https://pypi.org/project/stoneburner-atomics/)
[![License: MIT](https://img.shields.io/pypi/l/stoneburner-atomics.svg)](https://github.com/babywyrm/stoneburner/blob/main/LICENSE)
[![CI](https://github.com/babywyrm/stoneburner/actions/workflows/ci.yml/badge.svg)](https://github.com/babywyrm/stoneburner/actions/workflows/ci.yml)

Local-first LLM evaluation: token cost, quality, and security suites.
The same commands cover a laptop Ollama box and a cloud API.

Install **[stoneburner-atomics](https://pypi.org/project/stoneburner-atomics/)**.
The CLI and the import stay **`atomics`**. (`atomics` on PyPI is a different
package; `stoneburner` is too similar to an existing `stone-burner`.)

This is a desk tool, not a research harness and not an unsupervised agent.
It records cost, quality, and security-suite results in SQLite. A
finished-looking percentage on a partial run is the failure mode it is
built to avoid: incomplete coverage prints `n/a (scored/total scored)`
and JSON nulls the headline.

`atomics doctor` ends with one `Next:` command when the check is healthy.
Typical first-run output (Ollama on localhost, no cloud key):

```text
$ atomics doctor
Python 3.13.11 OK
Platform: Darwin (arm64)
Database path: data/atomics.db
SQLite database OK (readable / creatable)
ANTHROPIC_API_KEY not set (optional; needed for Claude)
OPENAI_API_KEY not set (optional; needed for OpenAI)
inference.env: not found (optional; $INFERENCE_ENV or /etc/agentic/inference.env)
Ollama endpoint: http://localhost:11434
Ollama reachable — 3 model(s): qwen2.5:7b, gemma3:4b, llama3.2:3b

Next: atomics provider-test --provider ollama --no-thinking
      Ollama is reachable.
```

```text
$ atomics toolcall --provider ollama --channel tools --runs 3 --no-thinking

Summary
  tool-capable: yes
  outcomes: safe call=6  no call=14
  channel divergence (resisted in prose, complied with tools): not measured (no qualifying fixtures)
  response divergence (dangerous call, refusing text): not measured (no qualifying fixtures)
  cost: $0.0000
```

A tools-only first run is valid. Channel divergence needs a second model
as judge. Thinking models that spend the token budget on hidden reasoning
are recorded as `thinking_budget`, not as a mystery generation failure.

## Install

Ollama on `http://localhost:11434` is the one-box path. No cloud key required.

```bash
uv tool install stoneburner-atomics
atomics doctor
atomics provider-test --provider ollama --no-thinking
atomics toolcall --provider ollama --channel tools --runs 3 --no-thinking
```

`--no-thinking` keeps reasoning models from spending the whole token budget
on hidden chain-of-thought.

```bash
uv tool install 'stoneburner-atomics[api,mcp]'
uv add 'stoneburner-atomics[rag]'          # from another project
```

From a clone, `uv sync --all-extras`. Bare `uv sync` drops the API, MCP,
RAG, and test extras. `atomics server`, `atomics mcp`, and `atomics repl`
need those extras and a running API server.

Cloud providers take the same `--provider` / `--effort` flags once a key
is set. Providers: Claude, Bedrock, OpenAI, Ollama, vLLM, llama.cpp, Groq,
Gemini, Together, brain-gateway.

## Docs

Recipes live in [QUICKSTART](https://github.com/babywyrm/stoneburner/blob/main/QUICKSTART.md).
Flags live in [CLI_REFERENCE](https://github.com/babywyrm/stoneburner/blob/main/docs/CLI_REFERENCE.md).
Links are absolute so they work on PyPI as well as GitHub.

| If you want | Read |
|-------------|------|
| Copy-paste recipes | [QUICKSTART](https://github.com/babywyrm/stoneburner/blob/main/QUICKSTART.md) |
| Every flag | [CLI_REFERENCE](https://github.com/babywyrm/stoneburner/blob/main/docs/CLI_REFERENCE.md) |
| Quality / compare | [COMPARING](https://github.com/babywyrm/stoneburner/blob/main/docs/COMPARING.md) |
| Security suites | [SECURITY_SUITES](https://github.com/babywyrm/stoneburner/blob/main/docs/SECURITY_SUITES.md) · [leaderboard](https://github.com/babywyrm/stoneburner/blob/main/docs/LEADERBOARD.md) · [red/blue](https://github.com/babywyrm/stoneburner/blob/main/docs/LEADERBOARD-REDBLUE.md) |
| Load / capacity | [LOAD_TESTING](https://github.com/babywyrm/stoneburner/blob/main/docs/LOAD_TESTING.md) |
| Thinking / `--effort` | [THINKING](https://github.com/babywyrm/stoneburner/blob/main/docs/THINKING.md) |
| `inference.env` | [INFERENCE_ENV](https://github.com/babywyrm/stoneburner/blob/main/docs/INFERENCE_ENV.md) |
| HTTP API, fleet, dashboard | [API_SERVER](https://github.com/babywyrm/stoneburner/blob/main/docs/API_SERVER.md) |
| MCP | [MCP_SERVER](https://github.com/babywyrm/stoneburner/blob/main/docs/MCP_SERVER.md) |
| Human REPL | [REPL](https://github.com/babywyrm/stoneburner/blob/main/docs/REPL.md) |
| Contribute | [CONTRIBUTING](https://github.com/babywyrm/stoneburner/blob/main/CONTRIBUTING.md) · [ARCHITECTURE](https://github.com/babywyrm/stoneburner/blob/main/ARCHITECTURE.md) |

## License

MIT — see [LICENSE](https://github.com/babywyrm/stoneburner/blob/main/LICENSE).
