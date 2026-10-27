# Supported Surfaces

What Stoneburner supports, and what each surface is tested against.
Two kinds of evidence count:

- **CI:** the test suite in `.github/workflows/ci.yml`. Tests mock
  providers and HTTP; a guard fails any test that reaches the network.
- **Live on record:** a real generate against the backend, written up in
  `ROADMAP.md`, `CHANGELOG.md`, or a results doc, with its date.

A surface with CI but no live run is wired and unit-tested, not proved
against the real service. Treat its first live run as a test of the tool.

## Runtime

| | Supported |
|---|---|
| Python | 3.11, 3.12, 3.13 |
| OS in CI | `ubuntu-latest`, `macos-latest` |
| Not tested | Windows |

CI syncs the `dev`, `api`, `mcp`, `claude`, `openai`, and `bedrock`
extras. The `rag` extra is not installed there: it pulls torch. The RAG
index and retrieval tests skip without it and run only where it is
installed.

## Providers

Build every provider through `make_provider()`. `toolcall` needs a
tool-capable one; on any other provider it reports the model as
incapable instead of scoring it.

| Provider | Tool calls | Live on record |
|---|---|---|
| `ollama` | yes, native `/api/chat` | yes: fleet run 2026-10-04 and later |
| `vllm` | yes, OpenAI-compatible | yes: 2026-10-16, `Qwen2.5-3B-Instruct-AWQ` |
| `llamacpp` | yes, OpenAI-compatible | yes: 2026-10-16, `granite4.2:3b`, Metal and CPU |
| `openai` | yes | yes: 2026-07-08 comparison, 2026-09-13 battery cut |
| `claude` | yes, Anthropic `tool_use` | yes: 2026-07-08 comparison, 2026-09-13 battery cut |
| `bedrock` | no | no |
| `groq` | yes, OpenAI-compatible | no |
| `together` | yes, OpenAI-compatible | no |
| `gemini` | yes, OpenAI-compatible | no |
| `brain-gateway` | no | no |

`vllm` was also run against Ollama's `/v1` (2026-10-04). That proves the
OpenAI-compatible wire path, not vLLM itself. See `ROADMAP.md` for the
numbers behind each live row and what they do not show.

## Suites

Every suite runs in CI against mocked providers. `eval` and `toolcall`
also run end to end: the real CLI against a stub inference server on a
loopback socket (`tests/test_e2e_http.py`).

| Surface | CLI | API |
|---|---|---|
| `eval` (accuracy), `rag`, `multiturn`, `adversarial`, `codegen`, `refusal`, `redblue`, `toolcall`, `codereview` | yes | `POST /api/v1/evals` |
| `battery`, `sweep` | yes | `POST /api/v1/batteries`, `/sweeps` |
| `stress`, `soak` | yes | `POST /api/v1/stress`, `/soak`, capped |
| `qa`, `probe`, `archreview`, `judge-calibrate`, `judge-agreement`, `scenario`, `labcompare` | yes | no |

Live runs on record: `toolcall` on every live provider above;
`redblue`, `refusal`, and `codereview` across the 2026-10-04 fleet
(judge `gemma4:26b`); `adversarial` and `redblue` in
`docs/FRONTIER_COMPARISON.md`.

## API, MCP, and REPL

| Surface | Tested against |
|---|---|
| `atomics server` | In-process `TestClient` on `127.0.0.1`: auth, Host allowlist, limits, jobs, routes |
| `atomics mcp` | stdio proxy over the API; tool contract checked against the API routes |
| `atomics repl` | API client; parse, dispatch, wait, and display tests |
| Distributed workers | coordinator and worker end to end in-process |

Endpoints are listed in [API_SERVER.md](API_SERVER.md).

## Not supported

- HTTP transport for MCP. `atomics mcp` stays stdio; reach a remote API
  with `--api-url`.
- Tool calling on `bedrock` and `brain-gateway`.
