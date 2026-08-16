# API models + provider-test Implementation Plan

> Executed inline 2026-08-16.

**Goal:** Agents can list loaded models and probe that one answers, through the API, then MCP.

**Architecture:** Sync `GET /models` and `POST /provider-test`. Fixed 2+2 prompt. MCP proxies only.

---

Tasks: discovery module, routes, MCP client/tools, contract tests, docs, live smoke on laptop Ollama.
