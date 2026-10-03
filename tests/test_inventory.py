"""Model inventory: sources, readers, probes, and the models command."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from click.testing import CliRunner

from atomics.cli import cli
from atomics.inventory import Capability, ModelRecord, readers
from atomics.inventory.readers import read_ollama, read_openai

# Trimmed from the laptop's Ollama 0.34.4 on 2026-10-02. Digests shortened.
TAGS = {
    "models": [
        {
            "name": "phi4-mini-reasoning:3.8b",
            "size": 3155072553,
            "digest": "3ca8c2865ce9",
            "capabilities": ["tools", "completion"],
            "details": {"parameter_size": "3.8B", "quantization_level": "Q4_K_M"},
        },
        {
            "name": "ornith-1.5:9b",
            "size": 6600000000,
            "digest": "e5df7dcdd8a2",
            "capabilities": ["tools", "thinking", "completion", "vision"],
            "details": {"parameter_size": "9B", "quantization_level": "Q4_K_M"},
        },
        {
            "name": "nomic-embed-text:latest",
            "size": 274302450,
            "digest": "0a109f422b47",
            "capabilities": ["embedding"],
            "details": {"parameter_size": "137M", "quantization_level": "F16"},
        },
    ]
}
VERSION = {"version": "0.34.4"}

Body = object | Callable[[httpx.Request], object]


def _client(routes: dict[str, Body]) -> httpx.AsyncClient:
    """Answer by URL path. A callable body sees the request; a Response passes through."""

    def handle(request: httpx.Request) -> httpx.Response:
        body = routes.get(request.url.path)
        if body is None:
            return httpx.Response(404)
        if callable(body):
            body = body(request)
        return body if isinstance(body, httpx.Response) else httpx.Response(200, json=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


def _down(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


def _invoke(monkeypatch, routes: dict[str, Body], *args: str):
    monkeypatch.setattr(readers, "new_client", lambda: _client(routes))
    return CliRunner().invoke(cli, ["models", *args], env={"COLUMNS": "200"})


def test_declared_capability_wins_over_name_table():
    rec = ModelRecord(
        host="h", name="phi4-mini-reasoning:3.8b", declared=frozenset({"tools", "completion"})
    )
    assert rec.capability("thinking") == Capability(False, "declared")
    assert rec.disagreements == ["thinking: declared=false name-table=true"]


def test_probe_wins_and_its_disagreement_is_listed():
    rec = ModelRecord(host="h", name="ornith-1.5:9b", declared=frozenset({"tools", "completion"}))
    rec.probed["tools"] = False
    assert rec.capability("tools") == Capability(False, "probe")
    assert "tools: probe=false declared=true" in rec.disagreements


def test_tools_never_come_from_a_name():
    rec = ModelRecord(host="h", name="qwen3:8b")
    assert rec.capability("tools") == Capability(None, "unknown")
    assert rec.capability("thinking") == Capability(True, "name-table")


def test_no_completion_is_not_evaluable():
    rec = ModelRecord(host="h", name="nomic-embed-text:latest", declared=frozenset({"embedding"}))
    assert rec.evaluable is False


async def test_read_ollama_records_declared_capabilities():
    async with _client({"/api/tags": TAGS, "/api/version": VERSION}) as client:
        host, models = await read_ollama(client, "http://laptop:11434/", label="laptop")
    assert (host.label, host.provider, host.version) == ("laptop", "ollama", "0.34.4")
    phi = models[0]
    assert phi.declared == frozenset({"tools", "completion"})
    assert (phi.digest, phi.quantization, phi.parameter_size) == ("3ca8c2865ce9", "Q4_K_M", "3.8B")


async def test_read_ollama_without_capabilities_uses_name_table():
    old = {"models": [{"name": "qwen3:8b", "size": 1, "digest": "d", "details": {}}]}
    async with _client({"/api/tags": old}) as client:
        host, models = await read_ollama(client, "http://old:11434", label="old")
    assert host.version is None
    assert models[0].declared is None
    assert models[0].capability("thinking").source == "name-table"


async def test_read_ollama_unreachable_raises_connection_error():
    async with httpx.AsyncClient(transport=httpx.MockTransport(_down)) as client:
        with pytest.raises(ConnectionError, match="Cannot reach"):
            await read_ollama(client, "http://gone:11434", label="gone")


async def test_read_openai_lists_model_ids():
    listing = {"data": [{"id": "Qwen/Qwen3-8B", "max_model_len": 32768}]}
    async with _client({"/v1/models": listing}) as client:
        host, models = await read_openai(client, "http://gpu:8000/v1", label="gpu")
    assert host.provider == "vllm"
    assert [m.name for m in models] == ["Qwen/Qwen3-8B"]


def test_models_table_and_json(monkeypatch, tmp_path):
    out = tmp_path / "inv.json"
    result = _invoke(
        monkeypatch,
        {"/api/tags": TAGS, "/api/version": VERSION},
        "--host",
        "http://laptop:11434",
        "--json-out",
        str(out),
    )
    assert result.exit_code == 0, result.output
    assert "phi4-mini-reasoning:3.8b thinking: declared=false name-table=true" in result.output
    assert "nomic-embed-text:latest not evaluable" in result.output
    assert " phi4-mini-reasoning:3.8b " in result.output
    data = json.loads(out.read_text())
    assert data["schema"] == 1
    assert data["hosts"] == [
        {"label": "laptop", "provider": "ollama", "version": "0.34.4", "error": None}
    ]
    phi, _, embed = data["models"]
    assert phi["capabilities"]["thinking"] == {"value": False, "source": "declared"}
    assert phi["capabilities"]["audio"] == {"value": False, "source": "declared"}
    assert embed["evaluable"] is False


def test_models_json_out_creates_the_directory(monkeypatch, tmp_path):
    out = tmp_path / "logs" / "inventory" / "laptop.json"
    result = _invoke(monkeypatch, {"/api/tags": TAGS}, "--json-out", str(out))
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["schema"] == 1


def test_models_vllm_host_option(monkeypatch):
    listing = {"data": [{"id": "qwen2.5:1.5b"}, {"id": "qwen3.5:0.8b"}]}
    result = _invoke(
        monkeypatch, {"/v1/models": listing}, "-p", "vllm", "--vllm-host", "http://fake:8000/v1"
    )
    assert result.exit_code == 0, result.output
    assert "qwen2.5:1.5b" in result.output
    assert "fake vllm" in result.output


def test_models_unreachable_host_exits_1(monkeypatch):
    monkeypatch.setattr(
        readers, "new_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(_down))
    )
    result = CliRunner().invoke(cli, ["models", "--host", "http://gone:11434"])
    assert result.exit_code == 1
    assert "Cannot reach" in result.output
