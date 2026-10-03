"""Model inventory: sources, readers, probes, and the models command."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from click.testing import CliRunner

from atomics.cli import cli
from atomics.eval.judge import JudgeResult
from atomics.eval.redblue.fixtures import BLUE_FIXTURES
from atomics.eval.toolcall.catalog import PROBE_TOOL
from atomics.inventory import Capability, ModelRecord, Reply, mark_digest_mismatches, readers
from atomics.inventory import probe as probe_mod
from atomics.inventory.probe import (
    BAD_ANSWER,
    GOOD_ANSWER,
    JUDGE_FIXTURE_ID,
    probe_judge,
    probe_model,
    thinking_verdict,
)
from atomics.inventory.readers import (
    read_llamacpp,
    read_ollama,
    read_openai,
    take_inventory,
    unique_labels,
)
from atomics.providers.base import ProviderResponse
from atomics.providers.toolcalls import ToolCall

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
        host, models = await read_ollama(
            client, "http://laptop:11434/", label="laptop", context_tokens=None
        )
    assert (host.label, host.provider, host.version) == ("laptop", "ollama", "0.34.4")
    phi = models[0]
    assert phi.declared == frozenset({"tools", "completion"})
    assert (phi.digest, phi.quantization, phi.parameter_size) == ("3ca8c2865ce9", "Q4_K_M", "3.8B")


async def test_read_ollama_without_capabilities_uses_name_table():
    old = {"models": [{"name": "qwen3:8b", "size": 1, "digest": "d", "details": {}}]}
    async with _client({"/api/tags": old}) as client:
        host, models = await read_ollama(
            client, "http://old:11434", label="old", context_tokens=None
        )
    assert host.version is None
    assert models[0].declared is None
    assert models[0].capability("thinking").source == "name-table"


async def test_read_ollama_unreachable_raises_connection_error():
    async with httpx.AsyncClient(transport=httpx.MockTransport(_down)) as client:
        with pytest.raises(ConnectionError, match="Cannot reach"):
            await read_ollama(client, "http://gone:11434", label="gone", context_tokens=None)


async def test_read_openai_lists_model_ids():
    listing = {"data": [{"id": "Qwen/Qwen3-8B", "max_model_len": 32768}]}
    async with _client({"/v1/models": listing}) as client:
        host, models = await read_openai(
            client, "http://gpu:8000/v1", label="gpu", context_tokens=None
        )
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
        {
            "label": "laptop",
            "provider": "ollama",
            "version": "0.34.4",
            "error": None,
            "loaded": [],
        }
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


# /api/show model_info, trimmed. phi3's rope key also ends in context_length.
SHOWS = {
    "phi4-mini-reasoning:3.8b": {
        "capabilities": ["tools", "completion"],
        "model_info": {
            "general.architecture": "phi3",
            "phi3.context_length": 131072,
            "phi3.rope.scaling.original_context_length": 4096,
        },
    },
    "ornith-1.5:9b": {
        "model_info": {"general.architecture": "qwen35", "qwen35.context_length": 262144}
    },
}
PS = {"models": [{"name": "ornith-1.5:9b", "context_length": 4096}]}


def _show(request: httpx.Request) -> object:
    name = json.loads(request.content)["model"]
    return SHOWS.get(name, httpx.Response(500))


OLLAMA = {"/api/tags": TAGS, "/api/version": VERSION, "/api/show": _show, "/api/ps": PS}


async def test_ollama_context_reality():
    async with _client(OLLAMA) as client:
        host, (phi, ornith, embed) = await read_ollama(
            client, "http://laptop:11434", label="laptop", context_tokens=None
        )
    assert (phi.declared_context, phi.requested_context, phi.loaded_context) == (
        131072,
        8192,
        None,
    )
    assert phi.flags == []
    assert ornith.loaded_context == 4096
    assert ornith.flags == ["loaded-at-other-context"]
    assert host.loaded == [("ornith-1.5:9b", 4096)]
    assert embed.errors and embed.errors[0].startswith("show: Cannot reach")


async def test_requested_above_declared_is_flagged():
    async with _client(OLLAMA) as client:
        _, (phi, _, _) = await read_ollama(
            client, "http://laptop:11434", label="laptop", context_tokens=200000
        )
    assert phi.requested_context == 200000
    assert "requested-above-declared" in phi.flags


async def test_show_fills_capabilities_when_tags_lack_them():
    old = {
        "models": [{"name": "phi4-mini-reasoning:3.8b", "size": 1, "digest": "d", "details": {}}]
    }
    async with _client({"/api/tags": old, "/api/show": _show}) as client:
        _, (phi,) = await read_ollama(client, "http://old:11434", label="old", context_tokens=None)
    assert phi.declared == frozenset({"tools", "completion"})


async def test_openai_reads_max_model_len():
    listing = {"data": [{"id": "Qwen/Qwen3-8B", "max_model_len": 32768}]}
    async with _client({"/v1/models": listing}) as client:
        _, (m,) = await read_openai(client, "http://gpu:8000/v1", label="gpu", context_tokens=None)
    assert (m.declared_context, m.requested_context) == (32768, None)


def test_models_json_carries_context(monkeypatch, tmp_path):
    out = tmp_path / "inv.json"
    result = _invoke(monkeypatch, OLLAMA, "--context-tokens", "16384", "--json-out", str(out))
    assert result.exit_code == 0, result.output
    assert "loaded: ornith-1.5:9b (4096)" in result.output
    assert "ornith-1.5:9b loaded-at-other-context" in result.output
    data = json.loads(out.read_text())
    assert data["hosts"][0]["loaded"] == [{"name": "ornith-1.5:9b", "context": 4096}]
    phi = data["models"][0]
    assert (phi["declared_context"], phi["requested_context"]) == (131072, 16384)
    assert data["models"][1]["flags"] == ["loaded-at-other-context"]


# llama-server shapes: newer builds nest n_ctx, older ones put it at the top.
LLAMA_MODELS = {"object": "list", "data": [{"id": "qwen3-8b-q4_k_m.gguf", "object": "model"}]}


@pytest.mark.parametrize(
    "props",
    [{"default_generation_settings": {"n_ctx": 16384}}, {"n_ctx": 16384}],
)
async def test_llamacpp_reads_n_ctx(props):
    async with _client({"/v1/models": LLAMA_MODELS, "/props": props}) as client:
        host, (m,) = await read_llamacpp(
            client, "http://box:8080", label="box", context_tokens=None
        )
    assert host.provider == "llamacpp"
    assert (m.name, m.declared_context, m.declared) == ("qwen3-8b-q4_k_m.gguf", 16384, None)


async def test_llamacpp_without_props_still_lists():
    async with _client({"/v1/models": LLAMA_MODELS}) as client:
        _, (m,) = await read_llamacpp(client, "http://box:8080", label="box", context_tokens=None)
    assert m.declared_context is None


def test_models_llamacpp_provider(monkeypatch):
    routes = {"/v1/models": LLAMA_MODELS, "/props": {"n_ctx": 16384}}
    result = _invoke(monkeypatch, routes, "-p", "llamacpp", "--host", "http://box:8080")
    assert result.exit_code == 0, result.output
    assert "qwen3-8b-q4_k_m.gguf" in result.output
    assert "box llamacpp" in result.output


def _resp(text: str = "Paris", *, out: int = 4, thinking: int = 0, calls=()) -> ProviderResponse:
    return ProviderResponse(
        text=text,
        input_tokens=10,
        output_tokens=out,
        total_tokens=10 + out,
        model="m",
        latency_ms=120.0,
        estimated_cost_usd=0.0,
        tokens_per_second=40.0,
        thinking_tokens=thinking,
        tool_calls=tuple(calls),
    )


class _Scripted:
    """Answers thinking-off with `off`, thinking-on with `on`, tools with a call or not."""

    supports_tools = True

    def __init__(self, *, off=None, on=None, tool=True, error=None):
        self.off, self.on, self.tool, self.error = off or _resp(), on or _resp(), tool, error
        self.calls: list[tuple[str, object]] = []

    async def generate(self, prompt, *, model=None, max_tokens=0, thinking=None, **_):
        self.calls.append(("generate", thinking))
        if self.error:
            raise self.error
        return self.on if thinking else self.off

    async def generate_with_tools(self, prompt, *, tools, thinking=None, **_):
        self.calls.append(("tools", thinking))
        return _resp(calls=(ToolCall(name=PROBE_TOOL),) if self.tool else ())


def _rec(name="ministral-3:8b", caps=("completion", "tools")) -> ModelRecord:
    return ModelRecord(host="laptop", name=name, declared=frozenset(caps))


async def test_probe_records_answer_speed_and_tools():
    rec, fake = _rec(), _Scripted()
    await probe_model(fake, rec)
    assert rec.probe.off.answered and rec.probe.off.tokens_per_second == 40.0
    assert rec.capability("completion") == Capability(True, "probe")
    assert rec.capability("tools") == Capability(True, "probe")
    assert fake.calls == [("generate", False), ("tools", False)]


async def test_declared_tools_that_never_call_are_a_disagreement():
    rec = _rec()
    await probe_model(_Scripted(tool=False), rec)
    assert "tools: probe=false declared=true" in rec.disagreements


async def test_tools_declared_absent_are_not_probed():
    rec, fake = _rec(caps=("completion",)), _Scripted()
    await probe_model(fake, rec)
    assert ("tools", False) not in fake.calls
    assert "tools" not in rec.probed


async def test_probe_error_is_recorded_and_nothing_is_probed():
    rec = _rec()
    await probe_model(_Scripted(error=ConnectionError("Cannot connect")), rec)
    assert rec.probe.off is None
    assert rec.probed == {}
    assert rec.errors and rec.errors[0].startswith("probe:")


def test_models_probe_filters_and_shows_speed(monkeypatch, tmp_path):
    fake = _Scripted()
    monkeypatch.setattr(readers, "probe_provider", lambda *a, **k: fake)
    out = tmp_path / "inv.json"
    result = _invoke(
        monkeypatch,
        OLLAMA,
        "--host",
        "http://laptop:11434",
        "--probe",
        "-m",
        "phi4*",
        "--json-out",
        str(out),
    )
    assert result.exit_code == 0, result.output
    assert "probed laptop phi4-mini-reasoning:3.8b" in result.output
    data = json.loads(out.read_text())
    assert [m["name"] for m in data["models"]] == ["phi4-mini-reasoning:3.8b"]
    assert data["models"][0]["probe"]["off"]["tokens_per_second"] == 40.0


def test_models_probe_skips_models_that_cannot_complete(monkeypatch, tmp_path):
    fake = _Scripted()
    monkeypatch.setattr(readers, "probe_provider", lambda *a, **k: fake)
    out = tmp_path / "inv.json"
    result = _invoke(
        monkeypatch,
        OLLAMA,
        "--host",
        "http://laptop:11434",
        "--probe",
        "-m",
        "nomic*",
        "--json-out",
        str(out),
    )
    assert result.exit_code == 0, result.output
    assert fake.calls == []
    assert json.loads(out.read_text())["models"][0]["probe"] is None


def _reply(*, out: int = 4, thinking: int = 0, answered: bool = True) -> Reply:
    return Reply(answered, "completed", out, thinking, 100.0, 40.0)


@pytest.mark.parametrize(
    ("off", "on", "verdict"),
    [
        (_reply(), _reply(out=60, thinking=50), "off-works"),
        (_reply(out=60, thinking=40), _reply(out=60, thinking=50), "off-ignored"),
        (_reply(out=4), _reply(out=120), "inline"),
        (_reply(out=4), _reply(out=6), "no-channel"),
    ],
)
def test_thinking_verdict(off, on, verdict):
    assert thinking_verdict(off, on) == verdict


async def test_thinking_is_probed_only_when_claimed():
    rec, fake = _rec(), _Scripted()
    await probe_model(fake, rec)
    assert [c for c in fake.calls if c[0] == "generate"] == [("generate", False)]
    assert rec.probe.verdict is None


async def test_probe_settles_a_name_table_disagreement():
    rec = _rec(name="phi4-mini-reasoning:3.8b")
    fake = _Scripted(on=_resp(out=60, thinking=50))
    await probe_model(fake, rec)
    assert rec.probe.verdict == "off-works"
    assert rec.probe.recommended == "--no-thinking"
    assert rec.capability("thinking") == Capability(True, "probe")
    assert "thinking: probe=true declared=false name-table=true" in rec.disagreements
    assert fake.calls[-1] == ("tools", False)


async def test_off_ignored_recommends_thinking_and_the_tool_probe_follows():
    rec = _rec(caps=("completion", "tools", "thinking"))
    fake = _Scripted(off=_resp(out=60, thinking=40), on=_resp(out=60, thinking=50))
    await probe_model(fake, rec)
    assert rec.probe.verdict == "off-ignored"
    assert rec.probe.recommended == "--thinking"
    assert fake.calls[-1] == ("tools", True)


async def test_an_answer_with_thinking_on_counts_as_completion():
    rec = _rec(caps=("completion", "thinking"))
    await probe_model(_Scripted(off=_resp(text=""), on=_resp(out=60, thinking=50)), rec)
    assert rec.probed["completion"] is True


def _hosts(per_host: dict[str, dict[str, Body]]) -> httpx.AsyncClient:
    """Route by hostname first; a host missing from the map refuses connections."""

    def handle(request: httpx.Request) -> httpx.Response:
        routes = per_host.get(request.url.host)
        if routes is None:
            raise httpx.ConnectError("refused", request=request)
        body = routes.get(request.url.path)
        if body is None:
            return httpx.Response(404)
        return httpx.Response(200, json=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


def _tags(*pairs: tuple[str, str]) -> dict[str, object]:
    return {
        "models": [
            {"name": n, "digest": d, "size": 1, "capabilities": ["completion"], "details": {}}
            for n, d in pairs
        ]
    }


def test_digest_mismatch_flags_every_copy():
    a = ModelRecord(host="laptop", name="gemma4:26b", digest="aaa")
    b = ModelRecord(host="beefy", name="gemma4:26b", digest="bbb")
    c = ModelRecord(host="brainbox", name="gemma4:26b", digest="aaa")
    d = ModelRecord(host="beefy", name="qwen3:8b", digest="ccc")
    mark_digest_mismatches([a, b, c, d])
    assert [m.flags for m in (a, b, c, d)] == [["digest-mismatch"]] * 3 + [[]]


def test_unique_labels_add_the_port_only_when_needed():
    assert unique_labels(["http://box:11434", "http://box:11435", "http://beefy:11434"]) == [
        "box:11434",
        "box:11435",
        "beefy",
    ]


async def test_unreachable_host_is_reported_and_skipped(monkeypatch):
    per_host = {"laptop": {"/api/tags": _tags(("gemma4:26b", "aaa"))}}
    monkeypatch.setattr(readers, "new_client", lambda: _hosts(per_host))
    inv = await take_inventory("ollama", ["http://laptop:11434", "http://brainbox:11434"])
    assert [h.label for h in inv.hosts] == ["laptop", "brainbox"]
    assert inv.hosts[1].error and "Cannot reach" in inv.hosts[1].error
    assert [m.host for m in inv.models] == ["laptop"]


async def test_every_host_down_raises(monkeypatch):
    monkeypatch.setattr(readers, "new_client", lambda: _hosts({}))
    with pytest.raises(ConnectionError, match="brainbox"):
        await take_inventory("ollama", ["http://laptop:11434", "http://brainbox:11434"])


def test_models_spans_hosts_and_flags_digests(monkeypatch, tmp_path):
    per_host = {
        "laptop": {"/api/tags": _tags(("gemma4:26b", "aaa"))},
        "beefy": {"/api/tags": _tags(("gemma4:26b", "bbb"), ("qwen3:8b", "ccc"))},
    }
    monkeypatch.setattr(readers, "new_client", lambda: _hosts(per_host))
    out = tmp_path / "inv.json"
    args = ["--host", "http://laptop:11434", "--host", "http://beefy:11434"]
    result = CliRunner().invoke(cli, ["models", *args, "--json-out", str(out)])
    assert result.exit_code == 0, result.output
    assert "beefy/gemma4:26b digest-mismatch" in result.output
    data = json.loads(out.read_text())
    assert [(m["host"], m["name"]) for m in data["models"]] == [
        ("laptop", "gemma4:26b"),
        ("beefy", "gemma4:26b"),
        ("beefy", "qwen3:8b"),
    ]


def _judged(scores: dict[str, float | None]):
    """Fake score_response: score by answer; None means the reply did not parse."""

    async def score(prompt, response, *, judge_provider, judge_model=None, **_):
        value = scores[response]
        return JudgeResult(
            score=value or 0.0,
            accuracy=0,
            completeness=0,
            format_score=0,
            rationale="",
            judge_model=judge_model or "",
            parse_failed=value is None,
        )

    return score


def test_judge_fixture_is_a_blue_fixture():
    assert JUDGE_FIXTURE_ID in {f.id for f in BLUE_FIXTURES}


@pytest.mark.parametrize(
    ("good", "bad", "fit"),
    [
        (0.9, 0.1, True),
        (0.7, 0.6, False),
        (0.4, 0.6, False),
        (None, 0.1, False),
    ],
)
async def test_judge_fitness(monkeypatch, good, bad, fit):
    monkeypatch.setattr(probe_mod, "score_response", _judged({GOOD_ANSWER: good, BAD_ANSWER: bad}))
    rec = _rec()
    await probe_judge(_Scripted(), rec)
    assert (rec.judge.good, rec.judge.bad, rec.judge.fit) == (good, bad, fit)
    assert rec.to_dict()["judge"] == {"good": good, "bad": bad, "fit": fit}


async def test_judge_error_is_recorded(monkeypatch):
    async def boom(*a, **k):
        raise ConnectionError("Cannot connect")

    monkeypatch.setattr(probe_mod, "score_response", boom)
    rec = _rec()
    await probe_judge(_Scripted(), rec)
    assert rec.judge is None
    assert rec.errors[0].startswith("judge:")


def test_models_probe_judge(monkeypatch, tmp_path):
    fake = _Scripted()
    monkeypatch.setattr(readers, "probe_provider", lambda *a, **k: fake)
    monkeypatch.setattr(probe_mod, "score_response", _judged({GOOD_ANSWER: 0.9, BAD_ANSWER: 0.1}))
    out = tmp_path / "inv.json"
    args = ["--host", "http://laptop:11434", "--probe-judge", "-m", "phi4*", "--json-out", str(out)]
    result = _invoke(monkeypatch, OLLAMA, *args)
    assert result.exit_code == 0, result.output
    model = json.loads(out.read_text())["models"][0]
    assert model["judge"] == {"good": 0.9, "bad": 0.1, "fit": True}
    assert model["probe"] is None
    assert fake.calls == []
