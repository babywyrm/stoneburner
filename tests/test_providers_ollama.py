"""Tests for the Ollama provider adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atomics.providers.base import BaseProvider, ProviderResponse
from atomics.providers.ollama import OllamaProvider


def test_ollama_implements_interface():
    provider = OllamaProvider(host="http://fake:11434")
    assert isinstance(provider, BaseProvider)
    assert provider.name == "ollama"


def test_ollama_default_host():
    provider = OllamaProvider()
    assert provider._host == "http://localhost:11434"
    assert provider._default_model == "qwen2.5:7b"


def test_ollama_custom_host():
    provider = OllamaProvider(host="http://gpu-box:11434", default_model="qwen3:4b")
    assert provider._host == "http://gpu-box:11434"
    assert provider._default_model == "qwen3:4b"


@pytest.mark.asyncio
async def test_ollama_generate_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "response": "MCP is a protocol for tool integration.",
        "eval_count": 42,
        "prompt_eval_count": 15,
        "eval_duration": 330_000_000,  # 0.33 seconds → ~127 tok/s
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    provider = OllamaProvider(host="http://fake:11434", client=mock_client)
    resp = await provider.generate("What is MCP?", system="Be helpful.")

    assert resp.text == "MCP is a protocol for tool integration."
    assert resp.input_tokens == 15
    assert resp.output_tokens == 42
    assert resp.total_tokens == 57
    assert resp.model == "qwen2.5:7b"
    assert resp.estimated_cost_usd == 0.0
    assert resp.tokens_per_second is not None
    assert resp.tokens_per_second == pytest.approx(42 / 0.33, rel=0.01)
    # Ollama times pure decode (eval_duration), so it reports the generation basis.
    assert resp.tps_basis == "generation"


@pytest.mark.asyncio
async def test_ollama_generate_with_model_override():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "response": "ok",
        "eval_count": 5,
        "prompt_eval_count": 3,
        "eval_duration": 100_000_000,
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    provider = OllamaProvider(host="http://fake:11434", client=mock_client)
    resp = await provider.generate("test", model="qwen3:4b")

    assert resp.model == "qwen3:4b"
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    body = call_kwargs[1].get("json") or call_kwargs[0][1] if len(call_kwargs[0]) > 1 else call_kwargs[1]["json"]
    assert body["model"] == "qwen3:4b"


def test_ollama_default_timeout_is_generous():
    # Thinking models can reason well past the old 120s cap.
    assert OllamaProvider()._timeout == 300.0


@pytest.mark.asyncio
async def test_ollama_generate_uses_configured_timeout():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "response": "ok", "eval_count": 5, "prompt_eval_count": 3, "eval_duration": 1,
    }
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    provider = OllamaProvider(host="http://fake:11434", timeout=42.0, client=mock_client)
    await provider.generate("hi")
    assert mock_client.post.call_args.kwargs["timeout"] == 42.0


@pytest.mark.asyncio
async def test_ollama_generate_sets_num_predict_from_max_tokens():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "response": "ok", "eval_count": 5, "prompt_eval_count": 3, "eval_duration": 1,
    }
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    provider = OllamaProvider(host="http://fake:11434", client=mock_client)
    await provider.generate("hi", max_tokens=321)

    body = mock_client.post.call_args.kwargs["json"]
    assert body["options"]["num_predict"] == 321


@pytest.mark.asyncio
async def test_ollama_generate_sets_configured_context_tokens():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "response": "ok", "eval_count": 5, "prompt_eval_count": 3, "eval_duration": 1,
    }
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    provider = OllamaProvider(
        host="http://fake:11434",
        context_tokens=20096,
        client=mock_client,
    )
    await provider.generate("hi", max_tokens=2048)

    body = mock_client.post.call_args.kwargs["json"]
    assert body["options"]["num_predict"] == 2048
    assert body["options"]["num_ctx"] == 20096


@pytest.mark.asyncio
async def test_ollama_generate_forwards_native_think_flag():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "response": "ok", "eval_count": 5, "prompt_eval_count": 3, "eval_duration": 1,
    }
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    provider = OllamaProvider(host="http://fake:11434", client=mock_client)
    await provider.generate("hi", thinking=False)

    body = mock_client.post.call_args.kwargs["json"]
    assert body["think"] is False


@pytest.mark.asyncio
async def test_ollama_generate_zero_eval_duration():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "response": "ok",
        "eval_count": 5,
        "prompt_eval_count": 3,
        "eval_duration": 0,
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    provider = OllamaProvider(host="http://fake:11434", client=mock_client)
    resp = await provider.generate("test")

    assert resp.tokens_per_second is None


@pytest.mark.asyncio
async def test_ollama_generate_connection_error():
    import httpx

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    provider = OllamaProvider(host="http://fake:11434", client=mock_client)
    with pytest.raises(ConnectionError, match="fake:11434"):
        await provider.generate("test")


@pytest.mark.asyncio
async def test_ollama_health_check_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "models": [{"name": "qwen2.5:7b"}, {"name": "qwen3:4b"}]
    }

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    provider = OllamaProvider(host="http://fake:11434", client=mock_client)
    assert await provider.health_check() is True


@pytest.mark.asyncio
async def test_ollama_health_check_failure():
    import httpx

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    provider = OllamaProvider(host="http://fake:11434", client=mock_client)
    assert await provider.health_check() is False


def test_provider_response_has_tokens_per_second():
    resp = ProviderResponse(
        text="test",
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        model="test",
        latency_ms=100.0,
        estimated_cost_usd=0.0,
        tokens_per_second=127.5,
    )
    assert resp.tokens_per_second == 127.5


def test_provider_response_tokens_per_second_default_none():
    resp = ProviderResponse(
        text="test",
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        model="test",
        latency_ms=100.0,
        estimated_cost_usd=0.0,
    )
    assert resp.tokens_per_second is None


def test_ollama_model_classes():
    from atomics.model_classes import ModelClass, classify_model

    assert classify_model("qwen2.5:7b") == ModelClass.MID
    assert classify_model("qwen2.5:1.5b") == ModelClass.LIGHT
    assert classify_model("qwen3:4b") == ModelClass.MID
    assert classify_model("qwen3.5:0.8b") == ModelClass.LIGHT
    assert classify_model("llama3.2:3b") == ModelClass.MID


def test_ollama_config_defaults(monkeypatch, tmp_path):
    # Clear any process-level env vars AND chdir to a tmp dir so that the
    # project's .env file (which points to the GPU host) is not loaded.
    monkeypatch.delenv("ATOMICS_OLLAMA_HOST", raising=False)
    monkeypatch.delenv("ATOMICS_OLLAMA_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)

    from atomics.config import AtomicsSettings

    s = AtomicsSettings()
    assert s.ollama_host == "http://localhost:11434"
    assert s.ollama_model == "qwen2.5:7b"


def test_ollama_config_env_override(monkeypatch):
    from atomics.config import AtomicsSettings

    monkeypatch.setenv("ATOMICS_OLLAMA_HOST", "http://gpu-box:11434")
    monkeypatch.setenv("ATOMICS_OLLAMA_MODEL", "qwen3:4b")
    s = AtomicsSettings()
    assert s.ollama_host == "http://gpu-box:11434"
    assert s.ollama_model == "qwen3:4b"


@pytest.mark.asyncio
async def test_ollama_list_models():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "models": [
            {
                "name": "qwen2.5:7b",
                "size": 4_700_000_000,
                "details": {"parameter_size": "7.6B", "family": "qwen2.5"},
            },
            {
                "name": "mistral:7b",
                "size": 4_400_000_000,
                "details": {"parameter_size": "7.2B", "family": "mistral"},
            },
            {
                "name": "some-unknown:1b",
                "size": 1_000_000_000,
                "details": {"parameter_size": "1B", "family": "unknown"},
            },
        ]
    }

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    provider = OllamaProvider(host="http://fake:11434", client=mock_client)
    models = await provider.list_models()

    assert len(models) == 3
    assert models[0]["name"] == "qwen2.5:7b"
    assert models[1]["name"] == "mistral:7b"
    assert models[2]["name"] == "some-unknown:1b"
    for m in models:
        assert "name" in m
        assert "size_gb" in m
        assert "model_class" in m
        assert "thinking" in m
    assert models[0]["model_class"] == "mid"
    assert models[2]["model_class"] == "unknown"


@pytest.mark.asyncio
async def test_ollama_list_models_connection_error():
    import httpx

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

    provider = OllamaProvider(host="http://fake:11434", client=mock_client)
    with pytest.raises(ConnectionError, match="fake:11434"):
        await provider.list_models()
