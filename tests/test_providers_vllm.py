"""Tests for the vLLM / OpenAI-compatible provider adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from atomics.providers.base import BaseProvider
from atomics.providers.vllm import VllmProvider

# ---------------------------------------------------------------------------
# Interface / construction
# ---------------------------------------------------------------------------

def test_vllm_implements_interface():
    provider = VllmProvider(base_url="http://fake:8000/v1")
    assert isinstance(provider, BaseProvider)
    assert provider.name == "vllm"


def test_vllm_default_url():
    provider = VllmProvider()
    assert provider._base_url == "http://localhost:8000/v1"
    assert provider._default_model == "qwen2.5:3b"


def test_vllm_custom_url():
    provider = VllmProvider(base_url="http://gpu-host:8000/v1", default_model="qwen3.5:0.8b")
    assert provider._base_url == "http://gpu-host:8000/v1"
    assert provider._default_model == "qwen3.5:0.8b"


def test_vllm_trailing_slash_stripped():
    provider = VllmProvider(base_url="http://fake:8000/v1/")
    assert provider._base_url == "http://fake:8000/v1"


def test_vllm_default_timeout_is_generous():
    assert VllmProvider()._timeout == 300.0


@pytest.mark.asyncio
async def test_vllm_generate_uses_configured_timeout():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    }
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    provider = VllmProvider(base_url="http://fake:8000/v1", timeout=37.0, client=mock_client)
    await provider.generate("hi")
    assert mock_client.post.call_args.kwargs["timeout"] == 37.0


# ---------------------------------------------------------------------------
# generate() — happy path
# ---------------------------------------------------------------------------

def _mock_openai_response(content: str, inp: int = 15, out: int = 42) -> MagicMock:
    mock = MagicMock()
    mock.status_code = 200
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "choices": [{"message": {"content": content, "role": "assistant"}}],
        "usage": {
            "prompt_tokens": inp,
            "completion_tokens": out,
            "total_tokens": inp + out,
        },
        "model": "qwen2.5:3b",
    }
    return mock


@pytest.mark.asyncio
async def test_vllm_generate_success():
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_mock_openai_response("MCP is a protocol."))

    provider = VllmProvider(base_url="http://fake:8000/v1", client=mock_client)
    resp = await provider.generate("What is MCP?", system="Be helpful.")

    assert resp.text == "MCP is a protocol."
    assert resp.input_tokens == 15
    assert resp.output_tokens == 42
    assert resp.total_tokens == 57
    assert resp.model == "qwen2.5:3b"
    assert resp.estimated_cost_usd == 0.0
    assert resp.tokens_per_second is not None


@pytest.mark.asyncio
async def test_vllm_generate_model_override():
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_mock_openai_response("ok", inp=3, out=5))

    provider = VllmProvider(base_url="http://fake:8000/v1", client=mock_client)
    resp = await provider.generate("test", model="qwen3.5:0.8b")

    assert resp.model == "qwen3.5:0.8b"
    call_body = mock_client.post.call_args[1]["json"]
    assert call_body["model"] == "qwen3.5:0.8b"


@pytest.mark.asyncio
async def test_vllm_generate_uses_correct_endpoint():
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_mock_openai_response("ok"))

    provider = VllmProvider(base_url="http://fake:8000/v1", client=mock_client)
    await provider.generate("hello")

    url = mock_client.post.call_args[0][0]
    assert url == "http://fake:8000/v1/chat/completions"


@pytest.mark.asyncio
async def test_vllm_generate_sends_messages_format():
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_mock_openai_response("ok"))

    provider = VllmProvider(base_url="http://fake:8000/v1", client=mock_client)
    await provider.generate("user question", system="custom system")

    body = mock_client.post.call_args[1]["json"]
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "custom system"
    assert body["messages"][1]["role"] == "user"
    assert body["messages"][1]["content"] == "user question"


@pytest.mark.asyncio
async def test_vllm_generate_zero_output_tokens_no_tps():
    mock = MagicMock()
    mock.status_code = 200
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "choices": [{"message": {"content": "", "role": "assistant"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
    }
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock)

    provider = VllmProvider(base_url="http://fake:8000/v1", client=mock_client)
    resp = await provider.generate("test")
    assert resp.tokens_per_second is None


# ---------------------------------------------------------------------------
# Thinking mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vllm_thinking_flag_sent_for_qwen3():
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_mock_openai_response("ok"))

    provider = VllmProvider(base_url="http://fake:8000/v1", client=mock_client)
    await provider.generate("test", model="qwen3.5:0.8b", thinking=True)

    body = mock_client.post.call_args[1]["json"]
    assert body.get("chat_template_kwargs") == {"enable_thinking": True}


@pytest.mark.asyncio
async def test_vllm_thinking_off_for_qwen3():
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_mock_openai_response("ok"))

    provider = VllmProvider(base_url="http://fake:8000/v1", client=mock_client)
    await provider.generate("test", model="qwen3.5:0.8b", thinking=False)

    body = mock_client.post.call_args[1]["json"]
    assert body.get("chat_template_kwargs") == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_vllm_no_thinking_flag_for_non_thinking_model():
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_mock_openai_response("ok"))

    provider = VllmProvider(base_url="http://fake:8000/v1", client=mock_client)
    await provider.generate("test", model="qwen2.5:3b", thinking=False)

    body = mock_client.post.call_args[1]["json"]
    assert "chat_template_kwargs" not in body


# ---------------------------------------------------------------------------
# Thinking — reasoning_content passthrough
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vllm_reasoning_content_captured():
    """reasoning_content from the response is surfaced as thinking_text."""
    mock = MagicMock()
    mock.status_code = 200
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "choices": [{
            "message": {
                "content": "The answer is 42.",
                "role": "assistant",
                "reasoning_content": "Let me think step by step about this...",
            }
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock)

    provider = VllmProvider(base_url="http://fake:8000/v1", client=mock_client)
    resp = await provider.generate("test", model="qwen3.5:0.8b", thinking=True)

    assert resp.text == "The answer is 42."
    assert "Let me think" in resp.thinking_text
    assert resp.thinking_tokens > 0


# ---------------------------------------------------------------------------
# Connection error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vllm_generate_connection_error():
    import httpx

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

    provider = VllmProvider(base_url="http://fake:8000/v1", client=mock_client)
    with pytest.raises(ConnectionError, match="fake:8000"):
        await provider.generate("test")


# ---------------------------------------------------------------------------
# list_models()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vllm_list_models():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"id": "qwen2.5:1.5b", "object": "model"},
            {"id": "qwen2.5:3b", "object": "model"},
            {"id": "qwen3.5:0.8b", "object": "model"},
        ],
        "object": "list",
    }
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    provider = VllmProvider(base_url="http://fake:8000/v1", client=mock_client)
    models = await provider.list_models()

    assert len(models) == 3
    names = [m["name"] for m in models]
    assert "qwen2.5:1.5b" in names
    assert "qwen3.5:0.8b" in names
    for m in models:
        assert "name" in m
        assert "model_class" in m
        assert "thinking" in m


@pytest.mark.asyncio
async def test_vllm_list_models_uses_correct_endpoint():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"data": []}
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    provider = VllmProvider(base_url="http://fake:8000/v1", client=mock_client)
    await provider.list_models()

    url = mock_client.get.call_args[0][0]
    assert url == "http://fake:8000/v1/models"


@pytest.mark.asyncio
async def test_vllm_list_models_connection_error():
    import httpx

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

    provider = VllmProvider(base_url="http://fake:8000/v1", client=mock_client)
    with pytest.raises(ConnectionError, match="fake:8000"):
        await provider.list_models()


# ---------------------------------------------------------------------------
# health_check()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vllm_health_check_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    provider = VllmProvider(base_url="http://fake:8000/v1", client=mock_client)
    assert await provider.health_check() is True


@pytest.mark.asyncio
async def test_vllm_health_check_failure():
    import httpx

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

    provider = VllmProvider(base_url="http://fake:8000/v1", client=mock_client)
    assert await provider.health_check() is False


@pytest.mark.asyncio
async def test_vllm_health_check_non_200():
    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    provider = VllmProvider(base_url="http://fake:8000/v1", client=mock_client)
    assert await provider.health_check() is False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_vllm_config_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("ATOMICS_VLLM_HOST", raising=False)
    monkeypatch.delenv("ATOMICS_VLLM_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)

    from atomics.config import AtomicsSettings
    s = AtomicsSettings()
    assert s.vllm_host == "http://localhost:8000/v1"
    assert s.vllm_model == "qwen2.5:3b"


def test_vllm_config_env_override(monkeypatch):
    from atomics.config import AtomicsSettings
    monkeypatch.setenv("ATOMICS_VLLM_HOST", "http://gpu-host:8000/v1")
    monkeypatch.setenv("ATOMICS_VLLM_MODEL", "qwen3.5:0.8b")
    s = AtomicsSettings()
    assert s.vllm_host == "http://gpu-host:8000/v1"
    assert s.vllm_model == "qwen3.5:0.8b"


# ---------------------------------------------------------------------------
# model_classes / thinking detection
# ---------------------------------------------------------------------------

def test_vllm_thinking_model_detection():
    from atomics.providers.vllm import _model_supports_thinking
    assert _model_supports_thinking("qwen3.5:0.8b") is True
    assert _model_supports_thinking("qwen3:1.7b") is True
    assert _model_supports_thinking("qwen2.5:3b") is False
    assert _model_supports_thinking("deepseek-r1:14b") is True
    assert _model_supports_thinking("mistral:7b") is False


def test_vllm_model_classes():
    from atomics.model_classes import ModelClass, classify_model
    assert classify_model("qwen2.5:1.5b") == ModelClass.LIGHT
    assert classify_model("qwen2.5:3b") == ModelClass.MID
    assert classify_model("qwen3.5:0.8b") == ModelClass.LIGHT
