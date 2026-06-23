"""Tests for the brain-gateway (camazotz) provider adapter."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from atomics.providers.base import BaseProvider, ProviderResponse
from atomics.providers.brain_gateway import BrainGatewayProvider


def test_brain_gateway_implements_interface():
    provider = BrainGatewayProvider(url="http://fake:8080")
    assert isinstance(provider, BaseProvider)
    assert provider.name == "brain-gateway"


def test_brain_gateway_defaults():
    provider = BrainGatewayProvider()
    assert provider._url == "http://localhost:8080"
    assert provider._default_model is None


def test_brain_gateway_custom_url():
    provider = BrainGatewayProvider(
        url="http://nuc:30080", default_model="claude-haiku-4-5"
    )
    assert provider._url == "http://nuc:30080"
    assert provider._default_model == "claude-haiku-4-5"


def test_brain_gateway_strips_trailing_slash():
    provider = BrainGatewayProvider(url="http://nuc:30080/")
    assert provider._url == "http://nuc:30080"


@pytest.mark.asyncio
async def test_brain_gateway_generate_success():
    inner_text = json.dumps({
        "answer": "4",
        "prompt_source": "default",
        "_usage": {
            "input_tokens": 12,
            "output_tokens": 3,
            "cost_usd": 0.000081,
            "model": "claude-sonnet-4-6",
        },
    })

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [{"type": "text", "text": inner_text}],
            "isError": False,
        },
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.put = AsyncMock(return_value=MagicMock(status_code=200))

    provider = BrainGatewayProvider(url="http://fake:8080", client=mock_client)
    resp = await provider.generate("What is 2+2?")

    assert isinstance(resp, ProviderResponse)
    assert resp.text == "4"
    assert resp.input_tokens == 12
    assert resp.output_tokens == 3
    assert resp.total_tokens == 15
    assert resp.model == "claude-sonnet-4-6"
    assert resp.estimated_cost_usd == 0.000081
    assert resp.latency_ms > 0

    call_args = mock_client.post.call_args
    payload = call_args.kwargs.get("json") or call_args[1].get("json")
    assert payload["method"] == "tools/call"
    assert payload["params"]["name"] == "config.ask_agent"
    assert payload["params"]["arguments"]["question"] == "What is 2+2?"


@pytest.mark.asyncio
async def test_brain_gateway_generate_with_model_switch():
    inner_text = json.dumps({
        "answer": "test",
        "_usage": {"input_tokens": 5, "output_tokens": 2, "cost_usd": 0.0, "model": "llama3.2:3b"},
    })

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": inner_text}], "isError": False},
    }

    mock_put_response = MagicMock()
    mock_put_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.put = AsyncMock(return_value=mock_put_response)

    provider = BrainGatewayProvider(
        url="http://fake:8080", default_model="llama3.2:3b", client=mock_client
    )
    resp = await provider.generate("Hello")

    mock_client.put.assert_called_once()
    put_args = mock_client.put.call_args
    put_json = put_args.kwargs.get("json") or put_args[1].get("json")
    assert put_json == {"model": "llama3.2:3b"}
    assert resp.model == "llama3.2:3b"


@pytest.mark.asyncio
async def test_brain_gateway_generate_rpc_error():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -32000, "message": "Rate limit exceeded"},
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.put = AsyncMock(return_value=MagicMock(status_code=200))

    provider = BrainGatewayProvider(url="http://fake:8080", client=mock_client)

    with pytest.raises(RuntimeError, match="Rate limit exceeded"):
        await provider.generate("test")


@pytest.mark.asyncio
async def test_brain_gateway_generate_empty_content():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [], "isError": False},
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.put = AsyncMock(return_value=MagicMock(status_code=200))

    provider = BrainGatewayProvider(url="http://fake:8080", client=mock_client)

    with pytest.raises(RuntimeError, match="empty content"):
        await provider.generate("test")


@pytest.mark.asyncio
async def test_brain_gateway_health_check_success():
    mock_response = MagicMock()
    mock_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    provider = BrainGatewayProvider(url="http://fake:8080", client=mock_client)
    assert await provider.health_check() is True


@pytest.mark.asyncio
async def test_brain_gateway_health_check_failure():
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=Exception("connection refused"))

    provider = BrainGatewayProvider(url="http://fake:8080", client=mock_client)
    assert await provider.health_check() is False


@pytest.mark.asyncio
async def test_brain_gateway_connection_error():
    import httpx

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    mock_client.put = AsyncMock(return_value=MagicMock(status_code=200))

    provider = BrainGatewayProvider(url="http://fake:8080", client=mock_client)

    with pytest.raises(ConnectionError, match="Cannot connect to brain-gateway"):
        await provider.generate("test")


@pytest.mark.asyncio
async def test_brain_gateway_no_usage_field():
    inner_text = json.dumps({"answer": "hello", "prompt_source": "default"})

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": inner_text}], "isError": False},
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    provider = BrainGatewayProvider(url="http://fake:8080", client=mock_client)
    resp = await provider.generate("test")

    assert resp.text == "hello"
    assert resp.input_tokens == 0
    assert resp.output_tokens == 0
    assert resp.estimated_cost_usd == 0.0
    assert resp.model == "unknown"
