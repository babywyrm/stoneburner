"""Tests for thinking/extended reasoning support across providers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from atomics.model_classes import THINKING_CAPABLE, supports_thinking
from atomics.providers.base import ProviderResponse


class TestThinkingRegistry:
    def test_known_thinking_models(self):
        assert supports_thinking("o3")
        assert supports_thinking("o4-mini")
        assert supports_thinking("qwen3:14b")
        assert supports_thinking("claude-sonnet-4-6")
        assert supports_thinking("gpt-5")

    def test_non_thinking_models(self):
        assert not supports_thinking("gpt-4o")
        assert not supports_thinking("gpt-4o-mini")
        assert not supports_thinking("qwen2.5:7b")
        assert not supports_thinking("llama3.2:3b")

    def test_prefix_auto_detect(self):
        assert supports_thinking("qwen3:72b")
        assert supports_thinking("o3-mini-2025")
        assert supports_thinking("o4-mini-latest")

    def test_thinking_capable_is_frozen(self):
        with pytest.raises(AttributeError):
            THINKING_CAPABLE.add("new-model")


class TestProviderResponseThinking:
    def test_defaults(self):
        resp = ProviderResponse(
            text="hi",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            model="test",
            latency_ms=100.0,
            estimated_cost_usd=0.0,
        )
        assert resp.thinking_tokens == 0
        assert resp.thinking_text == ""

    def test_with_thinking(self):
        resp = ProviderResponse(
            text="answer",
            input_tokens=10,
            output_tokens=50,
            total_tokens=60,
            model="test",
            latency_ms=100.0,
            estimated_cost_usd=0.0,
            thinking_tokens=40,
            thinking_text="let me think about this...",
        )
        assert resp.thinking_tokens == 40
        assert resp.thinking_text == "let me think about this..."


class TestOllamaThinkingParsing:
    def test_strip_thinking_tags(self):
        from atomics.providers.ollama import _strip_thinking

        raw = "<think>reasoning here</think>the answer is 4"
        clean, thinking = _strip_thinking(raw)
        assert clean == "the answer is 4"
        assert thinking == "reasoning here"

    def test_strip_multiple_thinking_blocks(self):
        from atomics.providers.ollama import _strip_thinking

        raw = "<think>step 1</think>partial <think>step 2</think>final answer"
        clean, thinking = _strip_thinking(raw)
        assert clean == "partial final answer"
        assert "step 1" in thinking
        assert "step 2" in thinking

    def test_no_thinking_tags(self):
        from atomics.providers.ollama import _strip_thinking

        raw = "just a plain answer"
        clean, thinking = _strip_thinking(raw)
        assert clean == "just a plain answer"
        assert thinking == ""

    def test_model_supports_thinking(self):
        from atomics.providers.ollama import _model_supports_thinking

        assert _model_supports_thinking("qwen3:14b")
        assert _model_supports_thinking("qwen3:1.7b")
        assert not _model_supports_thinking("qwen2.5:7b")
        assert not _model_supports_thinking("llama3.2:3b")


class TestOllamaProviderThinking:
    @pytest.mark.asyncio
    async def test_thinking_auto_enabled_for_qwen3(self):
        from atomics.providers.ollama import OllamaProvider

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "response": "<think>Let me reason</think>4",
            "eval_count": 20,
            "prompt_eval_count": 10,
            "eval_duration": 1_000_000_000,
            "total_duration": 2_000_000_000,
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        provider = OllamaProvider(default_model="qwen3:14b", client=mock_client)
        resp = await provider.generate("What is 2+2?")

        assert resp.text == "4"
        assert resp.thinking_text == "Let me reason"
        assert resp.thinking_tokens > 0

    @pytest.mark.asyncio
    async def test_thinking_disabled_explicit(self):
        from atomics.providers.ollama import OllamaProvider

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "response": "4",
            "eval_count": 5,
            "prompt_eval_count": 10,
            "eval_duration": 500_000_000,
            "total_duration": 1_000_000_000,
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        provider = OllamaProvider(default_model="qwen3:14b", client=mock_client)
        resp = await provider.generate("What is 2+2?", thinking=False)

        assert resp.text == "4"
        assert resp.thinking_text == ""

        call_args = mock_client.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert body["prompt"].startswith("/no_think")


class TestClaudeProviderThinking:
    @pytest.mark.asyncio
    async def test_thinking_enabled(self):
        from atomics.providers.claude import ClaudeProvider

        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "Let me analyze this carefully"

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "The answer is 4"

        usage = MagicMock()
        usage.input_tokens = 20
        usage.output_tokens = 50
        usage.thinking_tokens = 30

        mock_response = MagicMock()
        mock_response.content = [thinking_block, text_block]
        mock_response.usage = usage
        mock_response.model_dump.return_value = {}

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        provider = ClaudeProvider(api_key="test", client=mock_client)
        resp = await provider.generate("What is 2+2?", thinking=True)

        assert resp.text == "The answer is 4"
        assert resp.thinking_text == "Let me analyze this carefully"
        assert resp.thinking_tokens == 30

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "thinking" in call_kwargs
        assert call_kwargs["thinking"]["type"] == "enabled"

    @pytest.mark.asyncio
    async def test_thinking_disabled_default(self):
        from atomics.providers.claude import ClaudeProvider

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "4"

        usage = MagicMock()
        usage.input_tokens = 10
        usage.output_tokens = 5
        usage.thinking_tokens = 0

        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_response.usage = usage
        mock_response.model_dump.return_value = {}

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        provider = ClaudeProvider(api_key="test", client=mock_client)
        resp = await provider.generate("What is 2+2?")

        assert resp.text == "4"
        assert resp.thinking_tokens == 0

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "thinking" not in call_kwargs


class TestOpenAIProviderThinking:
    @pytest.mark.asyncio
    async def test_reasoning_token_extraction(self):
        from atomics.providers.openai import OpenAIProvider

        details = MagicMock()
        details.reasoning_tokens = 150

        usage = MagicMock()
        usage.prompt_tokens = 20
        usage.completion_tokens = 200
        usage.completion_tokens_details = details

        message = MagicMock()
        message.content = "42"
        message.reasoning_content = None

        choice = MagicMock()
        choice.message = message

        mock_response = MagicMock()
        mock_response.choices = [choice]
        mock_response.usage = usage
        mock_response.model_dump.return_value = {}

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenAIProvider(api_key="test", default_model="o3", client=mock_client)
        resp = await provider.generate("What is 6*7?", thinking=True)

        assert resp.text == "42"
        assert resp.thinking_tokens == 150
        assert resp.output_tokens == 200
        visible_out = resp.output_tokens - resp.thinking_tokens
        assert visible_out == 50


class TestOpenAIResponsesReasoning:
    @pytest.mark.asyncio
    async def test_responses_reasoning_from_output_tokens_details(self):
        """Responses API nests reasoning under usage.output_tokens_details."""
        from atomics.auth import AuthStrategy
        from atomics.providers.openai import OpenAIProvider

        details = MagicMock()
        details.reasoning_tokens = 120

        usage = MagicMock()
        usage.input_tokens = 20
        usage.output_tokens = 200
        usage.output_tokens_details = details
        # A spurious top-level attr must be ignored in favor of the nested one.
        usage.reasoning_tokens = 0

        response = MagicMock()
        response.output_text = "42"
        response.usage = usage
        response.model_dump.return_value = {}

        auth = MagicMock(spec=AuthStrategy)
        auth.get_headers = AsyncMock(return_value={"Authorization": "Bearer tok"})

        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=response)

        provider = OpenAIProvider(default_model="o3", client=mock_client, auth=auth)
        resp = await provider.generate("What is 6*7?", thinking=True)

        assert resp.text == "42"
        assert resp.thinking_tokens == 120
        assert resp.output_tokens - resp.thinking_tokens == 80


class TestOllamaThinkingTokenEstimate:
    @pytest.mark.asyncio
    async def test_thinking_tokens_anchored_to_eval_count(self):
        """Estimate is a char-proportional slice of the real eval_count, not a word count."""
        from atomics.providers.ollama import OllamaProvider

        # 36 reasoning chars, 4 answer chars -> ~90% of 100 generated tokens.
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "response": "<think>aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa</think>bbbb",
            "eval_count": 100,
            "prompt_eval_count": 10,
            "eval_duration": 1_000_000_000,
            "total_duration": 2_000_000_000,
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        provider = OllamaProvider(default_model="qwen3:14b", client=mock_client)
        resp = await provider.generate("hi")

        assert resp.thinking_text == "a" * 36
        assert resp.thinking_tokens == round(100 * 36 / 40)  # == 90
        # Never exceeds the real output token count.
        assert resp.thinking_tokens <= resp.output_tokens

    @pytest.mark.asyncio
    async def test_no_thinking_means_zero_thinking_tokens(self):
        from atomics.providers.ollama import OllamaProvider

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "response": "plain answer, no tags",
            "eval_count": 50,
            "prompt_eval_count": 10,
            "eval_duration": 1_000_000_000,
            "total_duration": 2_000_000_000,
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        provider = OllamaProvider(default_model="qwen2.5:7b", client=mock_client)
        resp = await provider.generate("hi")
        assert resp.thinking_tokens == 0


class TestTaskResultThinking:
    def test_thinking_fields_on_task_result(self):
        from atomics.models import TaskCategory, TaskResult

        result = TaskResult(
            run_id="test-run",
            category=TaskCategory.GENERAL_QA,
            task_name="test",
            provider="claude",
            model="claude-sonnet-4-6",
            thinking_tokens=500,
            thinking_enabled=True,
        )
        assert result.thinking_tokens == 500
        assert result.thinking_enabled is True

    def test_thinking_defaults(self):
        from atomics.models import TaskCategory, TaskResult

        result = TaskResult(
            run_id="test-run",
            category=TaskCategory.GENERAL_QA,
            task_name="test",
            provider="ollama",
            model="qwen2.5:7b",
        )
        assert result.thinking_tokens == 0
        assert result.thinking_enabled is False


class TestRunnerThinkingPassthrough:
    @pytest.mark.asyncio
    async def test_execute_task_passes_thinking(self):
        from atomics.core.runner import execute_task
        from atomics.models import TaskCategory, TaskComplexity, TaskDefinition

        task = TaskDefinition(
            category=TaskCategory.GENERAL_QA,
            name="test_task",
            prompt_template="{prompt}",
            complexity=TaskComplexity.LIGHT,
        )

        mock_provider = AsyncMock()
        mock_provider.name = "claude"
        mock_provider.generate = AsyncMock(return_value=ProviderResponse(
            text="answer",
            input_tokens=10,
            output_tokens=50,
            total_tokens=60,
            model="claude-sonnet-4-6",
            latency_ms=500.0,
            estimated_cost_usd=0.001,
            thinking_tokens=30,
            thinking_text="reasoning...",
        ))

        result = await execute_task(
            task,
            "test prompt",
            provider=mock_provider,
            run_id="test-run",
            model="claude-sonnet-4-6",
            thinking=True,
            thinking_budget=10000,
        )

        call_kwargs = mock_provider.generate.call_args.kwargs
        assert call_kwargs["thinking"] is True
        assert call_kwargs["thinking_budget"] == 10000
        assert result.thinking_tokens == 30
        assert result.thinking_enabled is True
