"""Tests for the task runner using a mock provider."""

import pytest

from atomics.models import TaskCategory, TaskComplexity, TaskDefinition, TaskStatus
from atomics.providers.base import BaseProvider, ProviderResponse
from atomics.core.runner import execute_task


class MockProvider(BaseProvider):
    """Deterministic provider for testing."""

    def __init__(self, text: str = "mock response", fail: bool = False) -> None:
        self._text = text
        self._fail = fail

    @property
    def name(self) -> str:
        return "mock"

    async def generate(self, prompt, *, system="", model=None, max_tokens=1024):
        if self._fail:
            raise ConnectionError("mock failure")
        return ProviderResponse(
            text=self._text,
            input_tokens=50,
            output_tokens=100,
            total_tokens=150,
            model=model or "mock-model",
            latency_ms=42.0,
            estimated_cost_usd=0.001,
        )

    async def health_check(self):
        return not self._fail


SAMPLE_TASK = TaskDefinition(
    category=TaskCategory.GENERAL_QA,
    name="test_task",
    prompt_template="Tell me about {topic}",
    complexity=TaskComplexity.LIGHT,
    weight=1.0,
    max_output_tokens=256,
)


@pytest.mark.asyncio
async def test_execute_task_success():
    provider = MockProvider(text="hello world")
    result = await execute_task(SAMPLE_TASK, "testing", provider=provider, run_id="run-test")

    assert result.status == TaskStatus.SUCCESS
    assert result.response == "hello world"
    assert result.input_tokens == 50
    assert result.output_tokens == 100
    assert result.total_tokens == 150
    assert result.latency_ms == 42.0
    assert result.estimated_cost_usd == 0.001
    assert result.run_id == "run-test"
    assert result.provider == "mock"
    assert result.completed_at is not None


@pytest.mark.asyncio
async def test_execute_task_failure():
    provider = MockProvider(fail=True)
    result = await execute_task(SAMPLE_TASK, "testing", provider=provider, run_id="run-fail")

    assert result.status == TaskStatus.FAILED
    assert result.error_class == "ConnectionError"
    assert "mock failure" in result.error_message
    assert result.completed_at is not None


@pytest.mark.asyncio
async def test_execute_task_formats_prompt():
    provider = MockProvider()
    result = await execute_task(SAMPLE_TASK, "kubernetes", provider=provider, run_id="r1")
    assert result.prompt == "Tell me about kubernetes"


@pytest.mark.asyncio
async def test_execute_task_with_model_override():
    provider = MockProvider()
    result = await execute_task(
        SAMPLE_TASK, "test", provider=provider, run_id="r2", model="custom-model"
    )
    assert result.model == "custom-model"
