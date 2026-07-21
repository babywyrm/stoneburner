"""Tests for distributed worker task execution runner."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atomics.distributed.models import TaskAssignment
from atomics.distributed.worker_runner import execute_assignment
from atomics.models import TaskCategory, TaskResult, TaskStatus


@pytest.mark.asyncio
async def test_execute_assignment_with_mock_provider():
    assignment = TaskAssignment(
        assignment_id="a1",
        job_id="j1",
        task_spec={"task_name": "general_qa/quick_question", "prompt": "hello"},
    )
    fake_provider = MagicMock()
    fake_provider.name = "mock"

    fake_result = TaskResult(
        run_id="j1",
        category=TaskCategory.GENERAL_QA,
        task_name="quick_question",
        provider="mock",
        model="mock-model",
        status=TaskStatus.SUCCESS,
        prompt="hello",
        response="world",
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        latency_ms=12.5,
        completed_at=datetime.now(UTC),
    )

    with (
        patch(
            "atomics.distributed.worker_runner._make_provider",
            return_value=fake_provider,
        ) as make_provider,
        patch(
            "atomics.distributed.worker_runner.execute_task",
            new_callable=AsyncMock,
            return_value=fake_result,
        ) as exec_task,
        patch(
            "atomics.distributed.worker_runner.load_settings",
            return_value=MagicMock(name="settings"),
        ),
    ):
        result = await execute_assignment(
            assignment,
            provider_name="ollama",
            model="qwen3:14b",
        )

    make_provider.assert_called_once()
    assert make_provider.call_args.args[0] == "ollama"
    assert make_provider.call_args.args[1] == "qwen3:14b"

    exec_task.assert_awaited_once()
    call_kwargs = exec_task.await_args.kwargs
    assert call_kwargs["provider"] is fake_provider
    assert call_kwargs["run_id"] == "j1"
    assert call_kwargs["model"] == "qwen3:14b"
    task_arg = exec_task.await_args.args[0]
    prompt_arg = exec_task.await_args.args[1]
    assert task_arg.name == "quick_question"
    assert prompt_arg == "hello"

    assert isinstance(result, dict)
    assert result["run_id"] == "j1"
    assert result["task_name"] == "quick_question"
    assert result["status"] == "success"
    assert result["response"] == "world"
    assert result["provider"] == "mock"


@pytest.mark.asyncio
async def test_execute_assignment_builds_fallback_task():
    assignment = TaskAssignment(
        assignment_id="a2",
        job_id="j2",
        task_spec={
            "task_name": "custom_suite/custom_task",
            "prompt": "do the thing",
            "max_output_tokens": 512,
        },
    )
    fake_result = TaskResult(
        run_id="j2",
        category=TaskCategory.GENERAL_QA,
        task_name="custom_task",
        provider="mock",
        model="m",
        status=TaskStatus.SUCCESS,
        response="ok",
    )

    with (
        patch(
            "atomics.distributed.worker_runner._make_provider",
            return_value=MagicMock(name="provider"),
        ),
        patch(
            "atomics.distributed.worker_runner.execute_task",
            new_callable=AsyncMock,
            return_value=fake_result,
        ) as exec_task,
        patch(
            "atomics.distributed.worker_runner.load_settings",
            return_value=MagicMock(name="settings"),
        ),
    ):
        result = await execute_assignment(assignment)

    task_arg = exec_task.await_args.args[0]
    assert task_arg.name == "custom_task"
    assert task_arg.prompt_template == "{prompt}"
    assert task_arg.max_output_tokens == 512
    assert result["task_name"] == "custom_task"
