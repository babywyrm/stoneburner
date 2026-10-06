"""--show-prompt tracing and the `atomics prompts` catalog."""

from __future__ import annotations

import asyncio
import io
from collections.abc import Iterator

import pytest
from click.testing import CliRunner
from rich.console import Console

from atomics.cli import cli
from atomics.config import AtomicsSettings
from atomics.eval import runner as eval_runner
from atomics.prompts import catalog, names_for_system
from atomics.providers import trace
from atomics.providers.factory import make_provider
from atomics.providers.ollama import OllamaProvider
from tests.conftest import MockProvider


class _WindowedMock(MockProvider):
    def _num_ctx(self) -> int:
        return 8192


class _Failing(MockProvider):
    async def generate(self, prompt, **kwargs):  # type: ignore[no-untyped-def]
        raise TimeoutError("slow")


@pytest.fixture
def shown() -> Iterator[io.StringIO]:
    buf = io.StringIO()
    trace.enable(Console(file=buf, width=200, highlight=False))
    yield buf
    trace.disable()


def test_off_by_default_leaves_providers_unwrapped():
    provider = make_provider("ollama", "m", None, AtomicsSettings())
    assert isinstance(provider, OllamaProvider)


def test_on_wraps_factory_providers(shown):
    provider = make_provider("ollama", "m", None, AtomicsSettings())
    assert isinstance(provider, trace.TracedProvider)
    assert trace.traced(provider) is provider


def test_call_shows_named_system_prompt_and_context(shown):
    provider = trace.traced(_WindowedMock())
    response = asyncio.run(
        provider.generate("What is 2+2?", system=eval_runner._SYSTEM_PROMPT, max_tokens=64)
    )
    out = shown.getvalue()
    assert response.text == "response #1"
    assert "[built-in eval]" in out
    assert eval_runner._SYSTEM_PROMPT in out
    assert "What is 2+2?" in out
    assert "max_tokens=64" in out and "num_ctx=8192" in out
    assert "input=30" in out and "template/formatting" in out
    assert "context 90/8192 tokens" in out


def test_failed_call_is_shown_and_reraised(shown):
    with pytest.raises(TimeoutError):
        asyncio.run(trace.traced(_Failing()).generate("hi", system="custom words"))
    out = shown.getvalue()
    assert "[custom]" in out and "error   TimeoutError" in out


def test_catalog_names_are_unique_and_shared_prompts_list_both():
    names = [e.name for e in catalog()]
    assert len(names) == len(set(names))
    adversarial = next(e for e in catalog() if e.name == "adversarial")
    assert names_for_system(adversarial.system) == ["adversarial", "toolcall"]


def test_prompts_command_lists_and_shows_one():
    runner = CliRunner()
    listing = runner.invoke(cli, ["prompts"])
    assert listing.exit_code == 0 and "eval.judge" in listing.output
    one = runner.invoke(cli, ["prompts", "eval.judge"])
    assert one.exit_code == 0 and "ACCURACY" in one.output
    assert runner.invoke(cli, ["prompts", "nope"]).exit_code == 2
