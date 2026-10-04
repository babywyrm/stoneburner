"""Optional provider SDKs stay off the CLI import path."""

import subprocess
import sys


def test_cli_import_does_not_load_optional_deps():
    sdks = "{'anthropic', 'openai', 'boto3', 'fastapi'}"
    code = f"import sys, atomics.cli; print(sorted({sdks} & set(sys.modules)))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_claude_without_sdk_names_the_extra(monkeypatch):
    import pytest

    from atomics.providers.claude import ClaudeProvider

    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(ImportError, match="--extra claude"):
        ClaudeProvider(api_key="k")


def test_distributed_owner_default_matches_the_api_caller() -> None:
    from atomics.api.callers import ANONYMOUS_CALLER
    from atomics.distributed.models import ANONYMOUS_OWNER

    assert ANONYMOUS_OWNER == ANONYMOUS_CALLER
