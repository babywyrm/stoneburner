"""`--budget` is accepted by every eval command and defaults to no ceiling.

The default matters as much as the flag. Imposing a ceiling on CLI runs would
break anyone doing a large sweep today, and the project's stated rule is to add
flags rather than change defaults.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from atomics.cli import cli
from atomics.commands.common import eval_budget_from
from atomics.eval.budget import EvalBudget, GuardedProvider
from atomics.providers.base import BaseProvider

EVAL_COMMANDS = [
    "eval",
    "adversarial",
    "redblue",
    "multiturn",
    "refusal",
    "codereview",
    "rag",
    "codegen",
    "toolcall",
]


@pytest.mark.parametrize("command", EVAL_COMMANDS)
def test_every_eval_command_accepts_a_budget(command):
    result = CliRunner().invoke(cli, [command, "--help"])
    assert result.exit_code == 0
    assert "--budget" in result.output


@pytest.mark.parametrize("command", EVAL_COMMANDS)
def test_the_budget_help_explains_the_shared_ceiling(command):
    result = CliRunner().invoke(cli, [command, "--help"])
    assert "judge" in result.output.split("--budget", 1)[1][:300]


class TestEvalBudgetFrom:
    def test_no_flag_means_no_ceiling(self):
        assert eval_budget_from(None) is None

    def test_a_value_becomes_a_budget(self):
        budget = eval_budget_from(2.5)
        assert isinstance(budget, EvalBudget)
        assert budget.budget_limit_usd == 2.5

    @pytest.mark.parametrize("value", [0.0, -3.0])
    def test_a_non_positive_value_is_a_usage_error(self, value):
        """click.BadParameter renders as a usage message, not a traceback."""
        import click

        with pytest.raises(click.BadParameter):
            eval_budget_from(value)

    def test_the_usage_error_names_the_flag(self):
        """param_hint is rendered by format_message, not by str()."""
        import click

        with pytest.raises(click.BadParameter) as excinfo:
            eval_budget_from(0)
        assert "--budget" in excinfo.value.format_message()

    def test_an_invalid_budget_is_a_clean_cli_error_not_a_traceback(self):
        result = CliRunner().invoke(cli, ["eval", "--budget", "0"])
        assert result.exit_code != 0
        assert "--budget" in result.output
        assert "Traceback" not in result.output


class TestDefaultRemainsUnmetered:
    def test_providers_pass_through_untouched_without_the_flag(self):
        from atomics.eval.budget import share_budget
        from tests.test_eval_budget import StubProvider

        model, judge = StubProvider(), StubProvider()
        result = share_budget(eval_budget_from(None), model, judge)

        assert result == (model, judge)
        assert not any(isinstance(p, GuardedProvider) for p in result)

    def test_providers_are_wrapped_when_the_flag_is_given(self):
        from atomics.eval.budget import share_budget
        from tests.test_eval_budget import StubProvider

        result = share_budget(eval_budget_from(1.0), StubProvider(), StubProvider())

        assert all(isinstance(p, GuardedProvider) for p in result)
        assert isinstance(result[0], BaseProvider), "must stay a usable provider"
