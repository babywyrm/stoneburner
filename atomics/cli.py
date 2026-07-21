"""CLI entry point — run, report, schedule, provider-test."""

from __future__ import annotations

import click

from atomics.commands import admin as admin_commands
from atomics.commands import api as api_commands
from atomics.commands import benchmark as benchmark_commands
from atomics.commands import distributed as distributed_commands
from atomics.commands import eval as eval_commands
from atomics.commands import load as load_commands
from atomics.commands import rag as rag_commands
from atomics.commands import security as security_commands
from atomics.commands import worker as worker_commands
from atomics.commands.auth import login, logout, secrets_group, whoami
from atomics.commands.common import setup_logging


@click.group()
@click.version_option(package_name="atomics")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable verbose/debug output.")
@click.option("--progress/--no-progress", default=True, show_default=True,
              help="Show real-time progress during long runs.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, progress: bool) -> None:
    """Atomics — Agentic token usage benchmarking platform."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["progress"] = progress
    if verbose:
        setup_logging("DEBUG", rich_tracebacks=True)
    else:
        setup_logging("WARNING")

cli.add_command(security_commands.refusal)
cli.add_command(security_commands.codereview)
cli.add_command(login)
cli.add_command(logout)
cli.add_command(whoami)
cli.add_command(secrets_group)
cli.add_command(admin_commands.doctor)
cli.add_command(admin_commands.schedule)
cli.add_command(admin_commands.schedule_status)
cli.add_command(admin_commands.export)
cli.add_command(admin_commands.models)
cli.add_command(admin_commands.provider_test)
cli.add_command(admin_commands.completion)
cli.add_command(load_commands.stress)
cli.add_command(load_commands.soak)
cli.add_command(load_commands.scenario)
cli.add_command(load_commands.capacity)
cli.add_command(load_commands.baselines)
cli.add_command(benchmark_commands.run)
cli.add_command(benchmark_commands.report)
cli.add_command(benchmark_commands.compare)
cli.add_command(benchmark_commands.tiers)
cli.add_command(benchmark_commands.sweep)
cli.add_command(benchmark_commands.labcompare)
cli.add_command(eval_commands.eval)
cli.add_command(eval_commands.advisor)
cli.add_command(rag_commands.codegen)
cli.add_command(security_commands.multiturn)
cli.add_command(rag_commands.rag)
cli.add_command(rag_commands.rag_index)
cli.add_command(rag_commands.rag_retrieval)
cli.add_command(security_commands.adversarial)
cli.add_command(security_commands.redblue)
cli.add_command(rag_commands.probe)
cli.add_command(rag_commands.qa)
cli.add_command(rag_commands.archreview)

# Re-export for tests/importers that still pull helpers from atomics.cli.
_write_generic_export = admin_commands._write_generic_export
_run_labcompare_sync = benchmark_commands._run_labcompare_sync
_parse_model_spec = security_commands._parse_model_spec
_make_provider = security_commands._make_provider
refusal = security_commands.refusal
codereview = security_commands.codereview

cli.add_command(api_commands.server)
cli.add_command(worker_commands.worker)
cli.add_command(distributed_commands.distributed)
