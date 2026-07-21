def test_worker_help():
    from click.testing import CliRunner

    from atomics.commands.worker import worker
    runner = CliRunner()
    result = runner.invoke(worker, ["--help"])
    assert result.exit_code == 0
    assert "coordinator" in result.output

def test_distributed_help():
    from click.testing import CliRunner

    from atomics.commands.distributed import distributed
    runner = CliRunner()
    result = runner.invoke(distributed, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "status" in result.output
