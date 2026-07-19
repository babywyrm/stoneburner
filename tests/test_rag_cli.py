from pathlib import Path

import pytest
from click.testing import CliRunner

try:
    import sentence_transformers
    import sqlite_vec
except ImportError:
    sqlite_vec = None
    sentence_transformers = None

_RAG_EXTRAS_MISSING = sqlite_vec is None or sentence_transformers is None

from atomics.cli import cli


def test_rag_cli_has_index_options():
    runner = CliRunner()
    result = runner.invoke(cli, ["rag", "--help"])
    assert result.exit_code == 0
    assert "--index" in result.output
    assert "--top-k" in result.output


@pytest.mark.skipif(_RAG_EXTRAS_MISSING, reason="rag extras not installed")
def test_rag_index_builds_database(tmp_path: Path):
    (tmp_path / "doc.txt").write_text("hello world, this is a test document")
    runner = CliRunner()
    db_path = tmp_path / "index.vec"
    result = runner.invoke(cli, ["rag-index", str(tmp_path), "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert db_path.exists()
    assert "chunks" in result.output.lower() or "stored" in result.output.lower()


def test_rag_retrieval_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["rag-retrieval", "--help"])
    assert result.exit_code == 0
    assert "--index" in result.output
    assert "--gold" in result.output


@pytest.mark.skipif(_RAG_EXTRAS_MISSING, reason="rag extras not installed")
def test_rag_retrieval_empty_gold_exits_cleanly(tmp_path: Path):
    (tmp_path / "doc.txt").write_text("hello world, this is a test document")
    runner = CliRunner()
    db_path = tmp_path / "index.vec"
    index_result = runner.invoke(cli, ["rag-index", str(tmp_path), "--db", str(db_path)])
    assert index_result.exit_code == 0, index_result.output

    gold_path = tmp_path / "gold.json"
    gold_path.write_text("{}")
    result = runner.invoke(
        cli,
        ["rag-retrieval", "--index", str(db_path), "--gold", str(gold_path)],
    )
    assert result.exit_code == 0, result.output
    assert "No queries in gold file" in result.output
    assert "0.000" in result.output
    assert "MRR: 0.000" in result.output
