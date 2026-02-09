"""Tests for CLI interface."""
from click.testing import CliRunner
from curation_tool.cli import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "curation" in result.output.lower() or "edit" in result.output.lower()


def test_cli_run_missing_config():
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--config", "nonexistent.yaml"])
    assert result.exit_code != 0
