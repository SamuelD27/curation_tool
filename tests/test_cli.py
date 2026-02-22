"""Tests for CLI interface."""
from click.testing import CliRunner
from curation_tool.cli import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "comfyui" in result.output.lower() or "curation" in result.output.lower()


def test_cli_run_missing_config():
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--config", "nonexistent.yaml"])
    assert result.exit_code != 0


def test_cli_health_command_exists():
    runner = CliRunner()
    result = runner.invoke(main, ["health", "--help"])
    assert result.exit_code == 0
    assert "comfyui" in result.output.lower() or "connectivity" in result.output.lower()


def test_cli_edit_command_exists():
    runner = CliRunner()
    result = runner.invoke(main, ["edit", "--help"])
    assert result.exit_code == 0
    assert "template" in result.output.lower()


def test_cli_face_presets():
    runner = CliRunner()
    result = runner.invoke(main, ["face", "presets"])
    assert result.exit_code == 0
    assert "headshot_20" in result.output


def test_cli_comfyui_url_option():
    runner = CliRunner()
    result = runner.invoke(main, ["--comfyui-url", "http://custom:9999", "--help"])
    assert result.exit_code == 0
