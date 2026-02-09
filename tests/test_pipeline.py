"""Smoke test for pipeline loading and inference."""
import pytest
from pathlib import Path


def test_pipeline_module_imports():
    from curation_tool.pipeline import load_pipeline, run_edit
    assert callable(load_pipeline)
    assert callable(run_edit)
