"""Tests for batch processing engine."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

from curation_tool.config import CurationJob, EditTask
from curation_tool.batch import run_batch


def test_run_batch_creates_output_files(tmp_path):
    """Test that batch runner creates output images and metadata."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    # Create test input images
    for name in ["img1.png", "img2.png"]:
        Image.new("RGB", (64, 64), color=(128, 128, 128)).save(input_dir / name)

    job = CurationJob(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        tasks=[
            EditTask(source_image="img1.png", prompt="make it red"),
            EditTask(source_image="img2.png", prompt="make it blue"),
        ],
    )

    # Mock the pipeline to avoid loading the real model in tests
    mock_pipeline = MagicMock()
    mock_output = MagicMock()
    mock_output.images = [Image.new("RGB", (64, 64), color=(255, 0, 0))]
    mock_pipeline.return_value = mock_output

    results = run_batch(job, pipeline=mock_pipeline)

    assert len(results) == 2
    assert output_dir.exists()
    for r in results:
        assert Path(r["output_path"]).exists()

    # Check metadata file created
    assert (output_dir / "metadata.jsonl").exists()
