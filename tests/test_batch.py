"""Tests for batch processing engine."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from curation_tool.batch import run_batch
from curation_tool.config import CurationJob, EditTask


class TestRunBatch:
    @patch("curation_tool.batch.run_edit")
    @patch("curation_tool.batch.ComfyUIClient")
    def test_creates_output_files(self, MockClient, mock_run_edit, tmp_path):
        # Setup mock client
        mock_client = MockClient.return_value
        mock_client.health_check.return_value = True
        mock_client.get_system_stats.return_value = {"devices": []}

        # Mock run_edit to return a fake image
        mock_run_edit.return_value = [Image.new("RGB", (64, 64), color=(255, 0, 0))]

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

        results = run_batch(job, comfyui_url="http://test:8188")

        assert len(results) == 2
        assert output_dir.exists()
        for r in results:
            assert Path(r["output_path"]).exists()
        assert (output_dir / "metadata.jsonl").exists()

    @patch("curation_tool.batch.ComfyUIClient")
    def test_health_check_fails(self, MockClient, tmp_path):
        mock_client = MockClient.return_value
        mock_client.health_check.return_value = False

        job = CurationJob(
            input_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            tasks=[EditTask(source_image="img.png", prompt="test")],
        )

        with pytest.raises(RuntimeError, match="not reachable"):
            run_batch(job, comfyui_url="http://test:8188")
