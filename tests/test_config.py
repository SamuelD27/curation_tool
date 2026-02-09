"""Tests for curation job configuration."""
import pytest
from curation_tool.config import CurationJob, EditTask


def test_edit_task_minimal():
    task = EditTask(
        source_image="input.png",
        prompt="Make the background blue",
    )
    assert task.source_image == "input.png"
    assert task.prompt == "Make the background blue"
    assert task.seed == 0
    assert task.num_steps == 40


def test_curation_job_from_yaml(tmp_path):
    config_file = tmp_path / "job.yaml"
    config_file.write_text("""
input_dir: ./raw_images
output_dir: ./curated
tasks:
  - source_image: "photo1.png"
    prompt: "Remove the background and replace with white"
    seed: 42
  - source_image: "photo2.png"
    prompt: "Change lighting to golden hour"
""")
    job = CurationJob.from_yaml(config_file)
    assert len(job.tasks) == 2
    assert job.input_dir == "./raw_images"
    assert job.output_dir == "./curated"


def test_curation_job_validates_empty_tasks():
    with pytest.raises(ValueError):
        CurationJob(input_dir=".", output_dir=".", tasks=[])
