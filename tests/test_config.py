"""Tests for curation job configuration."""
import pytest
from curation_tool.config import ComfyUISettings, CurationJob, EditTask, WorkflowSettings


def test_edit_task_minimal():
    task = EditTask(
        source_image="input.png",
        prompt="Make the background blue",
    )
    assert task.source_image == "input.png"
    assert task.prompt == "Make the background blue"
    assert task.seed is None
    assert task.num_steps is None


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


def test_comfyui_settings_defaults():
    settings = ComfyUISettings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 8188
    assert settings.url == "http://127.0.0.1:8188"


def test_workflow_settings_defaults():
    settings = WorkflowSettings()
    assert settings.template == "pulid_identity"
    assert settings.face_restore is True


def test_curation_job_has_comfyui_settings():
    job = CurationJob(
        input_dir=".",
        output_dir=".",
        tasks=[EditTask(source_image="a.png", prompt="test")],
    )
    assert job.comfyui.url == "http://127.0.0.1:8188"
    assert job.default_num_steps == 30
    assert job.default_cfg_scale == 3.5
