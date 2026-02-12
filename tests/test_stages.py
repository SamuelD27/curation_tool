"""Tests for stage config and task expansion."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from curation_tool.stages import StageConfig, expand_stage_tasks, run_stage


class TestExpandStageTasks:
    def test_refine_creates_n_candidates(self):
        stage = StageConfig(
            name="refine",
            type="refine",
            num_candidates=4,
            prompt="Make it perfect",
            seeds=[10, 20, 30, 40],
        )
        tasks = expand_stage_tasks(stage, source_image="photo.png")
        assert len(tasks) == 4
        assert [t.seed for t in tasks] == [10, 20, 30, 40]
        assert all(t.prompt == "Make it perfect" for t in tasks)

    def test_refine_auto_seeds(self):
        stage = StageConfig(
            name="refine",
            type="refine",
            num_candidates=3,
        )
        tasks = expand_stage_tasks(stage, source_image="photo.png", base_seed=100)
        assert len(tasks) == 3
        assert [t.seed for t in tasks] == [100, 101, 102]

    def test_refine_uses_default_prompt_when_none(self):
        stage = StageConfig(name="refine", type="refine", num_candidates=1)
        tasks = expand_stage_tasks(stage, source_image="photo.png")
        assert len(tasks) == 1
        assert "headshot" in tasks[0].prompt.lower()

    def test_angles_with_preset(self):
        stage = StageConfig(
            name="angles",
            type="angles",
            preset="headshot_20",
        )
        tasks = expand_stage_tasks(stage, source_image="photo.png")
        assert len(tasks) == 20

    def test_angles_with_custom_prompts(self):
        stage = StageConfig(
            name="angles",
            type="angles",
            prompts=["left view", "right view", "front view"],
        )
        tasks = expand_stage_tasks(stage, source_image="photo.png")
        assert len(tasks) == 3
        assert tasks[0].prompt == "left view"

    def test_body_with_preset(self):
        stage = StageConfig(
            name="body",
            type="body",
            preset="body_10",
        )
        tasks = expand_stage_tasks(stage, source_image="photo.png")
        assert len(tasks) == 10

    def test_angles_requires_preset_or_prompts(self):
        stage = StageConfig(name="angles", type="angles")
        with pytest.raises(ValueError, match="requires either"):
            expand_stage_tasks(stage, source_image="photo.png")

    def test_uses_stage_num_steps_and_cfg(self):
        stage = StageConfig(
            name="refine",
            type="refine",
            num_candidates=1,
            num_steps=30,
            cfg_scale=2.5,
        )
        tasks = expand_stage_tasks(stage, source_image="photo.png")
        assert tasks[0].num_steps == 30
        assert tasks[0].cfg_scale == 2.5

    def test_falls_back_to_default_num_steps_and_cfg(self):
        stage = StageConfig(name="refine", type="refine", num_candidates=1)
        tasks = expand_stage_tasks(
            stage, source_image="photo.png", default_num_steps=60, default_cfg_scale=5.0
        )
        assert tasks[0].num_steps == 60
        assert tasks[0].cfg_scale == 5.0

    def test_source_image_override(self):
        stage = StageConfig(
            name="refine",
            type="refine",
            source_image="override.png",
            num_candidates=1,
        )
        tasks = expand_stage_tasks(stage, source_image="fallback.png")
        assert tasks[0].source_image == "override.png"

    def test_seed_zero_preserved(self):
        stage = StageConfig(
            name="refine",
            type="refine",
            num_candidates=1,
            seeds=[0],
        )
        tasks = expand_stage_tasks(stage, source_image="photo.png", base_seed=99)
        assert tasks[0].seed == 0


class TestRunStage:
    def test_run_stage_creates_outputs(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        source = input_dir / "photo.png"
        Image.new("RGB", (64, 64)).save(source)

        output_dir = tmp_path / "output"

        mock_pipeline = MagicMock()
        mock_output = MagicMock()
        mock_output.images = [Image.new("RGB", (64, 64), color=(255, 0, 0))]
        mock_pipeline.return_value = mock_output

        stage = StageConfig(
            name="refine",
            type="refine",
            num_candidates=2,
            seeds=[10, 20],
            prompt="Improve this",
        )

        results = run_stage(
            stage,
            source_image_path=source,
            input_dir=input_dir,
            output_dir=output_dir,
            pipeline=mock_pipeline,
        )

        assert len(results) == 2
        stage_dir = output_dir / "stage_refine"
        assert stage_dir.exists()
        assert (stage_dir / "refine_0000.png").exists()
        assert (stage_dir / "refine_0001.png").exists()
        assert (stage_dir / "metadata.jsonl").exists()

        lines = (stage_dir / "metadata.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["stage"] == "refine"
        assert record["seed"] == 10
