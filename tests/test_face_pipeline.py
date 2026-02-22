"""Tests for multi-stage face pipeline."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from curation_tool.face_pipeline import (
    FacePipelineConfig,
    PipelineState,
    export_face_dataset,
    run_face_pipeline,
)
from curation_tool.stages import StageConfig


class TestFacePipelineConfig:
    def test_from_yaml(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("""
trigger_word: "sks_sam"
input_dir: ./input
output_dir: ./output
stages:
  - name: refine
    type: refine
    source_image: "photo.png"
    num_candidates: 2
    seeds: [42, 43]
    prompt: "Improve headshot"
  - name: angles
    type: angles
    preset: "headshot_20"
""")
        config = FacePipelineConfig.from_yaml(cfg_file)
        assert config.trigger_word == "sks_sam"
        assert len(config.stages) == 2
        assert config.stages[0].type == "refine"
        assert config.stages[1].preset == "headshot_20"

    def test_defaults(self):
        config = FacePipelineConfig(
            stages=[StageConfig(name="test", type="refine")]
        )
        assert config.trigger_word == "sks_person"
        assert config.default_seed == 42
        assert config.default_num_steps == 6
        assert config.comfyui.port == 8188

    def test_comfyui_url_property(self):
        config = FacePipelineConfig(
            stages=[StageConfig(name="test", type="refine")]
        )
        assert config.comfyui.url == "http://127.0.0.1:8188"


class TestPipelineState:
    def test_save_and_load(self, tmp_path):
        state = PipelineState(
            completed_stages=["refine"],
            picks={"refine": "/tmp/pick.png"},
            current_source="/tmp/pick.png",
        )
        state_path = tmp_path / "state.json"
        state.save(state_path)

        loaded = PipelineState.load(state_path)
        assert loaded.completed_stages == ["refine"]
        assert loaded.picks["refine"] == "/tmp/pick.png"
        assert loaded.current_source == "/tmp/pick.png"

    def test_empty_state(self):
        state = PipelineState()
        assert state.completed_stages == []
        assert state.picks == {}
        assert state.current_source is None


class TestRunFacePipeline:
    @patch("curation_tool.stages.run_edit")
    def test_refine_then_angles_with_pick(self, mock_run_edit, tmp_path):
        mock_run_edit.return_value = [Image.new("RGB", (64, 64), color=(255, 0, 0))]

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        source = input_dir / "photo.png"
        Image.new("RGB", (64, 64)).save(source)

        output_dir = tmp_path / "output"

        config = FacePipelineConfig(
            trigger_word="sks_test",
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            stages=[
                StageConfig(
                    name="refine",
                    type="refine",
                    source_image="photo.png",
                    num_candidates=2,
                    seeds=[42, 43],
                    prompt="Improve it",
                ),
                StageConfig(
                    name="angles",
                    type="angles",
                    prompts=["left", "right", "front"],
                ),
            ],
        )

        # Pre-supply the pick so it doesn't pause
        results = run_face_pipeline(
            config,
            comfyui_url="http://test:8188",
            interactive=False,
            picks={"refine": "refine_0000.png"},
        )

        # Should have 3 angle results
        assert len(results) == 3
        assert all(r["stage"] == "angles" for r in results)

        # State file should exist
        state_path = output_dir / "pipeline_state.json"
        assert state_path.exists()
        state = PipelineState.load(state_path)
        assert "refine" in state.completed_stages
        assert "angles" in state.completed_stages

    @patch("curation_tool.stages.run_edit")
    def test_pauses_without_pick(self, mock_run_edit, tmp_path):
        mock_run_edit.return_value = [Image.new("RGB", (64, 64), color=(255, 0, 0))]

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        source = input_dir / "photo.png"
        Image.new("RGB", (64, 64)).save(source)

        output_dir = tmp_path / "output"

        config = FacePipelineConfig(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            stages=[
                StageConfig(
                    name="refine",
                    type="refine",
                    source_image="photo.png",
                    num_candidates=2,
                    prompt="Improve it",
                ),
                StageConfig(
                    name="angles",
                    type="angles",
                    prompts=["left", "right"],
                ),
            ],
        )

        # Non-interactive, no picks -> should pause after refine
        results = run_face_pipeline(
            config,
            comfyui_url="http://test:8188",
            interactive=False,
            picks={},
        )

        # No angle results since pipeline paused
        assert len(results) == 0

        # But refine stage completed
        state = PipelineState.load(output_dir / "pipeline_state.json")
        assert "refine" in state.completed_stages
        assert "angles" not in state.completed_stages

    @patch("curation_tool.stages.run_edit")
    def test_resume_after_pick(self, mock_run_edit, tmp_path):
        mock_run_edit.return_value = [Image.new("RGB", (64, 64), color=(255, 0, 0))]

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        source = input_dir / "photo.png"
        Image.new("RGB", (64, 64)).save(source)

        output_dir = tmp_path / "output"

        config = FacePipelineConfig(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            stages=[
                StageConfig(
                    name="refine",
                    type="refine",
                    source_image="photo.png",
                    num_candidates=2,
                    prompt="Improve it",
                ),
                StageConfig(
                    name="angles",
                    type="angles",
                    prompts=["left", "right"],
                ),
            ],
        )

        # First run: pauses
        run_face_pipeline(
            config, comfyui_url="http://test:8188", interactive=False
        )

        # Now resume with pick
        picked = output_dir / "stage_refine" / "refine_0000.png"
        results = run_face_pipeline(
            config,
            comfyui_url="http://test:8188",
            interactive=False,
            picks={"refine": str(picked)},
        )

        assert len(results) == 2
        state = PipelineState.load(output_dir / "pipeline_state.json")
        assert "angles" in state.completed_stages


class TestExportFaceDataset:
    def test_exports_with_trigger_word(self, tmp_path):
        # Create fake results
        results = []
        src_dir = tmp_path / "stage_angles"
        src_dir.mkdir()
        for i in range(3):
            img_path = src_dir / f"angles_{i:04d}.png"
            Image.new("RGB", (64, 64)).save(img_path)
            results.append({
                "stage": "angles",
                "output_path": str(img_path),
                "prompt": f"prompt {i}",
            })

        config = FacePipelineConfig(
            trigger_word="sks_sam",
            output_dir=str(tmp_path),
            stages=[StageConfig(name="angles", type="angles", prompts=["a"])],
        )

        export_dir = export_face_dataset(config, results)

        assert export_dir.exists()
        pngs = sorted(export_dir.glob("*.png"))
        txts = sorted(export_dir.glob("*.txt"))
        assert len(pngs) == 3
        assert len(txts) == 3

        for txt in txts:
            assert txt.read_text().strip() == "sks_sam"

    def test_exports_with_caption_template(self, tmp_path):
        """Caption template from config is used when no trigger_word override."""
        results = []
        src_dir = tmp_path / "stage_body"
        src_dir.mkdir()
        for i in range(2):
            img_path = src_dir / f"body_{i:04d}.png"
            Image.new("RGB", (64, 64)).save(img_path)
            results.append({
                "stage": "body",
                "output_path": str(img_path),
                "prompt": f"full body pose {i}",
                "seed": 42 + i,
            })

        config = FacePipelineConfig(
            trigger_word="sks_sam",
            caption_template="{prompt}",
            output_dir=str(tmp_path),
            stages=[StageConfig(name="body", type="body", prompts=["a"])],
        )

        # trigger_word takes precedence (same as basic workflow)
        export_dir = export_face_dataset(config, results)
        txts = sorted(export_dir.glob("*.txt"))
        assert len(txts) == 2
        for txt in txts:
            assert txt.read_text().strip() == "sks_sam"

    def test_exports_with_caption_template_override(self, tmp_path):
        """CLI caption_template override works when passed explicitly."""
        results = []
        src_dir = tmp_path / "stage_body"
        src_dir.mkdir()
        for i in range(2):
            img_path = src_dir / f"body_{i:04d}.png"
            Image.new("RGB", (64, 64)).save(img_path)
            results.append({
                "stage": "body",
                "output_path": str(img_path),
                "prompt": f"full body pose {i}",
                "seed": 42 + i,
            })

        config = FacePipelineConfig(
            trigger_word="sks_sam",
            output_dir=str(tmp_path),
            stages=[StageConfig(name="body", type="body", prompts=["a"])],
        )

        # trigger_word still takes precedence via export_lora_dataset logic
        export_dir = export_face_dataset(config, results, caption_template="{prompt}")
        txts = sorted(export_dir.glob("*.txt"))
        for txt in txts:
            assert txt.read_text().strip() == "sks_sam"
