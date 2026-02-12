"""Tests for multi-stage face pipeline."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from curation_tool.face_pipeline import (
    FacePipelineConfig,
    PipelineState,
    export_face_dataset,
    run_face_pipeline,
)
from curation_tool.stages import StageConfig


def _make_mock_pipeline():
    mock = MagicMock()
    mock_output = MagicMock()
    mock_output.images = [Image.new("RGB", (64, 64), color=(255, 0, 0))]
    mock.return_value = mock_output
    return mock


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
        assert config.default_num_steps == 50


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
    def test_refine_then_angles_with_pick(self, tmp_path):
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

        mock_pipeline = _make_mock_pipeline()

        # Pre-supply the pick so it doesn't pause
        results = run_face_pipeline(
            config,
            pipeline=mock_pipeline,
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

    def test_pauses_without_pick(self, tmp_path):
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

        mock_pipeline = _make_mock_pipeline()

        # Non-interactive, no picks -> should pause after refine
        results = run_face_pipeline(
            config,
            pipeline=mock_pipeline,
            interactive=False,
            picks={},
        )

        # No angle results since pipeline paused
        assert len(results) == 0

        # But refine stage completed
        state = PipelineState.load(output_dir / "pipeline_state.json")
        assert "refine" in state.completed_stages
        assert "angles" not in state.completed_stages

    def test_resume_after_pick(self, tmp_path):
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

        mock_pipeline = _make_mock_pipeline()

        # First run: pauses
        run_face_pipeline(config, pipeline=mock_pipeline, interactive=False)

        # Now resume with pick
        picked = output_dir / "stage_refine" / "refine_0000.png"
        results = run_face_pipeline(
            config,
            pipeline=mock_pipeline,
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
