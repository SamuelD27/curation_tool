"""Tests for workflow builder."""
import json
from pathlib import Path

import pytest

from curation_tool.workflow_builder import WorkflowBuilder


class TestGetTemplate:
    def test_loads_existing_template(self, sample_workflows_dir):
        builder = WorkflowBuilder(sample_workflows_dir)
        wf = builder.get_template("flux_base")
        assert "3" in wf
        assert wf["3"]["class_type"] == "KSampler"

    def test_missing_template_raises(self, sample_workflows_dir):
        builder = WorkflowBuilder(sample_workflows_dir)
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            builder.get_template("nonexistent")

    def test_returns_deep_copy(self, sample_workflows_dir):
        builder = WorkflowBuilder(sample_workflows_dir)
        wf1 = builder.get_template("flux_base")
        wf2 = builder.get_template("flux_base")
        wf1["3"]["inputs"]["seed"] = 999
        assert wf2["3"]["inputs"]["seed"] != 999


class TestBuildEditWorkflow:
    def test_sets_prompt_and_seed(self, sample_workflows_dir):
        builder = WorkflowBuilder(sample_workflows_dir)
        wf = builder.build_edit_workflow(
            source_image="test.png",
            prompt="a beautiful sunset",
            seed=42,
            steps=20,
            cfg=4.0,
        )
        # KSampler should have updated seed/steps
        assert wf["3"]["inputs"]["seed"] == 42
        assert wf["3"]["inputs"]["steps"] == 20
        # Positive prompt
        assert wf["6"]["inputs"]["text"] == "a beautiful sunset"
        # FluxGuidance
        assert wf["20"]["inputs"]["guidance"] == 4.0

    def test_sets_dimensions(self, sample_workflows_dir):
        builder = WorkflowBuilder(sample_workflows_dir)
        wf = builder.build_edit_workflow(
            source_image="test.png",
            prompt="test",
            width=512,
            height=768,
        )
        assert wf["5"]["inputs"]["width"] == 512
        assert wf["5"]["inputs"]["height"] == 768


class TestBuildIdentityWorkflow:
    def test_sets_pulid_params(self, sample_workflows_dir):
        builder = WorkflowBuilder(sample_workflows_dir)
        wf = builder.build_identity_workflow(
            reference_image="ref.png",
            prompt="portrait photo",
            seed=10,
            weight=0.9,
            end_at=0.7,
            face_restore_fidelity=0.6,
        )
        assert wf["11"]["inputs"]["image"] == "ref.png"
        assert wf["14"]["inputs"]["weight"] == 0.9
        assert wf["14"]["inputs"]["end_at"] == 0.7
        assert wf["15"]["inputs"]["codeformer_fidelity"] == 0.6
        assert wf["6"]["inputs"]["text"] == "portrait photo"


class TestBuildCaptionWorkflow:
    def test_sets_trigger_word(self, sample_workflows_dir):
        builder = WorkflowBuilder(sample_workflows_dir)
        wf = builder.build_caption_workflow(
            source_image="img.png",
            trigger_word="sks_sam",
        )
        assert wf["1"]["inputs"]["image"] == "img.png"
        assert "sks_sam" in wf["4"]["inputs"]["system_message"]


class TestBuildUpscaleWorkflow:
    def test_sets_model_and_megapixels(self, sample_workflows_dir):
        builder = WorkflowBuilder(sample_workflows_dir)
        wf = builder.build_upscale_workflow(
            source_image="img.png",
            model_name="4x-UltraSharp.pth",
            megapixels=2.0,
        )
        assert wf["1"]["inputs"]["image"] == "img.png"
        assert wf["2"]["inputs"]["model_name"] == "4x-UltraSharp.pth"
        assert wf["4"]["inputs"]["megapixels"] == 2.0


class TestBuildFaceRestoreWorkflow:
    def test_sets_fidelity(self, sample_workflows_dir):
        builder = WorkflowBuilder(sample_workflows_dir)
        wf = builder.build_face_restore_workflow(
            source_image="face.png",
            fidelity=0.5,
        )
        assert wf["1"]["inputs"]["image"] == "face.png"
        assert wf["3"]["inputs"]["codeformer_fidelity"] == 0.5


class TestBuildFlux2Workflow:
    def test_pulid_txt2img(self, sample_workflows_dir):
        builder = WorkflowBuilder(sample_workflows_dir)
        wf = builder.build_flux2_workflow(
            prompt="portrait photo",
            reference_image="face_ref.png",
            seed=42,
            pulid_weight=0.9,
            pulid_end_at=0.7,
        )
        # LoadImage should have the face reference
        assert wf["100"]["inputs"]["image"] == "face_ref.png"
        # PuLID weight and end_at
        assert wf["14"]["inputs"]["weight"] == 0.9
        assert wf["14"]["inputs"]["end_at"] == 0.7
        # EmptyLatentImage (txt2img, not img2img)
        assert wf["5"]["class_type"] == "EmptyLatentImage"
        assert wf["5"]["inputs"]["width"] == 1024
        assert wf["5"]["inputs"]["height"] == 1024
        # KSampler uses EmptyLatentImage, full denoise
        assert wf["3"]["inputs"]["latent_image"] == ["5", 0]
        assert wf["3"]["inputs"]["denoise"] == 1.0
        assert wf["3"]["inputs"]["seed"] == 42

    def test_prompt_and_sampler(self, sample_workflows_dir):
        builder = WorkflowBuilder(sample_workflows_dir)
        wf = builder.build_flux2_workflow(
            prompt="woman walking on beach",
            cfg=4.0,
            steps=20,
            negative_prompt="blurry",
        )
        # CR Prompt List body_text
        assert wf["200"]["inputs"]["body_text"] == "woman walking on beach"
        # CLIPTextEncode positive
        assert wf["6"]["inputs"]["text"] == "woman walking on beach"
        # CLIPTextEncode negative
        assert wf["7"]["inputs"]["text"] == "blurry"
        # FluxGuidance
        assert wf["20"]["inputs"]["guidance"] == 4.0
        # KSampler scheduler is beta
        assert wf["3"]["inputs"]["scheduler"] == "beta"
        assert wf["3"]["inputs"]["steps"] == 20

    def test_face_restore_fidelity(self, sample_workflows_dir):
        builder = WorkflowBuilder(sample_workflows_dir)
        wf = builder.build_flux2_workflow(
            prompt="headshot",
            face_restore_fidelity=0.5,
        )
        assert wf["15"]["inputs"]["codeformer_fidelity"] == 0.5


class TestBuildQwenFaceWorkflow:
    def test_sets_source_and_prompt(self, sample_workflows_dir):
        builder = WorkflowBuilder(sample_workflows_dir)
        wf = builder.build_qwen_face_workflow(
            source_image="face.png",
            prompt="profile view",
            seed=99,
            steps=4,
        )
        assert wf["366"]["inputs"]["image"] == "face.png"
        assert wf["174"]["inputs"]["prompt"] == "profile view"
        assert wf["3"]["inputs"]["seed"] == 99
        assert wf["3"]["inputs"]["steps"] == 4
