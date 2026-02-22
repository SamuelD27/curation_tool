"""Shared test fixtures for curation tool tests."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


@pytest.fixture
def fake_image():
    """Create a small test image."""
    return Image.new("RGB", (64, 64), color=(128, 128, 128))


@pytest.fixture
def mock_comfyui_client():
    """Mock ComfyUIClient that returns a fake image."""
    client = MagicMock()
    client.health_check.return_value = True
    client.get_system_stats.return_value = {
        "devices": [{"name": "TestGPU", "vram_used": 1e9, "vram_total": 4e9}],
        "system": {"python_version": "3.12", "os": "linux"},
    }
    client.upload_image.return_value = "uploaded_ref.png"
    client.queue_prompt.return_value = "test-prompt-id"
    client.wait_for_result.return_value = {"outputs": {}}
    client.get_images.return_value = [Image.new("RGB", (64, 64), color=(255, 0, 0))]
    client.close.return_value = None
    return client


@pytest.fixture
def mock_workflow_builder(tmp_path):
    """Mock WorkflowBuilder that returns minimal workflow dicts."""
    builder = MagicMock()
    builder.build_edit_workflow.return_value = {"1": {"class_type": "KSampler", "inputs": {}}}
    builder.build_identity_workflow.return_value = {"1": {"class_type": "KSampler", "inputs": {}}}
    builder.build_caption_workflow.return_value = {"1": {"class_type": "Florence2Run", "inputs": {}}}
    builder.build_upscale_workflow.return_value = {"1": {"class_type": "UpscaleModelLoader", "inputs": {}}}
    builder.build_face_restore_workflow.return_value = {"1": {"class_type": "FaceRestoreModelLoader", "inputs": {}}}
    return builder


@pytest.fixture
def patch_pipeline(mock_comfyui_client, mock_workflow_builder):
    """Patch pipeline.run_edit to use mocked ComfyUI client and workflow builder."""
    fake_result = Image.new("RGB", (64, 64), color=(255, 0, 0))

    def mock_run_edit(images=None, prompt="", comfyui_url="", **kwargs):
        return [fake_result]

    with patch("curation_tool.pipeline.run_edit", side_effect=mock_run_edit) as mock:
        yield mock


@pytest.fixture
def sample_workflows_dir(tmp_path):
    """Create minimal workflow templates for testing."""
    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir()

    flux_base = {
        "3": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 30, "cfg": 1.0}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "test", "images": ["8", 0]}},
        "20": {"class_type": "FluxGuidance", "inputs": {"guidance": 3.5}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    }
    (wf_dir / "flux_base.json").write_text(json.dumps(flux_base))

    pulid = {
        **flux_base,
        "11": {"class_type": "LoadImage", "inputs": {"image": "reference.png"}},
        "14": {"class_type": "ApplyPulidFlux", "inputs": {"weight": 0.8, "end_at": 0.8}},
        "15": {"class_type": "FaceRestoreCFWithModel", "inputs": {"codeformer_fidelity": 0.8}},
    }
    (wf_dir / "pulid_identity.json").write_text(json.dumps(pulid))

    caption = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "input.png"}},
        "4": {"class_type": "OllamaGenerate", "inputs": {"system_message": "TRIGGER_WORD"}},
        "6": {"class_type": "SaveText|pysssss", "inputs": {"filename_prefix": "caption"}},
    }
    (wf_dir / "caption_pipeline.json").write_text(json.dumps(caption))

    upscale = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "input.png"}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "RealESRGAN_x4plus.pth"}},
        "4": {"class_type": "ImageScaleToTotalPixels", "inputs": {"megapixels": 1.0}},
    }
    (wf_dir / "upscale_resize.json").write_text(json.dumps(upscale))

    face_restore = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "input.png"}},
        "2": {"class_type": "FaceRestoreModelLoader", "inputs": {"model_name": "codeformer.pth"}},
        "3": {"class_type": "FaceRestoreCFWithModel", "inputs": {"codeformer_fidelity": 0.8}},
    }
    (wf_dir / "face_restore.json").write_text(json.dumps(face_restore))

    flux2_base = {
        "4": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux2-dev.safetensors", "weight_dtype": "default"}},
        "10": {"class_type": "CLIPLoader", "inputs": {"clip_name": "mistral_3_small_flux2_fp8.safetensors", "type": "flux2", "device": "default"}},
        "11": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-ae.safetensors"}},
        "100": {"class_type": "LoadImage", "inputs": {"image": "reference.png"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": 1, "height": 1024, "width": 1024}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["10", 0], "text": ""}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["10", 0], "text": ""}},
        "20": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["6", 0], "guidance": 3.0}},
        "14": {"class_type": "ApplyPulidFlux", "inputs": {"model": ["25", 0], "pulid_flux": ["12", 0], "eva_clip": ["19", 0], "face_analysis": ["13", 0], "image": ["100", 0], "weight": 0.8, "start_at": 0.0, "end_at": 0.8}},
        "3": {"class_type": "KSampler", "inputs": {"cfg": 1.0, "denoise": 1.0, "latent_image": ["5", 0], "model": ["14", 0], "negative": ["7", 0], "positive": ["20", 0], "sampler_name": "euler", "scheduler": "beta", "seed": 0, "steps": 28}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["11", 0]}},
        "15": {"class_type": "FaceRestoreCFWithModel", "inputs": {"facerestore_model": ["16", 0], "image": ["8", 0], "facedetection": "retinaface_resnet50", "codeformer_fidelity": 0.8}},
        "200": {"class_type": "CR Prompt List", "inputs": {"body_text": "", "prepend_text": "", "append_text": ""}},
        "9": {"class_type": "SaveImageKJ", "inputs": {"images": ["15", 0], "caption": ["200", 0], "filename_prefix": "curation_tool", "extension": "txt"}},
    }
    (wf_dir / "flux2_base.json").write_text(json.dumps(flux2_base))

    qwen_face_edit = {
        "37": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen.safetensors", "weight_dtype": "fp8_e4m3fn"}},
        "38": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_clip.safetensors", "type": "qwen_image", "device": "default"}},
        "39": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_vae.safetensors"}},
        "89": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["37", 0], "lora_name": "Lightning.safetensors", "strength_model": 0.8}},
        "66": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["89", 0], "shift": 3}},
        "366": {"class_type": "LoadImage", "inputs": {"image": "source.png"}},
        "126": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["366", 0], "upscale_method": "lanczos", "megapixels": 1.0}},
        "177": {"class_type": "GetImageSize", "inputs": {"image": ["126", 0]}},
        "176": {"class_type": "EmptyLatentImage", "inputs": {"width": ["177", 0], "height": ["177", 1], "batch_size": 1}},
        "174": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["38", 0], "vae": ["39", 0], "image1": ["126", 0], "prompt": ""}},
        "111": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["38", 0], "text": ""}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["66", 0], "positive": ["174", 0], "negative": ["111", 0], "latent_image": ["176", 0], "seed": 0, "steps": 6, "cfg": 1, "sampler_name": "euler", "scheduler": "beta", "denoise": 1}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["39", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "curation_tool"}},
    }
    (wf_dir / "qwen_face_edit.json").write_text(json.dumps(qwen_face_edit))

    return wf_dir
