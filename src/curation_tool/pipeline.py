"""Core pipeline for Qwen-Image-Edit-2509 inference."""
import logging
from pathlib import Path

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

logger = logging.getLogger(__name__)

_pipeline = None


def load_pipeline(
    model_id: str = "Qwen/Qwen-Image-Edit-2509",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> QwenImageEditPlusPipeline:
    """Load and cache the Qwen-Image-Edit-2509 pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    logger.info("Loading %s on %s with %s...", model_id, device, dtype)
    _pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_id, torch_dtype=dtype
    )
    _pipeline.to(device)
    _pipeline.set_progress_bar_config(disable=None)
    logger.info("Pipeline loaded.")
    return _pipeline


def run_edit(
    pipeline: QwenImageEditPlusPipeline,
    images: list[Image.Image],
    prompt: str,
    seed: int = 0,
    num_steps: int = 40,
    cfg_scale: float = 4.0,
    guidance_scale: float = 1.0,
) -> Image.Image:
    """Run a single image edit and return the result."""
    inputs = {
        "image": images,
        "prompt": prompt,
        "generator": torch.manual_seed(seed),
        "true_cfg_scale": cfg_scale,
        "negative_prompt": " ",
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": 1,
    }

    with torch.inference_mode():
        output = pipeline(**inputs)

    return output.images[0]
