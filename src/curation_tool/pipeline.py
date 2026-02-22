"""Core pipeline facade over ComfyUI for image generation."""
import logging
from typing import Callable

from PIL import Image

from curation_tool.comfyui_client import ComfyUIClient
from curation_tool.workflow_builder import WorkflowBuilder

logger = logging.getLogger(__name__)


def run_edit(
    images: list[Image.Image],
    prompt: str,
    seed: int = 0,
    num_steps: int = 30,
    cfg_scale: float = 3.5,
    negative_prompt: str = "",
    num_images: int = 1,
    template: str = "flux_base",
    reference_image: Image.Image | None = None,
    identity_strength: float = 0.8,
    face_restore: bool = True,
    face_restore_fidelity: float = 0.8,
    comfyui_url: str = "http://127.0.0.1:8188",
    progress_callback: Callable | None = None,
) -> list[Image.Image]:
    """Run image generation via ComfyUI and return results.

    Args:
        images: Source/reference images. First is primary.
        prompt: Scene description for Flux generation.
        seed: Random seed.
        num_steps: Sampling steps.
        cfg_scale: FluxGuidance scale.
        negative_prompt: Negative prompt text.
        num_images: Number of images to generate (sequential with seed+i).
        template: Workflow template name ("flux_base" or "pulid_identity").
        reference_image: Face reference for PuLID identity workflow.
        identity_strength: PuLID weight (0-1).
        face_restore: Enable CodeFormer face restoration.
        face_restore_fidelity: CodeFormer fidelity (0-1).
        comfyui_url: ComfyUI server URL.
        progress_callback: Optional (current, total) callback.

    Returns:
        List of generated PIL Images.
    """
    client = ComfyUIClient(base_url=comfyui_url)
    builder = WorkflowBuilder()
    results = []

    try:
        # Upload reference/source image for templates that use it
        ref_name = None
        needs_upload = (
            reference_image is not None
            or (template in ("pulid_identity", "qwen_face_edit", "flux2_base") and images)
        )
        if needs_upload:
            ref_img = reference_image if reference_image is not None else images[0]
            ref_name = client.upload_image(ref_img, "curation_ref.png")

        for i in range(num_images):
            current_seed = seed + i

            if template == "qwen_face_edit" and ref_name:
                workflow = builder.build_qwen_face_workflow(
                    source_image=ref_name,
                    prompt=prompt,
                    seed=current_seed,
                    steps=num_steps,
                )
            elif template == "pulid_identity" and ref_name:
                workflow = builder.build_identity_workflow(
                    reference_image=ref_name,
                    prompt=prompt,
                    seed=current_seed,
                    steps=num_steps,
                    cfg=cfg_scale,
                    negative_prompt=negative_prompt,
                    weight=identity_strength,
                    face_restore=face_restore,
                    face_restore_fidelity=face_restore_fidelity,
                )
            elif template == "flux2_base":
                workflow = builder.build_flux2_workflow(
                    prompt=prompt,
                    reference_image=ref_name or "reference.png",
                    seed=current_seed,
                    steps=num_steps,
                    cfg=cfg_scale,
                    negative_prompt=negative_prompt,
                    pulid_weight=identity_strength,
                    pulid_end_at=0.8,
                    face_restore_fidelity=face_restore_fidelity,
                )
            else:
                workflow = builder.build_edit_workflow(
                    source_image=ref_name or "",
                    prompt=prompt,
                    seed=current_seed,
                    steps=num_steps,
                    cfg=cfg_scale,
                    negative_prompt=negative_prompt,
                )

            prompt_id = client.queue_prompt(workflow)

            def _cb(val, mx, _i=i):
                if progress_callback:
                    progress_callback(val + _i * mx, num_images * mx)

            client.wait_for_result(prompt_id, progress_callback=_cb)
            generated = client.get_images(prompt_id)
            results.extend(generated)

            logger.info(
                "Generated image %d/%d (seed=%d, template=%s)",
                i + 1, num_images, current_seed, template,
            )
    finally:
        client.close()

    return results
