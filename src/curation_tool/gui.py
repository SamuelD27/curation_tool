"""Gradio GUI for Qwen-Image-Edit-2509 image editing."""
import logging
import tempfile
import time
from pathlib import Path

import gradio as gr
from PIL import Image

from curation_tool.face_pipeline import FacePipelineConfig, export_face_dataset
from curation_tool.pipeline import load_pipeline, run_edit
from curation_tool.presets import PRESETS
from curation_tool.stages import StageConfig, run_stage

logger = logging.getLogger(__name__)

_pipeline = None


def _ensure_pipeline(progress=None, model_id: str = "ovedrive/Qwen-Image-Edit-2509-4bit"):
    global _pipeline
    if _pipeline is None:
        if progress:
            progress(0, desc="Loading model (first run only)...")
        _pipeline = load_pipeline(model_id=model_id)
    return _pipeline


def generate(
    image1: Image.Image | None,
    image2: Image.Image | None,
    image3: Image.Image | None,
    prompt: str,
    negative_prompt: str,
    seed: int,
    num_steps: int,
    cfg_scale: float,
    guidance_scale: float,
    num_images: int,
    progress=gr.Progress(track_tqdm=True),
) -> list[Image.Image]:
    images = [img.convert("RGB") for img in [image1, image2, image3] if img is not None]
    if not images:
        raise gr.Error("Upload at least one reference image.")
    if not prompt.strip():
        raise gr.Error("Enter an edit prompt.")

    pipeline = _ensure_pipeline(progress=progress)

    num_images = int(num_images)
    progress(0, desc=f"Generating {num_images} image(s) ({int(num_steps)} steps)...")
    start = time.time()

    results = run_edit(
        pipeline=pipeline,
        images=images,
        prompt=prompt,
        seed=int(seed),
        num_steps=int(num_steps),
        cfg_scale=cfg_scale,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt.strip() or " ",
        num_images=num_images,
    )

    elapsed = time.time() - start
    logger.info("Generated %d image(s) in %.1fs", len(results), elapsed)
    return results


# --- Face Pipeline stage handlers ---


def run_refine_stage(
    base_image: Image.Image | None,
    output_dir: str,
    prompt: str,
    num_candidates: int,
    seed: int,
    steps: int,
    cfg: float,
    progress=gr.Progress(track_tqdm=True),
) -> tuple[list[str], dict, str]:
    """Run the refine stage and return candidate images for gallery."""
    if base_image is None:
        raise gr.Error("Upload a base photo first.")
    if not prompt.strip():
        raise gr.Error("Enter a refine prompt.")

    pipeline = _ensure_pipeline(progress=progress)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save base image to a temp input dir
    input_dir = out / "_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    base_path = input_dir / "base.png"
    base_image.convert("RGB").save(base_path)

    stage = StageConfig(
        name="refine",
        type="refine",
        prompt=prompt,
        num_candidates=int(num_candidates),
    )

    progress(0, desc=f"Refining {int(num_candidates)} candidate(s)...")
    results = run_stage(
        stage,
        source_image_path=base_path,
        input_dir=input_dir,
        output_dir=out,
        pipeline=pipeline,
        base_seed=int(seed),
        default_num_steps=int(steps),
        default_cfg_scale=cfg,
    )

    gallery_images = [r["output_path"] for r in results]
    state = {
        "picked_image_path": None,
        "angle_results": [],
        "stage_dir": str(out),
        "refine_results": results,
    }

    status = f"Refine complete: {len(results)} candidates generated."
    return gallery_images, state, status


def pick_candidate(evt: gr.SelectData, state: dict) -> tuple[str | None, dict]:
    """Handle gallery selection to pick a refine candidate."""
    if not state or "refine_results" not in state:
        raise gr.Error("Run refine first.")

    idx = evt.index
    results = state["refine_results"]
    if idx < 0 or idx >= len(results):
        raise gr.Error(f"Invalid selection index: {idx}")

    picked_path = results[idx]["output_path"]
    state = {**state, "picked_image_path": picked_path}
    return picked_path, state


def run_angles_stage(
    preset: str,
    seed: int,
    steps: int,
    cfg: float,
    state: dict,
    progress=gr.Progress(track_tqdm=True),
) -> tuple[list[str], dict, str]:
    """Run the angles stage using the picked candidate."""
    if not state or not state.get("picked_image_path"):
        raise gr.Error("Pick a refine candidate first.")

    pipeline = _ensure_pipeline(progress=progress)

    picked_path = Path(state["picked_image_path"])
    out = Path(state["stage_dir"])

    stage = StageConfig(
        name="angles",
        type="angles",
        preset=preset,
    )

    num_prompts = len(PRESETS[preset])
    progress(0, desc=f"Generating {num_prompts} angle variations...")
    results = run_stage(
        stage,
        source_image_path=picked_path,
        input_dir=picked_path.parent,
        output_dir=out,
        pipeline=pipeline,
        base_seed=int(seed),
        default_num_steps=int(steps),
        default_cfg_scale=cfg,
    )

    gallery_images = [r["output_path"] for r in results]
    state = {**state, "angle_results": results}

    status = f"Angles complete: {len(results)} variations generated."
    return gallery_images, state, status


def export_dataset(
    output_dir: str,
    trigger_word: str,
    state: dict,
) -> str:
    """Export the angle results as a LoRA dataset."""
    if not state or not state.get("angle_results"):
        raise gr.Error("Generate angles first.")

    config = FacePipelineConfig(
        trigger_word=trigger_word.strip() or "sks_person",
        output_dir=output_dir,
        stages=[],  # Not used by export_face_dataset
    )

    export_path = export_face_dataset(config, state["angle_results"])
    return f"Dataset exported to {export_path} ({len(state['angle_results'])} images)"


# --- App builder ---


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Qwen-Image-Edit Curation Tool") as app:
        with gr.Tabs():
            # --- Single Edit tab (existing UI) ---
            with gr.Tab("Single Edit"):
                gr.Markdown(
                    "# Qwen-Image-Edit-2509\n"
                    "Upload 1-3 reference images + prompt. First run loads model (~2 min)."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        img1 = gr.Image(label="Image 1 (required)", type="pil")
                        img2 = gr.Image(label="Image 2 (optional)", type="pil")
                        img3 = gr.Image(label="Image 3 (optional)", type="pil")
                        prompt = gr.Textbox(
                            label="Edit Prompt", lines=3,
                            placeholder="Describe the edit you want...",
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            lines=2,
                            value="blurry, low quality, distorted, deformed, artifacts",
                        )
                        with gr.Row():
                            seed = gr.Number(label="Seed", value=0, precision=0)
                            num_images = gr.Slider(
                                label="Batch Size", minimum=1, maximum=8, value=1, step=1,
                            )
                        with gr.Row():
                            steps = gr.Slider(
                                label="Steps", minimum=1, maximum=100, value=50, step=1,
                            )
                            cfg = gr.Slider(
                                label="True CFG Scale", minimum=1.0, maximum=15.0,
                                value=4.0, step=0.5,
                            )
                            guidance = gr.Slider(
                                label="Guidance Scale", minimum=0.0, maximum=10.0,
                                value=1.0, step=0.5,
                            )
                        generate_btn = gr.Button("Generate", variant="primary")

                    with gr.Column(scale=1):
                        output = gr.Gallery(label="Results", columns=2, height="auto")

                generate_btn.click(
                    fn=generate,
                    inputs=[
                        img1, img2, img3, prompt, negative_prompt,
                        seed, steps, cfg, guidance, num_images,
                    ],
                    outputs=output,
                )

            # --- Face Pipeline tab ---
            with gr.Tab("Face Pipeline"):
                gr.Markdown(
                    "# Face Pipeline\n"
                    "Refine -> Pick -> Angles -> Export. "
                    "Generate a LoRA training dataset from a single photo."
                )

                fp_state = gr.State(value={
                    "picked_image_path": None,
                    "angle_results": [],
                    "stage_dir": "",
                    "refine_results": [],
                })

                with gr.Row():
                    # Left column: controls
                    with gr.Column(scale=1):
                        fp_base_image = gr.Image(
                            label="Base Photo", type="pil",
                        )
                        fp_output_dir = gr.Textbox(
                            label="Output Directory", value="./face_output",
                        )
                        fp_trigger_word = gr.Textbox(
                            label="Trigger Word", value="sks_person",
                        )
                        fp_refine_prompt = gr.Textbox(
                            label="Refine Prompt", lines=3,
                            value="Improve this photo to a perfect professional headshot, "
                                  "sharp focus, studio lighting, neutral background",
                        )
                        fp_num_candidates = gr.Slider(
                            label="Num Candidates", minimum=1, maximum=8,
                            value=4, step=1,
                        )
                        fp_seed = gr.Number(label="Base Seed", value=42, precision=0)
                        with gr.Row():
                            fp_steps = gr.Slider(
                                label="Steps", minimum=1, maximum=100,
                                value=50, step=1,
                            )
                            fp_cfg = gr.Slider(
                                label="CFG Scale", minimum=1.0, maximum=15.0,
                                value=4.0, step=0.5,
                            )

                        fp_refine_btn = gr.Button("Run Refine", variant="primary")

                        fp_preset = gr.Dropdown(
                            label="Preset",
                            choices=list(PRESETS.keys()),
                            value="headshot_20",
                        )
                        fp_angles_btn = gr.Button(
                            "Generate Angles", interactive=False,
                        )
                        fp_export_btn = gr.Button(
                            "Export Dataset", interactive=False,
                        )

                    # Right column: results
                    with gr.Column(scale=1):
                        fp_refine_gallery = gr.Gallery(
                            label="Refine Candidates (click to pick)",
                            columns=2, height="auto",
                        )
                        fp_picked_preview = gr.Image(
                            label="Picked Candidate", type="filepath",
                        )
                        fp_angles_gallery = gr.Gallery(
                            label="Angle / Body Results",
                            columns=4, height="auto",
                        )
                        fp_status = gr.Textbox(
                            label="Status", interactive=False, lines=3,
                        )

                # Wire up events
                fp_refine_btn.click(
                    fn=run_refine_stage,
                    inputs=[
                        fp_base_image, fp_output_dir, fp_refine_prompt,
                        fp_num_candidates, fp_seed, fp_steps, fp_cfg,
                    ],
                    outputs=[fp_refine_gallery, fp_state, fp_status],
                ).then(
                    fn=lambda: gr.update(interactive=False),
                    outputs=fp_angles_btn,
                )

                fp_refine_gallery.select(
                    fn=pick_candidate,
                    inputs=[fp_state],
                    outputs=[fp_picked_preview, fp_state],
                ).then(
                    fn=lambda: gr.update(interactive=True),
                    outputs=fp_angles_btn,
                )

                fp_angles_btn.click(
                    fn=run_angles_stage,
                    inputs=[fp_preset, fp_seed, fp_steps, fp_cfg, fp_state],
                    outputs=[fp_angles_gallery, fp_state, fp_status],
                ).then(
                    fn=lambda: gr.update(interactive=True),
                    outputs=fp_export_btn,
                )

                fp_export_btn.click(
                    fn=export_dataset,
                    inputs=[fp_output_dir, fp_trigger_word, fp_state],
                    outputs=fp_status,
                )

    return app


def launch(host: str = "0.0.0.0", port: int = 7860, share: bool = False):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("Starting Gradio app on %s:%d", host, port)
    app = build_app()
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        strict_cors=False,
        ssr_mode=False,
    )
