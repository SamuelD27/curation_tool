"""Gradio GUI for image editing with ComfyUI backend."""
import logging
import time
from pathlib import Path

import gradio as gr
from PIL import Image

from curation_tool.comfyui_client import ComfyUIClient
from curation_tool.face_pipeline import FacePipelineConfig, export_face_dataset
from curation_tool.pipeline import run_edit
from curation_tool.presets import PRESETS
from curation_tool.stages import StageConfig, run_stage

logger = logging.getLogger(__name__)

_comfyui_url = "http://127.0.0.1:8188"


def _check_comfyui(url: str) -> str:
    """Return status string for the ComfyUI connection."""
    client = ComfyUIClient(base_url=url)
    try:
        if client.health_check():
            stats = client.get_system_stats()
            devices = stats.get("devices", [])
            if devices:
                dev = devices[0]
                vram = dev.get("vram_used", 0) / 1e6
                vram_total = dev.get("vram_total", 0) / 1e6
                return f"Connected ({dev.get('name', 'GPU')}: {vram:.0f}/{vram_total:.0f} MB)"
            return "Connected"
        return "Not reachable"
    except Exception as e:
        return f"Error: {e}"
    finally:
        client.close()


def generate(
    comfyui_url: str,
    image1: Image.Image | None,
    prompt: str,
    negative_prompt: str,
    seed: int,
    num_steps: int,
    cfg_scale: float,
    num_images: int,
    template: str,
    identity_strength: float,
    progress=gr.Progress(track_tqdm=True),
) -> list[Image.Image]:
    if not prompt.strip():
        raise gr.Error("Enter a scene description prompt.")

    url = comfyui_url.strip() or _comfyui_url

    ref_image = image1.convert("RGB") if image1 is not None else None
    images = [ref_image] if ref_image else []

    num_images = int(num_images)
    progress(0, desc=f"Generating {num_images} image(s) ({int(num_steps)} steps)...")
    start = time.time()

    results = run_edit(
        images=images,
        prompt=prompt,
        seed=int(seed),
        num_steps=int(num_steps),
        cfg_scale=cfg_scale,
        negative_prompt=negative_prompt.strip() or "",
        num_images=num_images,
        template=template,
        reference_image=ref_image,
        identity_strength=identity_strength,
        comfyui_url=url,
    )

    elapsed = time.time() - start
    logger.info("Generated %d image(s) in %.1fs", len(results), elapsed)
    return results


# --- Face Pipeline stage handlers ---


def run_refine_stage(
    comfyui_url: str,
    base_image: Image.Image | None,
    output_dir: str,
    prompt: str,
    num_candidates: int,
    seed: int,
    steps: int,
    cfg: float,
    template: str,
    progress=gr.Progress(track_tqdm=True),
) -> tuple[list[str], dict, str]:
    """Run the refine stage and return candidate images for gallery."""
    if base_image is None:
        raise gr.Error("Upload a base photo first.")
    if not prompt.strip():
        raise gr.Error("Enter a refine prompt.")

    url = comfyui_url.strip() or _comfyui_url
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

    progress(0, desc=f"Refining {int(num_candidates)} candidate(s) ({template})...")
    results = run_stage(
        stage,
        source_image_path=base_path,
        input_dir=input_dir,
        output_dir=out,
        comfyui_url=url,
        base_seed=int(seed),
        default_num_steps=int(steps),
        default_cfg_scale=cfg,
        template=template,
    )

    gallery_images = [r["output_path"] for r in results]
    state = {
        "picked_image_path": None,
        "angle_results": [],
        "stage_dir": str(out),
        "refine_results": results,
        "comfyui_url": url,
        "template": template,
    }

    status = f"Refine complete: {len(results)} candidates generated ({template})."
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

    url = state.get("comfyui_url", _comfyui_url)
    template = state.get("template", "qwen_face_edit")
    picked_path = Path(state["picked_image_path"])
    out = Path(state["stage_dir"])

    stage = StageConfig(
        name="angles",
        type="angles",
        preset=preset,
    )

    num_prompts = len(PRESETS[preset])
    progress(0, desc=f"Generating {num_prompts} angle variations ({template})...")
    results = run_stage(
        stage,
        source_image_path=picked_path,
        input_dir=picked_path.parent,
        output_dir=out,
        comfyui_url=url,
        base_seed=int(seed),
        default_num_steps=int(steps),
        default_cfg_scale=cfg,
        template=template,
    )

    gallery_images = [r["output_path"] for r in results]
    state = {**state, "angle_results": results}

    status = f"Angles complete: {len(results)} variations generated ({template})."
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
    with gr.Blocks(title="Curation Tool (ComfyUI Backend)") as app:
        with gr.Tabs():
            # --- Single Edit tab ---
            with gr.Tab("Single Edit"):
                gr.Markdown(
                    "# Curation Tool (ComfyUI Backend)\n"
                    "Generate images via Flux + PuLID identity preservation."
                )

                with gr.Row():
                    comfyui_url_input = gr.Textbox(
                        label="ComfyUI URL",
                        value=_comfyui_url,
                        scale=3,
                    )
                    status_btn = gr.Button("Check Status", scale=1)
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False,
                        scale=2,
                    )

                status_btn.click(
                    fn=_check_comfyui,
                    inputs=[comfyui_url_input],
                    outputs=[status_text],
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        img1 = gr.Image(label="Reference Image (optional, for PuLID)", type="pil")
                        prompt = gr.Textbox(
                            label="Scene Description Prompt", lines=3,
                            placeholder="Professional headshot portrait, soft studio lighting...",
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            lines=2,
                            value="",
                        )
                        template = gr.Dropdown(
                            label="Workflow Template",
                            choices=["flux_base", "flux2_base", "pulid_identity", "qwen_face_edit"],
                            value="qwen_face_edit",
                        )
                        with gr.Row():
                            seed = gr.Number(label="Seed", value=0, precision=0)
                            num_images = gr.Slider(
                                label="Batch Size", minimum=1, maximum=8, value=1, step=1,
                            )
                        with gr.Row():
                            steps = gr.Slider(
                                label="Steps", minimum=1, maximum=100, value=30, step=1,
                            )
                            cfg = gr.Slider(
                                label="Guidance Scale", minimum=1.0, maximum=15.0,
                                value=3.5, step=0.5,
                            )
                        identity_strength = gr.Slider(
                            label="Identity Strength (PuLID)",
                            minimum=0.0, maximum=1.5, value=0.8, step=0.05,
                            visible=True,
                        )
                        generate_btn = gr.Button("Generate", variant="primary")

                    with gr.Column(scale=1):
                        output = gr.Gallery(label="Results", columns=2, height="auto")

                generate_btn.click(
                    fn=generate,
                    inputs=[
                        comfyui_url_input, img1, prompt, negative_prompt,
                        seed, steps, cfg, num_images, template, identity_strength,
                    ],
                    outputs=output,
                )

            # --- Face Pipeline tab ---
            with gr.Tab("Face Pipeline"):
                gr.Markdown(
                    "# Face Pipeline (ComfyUI)\n"
                    "Refine -> Pick -> Angles -> Export. "
                    "Generate a LoRA training dataset from a single photo."
                )

                fp_state = gr.State(value={
                    "picked_image_path": None,
                    "angle_results": [],
                    "stage_dir": "",
                    "refine_results": [],
                    "comfyui_url": _comfyui_url,
                    "template": "qwen_face_edit",
                })

                with gr.Row():
                    fp_comfyui_url = gr.Textbox(
                        label="ComfyUI URL", value=_comfyui_url,
                    )

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
                        fp_template = gr.Dropdown(
                            label="Workflow",
                            choices=["qwen_face_edit", "pulid_identity", "flux2_base"],
                            value="qwen_face_edit",
                        )
                        fp_refine_prompt = gr.Textbox(
                            label="Refine Prompt", lines=3,
                            value="Professional headshot portrait, facing directly forward, "
                                  "centered composition, sharp focus, studio lighting, "
                                  "clean neutral background, 85mm lens",
                        )
                        fp_num_candidates = gr.Slider(
                            label="Num Candidates", minimum=1, maximum=8,
                            value=4, step=1,
                        )
                        fp_seed = gr.Number(label="Base Seed", value=42, precision=0)
                        with gr.Row():
                            fp_steps = gr.Slider(
                                label="Steps", minimum=1, maximum=100,
                                value=6, step=1,
                            )
                            fp_cfg = gr.Slider(
                                label="Guidance Scale", minimum=1.0, maximum=15.0,
                                value=3.5, step=0.5,
                            )

                        fp_refine_btn = gr.Button("Run Refine", variant="primary")

                        fp_preset = gr.Dropdown(
                            label="Preset",
                            choices=list(PRESETS.keys()),
                            value="qwen_headshot_20",
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

                # Update defaults when workflow template changes
                def _on_template_change(tmpl):
                    if tmpl == "pulid_identity":
                        return (
                            gr.update(value=30),         # steps
                            gr.update(value=3.5),        # cfg
                            gr.update(value="pulid_headshot_20"),  # preset
                        )
                    elif tmpl == "flux2_base":
                        return (
                            gr.update(value=28),          # steps
                            gr.update(value=3.0),         # cfg (FluxGuidance)
                            gr.update(value="flux2_headshot_20"),  # preset
                        )
                    else:  # qwen_face_edit
                        return (
                            gr.update(value=6),           # steps
                            gr.update(value=3.5),         # cfg
                            gr.update(value="qwen_headshot_20"),  # preset
                        )

                fp_template.change(
                    fn=_on_template_change,
                    inputs=[fp_template],
                    outputs=[fp_steps, fp_cfg, fp_preset],
                )

                # Wire up events
                fp_refine_btn.click(
                    fn=run_refine_stage,
                    inputs=[
                        fp_comfyui_url, fp_base_image, fp_output_dir, fp_refine_prompt,
                        fp_num_candidates, fp_seed, fp_steps, fp_cfg, fp_template,
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


def launch(
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
    comfyui_url: str = "http://127.0.0.1:8188",
):
    global _comfyui_url
    _comfyui_url = comfyui_url
    logger.info("Starting Gradio app on %s:%d (ComfyUI: %s)", host, port, comfyui_url)
    app = build_app()
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        strict_cors=False,
        ssr_mode=False,
    )
