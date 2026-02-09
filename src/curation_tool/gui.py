"""Gradio GUI for Qwen-Image-Edit-2509 image editing."""
import logging

import gradio as gr
from PIL import Image

from curation_tool.pipeline import load_pipeline, run_edit

logger = logging.getLogger(__name__)

_pipeline = None


def _ensure_pipeline(model_id: str = "Qwen/Qwen-Image-Edit-2509"):
    global _pipeline
    if _pipeline is None:
        _pipeline = load_pipeline(model_id=model_id)
    return _pipeline


def generate(
    image1: Image.Image | None,
    image2: Image.Image | None,
    image3: Image.Image | None,
    prompt: str,
    seed: int,
    num_steps: int,
    cfg_scale: float,
) -> Image.Image | None:
    images = [img.convert("RGB") for img in [image1, image2, image3] if img is not None]
    if not images:
        raise gr.Error("Upload at least one reference image.")
    if not prompt.strip():
        raise gr.Error("Enter an edit prompt.")

    pipeline = _ensure_pipeline()
    result = run_edit(
        pipeline=pipeline,
        images=images,
        prompt=prompt,
        seed=seed,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
    )
    return result


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Qwen-Image-Edit Curation Tool") as app:
        gr.Markdown("# Qwen-Image-Edit-2509\nUpload 1-3 reference images + prompt to generate edited images.")

        with gr.Row():
            with gr.Column(scale=1):
                img1 = gr.Image(label="Image 1 (required)", type="pil")
                img2 = gr.Image(label="Image 2 (optional)", type="pil")
                img3 = gr.Image(label="Image 3 (optional)", type="pil")
                prompt = gr.Textbox(label="Edit Prompt", lines=3, placeholder="Describe the edit you want...")
                with gr.Row():
                    seed = gr.Number(label="Seed", value=0, precision=0)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=40, step=1)
                    cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, value=4.0, step=0.5)
                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                output = gr.Image(label="Result", type="pil")

        generate_btn.click(
            fn=generate,
            inputs=[img1, img2, img3, prompt, seed, steps, cfg],
            outputs=output,
        )

    return app


def launch(host: str = "0.0.0.0", port: int = 7860, share: bool = False):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("Starting Gradio app on %s:%d", host, port)
    app = build_app()
    app.launch(server_name=host, server_port=port, share=share)
