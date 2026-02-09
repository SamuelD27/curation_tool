"""CLI interface for the curation tool."""
import logging
from pathlib import Path

import click

from curation_tool.config import CurationJob

logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def main(verbose: bool):
    """Qwen-Image-Edit dataset curation tool for LoRA training."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to YAML job config.")
@click.option("--model-id", default="Qwen/Qwen-Image-Edit-2509", help="HuggingFace model ID.")
@click.option("--device", default="cuda", help="Device to run on.")
def run(config: str, model_id: str, device: str):
    """Run a batch curation job from a YAML config file."""
    import torch
    from curation_tool.pipeline import load_pipeline
    from curation_tool.batch import run_batch

    job = CurationJob.from_yaml(Path(config))
    logger.info("Loaded job with %d tasks from %s", len(job.tasks), config)

    pipeline = load_pipeline(model_id=model_id, device=device)
    results = run_batch(job, pipeline=pipeline)

    click.echo(f"Done. {len(results)} images generated.")


@main.command()
@click.option("--image", "-i", required=True, type=click.Path(exists=True), help="Input image path.")
@click.option("--prompt", "-p", required=True, help="Edit instruction.")
@click.option("--output", "-o", default="output.png", help="Output image path.")
@click.option("--seed", default=0, type=int, help="Random seed.")
@click.option("--steps", default=40, type=int, help="Inference steps.")
@click.option("--model-id", default="Qwen/Qwen-Image-Edit-2509", help="HuggingFace model ID.")
def edit(image: str, prompt: str, output: str, seed: int, steps: int, model_id: str):
    """Edit a single image with a text prompt."""
    import torch
    from PIL import Image as PILImage
    from curation_tool.pipeline import load_pipeline, run_edit

    pipeline = load_pipeline(model_id=model_id)
    img = PILImage.open(image).convert("RGB")
    result = run_edit(pipeline, images=[img], prompt=prompt, seed=seed, num_steps=steps)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    click.echo(f"Saved to {output}")
