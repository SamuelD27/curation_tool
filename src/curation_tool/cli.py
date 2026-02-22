"""CLI interface for the curation tool."""
import logging
from pathlib import Path

import click

from curation_tool.config import CurationJob
from curation_tool.logging_config import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.option(
    "--comfyui-url",
    default=DEFAULT_COMFYUI_URL,
    envvar="COMFYUI_URL",
    show_envvar=True,
    help="ComfyUI server URL.",
)
@click.option("--log-file", default=None, type=click.Path(), help="Log file path.")
@click.pass_context
def main(ctx: click.Context, verbose: bool, comfyui_url: str, log_file: str | None):
    """Image dataset curation tool with ComfyUI backend."""
    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level, log_file=log_file)
    ctx.ensure_object(dict)
    ctx.obj["comfyui_url"] = comfyui_url


@main.command()
@click.pass_context
def health(ctx: click.Context):
    """Check ComfyUI connectivity and print system stats."""
    from curation_tool.comfyui_client import ComfyUIClient

    url = ctx.obj["comfyui_url"]
    client = ComfyUIClient(base_url=url)

    click.echo(f"Checking ComfyUI at {url}...")
    if not client.health_check():
        click.echo("FAIL: ComfyUI is not reachable.")
        raise SystemExit(1)

    click.echo("OK: ComfyUI is running.")
    stats = client.get_system_stats()

    devices = stats.get("devices", [])
    for dev in devices:
        vram_used = dev.get("vram_used", 0) / 1e6
        vram_total = dev.get("vram_total", 0) / 1e6
        click.echo(f"  GPU: {dev.get('name', 'unknown')} ({vram_used:.0f}/{vram_total:.0f} MB VRAM)")

    system = stats.get("system", {})
    click.echo(f"  Python: {system.get('python_version', 'unknown')}")
    click.echo(f"  OS: {system.get('os', 'unknown')}")
    client.close()


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to YAML job config.")
@click.option("--export-lora", is_flag=True, help="Export as LoRA training dataset (image+caption pairs).")
@click.option("--caption-template", default="{prompt}", help="Caption template. Use {prompt}, {source}, {seed}.")
@click.pass_context
def run(ctx: click.Context, config: str, export_lora: bool, caption_template: str):
    """Run a batch curation job from a YAML config file."""
    from curation_tool.batch import run_batch

    url = ctx.obj["comfyui_url"]
    job = CurationJob.from_yaml(Path(config))
    logger.info("Loaded job with %d tasks from %s", len(job.tasks), config)

    results = run_batch(job, comfyui_url=url)

    if export_lora:
        from curation_tool.dataset_export import export_lora_dataset
        lora_dir = Path(job.output_dir) / "lora_dataset"
        export_lora_dataset(results, lora_dir, caption_template=caption_template)
        click.echo(f"LoRA dataset exported to {lora_dir}")

    click.echo(f"Done. {len(results)} images generated.")


@main.command()
@click.option("--image", "-i", required=True, type=click.Path(exists=True), help="Input image path.")
@click.option("--prompt", "-p", required=True, help="Scene description prompt.")
@click.option("--output", "-o", default="output.png", help="Output image path.")
@click.option("--seed", default=0, type=int, help="Random seed.")
@click.option("--steps", default=30, type=int, help="Inference steps.")
@click.option("--template", "-t", default="flux_base", help="Workflow template (flux_base, pulid_identity, or qwen_face_edit).")
@click.pass_context
def edit(ctx: click.Context, image: str, prompt: str, output: str, seed: int, steps: int, template: str):
    """Generate an image with a text prompt (optionally identity-preserving)."""
    from PIL import Image as PILImage
    from curation_tool.pipeline import run_edit

    url = ctx.obj["comfyui_url"]
    img = PILImage.open(image).convert("RGB")
    results = run_edit(
        images=[img],
        prompt=prompt,
        seed=seed,
        num_steps=steps,
        template=template,
        comfyui_url=url,
    )

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    results[0].save(output)
    click.echo(f"Saved to {output}")


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to.")
@click.option("--port", default=7860, type=int, help="Port to serve on.")
@click.option("--share", is_flag=True, help="Create a public Gradio share link.")
@click.pass_context
def gui(ctx: click.Context, host: str, port: int, share: bool):
    """Launch the Gradio web interface."""
    from curation_tool.gui import launch
    url = ctx.obj["comfyui_url"]
    launch(host=host, port=port, share=share, comfyui_url=url)


# --- Face pipeline commands ---

@main.group()
def face():
    """Multi-stage face dataset generation pipeline."""
    pass


@face.command("run")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to face pipeline YAML config.")
@click.option("--pick", multiple=True, help="Pre-supply picks as stage=path (e.g. --pick refine=chosen.png).")
@click.option("--no-interactive", is_flag=True, help="Disable interactive prompts (pause instead).")
@click.option("--caption-template", default=None, help="Caption template. Use {prompt}, {source}, {seed}.")
@click.pass_context
def face_run(ctx: click.Context, config: str, pick: tuple[str, ...], no_interactive: bool, caption_template: str | None):
    """Run the face dataset pipeline from a YAML config."""
    from curation_tool.face_pipeline import FacePipelineConfig, run_face_pipeline, export_face_dataset

    url = ctx.obj["comfyui_url"]
    cfg = FacePipelineConfig.from_yaml(Path(config))
    picks = _parse_picks(pick)

    results = run_face_pipeline(
        cfg,
        comfyui_url=url,
        interactive=not no_interactive,
        picks=picks,
    )

    if results:
        export_dir = export_face_dataset(cfg, results, caption_template=caption_template)
        click.echo(f"LoRA dataset exported to {export_dir}")

    click.echo(f"Done. {len(results)} dataset images generated.")


@face.command("resume")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to face pipeline YAML config.")
@click.option("--pick", multiple=True, help="Supply picks as stage=path (e.g. --pick refine=chosen.png).")
@click.option("--caption-template", default=None, help="Caption template. Use {prompt}, {source}, {seed}.")
@click.pass_context
def face_resume(ctx: click.Context, config: str, pick: tuple[str, ...], caption_template: str | None):
    """Resume a paused face pipeline with picks."""
    from curation_tool.face_pipeline import FacePipelineConfig, run_face_pipeline, export_face_dataset

    url = ctx.obj["comfyui_url"]
    cfg = FacePipelineConfig.from_yaml(Path(config))
    picks = _parse_picks(pick)

    results = run_face_pipeline(
        cfg,
        comfyui_url=url,
        interactive=False,
        picks=picks,
    )

    if results:
        export_dir = export_face_dataset(cfg, results, caption_template=caption_template)
        click.echo(f"LoRA dataset exported to {export_dir}")

    click.echo(f"Done. {len(results)} dataset images generated.")


@face.command("presets")
def face_presets():
    """List available prompt presets."""
    from curation_tool.presets import PRESETS

    for name, prompts in sorted(PRESETS.items()):
        click.echo(f"\n{name} ({len(prompts)} prompts):")
        for i, p in enumerate(prompts):
            click.echo(f"  [{i:2d}] {p.prompt}")


def _parse_picks(pick_args: tuple[str, ...]) -> dict[str, str]:
    """Parse --pick stage=path arguments into a dict."""
    picks = {}
    for item in pick_args:
        if "=" not in item:
            raise click.BadParameter(f"Pick must be stage=path, got: {item}")
        stage_name, path = item.split("=", 1)
        picks[stage_name] = path
    return picks
