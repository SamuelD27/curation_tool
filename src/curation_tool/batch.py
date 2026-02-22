"""Batch processing engine for curation jobs."""
import json
import logging
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from curation_tool.comfyui_client import ComfyUIClient
from curation_tool.config import CurationJob
from curation_tool.pipeline import run_edit

logger = logging.getLogger(__name__)


def run_batch(
    job: CurationJob,
    comfyui_url: str | None = None,
) -> list[dict]:
    """Process all tasks in a curation job.

    Returns list of result dicts with input/output paths and metadata.
    """
    url = comfyui_url or job.comfyui.url

    # Health check
    client = ComfyUIClient(base_url=url)
    if not client.health_check():
        raise RuntimeError(f"ComfyUI is not reachable at {url}")
    stats = client.get_system_stats()
    devices = stats.get("devices", [])
    if devices:
        dev = devices[0]
        logger.info(
            "GPU: %s (VRAM %.0f/%.0f MB)",
            dev.get("name", "unknown"),
            dev.get("vram_used", 0) / 1e6,
            dev.get("vram_total", 0) / 1e6,
        )
    client.close()

    input_dir = Path(job.input_dir)
    output_dir = Path(job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    metadata_path = output_dir / "metadata.jsonl"

    with open(metadata_path, "w") as meta_f:
        for i, task in enumerate(tqdm(job.tasks, desc="Processing edits")):
            source_path = input_dir / task.source_image
            source_img = Image.open(source_path).convert("RGB")

            images = [source_img]
            for ref in task.reference_images:
                ref_img = Image.open(input_dir / ref).convert("RGB")
                images.append(ref_img)

            seed = task.seed if task.seed is not None else job.default_seed
            num_steps = task.num_steps if task.num_steps is not None else job.default_num_steps
            cfg_scale = task.cfg_scale if task.cfg_scale is not None else job.default_cfg_scale

            generated = run_edit(
                images=images,
                prompt=task.prompt,
                seed=seed,
                num_steps=num_steps,
                cfg_scale=cfg_scale,
                comfyui_url=url,
            )

            out_name = f"{Path(task.source_image).stem}_edited_{i:04d}.png"
            out_path = output_dir / out_name
            generated[0].save(out_path)

            record = {
                "index": i,
                "source": task.source_image,
                "prompt": task.prompt,
                "seed": seed,
                "num_steps": num_steps,
                "cfg_scale": cfg_scale,
                "output_path": str(out_path),
            }
            results.append(record)
            meta_f.write(json.dumps(record) + "\n")

            logger.info("[%d/%d] %s -> %s", i + 1, len(job.tasks), task.source_image, out_name)

    logger.info("Batch complete. %d images saved to %s", len(results), output_dir)
    return results
