"""Stage config model and task expansion for multi-stage face pipelines."""
import json
import logging
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from pydantic import BaseModel

from curation_tool.config import EditTask
from curation_tool.presets import get_preset

logger = logging.getLogger(__name__)


class StageConfig(BaseModel):
    """Configuration for a single pipeline stage."""

    name: str
    type: Literal["refine", "angles", "body"]
    source_image: str | None = None
    prompt: str | None = None
    num_candidates: int = 1
    seeds: list[int] | None = None
    preset: str | None = None
    prompts: list[str] | None = None
    num_steps: int | None = None
    cfg_scale: float | None = None


def expand_stage_tasks(
    stage: StageConfig,
    source_image: str,
    base_seed: int = 42,
    default_num_steps: int = 50,
    default_cfg_scale: float = 4.0,
) -> list[EditTask]:
    """Expand a stage config into concrete EditTasks. Pure function, no I/O."""
    num_steps = stage.num_steps if stage.num_steps is not None else default_num_steps
    cfg_scale = stage.cfg_scale if stage.cfg_scale is not None else default_cfg_scale
    src = stage.source_image or source_image

    if stage.type == "refine":
        seeds = stage.seeds or [base_seed + i for i in range(stage.num_candidates)]
        prompt = stage.prompt or "Improve this photo to a perfect professional headshot"
        return [
            EditTask(
                source_image=src,
                prompt=prompt,
                seed=s,
                num_steps=num_steps,
                cfg_scale=cfg_scale,
            )
            for s in seeds
        ]

    elif stage.type in ("angles", "body"):
        if stage.prompts:
            prompt_list = stage.prompts
        elif stage.preset:
            preset_prompts = get_preset(stage.preset)
            prompt_list = [p.prompt for p in preset_prompts]
        else:
            raise ValueError(
                f"Stage '{stage.name}' (type={stage.type}) requires either "
                "'preset' or 'prompts' to be set."
            )

        seeds = stage.seeds or [base_seed + i for i in range(len(prompt_list))]
        if len(seeds) < len(prompt_list):
            seeds = seeds + [
                seeds[-1] + i + 1 for i in range(len(prompt_list) - len(seeds))
            ]

        return [
            EditTask(
                source_image=src,
                prompt=p,
                seed=seeds[i],
                num_steps=num_steps,
                cfg_scale=cfg_scale,
            )
            for i, p in enumerate(prompt_list)
        ]

    else:
        raise ValueError(f"Unknown stage type: {stage.type}")


def run_stage(
    stage: StageConfig,
    source_image_path: Path,
    input_dir: Path,
    output_dir: Path,
    pipeline,
    base_seed: int = 42,
    default_num_steps: int = 50,
    default_cfg_scale: float = 4.0,
) -> list[dict]:
    """Run a single stage: expand tasks, call pipeline, save outputs + metadata."""
    stage_dir = output_dir / f"stage_{stage.name}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    tasks = expand_stage_tasks(
        stage,
        source_image=source_image_path.name,
        base_seed=base_seed,
        default_num_steps=default_num_steps,
        default_cfg_scale=default_cfg_scale,
    )

    source_img = Image.open(source_image_path).convert("RGB")
    results = []
    metadata_path = stage_dir / "metadata.jsonl"

    with open(metadata_path, "w") as meta_f:
        for i, task in enumerate(tasks):
            images = [source_img]

            inputs = {
                "image": images,
                "prompt": task.prompt,
                "generator": torch.manual_seed(task.seed),
                "true_cfg_scale": task.cfg_scale,
                "negative_prompt": "blurry, low quality, distorted, deformed, artifacts",
                "num_inference_steps": task.num_steps,
                "guidance_scale": 1.0,
                "num_images_per_prompt": 1,
            }

            with torch.inference_mode():
                output = pipeline(**inputs)

            out_name = f"{stage.name}_{i:04d}.png"
            out_path = stage_dir / out_name
            output.images[0].save(out_path)

            record = {
                "index": i,
                "stage": stage.name,
                "type": stage.type,
                "source": str(source_image_path.name),
                "prompt": task.prompt,
                "seed": task.seed,
                "num_steps": task.num_steps,
                "cfg_scale": task.cfg_scale,
                "output_path": str(out_path),
            }
            results.append(record)
            meta_f.write(json.dumps(record) + "\n")

            logger.info(
                "[%s %d/%d] %s -> %s",
                stage.name, i + 1, len(tasks), source_image_path.name, out_name,
            )

    logger.info("Stage '%s' complete. %d images saved to %s", stage.name, len(results), stage_dir)
    return results
