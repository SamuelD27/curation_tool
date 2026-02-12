"""Multi-stage face dataset generation pipeline with pause/resume."""
import json
import logging
import shutil
from pathlib import Path

import yaml
from pydantic import BaseModel

from curation_tool.stages import StageConfig, run_stage

logger = logging.getLogger(__name__)


class FacePipelineConfig(BaseModel):
    """Top-level config for the face dataset pipeline."""

    trigger_word: str = "sks_person"
    input_dir: str = "./input"
    output_dir: str = "./output"
    model_id: str = "ovedrive/Qwen-Image-Edit-2509-4bit"
    default_seed: int = 42
    default_num_steps: int = 50
    default_cfg_scale: float = 4.0
    stages: list[StageConfig]

    @classmethod
    def from_yaml(cls, path: Path) -> "FacePipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


class PipelineState(BaseModel):
    """Serializable pipeline state for pause/resume."""

    completed_stages: list[str] = []
    picks: dict[str, str] = {}
    current_source: str | None = None

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        return cls.model_validate_json(path.read_text())


def _resolve_source(
    stage: StageConfig,
    config: FacePipelineConfig,
    state: PipelineState,
    stages_so_far: list[StageConfig],
) -> Path:
    """Determine the source image path for a stage."""
    if state.current_source:
        return Path(state.current_source)

    if stage.source_image:
        return Path(config.input_dir) / stage.source_image

    # Default: first image in input_dir
    input_dir = Path(config.input_dir)
    images = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )
    if not images:
        raise FileNotFoundError(f"No images found in {input_dir}")
    return images[0]


def run_face_pipeline(
    config: FacePipelineConfig,
    pipeline=None,
    interactive: bool = True,
    picks: dict[str, str] | None = None,
) -> list[dict]:
    """Run the multi-stage face pipeline.

    Args:
        config: Pipeline configuration.
        pipeline: Loaded diffusion pipeline (or mock for testing).
        interactive: If True, pause for user picks via click.prompt().
        picks: Pre-supplied picks for scripted mode (stage_name -> image_path).

    Returns:
        All results from non-refine stages (the final dataset images).
    """
    picks = picks or {}
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = output_dir / "pipeline_state.json"
    if state_path.exists():
        state = PipelineState.load(state_path)
        logger.info("Resumed pipeline state: %d stages completed", len(state.completed_stages))
    else:
        state = PipelineState()

    all_results = []

    for stage in config.stages:
        if stage.name in state.completed_stages:
            logger.info("Skipping completed stage: %s", stage.name)
            continue

        source_path = _resolve_source(stage, config, state, config.stages)
        logger.info("Stage '%s': source=%s", stage.name, source_path)

        results = run_stage(
            stage,
            source_image_path=source_path,
            input_dir=Path(config.input_dir),
            output_dir=output_dir,
            pipeline=pipeline,
            base_seed=config.default_seed,
            default_num_steps=config.default_num_steps,
            default_cfg_scale=config.default_cfg_scale,
        )

        state.completed_stages.append(stage.name)

        if stage.type == "refine":
            pick_path = _handle_refine_pick(
                stage, results, picks, interactive, output_dir,
            )
            if pick_path is None:
                state.save(state_path)
                logger.info(
                    "Pipeline paused after stage '%s'. "
                    "Resume with --pick %s=<path> after choosing.",
                    stage.name, stage.name,
                )
                return all_results

            state.current_source = str(pick_path)
            state.picks[stage.name] = str(pick_path)
        else:
            all_results.extend(results)

        state.save(state_path)

    logger.info("Pipeline complete. %d dataset images generated.", len(all_results))
    return all_results


def _handle_refine_pick(
    stage: StageConfig,
    results: list[dict],
    picks: dict[str, str],
    interactive: bool,
    output_dir: Path,
) -> Path | None:
    """Handle the pick step after a refine stage.

    Returns the picked image path, or None if waiting for user input.
    """
    # Check pre-supplied picks
    if stage.name in picks:
        pick_path = Path(picks[stage.name])
        if not pick_path.is_absolute():
            pick_path = output_dir / f"stage_{stage.name}" / pick_path
        logger.info("Using pre-supplied pick for '%s': %s", stage.name, pick_path)
        return pick_path

    # Check for pick.png dropped in stage dir
    stage_dir = output_dir / f"stage_{stage.name}"
    pick_file = stage_dir / "pick.png"
    if pick_file.exists():
        logger.info("Found pick.png in %s", stage_dir)
        return pick_file

    # Interactive prompt
    if interactive:
        import click

        click.echo(f"\nRefine stage '{stage.name}' generated {len(results)} candidates:")
        for r in results:
            click.echo(f"  [{r['index']}] {Path(r['output_path']).name} (seed={r['seed']})")

        choice = click.prompt(
            "Choose candidate (index or filename)",
            type=str,
        )

        # Try as index first
        try:
            idx = int(choice)
            return Path(results[idx]["output_path"])
        except (ValueError, IndexError):
            pass

        # Try as filename
        candidate = stage_dir / choice
        if candidate.exists():
            return candidate

        click.echo(f"Invalid choice: {choice}")
        return None

    # Non-interactive, no pick provided -> pause
    return None


def export_face_dataset(
    config: FacePipelineConfig,
    results: list[dict],
) -> Path:
    """Export pipeline results as LoRA dataset with trigger-word captions."""
    export_dir = Path(config.output_dir) / "lora_dataset"
    export_dir.mkdir(parents=True, exist_ok=True)

    for i, record in enumerate(results):
        src = Path(record["output_path"])
        dst_img = export_dir / f"{i:05d}.png"
        dst_txt = export_dir / f"{i:05d}.txt"

        shutil.copy2(src, dst_img)
        dst_txt.write_text(config.trigger_word + "\n")

    logger.info(
        "Exported %d images with trigger word '%s' to %s",
        len(results), config.trigger_word, export_dir,
    )
    return export_dir
