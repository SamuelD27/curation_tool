"""Configuration schema for batch curation jobs."""
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator


class EditTask(BaseModel):
    """A single image edit task."""

    source_image: str
    prompt: str
    seed: int | None = None
    num_steps: int | None = None
    cfg_scale: float | None = None
    reference_images: list[str] = []


class CurationJob(BaseModel):
    """A batch curation job configuration."""

    input_dir: str
    output_dir: str
    tasks: list[EditTask]
    default_seed: int = 0
    default_num_steps: int = 40
    default_cfg_scale: float = 4.0

    @field_validator("tasks")
    @classmethod
    def tasks_not_empty(cls, v):
        if not v:
            raise ValueError("tasks must not be empty")
        return v

    @classmethod
    def from_yaml(cls, path: Path) -> "CurationJob":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
