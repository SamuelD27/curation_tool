"""Configuration schema for batch curation jobs."""
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator


class ComfyUISettings(BaseModel):
    """Connection settings for ComfyUI backend."""

    host: str = "127.0.0.1"
    port: int = 8188
    timeout: float = 300.0
    poll_interval: float = 0.5

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class WorkflowSettings(BaseModel):
    """Defaults for workflow generation."""

    template: str = "qwen_face_edit"
    face_restore: bool = True
    face_restore_fidelity: float = 0.8
    upscale_model: str = "RealESRGAN_x4plus.pth"


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
    default_num_steps: int = 30
    default_cfg_scale: float = 3.5
    comfyui: ComfyUISettings = ComfyUISettings()
    workflow: WorkflowSettings = WorkflowSettings()

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
