# Qwen-Image-Edit-2509 Dataset Curation Tool Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set up Qwen-Image-Edit-2509 as a local inference pipeline for generating and curating images for LoRA training datasets.

**Architecture:** Python CLI tool using HuggingFace `diffusers` with the `QwenImageEditPlusPipeline`. The tool reads source images + edit instructions from a config, runs inference on GB10 (128GB unified memory), and outputs curated datasets in a structured format ready for LoRA training. No quantization needed -- full bf16 fits comfortably in 128GB.

**Tech Stack:** Python 3.12, PyTorch (CUDA), diffusers (latest from git), Pillow, pydantic (config validation), click (CLI)

**Hardware:** NVIDIA GB10 Grace Blackwell, 128GB unified memory, ~40GB needed for bf16 model

**Key References:**
- [Qwen-Image-Edit-2509 on HuggingFace](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
- [Qwen-Image GitHub](https://github.com/QwenLM/Qwen-Image)
- [CivitAI GGUF variants](https://civitai.com/models/1981702/qwen-edit-2509-new-version) (not needed for our setup)
- [HuggingFace diffusers API](https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage)

---

## Task 1: Project Scaffolding & Environment Setup

**Files:**
- Create: `environment.yml`
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/curation_tool/__init__.py`

**Step 1: Initialize git repo**

```bash
cd /home/samsam/curation_tool
git init
git lfs install
```

**Step 2: Create conda environment file**

Write `environment.yml`:
```yaml
name: curation-tool-env
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - pip:
    - torch
    - pillow
    - pydantic>=2.0
    - click
    - tqdm
    - accelerate
    - transformers>=4.51.3
    - sentencepiece
    - protobuf
    - "diffusers @ git+https://github.com/huggingface/diffusers"
```

**Step 3: Create the conda environment and activate**

```bash
conda env create -f environment.yml
conda activate curation-tool-env
```

**Step 4: Verify PyTorch CUDA works on GB10**

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"
```

Expected: CUDA available, device name shows Blackwell GPU, memory shows available unified pool.

**Step 5: Create .gitignore**

Write `.gitignore`:
```
__pycache__/
*.pyc
.env
*.pt
*.pth
*.safetensors
*.ckpt
*.onnx
*.bin
outputs/
checkpoints/
.ipynb_checkpoints/
*.log
wandb/
datasets/
models/
.venv/
*.egg-info/
dist/
build/
```

**Step 6: Create minimal package structure**

Write `pyproject.toml`:
```toml
[project]
name = "curation-tool"
version = "0.1.0"
description = "Image dataset curation tool using Qwen-Image-Edit-2509"
requires-python = ">=3.12"

[project.scripts]
curate = "curation_tool.cli:main"
```

Write `src/curation_tool/__init__.py`:
```python
"""Image dataset curation tool using Qwen-Image-Edit-2509."""
```

**Step 7: Commit**

```bash
git add environment.yml pyproject.toml .gitignore src/
git commit -m "chore: scaffold project structure and conda environment"
```

---

## Task 2: Download & Cache Model Weights

**Files:**
- Create: `scripts/download_model.py`

**Step 1: Write the download script**

Write `scripts/download_model.py`:
```python
"""Download Qwen-Image-Edit-2509 model weights from HuggingFace."""
import logging
from diffusers import QwenImageEditPlusPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Downloading Qwen-Image-Edit-2509 (bf16, ~38GB)...")
    logger.info("This will cache to ~/.cache/huggingface/hub/")

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype="auto",
    )
    logger.info("Download complete. Model cached locally.")
    del pipeline


if __name__ == "__main__":
    main()
```

**Step 2: Run the download**

```bash
conda activate curation-tool-env
python scripts/download_model.py
```

Expected: Downloads ~38GB of model weights to HuggingFace cache. Takes 10-30 min depending on connection speed. Verify with:
```bash
du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2509/
```

**Step 3: Commit**

```bash
git add scripts/download_model.py
git commit -m "chore: add model download script"
```

---

## Task 3: Basic Single-Image Edit Inference Test

**Files:**
- Create: `scripts/test_inference.py`
- Create: `tests/test_pipeline.py`

**Step 1: Write the failing test**

Write `tests/__init__.py` (empty file).

Write `tests/test_pipeline.py`:
```python
"""Smoke test for pipeline loading and inference."""
import pytest
from pathlib import Path


def test_pipeline_module_imports():
    from curation_tool.pipeline import load_pipeline, run_edit
    assert callable(load_pipeline)
    assert callable(run_edit)
```

**Step 2: Run test to verify it fails**

```bash
conda activate curation-tool-env
python -m pytest tests/test_pipeline.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'curation_tool.pipeline'`

**Step 3: Write the pipeline module**

Write `src/curation_tool/pipeline.py`:
```python
"""Core pipeline for Qwen-Image-Edit-2509 inference."""
import logging
from pathlib import Path

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

logger = logging.getLogger(__name__)

_pipeline = None


def load_pipeline(
    model_id: str = "Qwen/Qwen-Image-Edit-2509",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> QwenImageEditPlusPipeline:
    """Load and cache the Qwen-Image-Edit-2509 pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    logger.info("Loading %s on %s with %s...", model_id, device, dtype)
    _pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_id, torch_dtype=dtype
    )
    _pipeline.to(device)
    _pipeline.set_progress_bar_config(disable=None)
    logger.info("Pipeline loaded.")
    return _pipeline


def run_edit(
    pipeline: QwenImageEditPlusPipeline,
    images: list[Image.Image],
    prompt: str,
    seed: int = 0,
    num_steps: int = 40,
    cfg_scale: float = 4.0,
    guidance_scale: float = 1.0,
) -> Image.Image:
    """Run a single image edit and return the result."""
    inputs = {
        "image": images,
        "prompt": prompt,
        "generator": torch.manual_seed(seed),
        "true_cfg_scale": cfg_scale,
        "negative_prompt": " ",
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": 1,
    }

    with torch.inference_mode():
        output = pipeline(**inputs)

    return output.images[0]
```

**Step 4: Run test to verify it passes**

```bash
pip install -e . && python -m pytest tests/test_pipeline.py -v
```

Expected: PASS

**Step 5: Write a quick manual inference test script**

Write `scripts/test_inference.py`:
```python
"""Quick manual test: edit a single image."""
import logging
from pathlib import Path

from PIL import Image

from curation_tool.pipeline import load_pipeline, run_edit

logging.basicConfig(level=logging.INFO)


def main():
    # Create a simple test image (solid color)
    test_img = Image.new("RGB", (512, 512), color=(100, 150, 200))
    output_dir = Path("outputs/test")
    output_dir.mkdir(parents=True, exist_ok=True)
    test_img.save(output_dir / "input.png")

    pipeline = load_pipeline()

    result = run_edit(
        pipeline=pipeline,
        images=[test_img],
        prompt="Transform this into a beautiful sunset landscape with mountains.",
        seed=42,
        num_steps=40,
    )
    result.save(output_dir / "output.png")
    print(f"Saved to {output_dir / 'output.png'}")
    print(f"Output size: {result.size}")


if __name__ == "__main__":
    main()
```

**Step 6: Run the inference test**

```bash
conda activate curation-tool-env
python scripts/test_inference.py
```

Expected: Model loads (~40s), generates one image (~30-90s on GB10), saves to `outputs/test/output.png`. Visually inspect the output.

**Step 7: Commit**

```bash
git add src/curation_tool/pipeline.py tests/ scripts/test_inference.py
git commit -m "feat: add core pipeline with single-image edit support"
```

---

## Task 4: Configuration Schema for Batch Curation Jobs

**Files:**
- Create: `src/curation_tool/config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

Write `tests/test_config.py`:
```python
"""Tests for curation job configuration."""
import pytest
from curation_tool.config import CurationJob, EditTask


def test_edit_task_minimal():
    task = EditTask(
        source_image="input.png",
        prompt="Make the background blue",
    )
    assert task.source_image == "input.png"
    assert task.prompt == "Make the background blue"
    assert task.seed == 0
    assert task.num_steps == 40


def test_curation_job_from_yaml(tmp_path):
    config_file = tmp_path / "job.yaml"
    config_file.write_text("""
input_dir: ./raw_images
output_dir: ./curated
tasks:
  - source_image: "photo1.png"
    prompt: "Remove the background and replace with white"
    seed: 42
  - source_image: "photo2.png"
    prompt: "Change lighting to golden hour"
""")
    job = CurationJob.from_yaml(config_file)
    assert len(job.tasks) == 2
    assert job.input_dir == "./raw_images"
    assert job.output_dir == "./curated"


def test_curation_job_validates_empty_tasks():
    with pytest.raises(ValueError):
        CurationJob(input_dir=".", output_dir=".", tasks=[])
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'curation_tool.config'`

**Step 3: Write the config module**

Write `src/curation_tool/config.py`:
```python
"""Configuration schema for batch curation jobs."""
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator


class EditTask(BaseModel):
    """A single image edit task."""
    source_image: str
    prompt: str
    seed: int = 0
    num_steps: int = 40
    cfg_scale: float = 4.0
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
```

**Step 4: Add pyyaml dependency**

Add `pyyaml` to `environment.yml` pip dependencies, then:
```bash
conda activate curation-tool-env
pip install pyyaml
```

**Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_config.py -v
```

Expected: All 3 tests PASS.

**Step 6: Commit**

```bash
git add src/curation_tool/config.py tests/test_config.py
git commit -m "feat: add YAML-based curation job configuration schema"
```

---

## Task 5: Batch Processing Engine

**Files:**
- Create: `src/curation_tool/batch.py`
- Create: `tests/test_batch.py`

**Step 1: Write the failing test**

Write `tests/test_batch.py`:
```python
"""Tests for batch processing engine."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

from curation_tool.config import CurationJob, EditTask
from curation_tool.batch import run_batch


def test_run_batch_creates_output_files(tmp_path):
    """Test that batch runner creates output images and metadata."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    # Create test input images
    for name in ["img1.png", "img2.png"]:
        Image.new("RGB", (64, 64), color=(128, 128, 128)).save(input_dir / name)

    job = CurationJob(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        tasks=[
            EditTask(source_image="img1.png", prompt="make it red"),
            EditTask(source_image="img2.png", prompt="make it blue"),
        ],
    )

    # Mock the pipeline to avoid loading the real model in tests
    mock_pipeline = MagicMock()
    mock_output = MagicMock()
    mock_output.images = [Image.new("RGB", (64, 64), color=(255, 0, 0))]
    mock_pipeline.return_value = mock_output

    results = run_batch(job, pipeline=mock_pipeline)

    assert len(results) == 2
    assert output_dir.exists()
    for r in results:
        assert Path(r["output_path"]).exists()

    # Check metadata file created
    assert (output_dir / "metadata.jsonl").exists()
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_batch.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'curation_tool.batch'`

**Step 3: Write the batch processor**

Write `src/curation_tool/batch.py`:
```python
"""Batch processing engine for curation jobs."""
import json
import logging
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from curation_tool.config import CurationJob

logger = logging.getLogger(__name__)


def run_batch(
    job: CurationJob,
    pipeline,
) -> list[dict]:
    """Process all tasks in a curation job.

    Returns list of result dicts with input/output paths and metadata.
    """
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

            seed = task.seed or job.default_seed
            num_steps = task.num_steps or job.default_num_steps
            cfg_scale = task.cfg_scale or job.default_cfg_scale

            inputs = {
                "image": images,
                "prompt": task.prompt,
                "generator": torch.manual_seed(seed),
                "true_cfg_scale": cfg_scale,
                "negative_prompt": " ",
                "num_inference_steps": num_steps,
                "guidance_scale": 1.0,
                "num_images_per_prompt": 1,
            }

            with torch.inference_mode():
                output = pipeline(**inputs)

            out_name = f"{Path(task.source_image).stem}_edited_{i:04d}.png"
            out_path = output_dir / out_name
            output.images[0].save(out_path)

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
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_batch.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/curation_tool/batch.py tests/test_batch.py
git commit -m "feat: add batch processing engine with metadata output"
```

---

## Task 6: CLI Interface

**Files:**
- Create: `src/curation_tool/cli.py`
- Create: `tests/test_cli.py`

**Step 1: Write the failing test**

Write `tests/test_cli.py`:
```python
"""Tests for CLI interface."""
from click.testing import CliRunner
from curation_tool.cli import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "curation" in result.output.lower() or "edit" in result.output.lower()


def test_cli_run_missing_config():
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--config", "nonexistent.yaml"])
    assert result.exit_code != 0
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_cli.py -v
```

Expected: FAIL

**Step 3: Write the CLI module**

Write `src/curation_tool/cli.py`:
```python
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
```

**Step 4: Run tests to verify they pass**

```bash
pip install -e . && python -m pytest tests/test_cli.py -v
```

Expected: Both tests PASS.

**Step 5: Commit**

```bash
git add src/curation_tool/cli.py tests/test_cli.py
git commit -m "feat: add CLI with single-edit and batch-run commands"
```

---

## Task 7: Example Job Config & End-to-End Smoke Test

**Files:**
- Create: `examples/sample_job.yaml`
- Create: `examples/README_examples.md`
- Create: `scripts/smoke_test.py`

**Step 1: Create example job config**

Write `examples/sample_job.yaml`:
```yaml
# Sample curation job for LoRA training dataset preparation
# Place source images in ./examples/input/ before running

input_dir: ./examples/input
output_dir: ./examples/output
default_seed: 42
default_num_steps: 40
default_cfg_scale: 4.0

tasks:
  - source_image: "sample.png"
    prompt: "Change the background to a clean white studio backdrop"
    seed: 42

  - source_image: "sample.png"
    prompt: "Change the lighting to dramatic side lighting with deep shadows"
    seed: 43

  - source_image: "sample.png"
    prompt: "Transform the style to look like a watercolor painting"
    seed: 44
```

**Step 2: Create a smoke test that generates a sample image and runs the full pipeline**

Write `scripts/smoke_test.py`:
```python
"""End-to-end smoke test: creates a test image, runs batch curation."""
import logging
from pathlib import Path

from PIL import Image

from curation_tool.config import CurationJob
from curation_tool.pipeline import load_pipeline
from curation_tool.batch import run_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    base = Path("examples")
    input_dir = base / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Generate a test image if none exists
    sample = input_dir / "sample.png"
    if not sample.exists():
        logger.info("Creating test image at %s", sample)
        img = Image.new("RGB", (512, 512), color=(120, 160, 200))
        img.save(sample)

    job = CurationJob.from_yaml(base / "sample_job.yaml")
    logger.info("Loaded job with %d tasks", len(job.tasks))

    pipeline = load_pipeline()
    results = run_batch(job, pipeline=pipeline)

    logger.info("Smoke test complete. Generated %d images.", len(results))
    for r in results:
        logger.info("  %s -> %s", r["source"], r["output_path"])


if __name__ == "__main__":
    main()
```

**Step 3: Run the smoke test**

```bash
conda activate curation-tool-env
python scripts/smoke_test.py
```

Expected: Loads model, processes 3 edits, saves outputs to `examples/output/`. Check:
- `examples/output/sample_edited_0000.png` (white background)
- `examples/output/sample_edited_0001.png` (dramatic lighting)
- `examples/output/sample_edited_0002.png` (watercolor style)
- `examples/output/metadata.jsonl` (3 JSON lines with full metadata)

**Step 4: Commit**

```bash
git add examples/sample_job.yaml scripts/smoke_test.py
git commit -m "feat: add example job config and end-to-end smoke test"
```

---

## Task 8: LoRA Dataset Output Format

**Files:**
- Modify: `src/curation_tool/batch.py`
- Create: `src/curation_tool/dataset_export.py`
- Create: `tests/test_dataset_export.py`

This task adds a LoRA-training-compatible output format: images + captions in the standard `image/caption` pair structure used by tools like `ai-toolkit`, `kohya_ss`, and `SimpleTuner`.

**Step 1: Write the failing test**

Write `tests/test_dataset_export.py`:
```python
"""Tests for LoRA dataset export."""
import json
from pathlib import Path

from PIL import Image

from curation_tool.dataset_export import export_lora_dataset


def test_export_creates_image_caption_pairs(tmp_path):
    """Verify export creates {name}.png + {name}.txt pairs."""
    results = []
    for i in range(3):
        img_path = tmp_path / "source" / f"img_{i}.png"
        img_path.parent.mkdir(exist_ok=True)
        Image.new("RGB", (64, 64)).save(img_path)
        results.append({
            "index": i,
            "source": f"img_{i}.png",
            "prompt": f"a photo of sks dog, variant {i}",
            "output_path": str(img_path),
        })

    export_dir = tmp_path / "lora_dataset"
    export_lora_dataset(results, export_dir, caption_template="{prompt}")

    assert export_dir.exists()
    files = list(export_dir.iterdir())
    pngs = [f for f in files if f.suffix == ".png"]
    txts = [f for f in files if f.suffix == ".txt"]
    assert len(pngs) == 3
    assert len(txts) == 3

    # Verify caption content
    for txt in txts:
        content = txt.read_text().strip()
        assert "sks dog" in content
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_dataset_export.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the export module**

Write `src/curation_tool/dataset_export.py`:
```python
"""Export curated images as LoRA training datasets."""
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def export_lora_dataset(
    results: list[dict],
    export_dir: Path,
    caption_template: str = "{prompt}",
) -> Path:
    """Export batch results as image+caption pairs for LoRA training.

    Creates:
        export_dir/
            00000.png + 00000.txt
            00001.png + 00001.txt
            ...
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    for i, record in enumerate(results):
        src = Path(record["output_path"])
        dst_img = export_dir / f"{i:05d}.png"
        dst_txt = export_dir / f"{i:05d}.txt"

        shutil.copy2(src, dst_img)

        caption = caption_template.format(**record)
        dst_txt.write_text(caption.strip() + "\n")

    logger.info("Exported %d image+caption pairs to %s", len(results), export_dir)
    return export_dir
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_dataset_export.py -v
```

Expected: PASS

**Step 5: Add --export-lora flag to CLI run command**

Modify `src/curation_tool/cli.py`, add to the `run` command:

In the `run` function, after `results = run_batch(...)`, add:
```python
    if export_lora:
        from curation_tool.dataset_export import export_lora_dataset
        lora_dir = Path(job.output_dir) / "lora_dataset"
        export_lora_dataset(results, lora_dir, caption_template=caption_template)
        click.echo(f"LoRA dataset exported to {lora_dir}")
```

And add these options to the `run` command:
```python
@click.option("--export-lora", is_flag=True, help="Export as LoRA training dataset (image+caption pairs).")
@click.option("--caption-template", default="{prompt}", help="Caption template. Use {prompt}, {source}, {seed}.")
```

**Step 6: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: All tests PASS.

**Step 7: Commit**

```bash
git add src/curation_tool/dataset_export.py tests/test_dataset_export.py src/curation_tool/cli.py
git commit -m "feat: add LoRA dataset export with image+caption pairs"
```

---

## Summary

After completing all 8 tasks, the tool supports:

1. **Single image edit**: `curate edit -i photo.png -p "change background to white" -o result.png`
2. **Batch processing**: `curate run -c job.yaml` with YAML-configured edit tasks
3. **LoRA export**: `curate run -c job.yaml --export-lora` outputs `{N}.png + {N}.txt` pairs
4. **Full metadata**: Every run produces `metadata.jsonl` for reproducibility

**Architecture:**
```
src/curation_tool/
    __init__.py          # Package
    pipeline.py          # Model loading + single inference
    config.py            # Pydantic YAML config schema
    batch.py             # Batch processing engine
    dataset_export.py    # LoRA dataset format exporter
    cli.py               # Click CLI (edit / run commands)
```

**Memory budget on GB10 (128GB unified):**
- Model (bf16): ~38GB
- Inference workspace: ~10-15GB
- System overhead: ~5GB
- Remaining for OS/other: ~60-70GB (plenty of headroom)
