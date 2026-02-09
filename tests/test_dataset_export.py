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
