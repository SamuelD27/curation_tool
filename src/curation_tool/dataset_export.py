"""Export curated images as LoRA training datasets."""
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def export_lora_dataset(
    results: list[dict],
    export_dir: Path,
    caption_template: str = "{prompt}",
    trigger_word: str | None = None,
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

        caption = trigger_word if trigger_word else caption_template.format(**record)
        dst_txt.write_text(caption.strip() + "\n")

    logger.info("Exported %d image+caption pairs to %s", len(results), export_dir)
    return export_dir
