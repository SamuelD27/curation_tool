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

    results = run_edit(
        pipeline=pipeline,
        images=[test_img],
        prompt="Transform this into a beautiful sunset landscape with mountains.",
        seed=42,
        num_steps=40,
    )
    results[0].save(output_dir / "output.png")
    print(f"Saved to {output_dir / 'output.png'}")
    print(f"Output size: {results[0].size}")


if __name__ == "__main__":
    main()
