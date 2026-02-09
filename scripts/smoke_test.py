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
