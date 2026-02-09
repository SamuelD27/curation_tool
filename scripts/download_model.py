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
