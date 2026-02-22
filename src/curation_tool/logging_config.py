"""Centralized logging setup for the curation tool."""
import logging
import logging.handlers
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
) -> None:
    """Configure root logger with console + optional rotating file handler.

    Args:
        level: Console log level (DEBUG, INFO, etc.).
        log_file: Path to log file. Defaults to ~/.cache/curation_tool/curation_tool.log.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicates on re-init
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s")

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler (always DEBUG)
    if log_file is None:
        log_dir = Path.home() / ".cache" / "curation_tool"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(log_dir / "curation_tool.log")

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for name in ("httpx", "httpcore", "PIL", "websocket"):
        logging.getLogger(name).setLevel(logging.WARNING)
