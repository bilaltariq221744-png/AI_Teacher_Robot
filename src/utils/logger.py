"""
AI Teacher Robot — Logging Configuration.

This module configures structured logging using the Loguru library. It
provides a centralized logger that can be used across all modules.

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("This is an info message")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from src.utils.constants import APP_NAME, LOGS_DIR


def setup_logger(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """Configure the global logger.

    This function sets up Loguru with:
        - Console output (stderr) with colored formatting.
        - Optional file output with rotation and retention.

    Args:
        level: The minimum log level (e.g., "DEBUG", "INFO", "WARNING").
        log_file: Optional path to a log file. If not provided, only console
            logging is enabled.
        rotation: File rotation policy (e.g., "10 MB", "1 day").
        retention: How long to keep old log files (e.g., "7 days", "30 days").
    """
    # Remove the default handler
    logger.remove()

    # Add console handler with colored output
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )

    # Add file handler if a log file path is provided
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_path),
            level=level,
            rotation=rotation,
            retention=retention,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
        )


def get_logger(name: str) -> "logger":
    """Get a logger instance for the given module name.

    This is a thin wrapper around Loguru's logger that binds the module name
    for easier filtering and identification in log output.

    Args:
        name: The name of the module (typically __name__).

    Returns:
        A configured Loguru logger instance.
    """
    return logger.bind(module=name)


# -----------------------------------------------------------------------------
# Initialize the logger on module import
# -----------------------------------------------------------------------------

# Set up default logging to console and a log file in the logs/ directory
_default_log_file: str = str(Path(LOGS_DIR) / f"{APP_NAME.lower().replace(' ', '_')}.log")

setup_logger(level="INFO", log_file=_default_log_file)
