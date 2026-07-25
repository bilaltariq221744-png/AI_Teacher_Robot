"""
AI Teacher Robot — Helper Functions.

This module contains general-purpose utility functions that are used across
multiple modules in the application. These include timing utilities, string
formatting, text processing, and other common operations.

Usage:
    from src.utils.helpers import format_duration, truncate_text
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def timer(description: str = "Operation") -> Iterator[None]:
    """Context manager for timing code execution.

    Logs the duration of the code block wrapped in this context manager.

    Args:
        description: A human-readable description of the operation being timed.

    Yields:
        None (the context manager is used for its side effect of timing).

    Example:
        with timer("STT transcription"):
            text = stt.transcribe(audio)
    """
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    start_time: float = time.perf_counter()
    try:
        yield
    finally:
        elapsed: float = time.perf_counter() - start_time
        logger.debug(f"{description} completed in {elapsed:.3f} seconds")


def format_duration(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string.

    Args:
        seconds: The duration in seconds.

    Returns:
        A human-readable duration string (e.g., "1m 30s", "2h 15m 45s").
    """
    if seconds < 0:
        return "0s"

    hours: int = int(seconds // 3600)
    minutes: int = int((seconds % 3600) // 60)
    secs: int = int(seconds % 60)

    parts: list[str] = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")

    return " ".join(parts)


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to a maximum length, adding an ellipsis if truncated.

    Args:
        text: The input text to truncate.
        max_length: The maximum number of characters to keep.

    Returns:
        The truncated text, with "..." appended if it was truncated.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def safe_get(data: dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely retrieve a value from a dictionary.

    Args:
        data: The dictionary to search.
        key: The key to look up.
        default: The default value to return if the key is not found.

    Returns:
        The value associated with the key, or the default value.
    """
    return data.get(key, default)


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries.

    The override dictionary takes precedence over the base dictionary.
    Nested dictionaries are merged recursively.

    Args:
        base: The base dictionary.
        override: The override dictionary (takes precedence).

    Returns:
        A new dictionary containing the merged result.
    """
    result: dict[str, Any] = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def is_valid_language(language: str) -> bool:
    """Check if a language code is supported.

    Args:
        language: The language code to check (e.g., "en", "ur").

    Returns:
        True if the language is supported, False otherwise.
    """
    from src.utils.constants import SUPPORTED_LANGUAGES

    return language in SUPPORTED_LANGUAGES
