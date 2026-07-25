"""
AI Teacher Robot — Project File Paths.

This module defines all file and directory paths used by the application.
Centralizing path definitions ensures consistency and makes it easy to
relocate directories if needed.

All paths are relative to the project root directory.
"""

from __future__ import annotations

from pathlib import Path

# -----------------------------------------------------------------------------
# Project Root
# -----------------------------------------------------------------------------

# The project root is two levels up from this file: src/config/paths.py -> src/ -> root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
"""The root directory of the AI Teacher Robot project."""

# -----------------------------------------------------------------------------
# Source Code
# -----------------------------------------------------------------------------

SRC_DIR: Path = PROJECT_ROOT / "src"
"""Directory containing the application source code."""

CONFIG_DIR: Path = SRC_DIR / "config"
"""Directory containing configuration files."""

# -----------------------------------------------------------------------------
# Data Directories
# -----------------------------------------------------------------------------

MODELS_DIR: Path = PROJECT_ROOT / "models"
"""Directory for downloaded / quantized model files (STT, TTS, LLM)."""

ASSETS_DIR: Path = PROJECT_ROOT / "assets"
"""Directory for asset files (images, sounds, fonts)."""

IMAGES_DIR: Path = ASSETS_DIR / "images"
"""Directory for image assets."""

SOUNDS_DIR: Path = ASSETS_DIR / "sounds"
"""Directory for sound effect files."""

FONTS_DIR: Path = ASSETS_DIR / "fonts"
"""Directory for font files (e.g., Urdu fonts)."""

LOGS_DIR: Path = PROJECT_ROOT / "logs"
"""Directory for runtime log files."""

TESTS_DIR: Path = PROJECT_ROOT / "tests"
"""Directory for test files."""

SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"
"""Directory for setup, install, and run scripts."""

DOCS_DIR: Path = PROJECT_ROOT / "docs"
"""Directory for documentation files."""

# -----------------------------------------------------------------------------
# Configuration Files
# -----------------------------------------------------------------------------

CONFIG_FILE: Path = CONFIG_DIR / "config.yaml"
"""Path to the default YAML configuration file."""

ENV_FILE: Path = PROJECT_ROOT / ".env"
"""Path to the environment variables file."""

ENV_EXAMPLE_FILE: Path = PROJECT_ROOT / ".env.example"
"""Path to the example environment variables file."""

# -----------------------------------------------------------------------------
# Model Files (populated by scripts)
# -----------------------------------------------------------------------------

STT_MODEL_DIR: Path = MODELS_DIR / "stt"
"""Directory for STT model files."""

TTS_MODEL_DIR: Path = MODELS_DIR / "tts"
"""Directory for TTS model files."""

LLM_MODEL_DIR: Path = MODELS_DIR / "llm"
"""Directory for LLM model files."""


def ensure_directories() -> None:
    """Create all required directories if they do not exist.

    This function should be called at application startup to ensure that
    all necessary directories are present before the application attempts
    to read from or write to them.
    """
    directories: list[Path] = [
        MODELS_DIR,
        ASSETS_DIR,
        IMAGES_DIR,
        SOUNDS_DIR,
        FONTS_DIR,
        LOGS_DIR,
        STT_MODEL_DIR,
        TTS_MODEL_DIR,
        LLM_MODEL_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
