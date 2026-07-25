"""
AI Teacher Robot — Project Constants.

This module defines project-wide constants such as the application name,
version, supported languages, and default configuration values.

All constants are defined here to avoid magic strings and numbers scattered
throughout the codebase.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Application Metadata
# -----------------------------------------------------------------------------

APP_NAME: str = "AI Teacher Robot"
"""The display name of the application."""

APP_VERSION: str = "0.1.0"
"""The current version of the application."""

APP_DESCRIPTION: str = (
    "An AI-powered educational robot for Grade 2-10 students, "
    "running locally on a Raspberry Pi with multilingual voice interaction."
)
"""A short description of the application."""

# -----------------------------------------------------------------------------
# Supported Languages
# -----------------------------------------------------------------------------

LANGUAGE_ENGLISH: str = "en"
"""Language code for English."""

LANGUAGE_URDU: str = "ur"
"""Language code for Urdu."""

SUPPORTED_LANGUAGES: list[str] = [LANGUAGE_ENGLISH, LANGUAGE_URDU]
"""List of all supported language codes."""

LANGUAGE_NAMES: dict[str, str] = {
    LANGUAGE_ENGLISH: "English",
    LANGUAGE_URDU: "Urdu",
}
"""Mapping of language codes to human-readable names."""

# -----------------------------------------------------------------------------
# Default Configuration Values
# -----------------------------------------------------------------------------

DEFAULT_LLM_MODEL: str = "llama3"
"""Default LLM model name to use."""

DEFAULT_LLM_API_URL: str = "http://localhost:11434/api/chat"
"""Default URL for the local LLM API (Ollama-compatible)."""

DEFAULT_TTS_ENGINE: str = "pyttsx3"
"""Default Text-to-Speech engine."""

DEFAULT_STT_ENGINE: str = "vosk"
"""Default Speech-to-Text engine."""

DEFAULT_SAMPLE_RATE: int = 16000
"""Default audio sample rate in Hz."""

DEFAULT_CHUNK_SIZE: int = 1024
"""Default audio chunk size for processing."""

# -----------------------------------------------------------------------------
# Hardware Configuration
# -----------------------------------------------------------------------------

DEFAULT_EYE_LEFT_GPIO: int = 18
"""Default GPIO pin for the left eye servo."""

DEFAULT_EYE_RIGHT_GPIO: int = 19
"""Default GPIO pin for the right eye servo."""

DEFAULT_MOUTH_GPIO: int = 13
"""Default GPIO pin for the mouth servo."""

SERVO_PWM_FREQUENCY: int = 50
"""Default PWM frequency for servo motors in Hz."""

# -----------------------------------------------------------------------------
# Timeouts and Limits
# -----------------------------------------------------------------------------

DEFAULT_LISTEN_TIMEOUT: int = 5
"""Default timeout (in seconds) for listening to voice input."""

DEFAULT_LLM_TIMEOUT: int = 30
"""Default timeout (in seconds) for LLM API requests."""

MAX_RESPONSE_LENGTH: int = 500
"""Maximum number of characters in an LLM response."""

# -----------------------------------------------------------------------------
# File Paths (relative to project root)
# -----------------------------------------------------------------------------

CONFIG_FILE: str = "src/config/config.yaml"
"""Path to the default configuration file."""

ENV_FILE: str = ".env"
"""Path to the environment variables file."""

LOGS_DIR: str = "logs"
"""Directory for log files."""

MODELS_DIR: str = "models"
"""Directory for model files."""

ASSETS_DIR: str = "assets"
"""Directory for asset files (images, sounds, fonts)."""
