"""
AI Teacher Robot — Settings Management.

This module provides the `Settings` class, which loads configuration from
a YAML file and environment variables. It uses the Singleton pattern to
ensure a single configuration instance throughout the application.

Usage:
    from src.config.settings import Settings

    settings = Settings()
    print(settings.llm_api_url)
    print(settings.supported_languages)
"""

from __future__ import annotations

import os
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

from src.config.paths import CONFIG_FILE, ENV_FILE
from src.utils.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EYE_LEFT_GPIO,
    DEFAULT_EYE_RIGHT_GPIO,
    DEFAULT_LLM_API_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LISTEN_TIMEOUT,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_MOUTH_GPIO,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_STT_ENGINE,
    DEFAULT_TTS_ENGINE,
    LANGUAGE_ENGLISH,
    MAX_RESPONSE_LENGTH,
    SERVO_PWM_FREQUENCY,
    SUPPORTED_LANGUAGES,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Settings:
    """Application configuration settings.

    This class loads configuration from a YAML file and environment variables.
    Environment variables take precedence over YAML file values, which in turn
    take precedence over default values.

    Attributes:
        app_name: The application name.
        app_version: The application version.
        debug: Whether debug mode is enabled.
        sample_rate: Audio sample rate in Hz.
        chunk_size: Audio chunk size for processing.
        stt_engine: The STT engine to use.
        stt_model_path: Path to the STT model directory.
        stt_language: Default language for STT.
        listen_timeout: Timeout for listening to voice input.
        language_detection_enabled: Whether language detection is enabled.
        default_language: Fallback language.
        llm_api_url: URL for the local LLM API.
        llm_model: LLM model name.
        llm_timeout: Timeout for LLM API requests.
        llm_max_tokens: Maximum tokens in LLM response.
        llm_temperature: Sampling temperature.
        system_prompt_en: System prompt for English.
        system_prompt_ur: System prompt for Urdu.
        tts_engine: The TTS engine to use.
        tts_language: Default language for TTS.
        tts_rate: Speech rate in words per minute.
        tts_volume: Volume level (0.0 to 1.0).
        tts_voice_id: Voice ID for TTS.
        hardware_enabled: Whether hardware control is enabled.
        eye_left_gpio: GPIO pin for left eye servo.
        eye_right_gpio: GPIO pin for right eye servo.
        mouth_gpio: GPIO pin for mouth servo.
        pwm_frequency: PWM frequency for servos.
        animation_speed: Animation speed multiplier.
        log_level: Log level.
        log_file: Path to log file.
        log_rotation: Log file rotation policy.
        log_retention: Log file retention policy.
    """

    # Class-level singleton instance
    _instance: Optional["Settings"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "Settings":
        """Ensure only one Settings instance exists (Singleton pattern).

        Returns:
            The single Settings instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize the Settings instance.

        Loads configuration from a YAML file and environment variables.
        If the instance has already been initialized, this method does
        nothing (Singleton pattern).

        Args:
            config_path: Optional path to a YAML configuration file.
                If not provided, the default config file is used.
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._config_path: str = config_path or str(CONFIG_FILE)
        self._config: dict[str, Any] = {}

        # Load environment variables from .env file
        load_dotenv(ENV_FILE)

        # Load YAML configuration
        self._load_yaml_config()

        # Apply configuration values
        self._apply_config()

        self._initialized: bool = True
        logger.info(f"Settings loaded from {self._config_path}")

    def _load_yaml_config(self) -> None:
        """Load configuration from the YAML file.

        If the file does not exist, default values are used.
        """
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
            logger.debug(f"YAML config loaded: {self._config_path}")
        except FileNotFoundError:
            logger.warning(
                f"Config file not found at {self._config_path}. "
                "Using default values."
            )
            self._config = {}
        except yaml.YAMLError as exc:
            logger.error(f"Error parsing YAML config: {exc}")
            self._config = {}

    def _get(
        self,
        section: str,
        key: str,
        default: Any,
        env_var: Optional[str] = None,
    ) -> Any:
        """Retrieve a configuration value.

        The lookup order is:
            1. Environment variable (if env_var is provided)
            2. YAML config file (section.key)
            3. Default value

        Args:
            section: The YAML section name (e.g., "llm").
            key: The key within the section (e.g., "api_url").
            default: The default value if not found.
            env_var: Optional environment variable name to check first.

        Returns:
            The configuration value.
        """
        # Check environment variable first
        if env_var and os.getenv(env_var) is not None:
            return os.getenv(env_var)

        # Check YAML config
        section_data: dict[str, Any] = self._config.get(section, {})
        if key in section_data:
            return section_data[key]

        # Return default
        return default

    def _apply_config(self) -> None:
        """Apply all configuration values to instance attributes."""
        # --- Application ---
        self.app_name: str = self._get("app", "name", "AI Teacher Robot")
        self.app_version: str = self._get("app", "version", "0.1.0")
        self.debug: bool = self._get("app", "debug", False)

        # --- Audio ---
        self.sample_rate: int = self._get(
            "audio", "sample_rate", DEFAULT_SAMPLE_RATE, "AUDIO_SAMPLE_RATE"
        )
        self.chunk_size: int = self._get(
            "audio", "chunk_size", DEFAULT_CHUNK_SIZE, "AUDIO_CHUNK_SIZE"
        )
        self.audio_channels: int = self._get("audio", "channels", 1)
        self.audio_device_index: Optional[int] = self._get(
            "audio", "device_index", None
        )

        # --- STT ---
        self.stt_engine: str = self._get(
            "stt", "engine", DEFAULT_STT_ENGINE, "STT_ENGINE"
        )
        self.stt_model_path: str = self._get("stt", "model_path", "models/stt")
        self.stt_language: str = self._get(
            "stt", "language", LANGUAGE_ENGLISH, "STT_LANGUAGE"
        )
        self.listen_timeout: int = self._get(
            "stt", "timeout", DEFAULT_LISTEN_TIMEOUT, "LISTEN_TIMEOUT"
        )

        # --- Language Detection ---
        self.language_detection_enabled: bool = self._get(
            "language_detection", "enabled", True
        )
        self.default_language: str = self._get(
            "language_detection", "default_language", LANGUAGE_ENGLISH
        )

        # --- LLM ---
        self.llm_api_url: str = self._get(
            "llm", "api_url", DEFAULT_LLM_API_URL, "LLM_API_URL"
        )
        self.llm_model: str = self._get(
            "llm", "model", DEFAULT_LLM_MODEL, "LLM_MODEL"
        )
        self.llm_timeout: int = self._get(
            "llm", "timeout", DEFAULT_LLM_TIMEOUT, "LLM_TIMEOUT"
        )
        self.llm_max_tokens: int = self._get(
            "llm", "max_tokens", MAX_RESPONSE_LENGTH, "LLM_MAX_TOKENS"
        )
        self.llm_temperature: float = self._get(
            "llm", "temperature", 0.7, "LLM_TEMPERATURE"
        )
        self.system_prompt_en: str = self._get(
            "llm", "system_prompt_en", ""
        )
        self.system_prompt_ur: str = self._get(
            "llm", "system_prompt_ur", ""
        )

        # --- TTS ---
        self.tts_engine: str = self._get(
            "tts", "engine", DEFAULT_TTS_ENGINE, "TTS_ENGINE"
        )
        self.tts_language: str = self._get(
            "tts", "language", LANGUAGE_ENGLISH, "TTS_LANGUAGE"
        )
        self.tts_rate: int = self._get("tts", "rate", 200, "TTS_RATE")
        self.tts_volume: float = self._get("tts", "volume", 1.0, "TTS_VOLUME")
        self.tts_voice_id: Optional[str] = self._get(
            "tts", "voice_id", None, "TTS_VOICE_ID"
        )

        # --- Hardware ---
        self.hardware_enabled: bool = self._get(
            "hardware", "enabled", True
        )
        self.eye_left_gpio: int = self._get(
            "hardware", "eye_left_gpio", DEFAULT_EYE_LEFT_GPIO
        )
        self.eye_right_gpio: int = self._get(
            "hardware", "eye_right_gpio", DEFAULT_EYE_RIGHT_GPIO
        )
        self.mouth_gpio: int = self._get(
            "hardware", "mouth_gpio", DEFAULT_MOUTH_GPIO
        )
        self.pwm_frequency: int = self._get(
            "hardware", "pwm_frequency", SERVO_PWM_FREQUENCY
        )
        self.animation_speed: float = self._get(
            "hardware", "animation_speed", 1.0
        )

        # --- Logging ---
        self.log_level: str = self._get("logging", "level", "INFO")
        self.log_file: str = self._get(
            "logging", "log_file", "logs/ai_teacher_robot.log"
        )
        self.log_rotation: str = self._get(
            "logging", "rotation", "10 MB"
        )
        self.log_retention: str = self._get(
            "logging", "retention", "7 days"
        )

        # --- Derived ---
        self.supported_languages: list[str] = list(SUPPORTED_LANGUAGES)

    def get_system_prompt(self, language: str) -> str:
        """Get the system prompt for the specified language.

        Args:
            language: The language code (e.g., "en", "ur").

        Returns:
            The system prompt string for the specified language.
            Falls back to the English prompt if the language is not found.
        """
        if language == "ur":
            return self.system_prompt_ur or self.system_prompt_en
        return self.system_prompt_en

    def __repr__(self) -> str:
        """Return a string representation of the Settings instance."""
        return (
            f"Settings(app_name={self.app_name!r}, "
            f"app_version={self.app_version!r}, "
            f"debug={self.debug!r})"
        )
