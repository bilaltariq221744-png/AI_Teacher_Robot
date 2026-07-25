"""
AI Teacher Robot — Test Configuration (pytest fixtures).

This module defines shared pytest fixtures used across all test modules.
Fixtures include mock settings, mock hardware, and other test utilities.

Usage:
    from tests.conftest import mock_settings

    def test_something(mock_settings):
        assert mock_settings.app_name == "AI Teacher Robot"
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_settings() -> MagicMock:
    """Provide a mock Settings instance for testing.

    This fixture returns a MagicMock that simulates the Settings class
    with all expected attributes. It is useful for testing modules that
    depend on configuration without loading the actual config file.

    Returns:
        A MagicMock simulating the Settings class.
    """
    settings = MagicMock()
    settings.app_name = "AI Teacher Robot"
    settings.app_version = "0.1.0"
    settings.debug = False
    settings.sample_rate = 16000
    settings.chunk_size = 1024
    settings.audio_channels = 1
    settings.audio_device_index = None
    settings.stt_engine = "vosk"
    settings.stt_model_path = "models/stt"
    settings.stt_language = "en"
    settings.listen_timeout = 5
    settings.language_detection_enabled = True
    settings.default_language = "en"
    settings.llm_api_url = "http://localhost:11434/api/chat"
    settings.llm_model = "llama3"
    settings.llm_timeout = 30
    settings.llm_max_tokens = 500
    settings.llm_temperature = 0.7
    settings.system_prompt_en = "You are a helpful AI teacher."
    settings.system_prompt_ur = "آپ ایک مددگار ای آئی اساتذہ ہیں۔"
    settings.tts_engine = "pyttsx3"
    settings.tts_language = "en"
    settings.tts_rate = 200
    settings.tts_volume = 1.0
    settings.tts_voice_id = None
    settings.hardware_enabled = False  # Disabled for testing
    settings.eye_left_gpio = 18
    settings.eye_right_gpio = 19
    settings.mouth_gpio = 13
    settings.pwm_frequency = 50
    settings.animation_speed = 1.0
    settings.log_level = "INFO"
    settings.log_file = "logs/ai_teacher_robot.log"
    settings.log_rotation = "10 MB"
    settings.log_retention = "7 days"
    settings.supported_languages = ["en", "ur"]
    return settings


@pytest.fixture
def mock_hardware_enabled_settings() -> MagicMock:
    """Provide a mock Settings instance with hardware enabled.

    Returns:
        A MagicMock simulating the Settings class with hardware_enabled=True.
    """
    settings = MagicMock()
    settings.hardware_enabled = True
    settings.eye_left_gpio = 18
    settings.eye_right_gpio = 19
    settings.mouth_gpio = 13
    settings.pwm_frequency = 50
    settings.animation_speed = 1.0
    return settings


@pytest.fixture
def sample_english_text() -> str:
    """Provide sample English text for testing.

    Returns:
        A sample English string.
    """
    return "What is the capital of France?"


@pytest.fixture
def sample_urdu_text() -> str:
    """Provide sample Urdu text for testing.

    Returns:
        A sample Urdu string.
    """
    return "فرانس کی ریاست کیا ہے؟"


@pytest.fixture
def sample_audio_data() -> bytes:
    """Provide sample audio data for testing.

    Returns:
        A bytes object simulating raw audio data.
    """
    return b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"
