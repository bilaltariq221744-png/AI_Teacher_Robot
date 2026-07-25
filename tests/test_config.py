"""
AI Teacher Robot — Tests for the Config Module.

This module contains unit tests for the configuration components:
    - Settings
    - paths

Usage:
    pytest tests/test_config.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.paths import (
    ASSETS_DIR,
    CONFIG_DIR,
    CONFIG_FILE,
    LOGS_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    SRC_DIR,
    ensure_directories,
)
from src.config.settings import Settings


class TestPaths:
    """Tests for the paths module."""

    def test_project_root(self):
        """Test that PROJECT_ROOT is correctly resolved."""
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()

    def test_src_dir(self):
        """Test that SRC_DIR is correctly resolved."""
        assert SRC_DIR.exists()
        assert SRC_DIR.is_dir()

    def test_config_dir(self):
        """Test that CONFIG_DIR is correctly resolved."""
        assert CONFIG_DIR.exists()
        assert CONFIG_DIR.is_dir()

    def test_config_file(self):
        """Test that CONFIG_FILE path is correct."""
        assert CONFIG_FILE.name == "config.yaml"

    def test_models_dir(self):
        """Test that MODELS_DIR path is correct."""
        assert MODELS_DIR.name == "models"

    def test_assets_dir(self):
        """Test that ASSETS_DIR path is correct."""
        assert ASSETS_DIR.name == "assets"

    def test_logs_dir(self):
        """Test that LOGS_DIR path is correct."""
        assert LOGS_DIR.name == "logs"

    def test_ensure_directories(self):
        """Test that ensure_directories() creates all required directories."""
        ensure_directories()
        assert MODELS_DIR.exists()
        assert ASSETS_DIR.exists()
        assert LOGS_DIR.exists()


class TestSettings:
    """Tests for the Settings class."""

    def test_singleton(self):
        """Test that Settings follows the Singleton pattern."""
        s1 = Settings()
        s2 = Settings()
        assert s1 is s2

    def test_app_name(self):
        """Test that app_name is loaded correctly."""
        settings = Settings()
        assert settings.app_name == "AI Teacher Robot"

    def test_app_version(self):
        """Test that app_version is loaded correctly."""
        settings = Settings()
        assert settings.app_version == "0.1.0"

    def test_debug(self):
        """Test that debug is loaded correctly."""
        settings = Settings()
        assert settings.debug is False

    def test_sample_rate(self):
        """Test that sample_rate is loaded correctly."""
        settings = Settings()
        assert settings.sample_rate == 16000

    def test_stt_engine(self):
        """Test that stt_engine is loaded correctly."""
        settings = Settings()
        assert settings.stt_engine == "vosk"

    def test_llm_api_url(self):
        """Test that llm_api_url is loaded correctly."""
        settings = Settings()
        assert "localhost" in settings.llm_api_url

    def test_llm_model(self):
        """Test that llm_model is loaded correctly."""
        settings = Settings()
        assert settings.llm_model == "llama3"

    def test_tts_engine(self):
        """Test that tts_engine is loaded correctly."""
        settings = Settings()
        assert settings.tts_engine == "pyttsx3"

    def test_supported_languages(self):
        """Test that supported_languages is loaded correctly."""
        settings = Settings()
        assert "en" in settings.supported_languages
        assert "ur" in settings.supported_languages

    def test_get_system_prompt_en(self):
        """Test get_system_prompt() for English."""
        settings = Settings()
        prompt = settings.get_system_prompt("en")
        assert len(prompt) > 0

    def test_get_system_prompt_ur(self):
        """Test get_system_prompt() for Urdu."""
        settings = Settings()
        prompt = settings.get_system_prompt("ur")
        assert len(prompt) > 0

    def test_repr(self):
        """Test __repr__() method."""
        settings = Settings()
        repr_str = repr(settings)
        assert "AI Teacher Robot" in repr_str
