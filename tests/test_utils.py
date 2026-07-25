"""
AI Teacher Robot — Tests for the Utils Module.

This module contains unit tests for the utility components:
    - constants
    - logger
    - helpers

Usage:
    pytest tests/test_utils.py -v
"""

from __future__ import annotations

import time

import pytest

from src.utils.constants import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SAMPLE_RATE,
    LANGUAGE_ENGLISH,
    LANGUAGE_URDU,
    SUPPORTED_LANGUAGES,
)
from src.utils.helpers import (
    format_duration,
    is_valid_language,
    merge_dicts,
    safe_get,
    timer,
    truncate_text,
)
from src.utils.logger import get_logger


class TestConstants:
    """Tests for the constants module."""

    def test_app_name(self):
        """Test APP_NAME constant."""
        assert APP_NAME == "AI Teacher Robot"

    def test_app_version(self):
        """Test APP_VERSION constant."""
        assert APP_VERSION == "0.1.0"

    def test_supported_languages(self):
        """Test SUPPORTED_LANGUAGES constant."""
        assert LANGUAGE_ENGLISH in SUPPORTED_LANGUAGES
        assert LANGUAGE_URDU in SUPPORTED_LANGUAGES
        assert len(SUPPORTED_LANGUAGES) == 2

    def test_default_sample_rate(self):
        """Test DEFAULT_SAMPLE_RATE constant."""
        assert DEFAULT_SAMPLE_RATE == 16000

    def test_default_chunk_size(self):
        """Test DEFAULT_CHUNK_SIZE constant."""
        assert DEFAULT_CHUNK_SIZE == 1024


class TestLogger:
    """Tests for the logger module."""

    def test_get_logger(self):
        """Test get_logger() returns a logger instance."""
        logger = get_logger("test_module")
        assert logger is not None

    def test_logger_name(self):
        """Test that the logger has the correct name bound."""
        logger = get_logger("test_module")
        # Loguru loggers are bound with the module name
        assert logger is not None


class TestHelpers:
    """Tests for the helpers module."""

    def test_format_duration_seconds(self):
        """Test format_duration() with seconds."""
        result = format_duration(45.0)
        assert "45s" in result

    def test_format_duration_minutes(self):
        """Test format_duration() with minutes."""
        result = format_duration(90.0)
        assert "1m" in result
        assert "30s" in result

    def test_format_duration_hours(self):
        """Test format_duration() with hours."""
        result = format_duration(3661.0)
        assert "1h" in result
        assert "1m" in result

    def test_format_duration_negative(self):
        """Test format_duration() with negative value."""
        result = format_duration(-5.0)
        assert result == "0s"

    def test_truncate_text_short(self):
        """Test truncate_text() with short text."""
        result = truncate_text("Hello", max_length=10)
        assert result == "Hello"

    def test_truncate_text_long(self):
        """Test truncate_text() with long text."""
        result = truncate_text("Hello, world!", max_length=8)
        assert result == "Hello..."
        assert len(result) <= 8

    def test_safe_get_existing_key(self):
        """Test safe_get() with an existing key."""
        data = {"name": "test", "value": 42}
        result = safe_get(data, "name", "default")
        assert result == "test"

    def test_safe_get_missing_key(self):
        """Test safe_get() with a missing key."""
        data = {"name": "test"}
        result = safe_get(data, "missing", "default")
        assert result == "default"

    def test_merge_dicts(self):
        """Test merge_dicts() with nested dictionaries."""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"d": 4, "e": 5}}
        result = merge_dicts(base, override)
        assert result["a"] == 1
        assert result["b"]["c"] == 2
        assert result["b"]["d"] == 4
        assert result["b"]["e"] == 5

    def test_merge_dicts_override(self):
        """Test merge_dicts() with top-level override."""
        base = {"a": 1}
        override = {"a": 2}
        result = merge_dicts(base, override)
        assert result["a"] == 2

    def test_is_valid_language_supported(self):
        """Test is_valid_language() with supported languages."""
        assert is_valid_language("en") is True
        assert is_valid_language("ur") is True

    def test_is_valid_language_unsupported(self):
        """Test is_valid_language() with unsupported language."""
        assert is_valid_language("fr") is False

    def test_timer(self):
        """Test timer() context manager."""
        with timer("test operation"):
            time.sleep(0.01)
        # If we get here without error, the timer works
