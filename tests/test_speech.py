"""
AI Teacher Robot — Tests for the Speech Module.

This module contains unit tests for the speech-related components:
    - AudioListener
    - SpeechToText
    - LanguageDetector

Usage:
    pytest tests/test_speech.py -v
"""

from __future__ import annotations

import pytest

from src.speech.language_detector import LanguageDetector
from src.speech.listener import AudioListener
from src.speech.stt import SpeechToText
from src.utils.constants import LANGUAGE_ENGLISH, LANGUAGE_URDU


class TestAudioListener:
    """Tests for the AudioListener class."""

    def test_init_default(self, mock_settings):
        """Test AudioListener initialization with default parameters."""
        listener = AudioListener(settings=mock_settings)
        assert listener.sample_rate == 16000
        assert listener.chunk_size == 1024
        assert listener.channels == 1
        assert listener.device_index is None
        assert not listener.is_listening()

    def test_init_custom(self):
        """Test AudioListener initialization with custom parameters."""
        listener = AudioListener(
            sample_rate=44100,
            chunk_size=2048,
            channels=2,
            device_index=1,
        )
        assert listener.sample_rate == 44100
        assert listener.chunk_size == 2048
        assert listener.channels == 2
        assert listener.device_index == 1

    def test_listen_not_started(self, mock_settings):
        """Test that listen() raises RuntimeError if not started."""
        listener = AudioListener(settings=mock_settings)
        with pytest.raises(RuntimeError, match="not started"):
            listener.listen(timeout=1.0)

    def test_context_manager(self, mock_settings):
        """Test AudioListener as a context manager."""
        with AudioListener(settings=mock_settings) as listener:
            assert listener.is_listening()
        assert not listener.is_listening()


class TestSpeechToText:
    """Tests for the SpeechToText class."""

    def test_init_default(self, mock_settings):
        """Test SpeechToText initialization with default parameters."""
        stt = SpeechToText(settings=mock_settings)
        assert stt.engine == "vosk"
        assert stt.language == "en"
        assert stt.model_path == "models/stt"

    def test_init_custom(self):
        """Test SpeechToText initialization with custom parameters."""
        stt = SpeechToText(
            engine="whisper",
            language="ur",
            model_path="models/whisper",
        )
        assert stt.engine == "whisper"
        assert stt.language == "ur"
        assert stt.model_path == "models/whisper"

    def test_transcribe_empty_audio(self, mock_settings):
        """Test transcribe() with empty audio data."""
        stt = SpeechToText(settings=mock_settings)
        result = stt.transcribe(b"")
        assert result == ""

    def test_transcribe_streaming(self, mock_settings):
        """Test transcribe_streaming() with multiple chunks."""
        stt = SpeechToText(settings=mock_settings)
        chunks = [b"\x00\x01", b"\x02\x03"]
        result = stt.transcribe_streaming(chunks)
        assert result == ""

    def test_set_language(self, mock_settings):
        """Test set_language() method."""
        stt = SpeechToText(settings=mock_settings)
        stt.set_language("ur")
        assert stt.language == "ur"


class TestLanguageDetector:
    """Tests for the LanguageDetector class."""

    def test_detect_english(self):
        """Test language detection for English text."""
        detector = LanguageDetector()
        result = detector.detect("What is the capital of France?")
        assert result == LANGUAGE_ENGLISH

    def test_detect_urdu(self):
        """Test language detection for Urdu text."""
        detector = LanguageDetector()
        result = detector.detect("فرانس کی ریاست کیا ہے؟")
        assert result == LANGUAGE_URDU

    def test_detect_empty_text(self):
        """Test language detection with empty text."""
        detector = LanguageDetector()
        result = detector.detect("")
        assert result == LANGUAGE_ENGLISH  # Default language

    def test_detect_whitespace_only(self):
        """Test language detection with whitespace-only text."""
        detector = LanguageDetector()
        result = detector.detect("   ")
        assert result == LANGUAGE_ENGLISH  # Default language

    def test_is_supported(self):
        """Test is_supported() method."""
        detector = LanguageDetector()
        assert detector.is_supported("en")
        assert detector.is_supported("ur")
        assert not detector.is_supported("fr")

    def test_default_language(self):
        """Test that the default language is used when detection is uncertain."""
        detector = LanguageDetector(default_language="ur")
        result = detector.detect("12345")
        assert result == "ur"
