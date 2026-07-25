"""
AI Teacher Robot — Tests for the TTS Module.

This module contains unit tests for the text-to-speech components:
    - TextToSpeechSynthesizer
    - Speaker

Usage:
    pytest tests/test_tts.py -v
"""

from __future__ import annotations

import pytest

from src.tts.synthesizer import TextToSpeechSynthesizer
from src.tts.speaker import Speaker


class TestTextToSpeechSynthesizer:
    """Tests for the TextToSpeechSynthesizer class."""

    def test_init_default(self, mock_settings):
        """Test TextToSpeechSynthesizer initialization with defaults."""
        synthesizer = TextToSpeechSynthesizer(settings=mock_settings)
        assert synthesizer.engine == "pyttsx3"
        assert synthesizer.language == "en"
        assert synthesizer.rate == 200
        assert synthesizer.volume == 1.0
        assert synthesizer.voice_id is None

    def test_init_custom(self):
        """Test TextToSpeechSynthesizer initialization with custom params."""
        synthesizer = TextToSpeechSynthesizer(
            engine="coqui",
            language="ur",
            rate=150,
            volume=0.8,
            voice_id="ur_voice_1",
        )
        assert synthesizer.engine == "coqui"
        assert synthesizer.language == "ur"
        assert synthesizer.rate == 150
        assert synthesizer.volume == 0.8
        assert synthesizer.voice_id == "ur_voice_1"

    def test_synthesize_empty_text(self, mock_settings):
        """Test synthesize() raises ValueError for empty text."""
        synthesizer = TextToSpeechSynthesizer(settings=mock_settings)
        with pytest.raises(ValueError, match="must not be empty"):
            synthesizer.synthesize("")

    def test_synthesize_whitespace_only(self, mock_settings):
        """Test synthesize() raises ValueError for whitespace-only text."""
        synthesizer = TextToSpeechSynthesizer(settings=mock_settings)
        with pytest.raises(ValueError, match="must not be empty"):
            synthesizer.synthesize("   ")

    def test_set_language(self, mock_settings):
        """Test set_language() method."""
        synthesizer = TextToSpeechSynthesizer(settings=mock_settings)
        synthesizer.set_language("ur")
        assert synthesizer.language == "ur"

    def test_set_rate(self, mock_settings):
        """Test set_rate() method."""
        synthesizer = TextToSpeechSynthesizer(settings=mock_settings)
        synthesizer.set_rate(300)
        assert synthesizer.rate == 300

    def test_set_volume(self, mock_settings):
        """Test set_volume() method."""
        synthesizer = TextToSpeechSynthesizer(settings=mock_settings)
        synthesizer.set_volume(0.5)
        assert synthesizer.volume == 0.5

    def test_set_volume_invalid(self, mock_settings):
        """Test set_volume() raises ValueError for invalid volume."""
        synthesizer = TextToSpeechSynthesizer(settings=mock_settings)
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            synthesizer.set_volume(1.5)

    def test_set_volume_negative(self, mock_settings):
        """Test set_volume() raises ValueError for negative volume."""
        synthesizer = TextToSpeechSynthesizer(settings=mock_settings)
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            synthesizer.set_volume(-0.5)

    def test_list_voices(self, mock_settings):
        """Test list_voices() returns a list."""
        synthesizer = TextToSpeechSynthesizer(settings=mock_settings)
        voices = synthesizer.list_voices()
        assert isinstance(voices, list)


class TestSpeaker:
    """Tests for the Speaker class."""

    def test_init_default(self, mock_settings):
        """Test Speaker initialization with defaults."""
        speaker = Speaker(settings=mock_settings)
        assert speaker.sample_rate == 16000
        assert speaker.volume == 1.0
        assert not speaker.is_playing()

    def test_init_custom(self):
        """Test Speaker initialization with custom parameters."""
        speaker = Speaker(sample_rate=44100, volume=0.5)
        assert speaker.sample_rate == 44100
        assert speaker.volume == 0.5

    def test_play_not_started(self, mock_settings):
        """Test play() raises RuntimeError if not started."""
        speaker = Speaker(settings=mock_settings)
        with pytest.raises(RuntimeError, match="not started"):
            speaker.play(b"audio data")

    def test_play_empty_data(self, mock_settings):
        """Test play() raises ValueError for empty data."""
        speaker = Speaker(settings=mock_settings)
        speaker.start()
        with pytest.raises(ValueError, match="must not be empty"):
            speaker.play(b"")
        speaker.stop()

    def test_set_volume(self, mock_settings):
        """Test set_volume() method."""
        speaker = Speaker(settings=mock_settings)
        speaker.set_volume(0.7)
        assert speaker.volume == 0.7

    def test_set_volume_invalid(self, mock_settings):
        """Test set_volume() raises ValueError for invalid volume."""
        speaker = Speaker(settings=mock_settings)
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            speaker.set_volume(2.0)

    def test_context_manager(self, mock_settings):
        """Test Speaker as a context manager."""
        with Speaker(settings=mock_settings) as speaker:
            assert not speaker.is_playing()
        assert not speaker.is_playing()
