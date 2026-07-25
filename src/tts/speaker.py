"""
AI Teacher Robot — Speaker (Audio Output).

This module plays audio data through the connected speaker. It wraps
PyAudio (or a similar library) to output audio samples.

Usage:
    from src.tts.speaker import Speaker

    speaker = Speaker()
    speaker.play(audio_data)
"""

from __future__ import annotations

from typing import Optional

from src.config.settings import Settings
from src.utils.constants import DEFAULT_SAMPLE_RATE
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Speaker:
    """Plays audio data through a speaker.

    This class wraps an audio output stream to play audio samples. It
    supports configurable sample rates and volume control.

    Attributes:
        sample_rate: The audio sample rate in Hz.
        volume: The playback volume (0.0 to 1.0).
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        volume: float = 1.0,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the Speaker.

        Args:
            sample_rate: The audio sample rate in Hz.
            volume: The playback volume (0.0 to 1.0).
            settings: Optional Settings instance for configuration.
        """
        if settings is not None:
            sample_rate = settings.sample_rate
            volume = settings.tts_volume

        self.sample_rate: int = sample_rate
        self.volume: float = volume

        # PyAudio instance (initialized lazily)
        self._pyaudio = None
        self._stream = None
        self._is_playing: bool = False

    def start(self) -> None:
        """Initialize the audio output stream.

        Raises:
            RuntimeError: If the audio output stream cannot be opened.
        """
        logger.info("Starting speaker...")

        # TODO: Initialize PyAudio output stream
        #   import pyaudio
        #   self._pyaudio = pyaudio.PyAudio()
        #   self._stream = self._pyaudio.open(
        #       format=pyaudio.paInt16,
        #       channels=1,
        #       rate=self.sample_rate,
        #       output=True,
        #   )

        # Placeholder: set a non-None stream for testing
        self._stream = True

        logger.info("Speaker started.")

    def stop(self) -> None:
        """Stop the audio output stream and release resources."""
        logger.info("Stopping speaker...")

        # TODO: Close the audio stream and terminate PyAudio
        #   if self._stream is not None:
        #       self._stream.stop_stream()
        #       self._stream.close()
        #   if self._pyaudio is not None:
        #       self._pyaudio.terminate()

        self._is_playing = False
        self._stream = None
        self._pyaudio = None
        logger.info("Speaker stopped.")

    def play(self, audio_data: bytes) -> None:
        """Play audio data through the speaker.

        Args:
            audio_data: Raw audio data as bytes (16-bit PCM, mono).

        Raises:
            RuntimeError: If the speaker is not started.
            ValueError: If the audio data is empty.
        """
        if not self._stream:
            raise RuntimeError(
                "Speaker is not started. Call start() first."
            )

        if not audio_data:
            raise ValueError("Audio data must not be empty.")

        logger.debug(f"Playing audio ({len(audio_data)} bytes)...")

        # TODO: Play the audio data
        #   self._stream.write(audio_data)

        self._is_playing = True
        logger.debug("Audio playback complete.")

    def play_file(self, file_path: str) -> None:
        """Play an audio file through the speaker.

        Args:
            file_path: Path to the audio file (WAV format).

        Raises:
            RuntimeError: If the speaker is not started.
            FileNotFoundError: If the file does not exist.
        """
        logger.info(f"Playing audio file: {file_path}")

        # TODO: Read and play the audio file
        #   import wave
        #   with wave.open(file_path, "rb") as wf:
        #       data = wf.readframes(wf.getnframes())
        #       self.play(data)

    def stop_playback(self) -> None:
        """Stop the current audio playback immediately."""
        logger.info("Stopping audio playback...")

        # TODO: Stop the current playback
        #   if self._stream:
        #       self._stream.stop_stream()

        self._is_playing = False

    def set_volume(self, volume: float) -> None:
        """Set the playback volume.

        Args:
            volume: Volume level (0.0 to 1.0).

        Raises:
            ValueError: If the volume is out of range.
        """
        if not 0.0 <= volume <= 1.0:
            raise ValueError("Volume must be between 0.0 and 1.0.")

        logger.info(f"Setting speaker volume to: {volume}")
        self.volume = volume

        # TODO: Apply volume to the audio stream
        #   This may require a volume control library or manual scaling.

    def is_playing(self) -> bool:
        """Check if audio is currently playing.

        Returns:
            True if audio is playing, False otherwise.
        """
        return self._is_playing

    def __enter__(self) -> "Speaker":
        """Context manager entry — starts the speaker."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit — stops the speaker."""
        self.stop()
