"""
AI Teacher Robot — Audio Listener.

This module captures audio from the microphone and provides it as raw audio
data for speech-to-text processing.

The listener uses PyAudio (or a similar library) to capture audio in real-time.
It supports configurable sample rates, chunk sizes, and audio device selection.

Usage:
    from src.speech.listener import AudioListener

    listener = AudioListener(sample_rate=16000)
    audio_data = listener.listen(timeout=5)
"""

from __future__ import annotations

from typing import Optional

from src.config.settings import Settings
from src.utils.constants import DEFAULT_CHUNK_SIZE, DEFAULT_SAMPLE_RATE
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioListener:
    """Captures audio from the microphone.

    This class wraps a PyAudio stream to capture audio input. It provides
    methods to start/stop listening and to capture a single utterance.

    Attributes:
        sample_rate: The audio sample rate in Hz.
        chunk_size: The number of audio frames per buffer.
        channels: The number of audio channels (1 = mono).
        device_index: The index of the audio input device (None = default).
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        channels: int = 1,
        device_index: Optional[int] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the AudioListener.

        Args:
            sample_rate: The audio sample rate in Hz.
            chunk_size: The number of audio frames per buffer.
            channels: The number of audio channels (1 = mono).
            device_index: The index of the audio input device.
                If None, the default device is used.
            settings: Optional Settings instance for configuration.
        """
        if settings is not None:
            sample_rate = settings.sample_rate
            chunk_size = settings.chunk_size
            channels = settings.audio_channels
            device_index = settings.audio_device_index

        self.sample_rate: int = sample_rate
        self.chunk_size: int = chunk_size
        self.channels: int = channels
        self.device_index: Optional[int] = device_index

        # PyAudio instance (initialized lazily)
        self._pyaudio = None
        self._stream = None
        self._is_listening: bool = False

    def start(self) -> None:
        """Start the audio input stream.

        Initializes PyAudio and opens an audio stream for recording.

        Raises:
            RuntimeError: If the audio stream cannot be opened.
        """
        logger.info("Starting audio listener...")

        # TODO: Initialize PyAudio
        #   import pyaudio
        #   self._pyaudio = pyaudio.PyAudio()
        #   self._stream = self._pyaudio.open(
        #       format=pyaudio.paInt16,
        #       channels=self.channels,
        #       rate=self.sample_rate,
        #       input=True,
        #       frames_per_buffer=self.chunk_size,
        #       input_device_index=self.device_index,
        #   )

        self._is_listening = True
        logger.info("Audio listener started.")

    def stop(self) -> None:
        """Stop the audio input stream and release resources."""
        logger.info("Stopping audio listener...")

        # TODO: Close the audio stream and terminate PyAudio
        #   if self._stream is not None:
        #       self._stream.stop_stream()
        #       self._stream.close()
        #   if self._pyaudio is not None:
        #       self._pyaudio.terminate()

        self._is_listening = False
        self._stream = None
        self._pyaudio = None
        logger.info("Audio listener stopped.")

    def listen(self, timeout: float = 5.0) -> bytes:
        """Capture audio from the microphone.

        Records audio until speech is detected or the timeout is reached.
        Uses voice activity detection (VAD) to determine when speech starts
        and ends.

        Args:
            timeout: Maximum time to wait for speech (in seconds).

        Returns:
            Raw audio data as bytes (16-bit PCM, mono).

        Raises:
            RuntimeError: If the listener is not started or audio capture fails.
        """
        if not self._is_listening:
            raise RuntimeError(
                "Audio listener is not started. Call start() first."
            )

        logger.info(f"Listening for audio (timeout={timeout}s)...")

        # TODO: Implement audio capture with VAD
        #   frames = []
        #   # Wait for speech to start
        #   # Record until speech ends or timeout
        #   # Return concatenated audio frames

        # Placeholder: return empty bytes
        return b""

    def is_listening(self) -> bool:
        """Check if the listener is currently active.

        Returns:
            True if the listener is active, False otherwise.
        """
        return self._is_listening

    def __enter__(self) -> "AudioListener":
        """Context manager entry — starts the listener."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit — stops the listener."""
        self.stop()
