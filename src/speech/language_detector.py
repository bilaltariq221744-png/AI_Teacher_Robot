"""
AI Teacher Robot — Language Detection.

This module detects the language of spoken input (English or Urdu) from
either transcribed text or raw audio features.

Usage:
    from src.speech.language_detector import LanguageDetector

    detector = LanguageDetector()
    language = detector.detect("What is the capital of Pakistan?")
    # Returns: "en"
"""

from __future__ import annotations

from typing import Optional

from src.config.settings import Settings
from src.utils.constants import LANGUAGE_ENGLISH, LANGUAGE_URDU, SUPPORTED_LANGUAGES
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LanguageDetector:
    """Detects the language of input text or audio.

    This class provides language detection for English and Urdu. It can
    operate on transcribed text (using character/script analysis) or on
    raw audio (using a statistical model).

    Attributes:
        supported_languages: List of supported language codes.
        default_language: The fallback language if detection fails.
    """

    # Unicode ranges for script detection
    # Arabic script (used for Urdu): U+0600 to U+06FF
    # Latin script (used for English): U+0000 to U+007F
    _ARABIC_SCRIPT_RANGES: list[tuple[int, int]] = [
        (0x0600, 0x06FF),  # Arabic
        (0x0750, 0x077F),  # Arabic Supplement
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    ]

    def __init__(
        self,
        supported_languages: Optional[list[str]] = None,
        default_language: str = LANGUAGE_ENGLISH,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the LanguageDetector.

        Args:
            supported_languages: List of supported language codes.
                If None, defaults to ["en", "ur"].
            default_language: The fallback language if detection fails.
            settings: Optional Settings instance for configuration.
        """
        if settings is not None:
            supported_languages = supported_languages or settings.supported_languages
            default_language = settings.default_language

        self.supported_languages: list[str] = (
            supported_languages or list(SUPPORTED_LANGUAGES)
        )
        self.default_language: str = default_language

        # TODO: Load a statistical language detection model
        #   (e.g., using langdetect or a custom model)
        self._model = None

    def detect(self, text: str) -> str:
        """Detect the language of the given text.

        Uses script analysis to determine whether the text is written in
        Latin script (English) or Arabic script (Urdu).

        Args:
            text: The input text to analyze.

        Returns:
            The detected language code (e.g., "en" or "ur").
            Returns the default language if detection is uncertain.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for language detection.")
            return self.default_language

        logger.debug(f"Detecting language for text: {text[:50]!r}...")

        # Count characters in each script
        arabic_count: int = 0
        latin_count: int = 0

        for char in text:
            if self._is_arabic_script(char):
                arabic_count += 1
            elif self._is_latin_script(char):
                latin_count += 1

        # Determine the language based on script counts
        if arabic_count > latin_count:
            language: str = LANGUAGE_URDU
        elif latin_count > arabic_count:
            language = LANGUAGE_ENGLISH
        else:
            # Equal or no recognizable characters — use default
            language = self.default_language

        logger.debug(
            f"Language detection: arabic={arabic_count}, "
            f"latin={latin_count}, result={language}"
        )

        return language

    def detect_from_audio(self, audio_data: bytes) -> str:
        """Detect the language from raw audio data.

        This method uses a statistical model to detect the language from
        audio features. It is less reliable than text-based detection.

        Args:
            audio_data: Raw audio data as bytes.

        Returns:
            The detected language code.
        """
        logger.debug("Detecting language from audio data...")

        # TODO: Implement audio-based language detection
        #   This could use a model like Vosk's language identification
        #   or a custom ML model.

        # Placeholder: fall back to text-based detection is not possible
        # without transcription, so return the default language.
        return self.default_language

    def _is_arabic_script(self, char: str) -> bool:
        """Check if a character belongs to the Arabic script.

        Args:
            char: A single character.

        Returns:
            True if the character is in the Arabic script range.
        """
        code_point: int = ord(char)
        for start, end in self._ARABIC_SCRIPT_RANGES:
            if start <= code_point <= end:
                return True
        return False

    def _is_latin_script(self, char: str) -> bool:
        """Check if a character belongs to the Latin script.

        Only matches Latin letters (A-Z, a-z), not digits or punctuation.

        Args:
            char: A single character.

        Returns:
            True if the character is a Latin letter.
        """
        return char.isalpha() and char.isascii()

    def is_supported(self, language: str) -> bool:
        """Check if a language is supported.

        Args:
            language: The language code to check.

        Returns:
            True if the language is supported, False otherwise.
        """
        return language in self.supported_languages
