"""
AI Teacher Robot — Speech Package.

This package handles all speech-related functionality:
    - Audio capture from the microphone (listener)
    - Speech-to-text conversion (stt)
    - Language detection (language_detector)

Usage:
    from src.speech.listener import AudioListener
    from src.speech.stt import SpeechToText
    from src.speech.language_detector import LanguageDetector
"""

from src.speech.language_detector import LanguageDetector
from src.speech.listener import AudioListener
from src.speech.stt import SpeechToText

__all__ = ["AudioListener", "SpeechToText", "LanguageDetector"]
