"""
AI Teacher Robot — Text-to-Speech (TTS) Package.

This package handles converting text into speech and playing it through
a speaker. It includes:
    - Synthesizer: Converts text to audio samples using a TTS engine.
    - Speaker: Plays audio samples through the connected speaker.

Usage:
    from src.tts.synthesizer import TextToSpeechSynthesizer
    from src.tts.speaker import Speaker

    synthesizer = TextToSpeechSynthesizer(language="en")
    audio = synthesizer.synthesize("Hello, how are you?")
    speaker = Speaker()
    speaker.play(audio)
"""

from src.tts.speaker import Speaker
from src.tts.synthesizer import TextToSpeechSynthesizer

__all__ = ["TextToSpeechSynthesizer", "Speaker"]
