"""
AI Teacher Robot — Application Entry Point.

This module is the main orchestrator for the AI Teacher Robot. It coordinates
the end-to-end flow:

    1. Initialize all modules (config, logger, speech, LLM, TTS, hardware).
    2. Enter the main listening loop.
    3. Capture audio -> STT -> detect language -> send to LLM -> TTS -> speak + animate.
    4. Handle errors and log all events.

Usage:
    python -m src.main
"""

from __future__ import annotations

import sys
from typing import Optional

from src.config.settings import Settings
from src.utils.constants import APP_NAME, APP_VERSION
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AITeacherRobot:
    """Main orchestrator for the AI Teacher Robot.

    This class initializes all subsystems and coordinates the conversation
    loop between the student and the robot.

    Attributes:
        settings: Application configuration settings.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize the AI Teacher Robot.

        Args:
            settings: Optional pre-configured Settings instance. If not
                provided, a default Settings instance will be created.
        """
        self.settings: Settings = settings or Settings()
        self._initialized: bool = False

    def initialize(self) -> None:
        """Initialize all subsystems.

        This method initializes the speech, LLM, TTS, and hardware modules
        in the correct order. It should be called once before starting the
        main loop.

        Raises:
            RuntimeError: If initialization fails for any subsystem.
        """
        logger.info(f"Initializing {APP_NAME} v{APP_VERSION}...")

        # TODO: Initialize config module (already done via Settings)
        # TODO: Initialize speech module (listener, STT, language detector)
        # TODO: Initialize LLM module (client, prompt manager, response handler)
        # TODO: Initialize TTS module (synthesizer, speaker)
        # TODO: Initialize hardware module (controller, eyes, mouth, animations)

        self._initialized = True
        logger.info(f"{APP_NAME} initialized successfully.")

    def run(self) -> None:
        """Start the main conversation loop.

        This method runs the main event loop that:
            1. Listens for student voice input.
            2. Converts speech to text.
            3. Detects the language (English or Urdu).
            4. Sends the text to the LLM.
            5. Converts the response to speech.
            6. Speaks the response and animates the robot.

        Raises:
            RuntimeError: If the robot has not been initialized.
        """
        if not self._initialized:
            raise RuntimeError(
                "Robot must be initialized before running. Call initialize() first."
            )

        logger.info("Starting AI Teacher Robot conversation loop...")

        # TODO: Implement the main conversation loop
        #   while True:
        #       audio = self.speech_listener.listen()
        #       text = self.stt.transcribe(audio)
        #       language = self.language_detector.detect(text)
        #       prompt = self.prompt_manager.build_prompt(text, language)
        #       response = self.llm_client.send(prompt)
        #       processed = self.response_handler.process(response)
        #       audio_output = self.tts_synthesizer.synthesize(processed, language)
        #       self.tts_speaker.play(audio_output)
        #       self.animations.speak()

        logger.info("AI Teacher Robot conversation loop ended.")

    def shutdown(self) -> None:
        """Clean up resources and shut down all subsystems.

        This method should be called when the application is shutting down
        to ensure all resources (GPIO, audio, etc.) are properly released.
        """
        logger.info("Shutting down AI Teacher Robot...")

        # TODO: Shut down hardware module (release GPIO)
        # TODO: Shut down TTS module (release audio resources)
        # TODO: Shut down speech module (release microphone)
        # TODO: Shut down LLM module (close connections)

        self._initialized = False
        logger.info("AI Teacher Robot shut down successfully.")


def main() -> None:
    """Application entry point.

    Creates an AITeacherRobot instance, initializes it, runs the main loop,
    and handles graceful shutdown on keyboard interrupt or error.
    """
    robot: Optional[AITeacherRobot] = None

    try:
        robot = AITeacherRobot()
        robot.initialize()
        robot.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    except Exception as exc:
        logger.error(f"An error occurred: {exc}", exc_info=True)
        sys.exit(1)
    finally:
        if robot is not None:
            robot.shutdown()


if __name__ == "__main__":
    main()
