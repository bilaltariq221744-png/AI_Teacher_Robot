"""
AI Teacher Robot — Hardware Controller.

This module is the main controller for all hardware components (servos,
GPIO). It initializes the GPIO pins, manages the eye and mouth controllers,
and provides a unified interface for robot animations.

Usage:
    from src.hardware.controller import HardwareController

    controller = HardwareController()
    controller.initialize()
    controller.animations.speak()
    controller.shutdown()
"""

from __future__ import annotations

from typing import Optional

from src.config.settings import Settings
from src.hardware.animations import AnimationController
from src.hardware.eyes import EyeController
from src.hardware.mouth import MouthController
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HardwareController:
    """Main controller for all robot hardware.

    This class initializes the GPIO pins, creates the eye and mouth
    controllers, and provides access to the animation controller.

    Attributes:
        enabled: Whether hardware control is enabled.
        eye_controller: Controller for the eye servos.
        mouth_controller: Controller for the mouth servo.
        animations: Animation controller for predefined sequences.
    """

    def __init__(
        self,
        enabled: bool = True,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the HardwareController.

        Args:
            enabled: Whether hardware control is enabled. If False, all
                hardware operations are no-ops (useful for testing).
            settings: Optional Settings instance for configuration.
        """
        if settings is not None:
            enabled = settings.hardware_enabled

        self.enabled: bool = enabled
        self._settings: Optional[Settings] = settings
        self._initialized: bool = False

        # Sub-controllers (initialized in initialize())
        self.eye_controller: Optional[EyeController] = None
        self.mouth_controller: Optional[MouthController] = None
        self.animations: Optional[AnimationController] = None

        # GPIO library (initialized lazily)
        self._gpio = None

    def initialize(self) -> None:
        """Initialize the hardware controller.

        Sets up the GPIO pins and creates the eye, mouth, and animation
        controllers.

        Raises:
            RuntimeError: If hardware is not enabled or GPIO setup fails.
        """
        if not self.enabled:
            logger.info("Hardware control is disabled. Skipping initialization.")
            self._initialized = True
            return

        logger.info("Initializing hardware controller...")

        # TODO: Initialize GPIO
        #   import RPi.GPIO as GPIO
        #   self._gpio = GPIO
        #   GPIO.setmode(GPIO.BCM)
        #   GPIO.setup(self._settings.eye_left_gpio, GPIO.OUT)
        #   GPIO.setup(self._settings.eye_right_gpio, GPIO.OUT)
        #   GPIO.setup(self._settings.mouth_gpio, GPIO.OUT)

        # Initialize sub-controllers
        self.eye_controller = EyeController(settings=self._settings)
        self.mouth_controller = MouthController(settings=self._settings)
        self.animations = AnimationController(
            eye_controller=self.eye_controller,
            mouth_controller=self.mouth_controller,
        )

        self._initialized = True
        logger.info("Hardware controller initialized successfully.")

    def shutdown(self) -> None:
        """Shut down the hardware controller.

        Cleans up GPIO pins and releases all resources.
        """
        logger.info("Shutting down hardware controller...")

        # TODO: Clean up GPIO
        #   if self._gpio:
        #       self._gpio.cleanup()

        self._initialized = False
        logger.info("Hardware controller shut down successfully.")

    def is_initialized(self) -> bool:
        """Check if the hardware controller is initialized.

        Returns:
            True if initialized, False otherwise.
        """
        return self._initialized

    def __enter__(self) -> "HardwareController":
        """Context manager entry — initializes the controller."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit — shuts down the controller."""
        self.shutdown()
