"""
AI Teacher Robot — Mouth Controller.

This module controls the mouth servo on the robot. It provides methods for
animating the mouth during speech (opening and closing).

Usage:
    from src.hardware.mouth import MouthController

    mouth = MouthController()
    mouth.open()
    mouth.close()
    mouth.speak()
"""

from __future__ import annotations

import time
from typing import Optional

from src.config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MouthController:
    """Controls the mouth servo on the robot.

    This class manages a single servo for the mouth and provides methods
    for opening, closing, and animating the mouth during speech.

    Attributes:
        gpio: GPIO pin for the mouth servo.
        pwm_frequency: PWM frequency for the servo.
    """

    # Servo angle constants (in degrees)
    MOUTH_CLOSED: int = 0
    MOUTH_OPEN: int = 60
    MOUTH_NEUTRAL: int = 30

    def __init__(
        self,
        gpio: int = 13,
        pwm_frequency: int = 50,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the MouthController.

        Args:
            gpio: GPIO pin for the mouth servo.
            pwm_frequency: PWM frequency for the servo in Hz.
            settings: Optional Settings instance for configuration.
        """
        if settings is not None:
            gpio = settings.mouth_gpio
            pwm_frequency = settings.pwm_frequency

        self.gpio: int = gpio
        self.pwm_frequency: int = pwm_frequency

        # PWM instance (initialized in initialize())
        self._pwm = None
        self._initialized: bool = False

    def initialize(self) -> None:
        """Initialize the mouth servo.

        Sets up the GPIO pin and starts PWM for the mouth servo.

        Raises:
            RuntimeError: If GPIO setup fails.
        """
        logger.info("Initializing mouth controller...")

        # TODO: Initialize PWM for mouth servo
        #   import RPi.GPIO as GPIO
        #   GPIO.setup(self.gpio, GPIO.OUT)
        #   self._pwm = GPIO.PWM(self.gpio, self.pwm_frequency)
        #   self._pwm.start(0)

        self._initialized = True
        logger.info("Mouth controller initialized.")

    def set_angle(self, angle: int) -> None:
        """Set the angle of the mouth servo.

        Args:
            angle: The servo angle (0-180 degrees).

        Raises:
            RuntimeError: If the controller is not initialized.
            ValueError: If the angle is out of range.
        """
        if not self._initialized:
            raise RuntimeError("Mouth controller is not initialized.")

        if not (0 <= angle <= 180):
            raise ValueError(f"Mouth angle must be 0-180, got {angle}.")

        logger.debug(f"Setting mouth angle: {angle}")

        # TODO: Set the servo angle
        #   duty = self._angle_to_duty(angle)
        #   self._pwm.ChangeDutyCycle(duty)

    def open(self) -> None:
        """Open the mouth."""
        logger.info("Opening mouth...")
        self.set_angle(self.MOUTH_OPEN)

    def close(self) -> None:
        """Close the mouth."""
        logger.info("Closing mouth...")
        self.set_angle(self.MOUTH_CLOSED)

    def speak(self, duration: float = 2.0) -> None:
        """Animate the mouth as if speaking.

        Opens and closes the mouth repeatedly for the specified duration,
        simulating speech.

        Args:
            duration: Duration of the speaking animation in seconds.
        """
        logger.info(f"Speaking animation (duration={duration}s)...")

        # TODO: Implement speaking animation
        #   end_time = time.time() + duration
        #   while time.time() < end_time:
        #       self.open()
        #       time.sleep(0.1)
        #       self.close()
        #       time.sleep(0.1)

    def _angle_to_duty(self, angle: int) -> float:
        """Convert a servo angle to a PWM duty cycle.

        Args:
            angle: The servo angle (0-180 degrees).

        Returns:
            The PWM duty cycle (typically 2.5 to 12.5 for 180-degree servos).
        """
        return 2.5 + (angle / 180.0) * 10.0

    def shutdown(self) -> None:
        """Clean up the mouth controller."""
        logger.info("Shutting down mouth controller...")

        # TODO: Stop PWM
        #   if self._pwm:
        #       self._pwm.stop()

        self._initialized = False
