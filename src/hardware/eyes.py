"""
AI Teacher Robot — Eye Controller.

This module controls the eye servos on the robot. It provides methods for
blinking, looking left/right, and setting eye positions.

Usage:
    from src.hardware.eyes import EyeController

    eyes = EyeController()
    eyes.blink()
    eyes.look_left()
"""

from __future__ import annotations

from typing import Optional

from src.config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EyeController:
    """Controls the eye servos on the robot.

    This class manages two servos (left and right eye) and provides
    methods for common eye movements and animations.

    Attributes:
        left_gpio: GPIO pin for the left eye servo.
        right_gpio: GPIO pin for the right eye servo.
        pwm_frequency: PWM frequency for the servos.
    """

    # Servo angle constants (in degrees)
    EYE_CENTER: int = 90
    EYE_LEFT: int = 60
    EYE_RIGHT: int = 120
    EYE_CLOSED: int = 0
    EYE_OPEN: int = 90

    def __init__(
        self,
        left_gpio: int = 18,
        right_gpio: int = 19,
        pwm_frequency: int = 50,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the EyeController.

        Args:
            left_gpio: GPIO pin for the left eye servo.
            right_gpio: GPIO pin for the right eye servo.
            pwm_frequency: PWM frequency for the servos in Hz.
            settings: Optional Settings instance for configuration.
        """
        if settings is not None:
            left_gpio = settings.eye_left_gpio
            right_gpio = settings.eye_right_gpio
            pwm_frequency = settings.pwm_frequency

        self.left_gpio: int = left_gpio
        self.right_gpio: int = right_gpio
        self.pwm_frequency: int = pwm_frequency

        # PWM instances (initialized in initialize())
        self._left_pwm = None
        self._right_pwm = None
        self._initialized: bool = False

    def initialize(self) -> None:
        """Initialize the eye servos.

        Sets up the GPIO pins and starts PWM for both eye servos.

        Raises:
            RuntimeError: If GPIO setup fails.
        """
        logger.info("Initializing eye controller...")

        # TODO: Initialize PWM for eye servos
        #   import RPi.GPIO as GPIO
        #   GPIO.setup(self.left_gpio, GPIO.OUT)
        #   GPIO.setup(self.right_gpio, GPIO.OUT)
        #   self._left_pwm = GPIO.PWM(self.left_gpio, self.pwm_frequency)
        #   self._right_pwm = GPIO.PWM(self.right_gpio, self.pwm_frequency)
        #   self._left_pwm.start(0)
        #   self._right_pwm.start(0)

        self._initialized = True
        logger.info("Eye controller initialized.")

    def set_angle(self, left_angle: int, right_angle: int) -> None:
        """Set the angle of both eye servos.

        Args:
            left_angle: Angle for the left eye (0-180 degrees).
            right_angle: Angle for the right eye (0-180 degrees).

        Raises:
            RuntimeError: If the controller is not initialized.
            ValueError: If the angle is out of range.
        """
        if not self._initialized:
            raise RuntimeError("Eye controller is not initialized.")

        if not (0 <= left_angle <= 180):
            raise ValueError(f"Left eye angle must be 0-180, got {left_angle}.")
        if not (0 <= right_angle <= 180):
            raise ValueError(f"Right eye angle must be 0-180, got {right_angle}.")

        logger.debug(f"Setting eye angles: left={left_angle}, right={right_angle}")

        # TODO: Set the servo angles
        #   left_duty = self._angle_to_duty(left_angle)
        #   right_duty = self._angle_to_duty(right_angle)
        #   self._left_pwm.ChangeDutyCycle(left_duty)
        #   self._right_pwm.ChangeDutyCycle(right_duty)

    def blink(self, duration: float = 0.3) -> None:
        """Blink both eyes.

        Args:
            duration: Duration of the blink in seconds.
        """
        logger.info("Blinking eyes...")

        # TODO: Implement blink animation
        #   self.set_angle(self.EYE_CLOSED, self.EYE_CLOSED)
        #   time.sleep(duration)
        #   self.set_angle(self.EYE_OPEN, self.EYE_OPEN)

    def look_left(self) -> None:
        """Move both eyes to the left."""
        logger.info("Looking left...")
        self.set_angle(self.EYE_LEFT, self.EYE_LEFT)

    def look_right(self) -> None:
        """Move both eyes to the right."""
        logger.info("Looking right...")
        self.set_angle(self.EYE_RIGHT, self.EYE_RIGHT)

    def look_center(self) -> None:
        """Move both eyes to the center."""
        logger.info("Looking center...")
        self.set_angle(self.EYE_CENTER, self.EYE_CENTER)

    def look_around(self) -> None:
        """Move eyes around in a scanning pattern."""
        logger.info("Looking around...")

        # TODO: Implement scanning animation
        #   self.look_left()
        #   time.sleep(0.5)
        #   self.look_right()
        #   time.sleep(0.5)
        #   self.look_center()

    def _angle_to_duty(self, angle: int) -> float:
        """Convert a servo angle to a PWM duty cycle.

        Args:
            angle: The servo angle (0-180 degrees).

        Returns:
            The PWM duty cycle (typically 2.5 to 12.5 for 180-degree servos).
        """
        return 2.5 + (angle / 180.0) * 10.0

    def shutdown(self) -> None:
        """Clean up the eye controller."""
        logger.info("Shutting down eye controller...")

        # TODO: Stop PWM
        #   if self._left_pwm:
        #       self._left_pwm.stop()
        #   if self._right_pwm:
        #       self._right_pwm.stop()

        self._initialized = False
