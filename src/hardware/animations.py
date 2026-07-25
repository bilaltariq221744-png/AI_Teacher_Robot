"""
AI Teacher Robot — Animation Controller.

This module provides predefined animation sequences for the robot's face
(eyes and mouth). Animations include idle, listening, speaking, and
error states.

Usage:
    from src.hardware.animations import AnimationController

    animations = AnimationController(eye_controller, mouth_controller)
    animations.idle()
    animations.speak()
"""

from __future__ import annotations

import time
from typing import Optional

from src.config.settings import Settings
from src.hardware.eyes import EyeController
from src.hardware.mouth import MouthController
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnimationController:
    """Controls predefined animation sequences for the robot.

    This class coordinates the eye and mouth controllers to produce
    meaningful animations that correspond to the robot's state
    (idle, listening, speaking, error).

    Attributes:
        eye_controller: The eye controller instance.
        mouth_controller: The mouth controller instance.
        animation_speed: Speed multiplier for animations.
    """

    def __init__(
        self,
        eye_controller: Optional[EyeController] = None,
        mouth_controller: Optional[MouthController] = None,
        animation_speed: float = 1.0,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the AnimationController.

        Args:
            eye_controller: The eye controller instance.
            mouth_controller: The mouth controller instance.
            animation_speed: Speed multiplier for animations (1.0 = normal).
            settings: Optional Settings instance for configuration.
        """
        if settings is not None:
            animation_speed = settings.animation_speed

        self.eye_controller: Optional[EyeController] = eye_controller
        self.mouth_controller: Optional[MouthController] = mouth_controller
        self.animation_speed: float = animation_speed

    def idle(self, duration: float = 3.0) -> None:
        """Perform the idle animation.

        The robot slowly blinks and looks around occasionally.

        Args:
            duration: Duration of the idle animation in seconds.
        """
        logger.info(f"Playing idle animation (duration={duration}s)...")

        # TODO: Implement idle animation
        #   end_time = time.time() + duration
        #   while time.time() < end_time:
        #       self.eye_controller.blink()
        #       time.sleep(2.0 / self.animation_speed)
        #       self.eye_controller.look_around()
        #       time.sleep(1.0 / self.animation_speed)

    def listening(self, duration: float = 2.0) -> None:
        """Perform the listening animation.

        The robot's eyes move side to side, indicating it is listening.

        Args:
            duration: Duration of the listening animation in seconds.
        """
        logger.info(f"Playing listening animation (duration={duration}s)...")

        # TODO: Implement listening animation
        #   end_time = time.time() + duration
        #   while time.time() < end_time:
        #       self.eye_controller.look_left()
        #       time.sleep(0.5 / self.animation_speed)
        #       self.eye_controller.look_right()
        #       time.sleep(0.5 / self.animation_speed)
        #   self.eye_controller.look_center()

    def speaking(self, duration: float = 2.0) -> None:
        """Perform the speaking animation.

        The robot's mouth opens and closes in sync with speech.

        Args:
            duration: Duration of the speaking animation in seconds.
        """
        logger.info(f"Playing speaking animation (duration={duration}s)...")

        # TODO: Implement speaking animation
        #   if self.mouth_controller:
        #       self.mouth_controller.speak(duration)

    def thinking(self, duration: float = 2.0) -> None:
        """Perform the thinking animation.

        The robot's eyes move up and down, indicating it is thinking.

        Args:
            duration: Duration of the thinking animation in seconds.
        """
        logger.info(f"Playing thinking animation (duration={duration}s)...")

        # TODO: Implement thinking animation
        #   end_time = time.time() + duration
        #   while time.time() < end_time:
        #       self.eye_controller.set_angle(120, 120)
        #       time.sleep(0.3 / self.animation_speed)
        #       self.eye_controller.set_angle(60, 60)
        #       time.sleep(0.3 / self.animation_speed)
        #   self.eye_controller.look_center()

    def error(self, duration: float = 1.0) -> None:
        """Perform the error animation.

        The robot's eyes blink rapidly, indicating an error.

        Args:
            duration: Duration of the error animation in seconds.
        """
        logger.info(f"Playing error animation (duration={duration}s)...")

        # TODO: Implement error animation
        #   end_time = time.time() + duration
        #   while time.time() < end_time:
        #       self.eye_controller.blink(duration=0.1)
        #       time.sleep(0.1 / self.animation_speed)

    def greet(self) -> None:
        """Perform the greeting animation.

        The robot waves its eyes and opens its mouth briefly.
        """
        logger.info("Playing greeting animation...")

        # TODO: Implement greeting animation
        #   self.eye_controller.look_left()
        #   time.sleep(0.3 / self.animation_speed)
        #   self.eye_controller.look_right()
        #   time.sleep(0.3 / self.animation_speed)
        #   self.eye_controller.look_center()
        #   if self.mouth_controller:
        #       self.mouth_controller.open()
        #       time.sleep(0.5 / self.animation_speed)
        #       self.mouth_controller.close()

    def stop(self) -> None:
        """Stop all animations and reset to neutral position."""
        logger.info("Stopping all animations...")

        # TODO: Reset to neutral position
        #   if self.eye_controller:
        #       self.eye_controller.look_center()
        #   if self.mouth_controller:
        #       self.mouth_controller.close()
