"""
AI Teacher Robot — Hardware Package.

This package controls the physical robot features (eyes, mouth) via GPIO
and PWM on the Raspberry Pi. It includes:
    - Controller: Main hardware controller that manages all servos.
    - Eyes: Controls the eye servos (blinking, looking around).
    - Mouth: Controls the mouth servo to animate during speech.
    - Animations: Predefined animation sequences (idle, listening, speaking).

Usage:
    from src.hardware.controller import HardwareController

    controller = HardwareController()
    controller.initialize()
    controller.animations.idle()
    controller.shutdown()
"""

from src.hardware.animations import AnimationController
from src.hardware.controller import HardwareController
from src.hardware.eyes import EyeController
from src.hardware.mouth import MouthController

__all__ = [
    "HardwareController",
    "EyeController",
    "MouthController",
    "AnimationController",
]
