"""
AI Teacher Robot — Tests for the Hardware Module.

This module contains unit tests for the hardware control components:
    - HardwareController
    - EyeController
    - MouthController
    - AnimationController

Usage:
    pytest tests/test_hardware.py -v
"""

from __future__ import annotations

import pytest

from src.hardware.animations import AnimationController
from src.hardware.controller import HardwareController
from src.hardware.eyes import EyeController
from src.hardware.mouth import MouthController


class TestHardwareController:
    """Tests for the HardwareController class."""

    def test_init_disabled(self):
        """Test HardwareController initialization with hardware disabled."""
        controller = HardwareController(enabled=False)
        assert not controller.enabled
        assert not controller.is_initialized()

    def test_init_enabled(self, mock_hardware_enabled_settings):
        """Test HardwareController initialization with hardware enabled."""
        controller = HardwareController(
            enabled=True, settings=mock_hardware_enabled_settings
        )
        assert controller.enabled
        assert not controller.is_initialized()

    def test_initialize_disabled(self):
        """Test initialize() when hardware is disabled."""
        controller = HardwareController(enabled=False)
        controller.initialize()
        assert controller.is_initialized()

    def test_initialize_enabled(self, mock_hardware_enabled_settings):
        """Test initialize() when hardware is enabled."""
        controller = HardwareController(
            enabled=True, settings=mock_hardware_enabled_settings
        )
        controller.initialize()
        assert controller.is_initialized()
        assert controller.eye_controller is not None
        assert controller.mouth_controller is not None
        assert controller.animations is not None

    def test_shutdown(self, mock_hardware_enabled_settings):
        """Test shutdown() method."""
        controller = HardwareController(
            enabled=True, settings=mock_hardware_enabled_settings
        )
        controller.initialize()
        controller.shutdown()
        assert not controller.is_initialized()

    def test_context_manager(self, mock_hardware_enabled_settings):
        """Test HardwareController as a context manager."""
        with HardwareController(
            enabled=True, settings=mock_hardware_enabled_settings
        ) as controller:
            assert controller.is_initialized()
        assert not controller.is_initialized()


class TestEyeController:
    """Tests for the EyeController class."""

    def test_init_default(self, mock_settings):
        """Test EyeController initialization with defaults."""
        eyes = EyeController(settings=mock_settings)
        assert eyes.left_gpio == 18
        assert eyes.right_gpio == 19
        assert eyes.pwm_frequency == 50

    def test_init_custom(self):
        """Test EyeController initialization with custom parameters."""
        eyes = EyeController(left_gpio=20, right_gpio=21, pwm_frequency=60)
        assert eyes.left_gpio == 20
        assert eyes.right_gpio == 21
        assert eyes.pwm_frequency == 60

    def test_set_angle_not_initialized(self, mock_settings):
        """Test set_angle() raises RuntimeError if not initialized."""
        eyes = EyeController(settings=mock_settings)
        with pytest.raises(RuntimeError, match="not initialized"):
            eyes.set_angle(90, 90)

    def test_set_angle_invalid_range(self, mock_settings):
        """Test set_angle() raises ValueError for out-of-range angle."""
        eyes = EyeController(settings=mock_settings)
        eyes._initialized = True
        with pytest.raises(ValueError, match="must be 0-180"):
            eyes.set_angle(-1, 90)
        with pytest.raises(ValueError, match="must be 0-180"):
            eyes.set_angle(90, 181)

    def test_angle_to_duty(self, mock_settings):
        """Test _angle_to_duty() conversion."""
        eyes = EyeController(settings=mock_settings)
        assert eyes._angle_to_duty(0) == 2.5
        assert eyes._angle_to_duty(180) == 12.5
        assert eyes._angle_to_duty(90) == 7.5

    def test_constants(self, mock_settings):
        """Test that servo angle constants are defined."""
        eyes = EyeController(settings=mock_settings)
        assert eyes.EYE_CENTER == 90
        assert eyes.EYE_LEFT == 60
        assert eyes.EYE_RIGHT == 120
        assert eyes.EYE_CLOSED == 0
        assert eyes.EYE_OPEN == 90


class TestMouthController:
    """Tests for the MouthController class."""

    def test_init_default(self, mock_settings):
        """Test MouthController initialization with defaults."""
        mouth = MouthController(settings=mock_settings)
        assert mouth.gpio == 13
        assert mouth.pwm_frequency == 50

    def test_init_custom(self):
        """Test MouthController initialization with custom parameters."""
        mouth = MouthController(gpio=14, pwm_frequency=60)
        assert mouth.gpio == 14
        assert mouth.pwm_frequency == 60

    def test_set_angle_not_initialized(self, mock_settings):
        """Test set_angle() raises RuntimeError if not initialized."""
        mouth = MouthController(settings=mock_settings)
        with pytest.raises(RuntimeError, match="not initialized"):
            mouth.set_angle(90)

    def test_set_angle_invalid_range(self, mock_settings):
        """Test set_angle() raises ValueError for out-of-range angle."""
        mouth = MouthController(settings=mock_settings)
        mouth._initialized = True
        with pytest.raises(ValueError, match="must be 0-180"):
            mouth.set_angle(-1)
        with pytest.raises(ValueError, match="must be 0-180"):
            mouth.set_angle(181)

    def test_angle_to_duty(self, mock_settings):
        """Test _angle_to_duty() conversion."""
        mouth = MouthController(settings=mock_settings)
        assert mouth._angle_to_duty(0) == 2.5
        assert mouth._angle_to_duty(180) == 12.5

    def test_constants(self, mock_settings):
        """Test that servo angle constants are defined."""
        mouth = MouthController(settings=mock_settings)
        assert mouth.MOUTH_CLOSED == 0
        assert mouth.MOUTH_OPEN == 60
        assert mouth.MOUTH_NEUTRAL == 30


class TestAnimationController:
    """Tests for the AnimationController class."""

    def test_init_default(self):
        """Test AnimationController initialization with defaults."""
        animations = AnimationController()
        assert animations.eye_controller is None
        assert animations.mouth_controller is None
        assert animations.animation_speed == 1.0

    def test_init_custom(self, mock_settings):
        """Test AnimationController initialization with custom parameters."""
        eye_ctrl = EyeController(settings=mock_settings)
        mouth_ctrl = MouthController(settings=mock_settings)
        animations = AnimationController(
            eye_controller=eye_ctrl,
            mouth_controller=mouth_ctrl,
            animation_speed=2.0,
        )
        assert animations.eye_controller is eye_ctrl
        assert animations.mouth_controller is mouth_ctrl
        assert animations.animation_speed == 2.0
