"""Hardware adapters for real-experiment workflows."""

from src.hardware.camera import CameraCaptureConfig, CaptureArtifacts, HardwareCamera
from src.hardware.mindvision_camera import MindVisionCamera, MindVisionCameraConfig
from src.hardware.motors import (
    AxisMotorConfig,
    BaselineState,
    MotionRecord,
    MotionSafetyLimitError,
    SessionMotionState,
    StepperMotorRig,
)

__all__ = [
    "AxisMotorConfig",
    "BaselineState",
    "CameraCaptureConfig",
    "CaptureArtifacts",
    "HardwareCamera",
    "MindVisionCamera",
    "MindVisionCameraConfig",
    "MotionRecord",
    "MotionSafetyLimitError",
    "SessionMotionState",
    "StepperMotorRig",
]
