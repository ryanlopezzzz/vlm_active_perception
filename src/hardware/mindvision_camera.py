"""MindVision camera adapter for real two-mirror cavity experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
from typing import Any

import numpy as np

from src.hardware.camera import CameraCaptureConfig, CaptureArtifacts, HardwareCamera

try:
    from src.hardware.vendor import mvsdk as _mvsdk
except Exception:  # pragma: no cover - depends on vendor SDK installation
    _mvsdk = None


@dataclass(frozen=True)
class MindVisionCameraConfig:
    camera_index: int = 0
    exposure_ms: float = 100.0
    analog_gain: int = 1
    average_frames: int = 1
    warmup_frames: int = 2
    frame_timeout_ms: int = 1000
    frame_retries: int = 3
    llm_image_format: str = "png"


class MindVisionCamera(HardwareCamera):
    """Capture MindVision SDK frames using the shared artifact pipeline."""

    def __init__(self, config: MindVisionCameraConfig, *, sdk: Any | None = None) -> None:
        super().__init__(
            CameraCaptureConfig(
                device_index=config.camera_index,
                average_frames=config.average_frames,
                warmup_frames=0,
                llm_image_format=config.llm_image_format,
            )
        )
        self.mindvision_config = config
        self._sdk = sdk if sdk is not None else _mvsdk
        self._camera_handle = 0
        self._frame_buffer = 0
        self._running = False
        self._mono_camera = False

    def open(self) -> None:
        if self._running:
            return
        if self._sdk is None:
            raise RuntimeError(
                "MindVision camera capture requires the native MindVision SDK and MVCAMSDK_X64 DLL."
            )
        devices = self._sdk.CameraEnumerateDevice()
        if not devices:
            raise RuntimeError("No MindVision cameras found. Check USB power and close DemoViewer.")
        index = self.mindvision_config.camera_index
        if index < 0 or index >= len(devices):
            raise RuntimeError(
                f"MindVision camera index {index} is unavailable; found {len(devices)} camera(s)."
            )

        try:
            self._camera_handle = self._sdk.CameraInit(devices[index], -1, -1)
            capability = self._sdk.CameraGetCapability(self._camera_handle)
            self._mono_camera = bool(capability.sIspCapacity.bMonoSensor)
            media_type = (
                self._sdk.CAMERA_MEDIA_TYPE_MONO8
                if self._mono_camera
                else self._sdk.CAMERA_MEDIA_TYPE_BGR8
            )
            self._sdk.CameraSetIspOutFormat(self._camera_handle, media_type)
            frame_buffer_size = (
                capability.sResolutionRange.iWidthMax
                * capability.sResolutionRange.iHeightMax
                * (1 if self._mono_camera else 3)
            )
            self._frame_buffer = self._sdk.CameraAlignMalloc(frame_buffer_size, 16)
            self._sdk.CameraSetAeState(self._camera_handle, 0)
            self._sdk.CameraSetExposureTime(
                self._camera_handle,
                int(self.mindvision_config.exposure_ms * 1000),
            )
            self._sdk.CameraSetAnalogGain(
                self._camera_handle,
                int(self.mindvision_config.analog_gain),
            )
            self._sdk.CameraSetTriggerMode(self._camera_handle, 0)
            self._sdk.CameraPlay(self._camera_handle)
            self._running = True
            for _ in range(self.mindvision_config.warmup_frames):
                self._read_frame()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._sdk is None:
            return
        if self._running and self._camera_handle:
            try:
                self._sdk.CameraPause(self._camera_handle)
            finally:
                self._running = False
        if self._camera_handle:
            try:
                self._sdk.CameraUnInit(self._camera_handle)
            finally:
                self._camera_handle = 0
        if self._frame_buffer:
            try:
                self._sdk.CameraAlignFree(self._frame_buffer)
            finally:
                self._frame_buffer = 0

    def _read_frame(self) -> np.ndarray:
        if not self._running:
            self.open()
        last_error: Exception | None = None
        for _ in range(self.mindvision_config.frame_retries):
            raw_data = None
            try:
                raw_data, frame_head = self._sdk.CameraGetImageBuffer(
                    self._camera_handle,
                    self.mindvision_config.frame_timeout_ms,
                )
                self._sdk.CameraImageProcess(
                    self._camera_handle,
                    raw_data,
                    self._frame_buffer,
                    frame_head,
                )
                if platform.system() == "Windows":
                    self._sdk.CameraFlipFrameBuffer(self._frame_buffer, frame_head, 1)
                frame_data = (self._sdk.c_ubyte * frame_head.uBytes).from_address(
                    self._frame_buffer
                )
                channels = (
                    1
                    if frame_head.uiMediaType == self._sdk.CAMERA_MEDIA_TYPE_MONO8
                    else 3
                )
                return np.frombuffer(frame_data, dtype=np.uint8).reshape(
                    frame_head.iHeight,
                    frame_head.iWidth,
                    channels,
                ).copy()
            except Exception as exc:
                last_error = exc
            finally:
                if raw_data is not None:
                    self._sdk.CameraReleaseImageBuffer(self._camera_handle, raw_data)
        raise RuntimeError(
            f"MindVision camera failed to return a frame after "
            f"{self.mindvision_config.frame_retries} attempt(s)."
        ) from last_error

    def save_capture(
        self,
        out_dir: str | Path,
        stem: str,
        *,
        average_frames: int | None = None,
        extra_metadata: dict[str, object] | None = None,
        save_raw_array: bool = True,
        save_metadata: bool = True,
    ) -> CaptureArtifacts:
        metadata = {
            "camera_backend": "mindvision",
            "camera_index": self.mindvision_config.camera_index,
            "exposure_ms": self.mindvision_config.exposure_ms,
            "analog_gain": self.mindvision_config.analog_gain,
            **(extra_metadata or {}),
        }
        return super().save_capture(
            out_dir,
            stem,
            average_frames=average_frames,
            extra_metadata=metadata,
            save_raw_array=save_raw_array,
            save_metadata=save_metadata,
        )
