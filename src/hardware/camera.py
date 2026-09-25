"""Camera capture helpers for real experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - import environment variance
    cv2 = None


@dataclass(frozen=True)
class CameraCaptureConfig:
    device_index: int = 0
    frame_width: int | None = None
    frame_height: int | None = None
    average_frames: int = 1
    warmup_frames: int = 2
    llm_image_format: str = "png"


@dataclass(frozen=True)
class CaptureArtifacts:
    raw_array_path: str | None
    llm_image_path: str
    metadata_path: str | None
    shape: tuple[int, ...]
    dtype: str


class HardwareCamera:
    """Capture experiment frames and persist both raw and LLM-ready artifacts."""

    LLM_IMAGE_WIDTH = 960
    LLM_IMAGE_HEIGHT = 540

    def __init__(
        self,
        config: CameraCaptureConfig,
        *,
        frame_provider: Callable[[], np.ndarray] | None = None,
    ) -> None:
        self.config = config
        self._frame_provider = frame_provider
        self._capture = None

    def open(self) -> None:
        if self._frame_provider is not None or self._capture is not None:
            return
        if cv2 is None:  # pragma: no cover - depends on environment
            raise RuntimeError("OpenCV is required for real camera capture.")
        capture = cv2.VideoCapture(self.config.device_index)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera device {self.config.device_index}.")
        if self.config.frame_width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config.frame_width))
        if self.config.frame_height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config.frame_height))
        self._capture = capture
        for _ in range(max(0, self.config.warmup_frames)):
            self._read_frame()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _read_frame(self) -> np.ndarray:
        if self._frame_provider is not None:
            frame = self._frame_provider()
            return np.asarray(frame)
        if self._capture is None:
            self.open()
        assert self._capture is not None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("Camera read failed.")
        return np.asarray(frame)

    def capture_array(self, average_frames: int | None = None) -> np.ndarray:
        frames = []
        count = average_frames if average_frames is not None else self.config.average_frames
        count = max(1, int(count))
        for _ in range(count):
            frame = self._read_frame()
            frames.append(frame.astype(np.float32))
        averaged = np.mean(frames, axis=0)
        return np.clip(averaged, 0, 255).astype(np.uint8)

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
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        frame = self.capture_array(average_frames=average_frames)

        raw_array_path = out_path / f"{stem}.npy"
        llm_image_path = out_path / f"{stem}.{self.config.llm_image_format.lower()}"
        metadata_path = out_path / f"{stem}.json"
        llm_frame = self._prepare_llm_image(frame)

        if save_raw_array:
            np.save(raw_array_path, frame)
        self._write_image(llm_image_path, llm_frame)

        if save_metadata:
            metadata = {
                "raw_array_path": str(raw_array_path) if save_raw_array else None,
                "llm_image_path": str(llm_image_path),
                "shape": list(frame.shape),
                "dtype": str(frame.dtype),
                "llm_image_shape": list(llm_frame.shape),
                "average_frames": average_frames if average_frames is not None else self.config.average_frames,
            }
            if extra_metadata:
                metadata.update(extra_metadata)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return CaptureArtifacts(
            raw_array_path=str(raw_array_path) if save_raw_array else None,
            llm_image_path=str(llm_image_path),
            metadata_path=str(metadata_path) if save_metadata else None,
            shape=tuple(frame.shape),
            dtype=str(frame.dtype),
        )

    def _prepare_llm_image(self, frame: np.ndarray) -> np.ndarray:
        if cv2 is None:  # pragma: no cover - depends on environment
            raise RuntimeError("OpenCV is required for image export.")
        working = frame
        if working.ndim == 3 and working.shape[2] >= 3:
            working = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        elif working.ndim == 3 and working.shape[2] == 1:
            working = working[:, :, 0]
        resized = cv2.resize(
            working,
            (self.LLM_IMAGE_WIDTH, self.LLM_IMAGE_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        return np.asarray(resized, dtype=np.uint8)

    def _write_image(self, path: Path, frame: np.ndarray) -> None:
        if cv2 is None:  # pragma: no cover - depends on environment
            raise RuntimeError("OpenCV is required for image export.")
        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Failed to write image: {path}")
