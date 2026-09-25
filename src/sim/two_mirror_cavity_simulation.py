"""Lab-derived image-space simulation of the two-mirror cavity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy import ndimage


AXIS_NAMES = ("x", "y")
DEFAULT_PROFILE_PATH = Path(__file__).resolve().parent / "assets" / "two_mirror_cavity_profiles.npz"


@dataclass(frozen=True)
class TwoMirrorCavityConfig:
    profile_path: str | Path = DEFAULT_PROFILE_PATH
    x_min_steps: int = -8000
    x_max_steps: int = 8000
    y_min_steps: int = -5000
    y_max_steps: int = 5000
    x_image_pixels_per_step: float = -0.00020900476190476185 * 960.0 / 2.0
    y_image_pixels_per_step: float = -0.0003231375000000002 * 540.0 / 2.0
    sensor_saturation: float = 1.0
    normalize_each_frame: bool = False
    interpolation_order: int = 1


@dataclass(frozen=True)
class TwoMirrorCavityBeamCenters:
    main_x_px: float
    main_y_px: float
    secondary_x_px: float
    secondary_y_px: float

    @property
    def separation_px(self) -> float:
        return float(np.hypot(self.secondary_x_px - self.main_x_px, self.secondary_y_px - self.main_y_px))


class TwoMirrorCavitySimulation:
    """Fixed main beam plus a two-axis, lab-scaled movable secondary beam."""

    def __init__(self, config: TwoMirrorCavityConfig | None = None) -> None:
        self.config = config or TwoMirrorCavityConfig()
        self._validate_config()
        profile_path = Path(self.config.profile_path)
        if not profile_path.is_file():
            raise FileNotFoundError(
                f"Two-mirror profile asset not found: {profile_path}. "
                "Run src/analysis/extract_two_mirror_cavity_profiles.py to generate it."
            )
        with np.load(profile_path) as profiles:
            self._main_intensity = np.asarray(profiles["main_intensity"], dtype=np.float64)
            self._secondary_aligned = np.asarray(profiles["secondary_intensity"], dtype=np.float64)
            main_center = np.asarray(profiles["main_center_px"], dtype=np.float64)
        if self._main_intensity.shape != self._secondary_aligned.shape:
            raise ValueError("Main and secondary profile shapes must match.")
        if self._main_intensity.ndim != 2:
            raise ValueError("Two-mirror intensity profiles must be 2D arrays.")
        if not np.all(np.isfinite(self._main_intensity)) or not np.all(np.isfinite(self._secondary_aligned)):
            raise ValueError("Two-mirror intensity profiles must contain only finite values.")
        if np.any(self._main_intensity < 0) or np.any(self._secondary_aligned < 0):
            raise ValueError("Two-mirror intensity profiles must be nonnegative.")
        if main_center.shape != (2,):
            raise ValueError("main_center_px must contain [x, y].")
        self._main_center_x_px = float(main_center[0])
        self._main_center_y_px = float(main_center[1])

    @property
    def frame_height_px(self) -> int:
        return int(self._main_intensity.shape[0])

    @property
    def frame_width_px(self) -> int:
        return int(self._main_intensity.shape[1])

    def zero_command(self) -> dict[str, int]:
        return {"x": 0, "y": 0}

    def clip_command(self, command: Mapping[str, int | float]) -> dict[str, int]:
        x = int(round(float(command.get("x", 0))))
        y = int(round(float(command.get("y", 0))))
        return {
            "x": int(np.clip(x, self.config.x_min_steps, self.config.x_max_steps)),
            "y": int(np.clip(y, self.config.y_min_steps, self.config.y_max_steps)),
        }

    def beam_centers(
        self,
        command: Mapping[str, int | float],
        *,
        clip_command: bool = True,
    ) -> TwoMirrorCavityBeamCenters:
        resolved = self._resolve_command(command, clip_command=clip_command)
        return TwoMirrorCavityBeamCenters(
            main_x_px=self._main_center_x_px,
            main_y_px=self._main_center_y_px,
            secondary_x_px=(
                self._main_center_x_px + self.config.x_image_pixels_per_step * resolved["x"]
            ),
            secondary_y_px=(
                self._main_center_y_px + self.config.y_image_pixels_per_step * resolved["y"]
            ),
        )

    def render_components(
        self,
        command: Mapping[str, int | float],
        *,
        clip_command: bool = True,
    ) -> dict[str, object]:
        resolved = self._resolve_command(command, clip_command=clip_command)
        centers = self.beam_centers(resolved, clip_command=False)
        secondary = ndimage.shift(
            self._secondary_aligned,
            shift=(
                centers.secondary_y_px - centers.main_y_px,
                centers.secondary_x_px - centers.main_x_px,
            ),
            order=self.config.interpolation_order,
            mode="constant",
            cval=0.0,
            prefilter=self.config.interpolation_order > 1,
        )
        secondary = np.maximum(secondary, 0.0)
        combined = self._main_intensity + secondary
        return {
            "command": resolved,
            "centers": centers,
            "main_intensity": self._main_intensity.copy(),
            "secondary_intensity": secondary,
            "combined_intensity": combined,
        }

    def render(
        self,
        command: Mapping[str, int | float],
        *,
        clip_command: bool = True,
    ) -> np.ndarray:
        image = np.asarray(
            self.render_components(command, clip_command=clip_command)["combined_intensity"],
            dtype=np.float64,
        )
        if self.config.normalize_each_frame:
            peak = float(np.max(image))
            if np.isfinite(peak) and peak > 0:
                image = image / peak
        return np.clip(image, 0.0, self.config.sensor_saturation)

    def _resolve_command(
        self,
        command: Mapping[str, int | float],
        *,
        clip_command: bool,
    ) -> dict[str, int]:
        if clip_command:
            return self.clip_command(command)
        return {
            axis: int(round(float(command.get(axis, 0))))
            for axis in AXIS_NAMES
        }

    def _validate_config(self) -> None:
        if self.config.x_min_steps >= self.config.x_max_steps:
            raise ValueError("x_min_steps must be smaller than x_max_steps.")
        if self.config.y_min_steps >= self.config.y_max_steps:
            raise ValueError("y_min_steps must be smaller than y_max_steps.")
        if self.config.sensor_saturation <= 0:
            raise ValueError("sensor_saturation must be positive.")
        if self.config.interpolation_order not in range(0, 6):
            raise ValueError("interpolation_order must be between 0 and 5.")
