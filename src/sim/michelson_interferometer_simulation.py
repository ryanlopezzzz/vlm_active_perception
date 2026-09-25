"""Primary Michelson interferometer simulation.

The model uses discrete motor steps and image pixels to render two coherent
Gaussian beams, making the control-to-image mapping direct and inspectable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np


AXIS_NAMES = (
    "mirror_1_axis_1",
    "mirror_1_axis_2",
    "mirror_2_axis_1",
    "mirror_2_axis_2",
)


@dataclass(frozen=True)
class MichelsonInterferometerConfig:
    frame_width_px: int = 960
    frame_height_px: int = 540
    steps_per_revolution: int = 4096
    max_rotations_from_center: float = 2.0
    beam_diameter_fraction_of_width: float = 0.25
    fringe_count_across_beam_diameter: float = 8.0
    first_beam_amplitude: float = 1.0
    second_beam_amplitude: float = 1.0
    normalize_each_frame: bool = False

    @property
    def center_x_px(self) -> float:
        return 0.5 * self.frame_width_px

    @property
    def center_y_px(self) -> float:
        return 0.5 * self.frame_height_px

    @property
    def beam_diameter_px(self) -> float:
        return self.frame_width_px * self.beam_diameter_fraction_of_width

    @property
    def beam_radius_px(self) -> float:
        """1/e^2 intensity radius."""
        return 0.5 * self.beam_diameter_px

    @property
    def gaussian_sigma_px(self) -> float:
        """Equivalent sigma for intensity exp(-r^2 / (2 sigma^2))."""
        return 0.5 * self.beam_radius_px

    @property
    def pixels_per_step(self) -> float:
        return (0.5 * self.frame_width_px) / self.steps_per_revolution

    @property
    def max_steps(self) -> int:
        return int(round(self.steps_per_revolution * self.max_rotations_from_center))

    @property
    def fringe_period_px(self) -> float:
        return self.beam_diameter_px / self.fringe_count_across_beam_diameter

    @property
    def fringe_radians_per_px(self) -> float:
        return 2.0 * np.pi / self.fringe_period_px

    def to_dict(self) -> dict[str, float | int | bool]:
        payload = asdict(self)
        payload.update(
            {
                "center_x_px": self.center_x_px,
                "center_y_px": self.center_y_px,
                "beam_diameter_px": self.beam_diameter_px,
                "beam_radius_px": self.beam_radius_px,
                "gaussian_sigma_px": self.gaussian_sigma_px,
                "pixels_per_step": self.pixels_per_step,
                "max_steps": self.max_steps,
                "fringe_period_px": self.fringe_period_px,
                "fringe_radians_per_px": self.fringe_radians_per_px,
            }
        )
        return payload


@dataclass(frozen=True)
class BeamCenters:
    first_x_px: float
    first_y_px: float
    second_x_px: float
    second_y_px: float

    @property
    def separation_px(self) -> float:
        return float(
            np.hypot(
                self.first_x_px - self.second_x_px,
                self.first_y_px - self.second_y_px,
            )
        )


class MichelsonInterferometerSimulation:
    """Coherent two-Gaussian Michelson image model with motor-step controls."""

    def __init__(self, config: MichelsonInterferometerConfig | None = None) -> None:
        self.config = config or MichelsonInterferometerConfig()
        if self.config.frame_width_px <= 0 or self.config.frame_height_px <= 0:
            raise ValueError("Frame dimensions must be positive.")
        if self.config.steps_per_revolution <= 0:
            raise ValueError("steps_per_revolution must be positive.")
        if self.config.beam_diameter_fraction_of_width <= 0:
            raise ValueError("beam_diameter_fraction_of_width must be positive.")
        if self.config.fringe_count_across_beam_diameter <= 0:
            raise ValueError("fringe_count_across_beam_diameter must be positive.")

        y = np.arange(self.config.frame_height_px, dtype=np.float64)
        x = np.arange(self.config.frame_width_px, dtype=np.float64)
        self._x_grid, self._y_grid = np.meshgrid(x, y)
        self._phase_ramp = self._build_phase_ramp()

    def zero_command(self) -> dict[str, int]:
        return {axis_name: 0 for axis_name in AXIS_NAMES}

    def clip_command(self, command: Mapping[str, int | float]) -> dict[str, int]:
        clipped: dict[str, int] = {}
        for axis_name in AXIS_NAMES:
            value = int(round(float(command.get(axis_name, 0))))
            clipped[axis_name] = int(np.clip(value, -self.config.max_steps, self.config.max_steps))
        return clipped

    def _resolve_command(
        self,
        command: Mapping[str, int | float],
        *,
        clip_command: bool,
    ) -> dict[str, int]:
        if clip_command:
            return self.clip_command(command)
        return {axis_name: int(round(float(command.get(axis_name, 0)))) for axis_name in AXIS_NAMES}

    def beam_centers(
        self,
        command: Mapping[str, int | float],
        *,
        clip_command: bool = True,
    ) -> BeamCenters:
        cmd = self._resolve_command(command, clip_command=clip_command)
        scale = self.config.pixels_per_step
        return BeamCenters(
            first_x_px=self.config.center_x_px - scale * cmd["mirror_1_axis_1"],
            first_y_px=self.config.center_y_px - scale * cmd["mirror_1_axis_2"],
            second_x_px=self.config.center_x_px - scale * cmd["mirror_2_axis_1"],
            second_y_px=self.config.center_y_px - scale * cmd["mirror_2_axis_2"],
        )

    def render_components(
        self,
        command: Mapping[str, int | float],
        *,
        clip_command: bool = True,
    ) -> dict[str, np.ndarray | BeamCenters | dict[str, int]]:
        cmd = self._resolve_command(command, clip_command=clip_command)
        centers = self.beam_centers(cmd, clip_command=False)
        first_field = self.config.first_beam_amplitude * self._gaussian_field(
            centers.first_x_px,
            centers.first_y_px,
        )
        second_field = self.config.second_beam_amplitude * self._gaussian_field(
            centers.second_x_px,
            centers.second_y_px,
        ) * np.exp(1j * self._phase_ramp)
        intensity = np.abs(first_field + second_field) ** 2
        return {
            "command": cmd,
            "centers": centers,
            "first_intensity": np.abs(first_field) ** 2,
            "second_intensity": np.abs(second_field) ** 2,
            "combined_intensity": intensity,
        }

    def render(
        self,
        command: Mapping[str, int | float],
        *,
        clip_command: bool = True,
    ) -> np.ndarray:
        components = self.render_components(command, clip_command=clip_command)
        image = np.asarray(components["combined_intensity"], dtype=np.float64)
        if self.config.normalize_each_frame:
            max_value = float(np.max(image))
            if np.isfinite(max_value) and max_value > 0:
                image = image / max_value
        return np.clip(image, 0.0, 1.0)

    def save_image(
        self,
        command: Mapping[str, int | float],
        path: str | Path,
        *,
        clip_command: bool = True,
    ) -> None:
        image = self.render(command, clip_command=clip_command)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dpi = 100
        fig = plt.figure(
            figsize=(self.config.frame_width_px / dpi, self.config.frame_height_px / dpi),
            dpi=dpi,
            frameon=False,
        )
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, origin="upper", interpolation="nearest")
        ax.set_axis_off()
        fig.savefig(path, dpi=dpi, bbox_inches=None, pad_inches=0)
        plt.close(fig)

    def _gaussian_field(self, center_x_px: float, center_y_px: float) -> np.ndarray:
        radius = self.config.beam_radius_px
        r2 = (self._x_grid - center_x_px) ** 2 + (self._y_grid - center_y_px) ** 2
        return np.exp(-r2 / (radius**2))

    def _build_phase_ramp(self) -> np.ndarray:
        """Linear phase whose gradient points toward the upper-right corner."""
        k = self.config.fringe_radians_per_px
        diagonal_coordinate = (
            (self._x_grid - self.config.center_x_px)
            + (self.config.center_y_px - self._y_grid)
        ) / np.sqrt(2.0)
        return k * diagonal_coordinate
