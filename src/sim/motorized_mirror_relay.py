"""Motorized disk-mirror relay, adapted from the four-mirror 3D ray model.

Geometry is in mm; mirror commands are absolute integer motor steps.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image


AXES = tuple(f"mirror_{i}_{axis}_steps" for i in range(1, 5)
             for axis in ("in_plane", "out_of_plane"))
RELAY_SEQUENCE = ["M1", "M2", "M3", "M4"]
NO_SIGNAL_OBJECTIVE = -1_000_000.0
CAMERA_CLEARANCE_MM = 42.5


@dataclass(frozen=True)
class RelayConfig:
    width_mm: float = 279.4
    height_mm: float = 203.2
    input_distance_mm: float = 120.0
    mirror_diameter_mm: float = 25.4
    camera_width_mm: float = 6.4
    camera_height_mm: float = 4.8
    frame_width_px: int = 640
    frame_height_px: int = 480
    beam_radius_mm: float = 0.8
    steps_per_revolution: int = 4096
    mirror_radians_per_revolution: float = 0.027
    hidden_offset_max_steps: int = 8192
    command_limit_steps: int = 8192

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if value == 0 and name != "hidden_offset_max_steps":
                raise ValueError(f"{name} must be positive")
        for name in ("frame_width_px", "frame_height_px", "steps_per_revolution",
                     "hidden_offset_max_steps", "command_limit_steps"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")

    @property
    def radians_per_step(self) -> float:
        return self.mirror_radians_per_revolution / self.steps_per_revolution


def unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


@dataclass(frozen=True)
class CameraPose:
    center_mm: tuple[float, float, float]
    normal_yaw_deg: float = 180.0
    normal_pitch_deg: float = 0.0

    def __post_init__(self) -> None:
        if len(self.center_mm) != 3 or not np.all(np.isfinite(self.center_mm)):
            raise ValueError("Camera center must contain three finite coordinates")
        if not np.all(np.isfinite([self.normal_yaw_deg, self.normal_pitch_deg])):
            raise ValueError("Camera angles must be finite")

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        yaw, pitch = np.deg2rad([self.normal_yaw_deg, self.normal_pitch_deg])
        normal = unit([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        vertical = np.array([0., 0., 1.]) - normal[2] * normal
        if np.linalg.norm(vertical) < 1e-12:
            vertical = np.array([0., 1., 0.]) - normal[1] * normal
        vertical = unit(vertical)
        return normal, unit(np.cross(vertical, normal)), vertical


class MotorizedMirrorRelay:
    def __init__(self, config: RelayConfig | None = None, *, seed: int = 0,
                 hidden_offsets: Mapping[str, int] | None = None):
        self.config = config or RelayConfig()
        c = self.config
        hidden_seed = np.random.SeedSequence(seed).spawn(3)[0]
        hidden_rng = np.random.default_rng(hidden_seed)
        self.hidden_offsets = (dict(hidden_offsets) if hidden_offsets is not None else
                               dict(zip(AXES, map(int, hidden_rng.integers(
                                   -c.hidden_offset_max_steps, c.hidden_offset_max_steps + 1, len(AXES))))))
        self._validate_steps(self.hidden_offsets, c.hidden_offset_max_steps)
        x, w, h = c.input_distance_mm, c.width_mm, c.height_mm
        self.nominal_centers = np.array([[x, 0, 0], [x, h, 0], [x+w, h, 0], [x+w, 0, 0.]])
        self.centers = self.nominal_centers.copy()
        self.ideal_normals = np.array([unit(v) for v in
                                      ([-1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 1, 0])])
        self.camera_placements = 0
        self.camera_image_count = 0
        self.camera_locations = self._camera_locations()
        self.current_camera_location: str | None = None
        self.actual_camera_pose: CameraPose | None = None

    def _camera_locations(self) -> dict[str, CameraPose]:
        """Eight nominal stations, with >=4 cm center-to-center clearance."""
        c, gap = self.config, CAMERA_CLEARANCE_MM
        if min(c.width_mm, c.height_mm) <= 2 * gap:
            raise ValueError("Relay legs must exceed 85 mm to fit two distinct camera stations")
        x, w, h = c.input_distance_mm, c.width_mm, c.height_mm
        return {
            "C1": CameraPose((x, gap, 0.), -90., 0.),
            "C2": CameraPose((x, h - gap, 0.), -90., 0.),
            "C3": CameraPose((x + gap, h, 0.), 180., 0.),
            "C4": CameraPose((x + w - gap, h, 0.), 180., 0.),
            "C5": CameraPose((x + w, h - gap, 0.), 90., 0.),
            "C6": CameraPose((x + w, gap, 0.), 90., 0.),
            "C7": CameraPose((x + w + gap, 0., 0.), 180., 0.),
            "C8": CameraPose((x + w + 82.5, 0., 0.), 180., 0.),
        }

    def validate_camera_choice(self, choice: str) -> None:
        if not isinstance(choice, str) or choice not in {*self.camera_locations, "keep"}:
            raise ValueError("Camera choice must be C1–C8 or 'keep'")

    @staticmethod
    def _validate_steps(command: Mapping[str, int], limit: int) -> None:
        if set(command) != set(AXES):
            raise ValueError("All eight motor axes are required")
        if any(type(v) is not int or abs(v) > limit for v in command.values()):
            raise ValueError(f"Motor settings must be integers within ±{limit} steps")

    def validate_command(self, command: Mapping[str, int]) -> None:
        self._validate_steps(command, self.config.command_limit_steps)

    def zero_command(self) -> dict[str, int]:
        return dict.fromkeys(AXES, 0)

    def normals(self, command: Mapping[str, int]) -> np.ndarray:
        self.validate_command(command)
        normals = []
        for i, base in enumerate(self.ideal_normals):
            a1 = np.array([0., 0., 1.])
            a2 = unit(np.cross(base, a1))
            angles = [(command[axis] + self.hidden_offsets[axis]) * self.config.radians_per_step
                      for axis in AXES[2*i:2*i+2]]
            normals.append(unit(base + angles[0] * np.cross(a1, base) + angles[1] * np.cross(a2, base)))
        return np.array(normals)

    @staticmethod
    def plane_intersection(p, d, center, normal):
        denom = float(np.dot(d, normal))
        if denom >= -1e-9:  # Reflecting/receiving front face only.
            return None
        distance = float(np.dot(np.asarray(center) - p, normal) / denom)
        if distance <= 1e-9:
            return None
        return distance, p + distance * d

    def trace(self, command: Mapping[str, int], pose: CameraPose) -> dict:
        """Deterministic central-ray trace; does not place a camera or consume RNG."""
        normals = self.normals(command)
        normal, horizontal, vertical = pose.basis()
        center = np.asarray(pose.center_mm)
        p, d = np.zeros(3), np.array([1., 0., 0.])
        sequence, path = [], [p.tolist()]
        off_camera_crossings = []
        for _ in range(10):
            candidates = []
            for i, (mc, mn) in enumerate(zip(self.centers, normals)):
                hit = self.plane_intersection(p, d, mc, mn)
                if hit is not None and np.linalg.norm(hit[1] - mc) <= self.config.mirror_diameter_mm / 2 + 1e-9:
                    candidates.append((hit[0], i, hit[1]))
            hit = self.plane_intersection(p, d, center, normal)
            if hit is not None:
                delta = hit[1] - center
                u, v = float(delta @ horizontal), float(delta @ vertical)
                # Only consider this segment, before the next mirror. A missed
                # finite camera must not block a later reflection or camera hit.
                if not candidates or hit[0] < min(item[0] for item in candidates):
                    crossing = {"screen_hit": abs(u) <= self.config.camera_width_mm / 2 and
                            abs(v) <= self.config.camera_height_mm / 2,
                            "plane_reached": True, "u_mm": u, "v_mm": v,
                            "beam_offset_mm": float(np.hypot(u, v)),
                            "mirror_hit_sequence": list(sequence), "path_mm": path + [hit[1].tolist()],
                            "final_direction": d.tolist()}
                    if crossing["screen_hit"]:
                        return crossing
                    off_camera_crossings.append(crossing)
            if not candidates:
                break
            _, i, p = min(candidates, key=lambda item: item[0])
            path.append(p.tolist())
            sequence.append(f"M{i+1}")
            d = unit(d - 2 * np.dot(d, normals[i]) * normals[i])
        if off_camera_crossings:
            # Retain Gaussian tails from the closest unobstructed plane crossing.
            return min(off_camera_crossings, key=lambda item: item["beam_offset_mm"])
        return {"screen_hit": False, "plane_reached": False, "u_mm": None, "v_mm": None,
                "beam_offset_mm": None, "mirror_hit_sequence": sequence,
                "path_mm": path, "final_direction": d.tolist()}

    def image(self, trace: dict) -> np.ndarray:
        c = self.config
        if not trace["plane_reached"]:
            return np.zeros((c.frame_height_px, c.frame_width_px), dtype=np.uint8)
        u = ((np.arange(c.frame_width_px) + 0.5) / c.frame_width_px - 0.5) * c.camera_width_mm
        # Top row is +vertical: PNG and physical coordinates agree.
        v = (0.5 - (np.arange(c.frame_height_px) + 0.5) / c.frame_height_px) * c.camera_height_mm
        radius_squared = (u[None, :] - trace["u_mm"])**2 + (v[:, None] - trace["v_mm"])**2
        return np.rint(255 * np.exp(-2 * radius_squared / c.beam_radius_mm**2)).astype(np.uint8)

    def place_camera(self, camera_choice: str) -> bool:
        """Place without exposing; keeping a station preserves its sampled pose."""
        self.validate_camera_choice(camera_choice)
        location = self.current_camera_location if camera_choice == "keep" else camera_choice
        if location is None:
            raise ValueError("Cannot keep the camera before its initial placement")
        pose = self.camera_locations[location]
        moved = location != self.current_camera_location
        if moved:
            self.actual_camera_pose = pose
            self.current_camera_location = location
            self.camera_placements += 1
        return moved

    def capture(self, command: Mapping[str, int], camera_choice: str, image_path: Path) -> dict:
        """Take one image; place only when changing stations (or on the first image)."""
        self.validate_command(command)
        moved = self.place_camera(camera_choice)
        location = self.current_camera_location
        pose = self.camera_locations[location]
        actual = self.actual_camera_pose
        assert actual is not None
        trace = self.trace(command, actual)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.image(trace)).save(image_path)
        self.camera_image_count += 1
        return {"nominal_camera": asdict(pose), "actual_camera": asdict(actual),
                "camera_location": location, "camera_moved": moved,
                "camera_placement_count": self.camera_placements,
                "image_path": str(image_path), **trace}

    def exit_pose(self) -> CameraPose:
        return self.camera_locations["C8"]

    def initial_conditions(self) -> dict:
        return {"hidden_offset_steps": self.hidden_offsets,
                "mirror_centers_mm": self.centers.tolist()}
