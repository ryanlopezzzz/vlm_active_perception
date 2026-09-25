"""Persistent lab adapter for the camera skeleton and Michelson motor rig.

Importing this adapter does not load the skeleton. Opening a real session imports
the skeleton, which connects to/enables the robot at import as in the lab script.
"""
from __future__ import annotations

from dataclasses import asdict
import importlib
import math
from pathlib import Path
import time

import numpy as np

from src.hardware.motors import AxisMotorConfig, StepperMotorRig
from src.sim.motorized_mirror_relay import AXES


STATION_IDS = tuple(f"C{i}" for i in range(1, 9))


def validate_lab_config(config: dict) -> None:
    """Reject unfilled placeholders before importing hardware or creating an LLM."""
    errors = []

    def number(value, name, *, positive=False, nonnegative=False, integer=False):
        valid = type(value) in (int, float) and math.isfinite(value)
        if integer:
            valid = valid and type(value) is int
        if positive:
            valid = valid and value > 0
        if nonnegative:
            valid = valid and value >= 0
        if not valid:
            errors.append(f"{name}: fill in a valid {'integer' if integer else 'number'}")

    for name in ("iterations", "repeats"):
        number(config.get(name), name, positive=True, integer=True)
    if config.get("max_parallel_jobs", 1) != 1:
        errors.append("max_parallel_jobs must be 1: all trials share the same lab hardware")
    if not isinstance(config.get("llm", {}).get("model"), str):
        errors.append("llm.model is required")
    hardware = config.get("hardware", {})
    for name in ("steps_per_revolution", "command_limit_steps"):
        number(hardware.get(name), f"hardware.{name}", positive=True, integer=True)
    for name in ("mirror_radians_per_revolution", "max_cumulative_revolutions"):
        number(hardware.get(name), f"hardware.{name}", positive=True)
    number(hardware.get("settle_time_sec"), "hardware.settle_time_sec", nonnegative=True)
    for name in ("session_state_path", "skeleton_module", "calibration_path"):
        if not isinstance(hardware.get(name), str) or not hardware[name].strip():
            errors.append(f"hardware.{name} is required")
    for name in ("backlash_compensation",):
        if type(hardware.get(name)) is not bool:
            errors.append(f"hardware.{name} must be boolean")
    controllers = config.get("mirror_controllers", {})
    if set(controllers) != {f"M{i}" for i in range(1, 5)}:
        errors.append("mirror_controllers must contain M1–M4")
    used_channels = set()
    for i in range(1, 5):
        controller = controllers.get(f"M{i}", {})
        address = controller.get("controller_ip")
        if not isinstance(address, str) or not address.strip() or address.upper() in {"TODO", "REPLACE_ME", "ANONYMIZED"}:
            errors.append(f"mirror_controllers.M{i}.controller_ip: fill in the controller address")
        for axis in ("in_plane", "out_of_plane"):
            channel = controller.get(axis, {})
            motor = channel.get("motor_number")
            number(motor, f"mirror_controllers.M{i}.{axis}.motor_number", positive=True, integer=True)
            if type(channel.get("direction_sign")) is not int or channel["direction_sign"] not in (-1, 1):
                errors.append(f"mirror_controllers.M{i}.{axis}.direction_sign must be -1 or 1")
            if isinstance(address, str) and type(motor) is int:
                key = (address, motor)
                if key in used_channels:
                    errors.append(f"Motor channel {key} is assigned to multiple axes")
                used_channels.add(key)
    stations = config.get("camera_stations", {})
    if set(stations) != set(STATION_IDS):
        errors.append("camera_stations must contain exactly C1–C8")
    for station_id in STATION_IDS:
        station = stations.get(station_id, {})
        for name in ("x_position_f", "y_position_f", "added_angle"):
            number(station.get(name), f"camera_stations.{station_id}.{name}")
        for name in ("view_x", "view_y"):
            if name in station:
                number(station[name], f"camera_stations.{station_id}.{name}")
    camera = config.get("camera", {})
    for name in ("width", "height", "llm_width", "llm_height"):
        number(camera.get(name), f"camera.{name}", positive=True, integer=True)
    for name in ("index", "skip_frames"):
        number(camera.get(name), f"camera.{name}", nonnegative=True, integer=True)
    for name in ("exposure", "auto_exposure"):
        number(camera.get(name), f"camera.{name}")
    number(camera.get("settle_s"), "camera.settle_s", nonnegative=True)
    if camera.get("backend") not in {"CAP_DSHOW", "CAP_MSMF", "CAP_ANY"}:
        errors.append("camera.backend must be CAP_DSHOW, CAP_MSMF, or CAP_ANY")
    if errors:
        raise ValueError("Lab configuration is incomplete or invalid:\n- " + "\n- ".join(errors))


def axis_configs(config: dict) -> list[AxisMotorConfig]:
    axes = []
    for i in range(1, 5):
        controller = config["mirror_controllers"][f"M{i}"]
        for direction in ("in_plane", "out_of_plane"):
            channel = controller[direction]
            axes.append(AxisMotorConfig(
                name=f"mirror_{i}_{direction}_steps", controller_ip=controller["controller_ip"],
                motor_number=channel["motor_number"], direction_sign=channel["direction_sign"],
                llm_step_limit=config["hardware"]["command_limit_steps"]))
    return axes


class RelayLabSession:
    def __init__(self, config: dict, *, skeleton=None, rig_factory=StepperMotorRig,
                 enable_beam_camera: bool = True, enforce_motor_limits: bool = True,
                 tracked_camera: bool = False):
        validate_lab_config(config)
        self.config = config
        self.skeleton = skeleton
        self.rig_factory = rig_factory
        self.enable_beam_camera = enable_beam_camera
        self.enforce_motor_limits = enforce_motor_limits
        self.tracked_camera = tracked_camera
        self.rig = None
        self.vision_cams = None
        self.elp = None
        self.current_camera_location = None
        self.camera_placements = 0
        self.camera_image_count = 0
        self.baseline_positions = {}
        self.opened = False

    def validate_command(self, command: dict) -> None:
        if not isinstance(command, dict) or set(command) != set(AXES):
            raise ValueError("All eight absolute motor-step settings are required")
        limit = self.config["hardware"]["command_limit_steps"]
        if any(type(value) is not int for value in command.values()):
            raise ValueError("Motor settings must be integers")
        if self.enforce_motor_limits and any(abs(value) > limit for value in command.values()):
            raise ValueError(f"Motor settings must be integers within ±{limit}")

    def validate_camera_choice(self, choice: str) -> None:
        if not isinstance(choice, str) or choice not in {*STATION_IDS, "keep"}:
            raise ValueError("Camera choice must be C1–C8 or keep")

    def open(self):
        if self.opened:
            return self
        hardware, camera = self.config["hardware"], self.config["camera"]
        if self.skeleton is None:
            if not self.tracked_camera and not Path(hardware["calibration_path"]).is_file():
                raise ValueError(f"Calibration file not found: {hardware['calibration_path']}")
            # Intentional: importing the colleague's module connects/enables xArm.
            self.skeleton = importlib.import_module(hardware["skeleton_module"])
        s = self.skeleton
        try:
            s.CALIBRATION_PATH = Path(hardware["calibration_path"]).resolve()
            if self.enable_beam_camera:
                s.ELP_INDEX = camera["index"]
                s.ELP_BACKEND = getattr(s.cv2, camera["backend"])
                s.ELP_WIDTH, s.ELP_HEIGHT = camera["width"], camera["height"]
                s.ELP_AUTO_EXPOSURE, s.ELP_N_SKIP = camera["auto_exposure"], camera["skip_frames"]
            self.rig = self.rig_factory(
                axis_configs(self.config), session_state_path=hardware["session_state_path"],
                steps_per_revolution=hardware["steps_per_revolution"],
                max_cumulative_revolutions=hardware["max_cumulative_revolutions"],
                backlash_compensation=hardware["backlash_compensation"],
                enforce_position_limit=self.enforce_motor_limits)
            # The batch starts from the actual tracked state, never from a fictitious
            # reset of motor counters. New state files declare the present pose zero.
            self.baseline_positions = dict(self.rig.state.current_positions)
            s.start()
            self.vision_cams = s.initialize_realsense_only() if self.tracked_camera else s.initialize_cams()
            if self.enable_beam_camera:
                self.elp = s.open_elp_camera()
            self.opened = True
            return self
        except BaseException:
            self.close()
            raise

    @property
    def current_command(self) -> dict:
        if self.rig is None:
            raise RuntimeError("Lab session is not open")
        return {axis: self.rig.state.current_positions[axis] - self.baseline_positions[axis] for axis in AXES}

    def set_motor_steps(self, command: dict) -> list[dict]:
        self.validate_command(command)
        if not self.opened:
            raise RuntimeError("Lab session is not open")
        targets = {axis: self.baseline_positions[axis] + command[axis] for axis in AXES}
        limit = round(self.rig.steps_per_revolution * self.rig.max_cumulative_revolutions)
        if self.enforce_motor_limits and any(abs(value) > limit for value in targets.values()):
            raise ValueError("Requested motor positions exceed the configured session displacement limit")
        deltas = {axis: target - self.rig.state.current_positions[axis] for axis, target in targets.items()
                  if target != self.rig.state.current_positions[axis]}
        records = self.rig.move_many(deltas)
        if records:
            time.sleep(self.config["hardware"]["settle_time_sec"])
        return [asdict(record) for record in records]

    def declare_camera_station(self, station: str) -> None:
        if station not in STATION_IDS:
            raise ValueError("Declare a station C1 through C8")
        self.current_camera_location = station

    def place_camera(self, camera_choice: str) -> dict:
        """Move between stations independently of beam image capture."""
        self.validate_camera_choice(camera_choice)
        if not self.opened:
            raise RuntimeError("Lab session is not open")
        if self.tracked_camera and self.current_camera_location is None:
            raise ValueError("Camera location unknown. Use locate C1 (or the actual station) first.")
        location = self.current_camera_location if camera_choice == "keep" else camera_choice
        if location is None:
            raise ValueError("Cannot keep camera before its first placement")
        station = self.config["camera_stations"][location]
        camera, s = self.config["camera"], self.skeleton
        step = s.CaptureStep(name=f"placement_{self.camera_placements}", x_position_f=station["x_position_f"],
                             y_position_f=station["y_position_f"], added_angle=station["added_angle"],
                             exposure=camera["exposure"], settle_s=camera["settle_s"])
        moved = location != self.current_camera_location
        if moved:
            try:
                if self.tracked_camera:
                    source = self.config["camera_stations"][self.current_camera_location]
                    s.place_camera_from_known_station(self.vision_cams, source, step)
                else:
                    s.place_camera_at_xy(self.vision_cams, step)
            except BaseException:
                if self.tracked_camera:
                    self.current_camera_location = None
                raise
            self.current_camera_location = location
            self.camera_placements += 1
            time.sleep(step.settle_s)
        return {"camera_location": location, "camera_moved": moved,
                "camera_placement_count": self.camera_placements, "robot_station": dict(station)}

    def capture(self, camera_choice: str, out_dir: Path, index: int) -> dict:
        if not self.opened or self.elp is None:
            raise RuntimeError("Beam camera is not open in this session")
        placement = self.place_camera(camera_choice)
        location = placement["camera_location"]
        station = self.config["camera_stations"][location]
        camera, s = self.config["camera"], self.skeleton
        step = s.CaptureStep(name=f"step_{index}", x_position_f=station["x_position_f"],
                             y_position_f=station["y_position_f"], added_angle=station["added_angle"],
                             exposure=camera["exposure"], settle_s=camera["settle_s"])
        raw_dir = out_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        s.SNAPSHOT_DIR = raw_dir
        frame = s.capture_beam_image(self.elp, step)
        if frame is None:
            raise RuntimeError(f"Beam capture failed at {location}")
        image_path = out_dir / f"step_{index}.png"
        metadata = prepare_llm_image(frame, image_path, camera["llm_width"], camera["llm_height"])
        self.camera_image_count += 1
        return {**placement, "raw_image_path": str(raw_dir / f"step_{index}.png"),
                "image_path": str(image_path), "exposure": camera["exposure"], **metadata}

    def close(self) -> None:
        try:
            if self.elp is not None:
                self.elp.release()
            if self.vision_cams is not None:
                *cameras, pipeline = self.vision_cams
                for camera in cameras:
                    if camera is not None:
                        camera.release()
                pipeline.stop()
        finally:
            self.elp = self.vision_cams = None
            self.opened = False

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()


def prepare_llm_image(frame: np.ndarray, path: Path, width: int, height: int) -> dict:
    """Keep full field of view; grayscale and letterbox instead of stretching."""
    from PIL import Image
    if frame.dtype != np.uint8:
        raise ValueError("Expected an 8-bit ELP camera frame")
    if frame.ndim == 2:
        gray = Image.fromarray(frame)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        gray = Image.fromarray(frame[:, :, 0])
    else:
        # OpenCV capture returns BGR/BGRA, while Pillow expects RGB/RGBA.
        channels = [2, 1, 0, 3] if frame.shape[2] == 4 else [2, 1, 0]
        gray = Image.fromarray(frame[:, :, channels]).convert("L")
    scale = min(width / gray.width, height / gray.height)
    resized_width = max(1, min(width, round(gray.width * scale)))
    resized_height = max(1, min(height, round(gray.height * scale)))
    resized = gray.resize((resized_width, resized_height), Image.Resampling.BOX)
    left, top = (width - resized_width) // 2, (height - resized_height) // 2
    image = np.zeros((height, width), dtype=np.uint8)
    image[top:top+resized_height, left:left+resized_width] = resized
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    return {"raw_shape": list(frame.shape), "llm_image_shape": list(image.shape),
            "image_content_box_px": [left, top, resized_width, resized_height]}
