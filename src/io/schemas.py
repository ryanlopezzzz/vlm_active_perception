"""Schema definitions and loaders for experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import yaml


ALLOWED_LLM_EFFORTS = {"none", "low", "medium", "high"}
ALLOWED_REAL_MI_METHODS = {"llm"}
ALLOWED_MICHELSON_INTERFEROMETER_COMPARISON_METHODS = {"llm", "bayes", "random"}
ALLOWED_MI_INITIAL_CONDITION_MODES = {
    "random",
    "both_beams_onscreen",
    "one_beam_onscreen",
    "no_beams_onscreen",
}
ALLOWED_MICHELSON_INTERFEROMETER_INITIAL_CONDITION_MODES = ALLOWED_MI_INITIAL_CONDITION_MODES
HARDWARE_ADDRESS_PLACEHOLDERS = {"", "TODO", "REPLACE_ME", "ANONYMIZED"}
MICHELSON_INTERFEROMETER_AXIS_NAMES = (
    "mirror_1_axis_1",
    "mirror_1_axis_2",
    "mirror_2_axis_1",
    "mirror_2_axis_2",
)


@dataclass(frozen=True)
class LLMConfig:
    model: str = "google/gemini-2.5-flash"
    effort: str = "high"
    max_tokens: int = 6000
    reasoning_max_tokens: int | None = 2000


@dataclass(frozen=True)
class RealMICameraConfig:
    device_index: int = 0
    frame_width: int | None = None
    frame_height: int | None = None
    average_frames: int = 1
    warmup_frames: int = 2
    llm_image_format: str = "png"


@dataclass(frozen=True)
class RealMIAxisConfig:
    name: str
    controller_ip: str
    motor_number: int
    direction_sign: int = 1
    hidden_offset_max_steps: int = 0
    llm_step_limit: int = 0
    preflight_test_steps: int = 0


@dataclass(frozen=True)
class RealMIHardwareConfig:
    baseline_state_path: str
    session_state_path: str
    steps_per_revolution: int = 4096
    max_cumulative_revolutions: float = 5.0
    backlash_compensation: bool = True
    settle_time_sec: float = 0.5
    dry_run: bool = False
    use_baseline_reference: bool = True


@dataclass(frozen=True)
class RealMIPreflightConfig:
    experiment_name: str
    output_root: str
    camera: RealMICameraConfig
    hardware: RealMIHardwareConfig
    axes: list[RealMIAxisConfig]
    max_parallel_jobs: int = 1


@dataclass(frozen=True)
class TwoMirrorCavityCameraConfig:
    camera_index: int = 0
    exposure_ms: float = 100.0
    analog_gain: int = 1
    average_frames: int = 1
    warmup_frames: int = 2
    frame_timeout_ms: int = 1000
    frame_retries: int = 3
    llm_image_format: str = "png"


@dataclass(frozen=True)
class TwoMirrorCavityHardwareConfig:
    session_state_path: str
    steps_per_revolution: int = 4096
    max_cumulative_revolutions: float = 5.0
    backlash_compensation: bool = True
    settle_time_sec: float = 0.5
    dry_run: bool = False


@dataclass(frozen=True)
class TwoMirrorCavityAxisConfig:
    name: str
    controller_ip: str
    motor_number: int
    direction_sign: int = 1
    logical_steps: int = -2000


@dataclass(frozen=True)
class TwoMirrorCavityHardwareSmokeConfig:
    experiment_name: str
    output_root: str
    camera: TwoMirrorCavityCameraConfig
    hardware: TwoMirrorCavityHardwareConfig
    axis: TwoMirrorCavityAxisConfig


@dataclass(frozen=True)
class TwoMirrorCavityRealLLMAxisConfig:
    name: str
    role: str
    controller_ip: str
    motor_number: int
    direction_sign: int
    hidden_offset_min_steps: int
    hidden_offset_max_steps: int
    llm_min_steps: int
    llm_max_steps: int
    calibration_slope_normalized_per_step: float
    calibration_offset_normalized: float = 0.0


@dataclass(frozen=True)
class TwoMirrorCavityRealLLMRunsConfig:
    experiment_name: str
    output_root: str
    repeats: int
    skip_repeat_indices: list[int]
    iterations: int
    probes_per_batch: int
    base_seed: int
    calibration_source_path: str | None
    calibration_origin_definition: str | None
    camera: TwoMirrorCavityCameraConfig
    hardware: TwoMirrorCavityHardwareConfig
    axes: list[TwoMirrorCavityRealLLMAxisConfig]
    llm: LLMConfig


@dataclass(frozen=True)
class RealMILLMRunsConfig:
    experiment_name: str
    output_root: str
    repeats: int
    iterations: int
    base_seed: int
    max_parallel_jobs: int
    methods: list[str]
    camera: RealMICameraConfig
    hardware: RealMIHardwareConfig
    axes: list[RealMIAxisConfig]
    llm: LLMConfig
    initial_condition_mode: str = "random"



@dataclass(frozen=True)
class MichelsonInterferometerAxisConfig:
    name: str
    direction_sign: int = 1
    hidden_offset_max_steps: int = 8192
    visible_step_limit: int = 8192


@dataclass(frozen=True)
class MichelsonInterferometerSimulationConfig:
    frame_width_px: int = 960
    frame_height_px: int = 540
    steps_per_revolution: int = 4096
    max_rotations_from_center: float = 2.0
    beam_diameter_fraction_of_width: float = 0.25
    fringe_count_across_beam_diameter: float = 8.0
    first_beam_amplitude: float = 1.0
    second_beam_amplitude: float = 1.0
    normalize_each_frame: bool = False
    initial_condition_mode: str = "random"


@dataclass(frozen=True)
class MichelsonInterferometerModelConfig:
    key: str
    llm: LLMConfig


@dataclass(frozen=True)
class MichelsonInterferometerCompareMethodsConfig:
    experiment_name: str
    output_root: str
    repeats: int
    iterations: int
    base_seed: int
    max_parallel_jobs: int
    methods: list[str]
    initial_conditions_path: str
    simulation: MichelsonInterferometerSimulationConfig
    axes: list[MichelsonInterferometerAxisConfig]
    llm_models: list[MichelsonInterferometerModelConfig]


@dataclass(frozen=True)
class TwoMirrorCavitySimulationCompareMethodsConfig:
    experiment_name: str
    output_root: str
    repeats: int
    iterations: int
    probes_per_batch: int
    base_seed: int
    max_parallel_jobs: int
    methods: list[str]
    initial_conditions_path: str
    profile_path: str
    axes: list[TwoMirrorCavityRealLLMAxisConfig]
    llm_models: list[MichelsonInterferometerModelConfig]


def _require_mapping(value: Any, ctx: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{ctx} must be a mapping/object.")
    return value


def _is_placeholder_hardware_address(value: str) -> bool:
    return value.strip().upper() in HARDWARE_ADDRESS_PLACEHOLDERS


def _load_real_mi_camera_config(root: dict[str, Any]) -> RealMICameraConfig:
    camera_raw = _require_mapping(root.get("camera", {}), "camera")
    llm_image_format = str(camera_raw.get("llm_image_format", "png")).lower()
    if llm_image_format not in {"png", "jpg", "jpeg"}:
        raise ValueError("camera.llm_image_format must be one of ['jpeg', 'jpg', 'png'].")
    return RealMICameraConfig(
        device_index=int(camera_raw.get("device_index", 0)),
        frame_width=None if camera_raw.get("frame_width") is None else int(camera_raw.get("frame_width")),
        frame_height=None if camera_raw.get("frame_height") is None else int(camera_raw.get("frame_height")),
        average_frames=int(camera_raw.get("average_frames", 1)),
        warmup_frames=int(camera_raw.get("warmup_frames", 2)),
        llm_image_format=llm_image_format,
    )


def _load_real_mi_hardware_config(root: dict[str, Any]) -> RealMIHardwareConfig:
    hardware_raw = _require_mapping(root.get("hardware", {}), "hardware")
    baseline_state_path = hardware_raw.get("baseline_state_path")
    session_state_path = hardware_raw.get("session_state_path")
    if not baseline_state_path:
        raise ValueError("hardware.baseline_state_path is required.")
    if not session_state_path:
        raise ValueError("hardware.session_state_path is required.")
    steps_per_revolution = int(hardware_raw.get("steps_per_revolution", 4096))
    max_cumulative_revolutions = float(hardware_raw.get("max_cumulative_revolutions", 5.0))
    if steps_per_revolution <= 0:
        raise ValueError("hardware.steps_per_revolution must be > 0.")
    if max_cumulative_revolutions <= 0:
        raise ValueError("hardware.max_cumulative_revolutions must be > 0.")
    return RealMIHardwareConfig(
        baseline_state_path=str(baseline_state_path),
        session_state_path=str(session_state_path),
        steps_per_revolution=steps_per_revolution,
        max_cumulative_revolutions=max_cumulative_revolutions,
        backlash_compensation=bool(hardware_raw.get("backlash_compensation", True)),
        settle_time_sec=float(hardware_raw.get("settle_time_sec", 0.5)),
        dry_run=bool(hardware_raw.get("dry_run", False)),
        use_baseline_reference=bool(hardware_raw.get("use_baseline_reference", True)),
    )


def _load_real_mi_axes(root: dict[str, Any]) -> list[RealMIAxisConfig]:
    axes_raw = root.get("axes")
    if not isinstance(axes_raw, list) or not axes_raw:
        raise ValueError("axes must be a non-empty list.")
    axes: list[RealMIAxisConfig] = []
    seen_names: set[str] = set()
    for idx, item in enumerate(axes_raw):
        axis_raw = _require_mapping(item, f"axes[{idx}]")
        name = str(axis_raw.get("name", "")).strip()
        if not name:
            raise ValueError(f"axes[{idx}].name is required.")
        if name in seen_names:
            raise ValueError(f"Duplicate axis name: {name}")
        seen_names.add(name)
        direction_sign = int(axis_raw.get("direction_sign", 1))
        if direction_sign not in {-1, 1}:
            raise ValueError(f"axes[{idx}].direction_sign must be -1 or 1.")
        axis = RealMIAxisConfig(
            name=name,
            controller_ip=str(axis_raw.get("controller_ip", "")),
            motor_number=int(axis_raw.get("motor_number")),
            direction_sign=direction_sign,
            hidden_offset_max_steps=int(axis_raw.get("hidden_offset_max_steps", 0)),
            llm_step_limit=int(axis_raw.get("llm_step_limit", 0)),
            preflight_test_steps=int(axis_raw.get("preflight_test_steps", 0)),
        )
        if _is_placeholder_hardware_address(axis.controller_ip) and not root.get("hardware", {}).get("dry_run", False):
            raise ValueError(f"axes[{idx}].controller_ip is required when not in dry_run mode.")
        if axis.hidden_offset_max_steps < 0 or axis.llm_step_limit < 0 or axis.preflight_test_steps < 0:
            raise ValueError(f"axes[{idx}] step limits must be >= 0.")
        axes.append(axis)
    return axes


def _load_two_mirror_cavity_camera_config(root: dict[str, Any]) -> TwoMirrorCavityCameraConfig:
    camera_raw = _require_mapping(root.get("camera", {}), "camera")
    config = TwoMirrorCavityCameraConfig(
        camera_index=int(camera_raw.get("camera_index", 0)),
        exposure_ms=float(camera_raw.get("exposure_ms", 100.0)),
        analog_gain=int(camera_raw.get("analog_gain", 1)),
        average_frames=int(camera_raw.get("average_frames", 1)),
        warmup_frames=int(camera_raw.get("warmup_frames", 2)),
        frame_timeout_ms=int(camera_raw.get("frame_timeout_ms", 1000)),
        frame_retries=int(camera_raw.get("frame_retries", 3)),
        llm_image_format=str(camera_raw.get("llm_image_format", "png")).lower(),
    )
    if config.camera_index < 0:
        raise ValueError("camera.camera_index must be >= 0.")
    if config.exposure_ms <= 0:
        raise ValueError("camera.exposure_ms must be > 0.")
    if config.analog_gain <= 0:
        raise ValueError("camera.analog_gain must be > 0.")
    if config.average_frames <= 0:
        raise ValueError("camera.average_frames must be > 0.")
    if config.warmup_frames < 0:
        raise ValueError("camera.warmup_frames must be >= 0.")
    if config.frame_timeout_ms <= 0 or config.frame_retries <= 0:
        raise ValueError("camera frame timeout and retries must be > 0.")
    if config.llm_image_format not in {"png", "jpg", "jpeg"}:
        raise ValueError("camera.llm_image_format must be one of ['jpeg', 'jpg', 'png'].")
    return config


def _load_two_mirror_cavity_hardware_config(root: dict[str, Any]) -> TwoMirrorCavityHardwareConfig:
    hardware_raw = _require_mapping(root.get("hardware", {}), "hardware")
    session_state_path = str(hardware_raw.get("session_state_path", "")).strip()
    if not session_state_path:
        raise ValueError("hardware.session_state_path is required.")
    config = TwoMirrorCavityHardwareConfig(
        session_state_path=session_state_path,
        steps_per_revolution=int(hardware_raw.get("steps_per_revolution", 4096)),
        max_cumulative_revolutions=float(hardware_raw.get("max_cumulative_revolutions", 5.0)),
        backlash_compensation=bool(hardware_raw.get("backlash_compensation", True)),
        settle_time_sec=float(hardware_raw.get("settle_time_sec", 0.5)),
        dry_run=bool(hardware_raw.get("dry_run", False)),
    )
    if config.steps_per_revolution <= 0:
        raise ValueError("hardware.steps_per_revolution must be > 0.")
    if config.max_cumulative_revolutions <= 0:
        raise ValueError("hardware.max_cumulative_revolutions must be > 0.")
    if config.settle_time_sec < 0:
        raise ValueError("hardware.settle_time_sec must be >= 0.")
    return config


def _load_two_mirror_cavity_axis_config(
    root: dict[str, Any], *, dry_run: bool
) -> TwoMirrorCavityAxisConfig:
    axis_raw = _require_mapping(root.get("axis", {}), "axis")
    config = TwoMirrorCavityAxisConfig(
        name=str(axis_raw.get("name", "")).strip(),
        controller_ip=str(axis_raw.get("controller_ip", "")).strip(),
        motor_number=int(axis_raw.get("motor_number", 0)),
        direction_sign=int(axis_raw.get("direction_sign", 1)),
        logical_steps=int(axis_raw.get("logical_steps", -2000)),
    )
    if not config.name:
        raise ValueError("axis.name is required.")
    if _is_placeholder_hardware_address(config.controller_ip) and not dry_run:
        raise ValueError("axis.controller_ip is required when not in dry_run mode.")
    if config.motor_number not in {1, 2, 3}:
        raise ValueError("axis.motor_number must be one of [1, 2, 3].")
    if config.direction_sign not in {-1, 1}:
        raise ValueError("axis.direction_sign must be -1 or 1.")
    if config.logical_steps == 0:
        raise ValueError("axis.logical_steps must be nonzero.")
    return config




def _load_mi_initial_condition_mode(root: dict[str, Any]) -> str:
    mode = str(root.get("initial_condition_mode", "random"))
    if mode not in ALLOWED_MI_INITIAL_CONDITION_MODES:
        raise ValueError(f"initial_condition_mode must be one of {sorted(ALLOWED_MI_INITIAL_CONDITION_MODES)}")
    return mode





def _load_michelson_interferometer_simulation_config(root: dict[str, Any]) -> MichelsonInterferometerSimulationConfig:
    simulation_raw = _require_mapping(root.get("simulation", {}), "simulation")
    config = MichelsonInterferometerSimulationConfig(
        frame_width_px=int(simulation_raw.get("frame_width_px", 960)),
        frame_height_px=int(simulation_raw.get("frame_height_px", 540)),
        steps_per_revolution=int(simulation_raw.get("steps_per_revolution", 4096)),
        max_rotations_from_center=float(simulation_raw.get("max_rotations_from_center", 2.0)),
        beam_diameter_fraction_of_width=float(simulation_raw.get("beam_diameter_fraction_of_width", 0.25)),
        fringe_count_across_beam_diameter=float(simulation_raw.get("fringe_count_across_beam_diameter", 8.0)),
        first_beam_amplitude=float(simulation_raw.get("first_beam_amplitude", 1.0)),
        second_beam_amplitude=float(simulation_raw.get("second_beam_amplitude", 1.0)),
        normalize_each_frame=bool(simulation_raw.get("normalize_each_frame", False)),
        initial_condition_mode=str(simulation_raw.get("initial_condition_mode", "random")),
    )
    if config.frame_width_px <= 0 or config.frame_height_px <= 0:
        raise ValueError("simulation frame dimensions must be > 0.")
    if config.steps_per_revolution <= 0:
        raise ValueError("simulation.steps_per_revolution must be > 0.")
    if config.max_rotations_from_center <= 0:
        raise ValueError("simulation.max_rotations_from_center must be > 0.")
    if config.beam_diameter_fraction_of_width <= 0:
        raise ValueError("simulation.beam_diameter_fraction_of_width must be > 0.")
    if config.fringe_count_across_beam_diameter <= 0:
        raise ValueError("simulation.fringe_count_across_beam_diameter must be > 0.")
    if config.first_beam_amplitude < 0 or config.second_beam_amplitude < 0:
        raise ValueError("simulation beam amplitudes must be >= 0.")
    if config.initial_condition_mode not in ALLOWED_MICHELSON_INTERFEROMETER_INITIAL_CONDITION_MODES:
        raise ValueError(
            "simulation.initial_condition_mode must be one of "
            f"{sorted(ALLOWED_MICHELSON_INTERFEROMETER_INITIAL_CONDITION_MODES)}."
        )
    return config


def _load_michelson_interferometer_axes(root: dict[str, Any]) -> list[MichelsonInterferometerAxisConfig]:
    default_axes = [{"name": name, "direction_sign": 1} for name in MICHELSON_INTERFEROMETER_AXIS_NAMES]
    axes_raw = root.get("axes", default_axes)
    if not isinstance(axes_raw, list) or not axes_raw:
        raise ValueError("axes must be a non-empty list.")
    axes: list[MichelsonInterferometerAxisConfig] = []
    seen_names: set[str] = set()
    for idx, item in enumerate(axes_raw):
        axis_raw = _require_mapping(item, f"axes[{idx}]")
        name = str(axis_raw.get("name", "")).strip()
        if not name:
            raise ValueError(f"axes[{idx}].name is required.")
        if name in seen_names:
            raise ValueError(f"Duplicate axis name: {name}")
        seen_names.add(name)
        direction_sign = int(axis_raw.get("direction_sign", 1))
        if direction_sign not in {-1, 1}:
            raise ValueError(f"axes[{idx}].direction_sign must be -1 or 1.")
        hidden_offset_max_steps = int(axis_raw.get("hidden_offset_max_steps", 8192))
        visible_step_limit = int(axis_raw.get("visible_step_limit", 8192))
        if hidden_offset_max_steps <= 0:
            raise ValueError(f"axes[{idx}].hidden_offset_max_steps must be > 0.")
        if visible_step_limit <= 0:
            raise ValueError(f"axes[{idx}].visible_step_limit must be > 0.")
        axes.append(
            MichelsonInterferometerAxisConfig(
                name=name,
                direction_sign=direction_sign,
                hidden_offset_max_steps=hidden_offset_max_steps,
                visible_step_limit=visible_step_limit,
            )
        )
    if tuple(axis.name for axis in axes) != MICHELSON_INTERFEROMETER_AXIS_NAMES:
        raise ValueError(f"axes must exactly match {list(MICHELSON_INTERFEROMETER_AXIS_NAMES)} in order.")
    return axes




def load_real_mi_preflight_config(path: str | Path) -> RealMIPreflightConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = _require_mapping(raw, "config")
    output_root = str(root.get("output_root", "runs"))
    experiment_name = str(root.get("experiment_name", "mi_real_preflight"))
    max_parallel_jobs = int(root.get("max_parallel_jobs", 1))
    if max_parallel_jobs <= 0:
        raise ValueError("max_parallel_jobs must be > 0.")
    return RealMIPreflightConfig(
        experiment_name=experiment_name,
        output_root=output_root,
        camera=_load_real_mi_camera_config(root),
        hardware=_load_real_mi_hardware_config(root),
        axes=_load_real_mi_axes(root),
        max_parallel_jobs=max_parallel_jobs,
    )


def load_two_mirror_cavity_hardware_smoke_config(
    path: str | Path,
) -> TwoMirrorCavityHardwareSmokeConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = _require_mapping(raw, "config")
    hardware = _load_two_mirror_cavity_hardware_config(root)
    return TwoMirrorCavityHardwareSmokeConfig(
        experiment_name=str(
            root.get("experiment_name", "two_mirror_cavity_hardware_smoke")
        ),
        output_root=str(root.get("output_root", "runs")),
        camera=_load_two_mirror_cavity_camera_config(root),
        hardware=hardware,
        axis=_load_two_mirror_cavity_axis_config(root, dry_run=hardware.dry_run),
    )


def _resolve_robotics_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path.resolve()


def _load_two_mirror_cavity_real_llm_calibration(
    root: dict[str, Any],
) -> tuple[dict[str, tuple[float, float]], str | None, str | None]:
    """Load an optional fitted calibration, selecting the newest result if requested."""
    raw = root.get("calibration")
    if raw is None:
        return {}, None, None
    calibration = _require_mapping(raw, "calibration")
    explicit_path = calibration.get("path")
    results_root = calibration.get("results_root")
    selection = str(calibration.get("selection", "latest")).strip().lower()
    if explicit_path is not None and results_root is not None:
        raise ValueError("calibration must define either path or results_root, not both.")
    if explicit_path is not None:
        calibration_path = _resolve_robotics_path(str(explicit_path))
        if calibration_path.is_dir():
            calibration_path = calibration_path / "motor_screen_calibration.json"
    else:
        if not results_root:
            raise ValueError(
                "calibration.results_root is required when calibration.path is absent."
            )
        if selection != "latest":
            raise ValueError("calibration.selection currently supports only 'latest'.")
        root_path = _resolve_robotics_path(str(results_root))
        candidates = sorted(
            root_path.glob("run_*/motor_screen_calibration.json"),
            key=lambda path: path.parent.name,
        )
        if not candidates:
            raise FileNotFoundError(
                "No motor_screen_calibration.json files were found under "
                f"{root_path}. Run the two-mirror-cavity grid analysis first."
            )
        calibration_path = candidates[-1]
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    matrix = payload.get("matrix")
    offset = payload.get("offset")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in matrix)
        or not isinstance(offset, list)
        or len(offset) != 2
    ):
        raise ValueError(
            f"Calibration must contain a 2x2 matrix and two-element offset: {calibration_path}"
        )
    try:
        numeric_matrix = [[float(value) for value in row] for row in matrix]
        numeric_offset = [float(value) for value in offset]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Calibration matrix and offset must be numeric: {calibration_path}"
        ) from exc
    values = [value for row in numeric_matrix for value in row] + numeric_offset
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Calibration contains non-finite values: {calibration_path}")
    if abs(numeric_matrix[0][1]) > 1e-12 or abs(numeric_matrix[1][0]) > 1e-12:
        raise ValueError(
            "Two-mirror-cavity real LLM runs require the independent-axis, "
            f"no-cross-coupling calibration model: {calibration_path}"
        )
    if abs(numeric_matrix[0][0]) < 1e-12 or abs(numeric_matrix[1][1]) < 1e-12:
        raise ValueError(f"Calibration slopes must be nonzero: {calibration_path}")
    return (
        {
            "x": (numeric_matrix[0][0], numeric_offset[0]),
            "y": (numeric_matrix[1][1], numeric_offset[1]),
        },
        str(calibration_path.resolve()),
        (
            None
            if payload.get("origin_definition") is None
            else str(payload["origin_definition"])
        ),
    )


def load_two_mirror_cavity_real_llm_runs_config(
    path: str | Path,
) -> TwoMirrorCavityRealLLMRunsConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = _require_mapping(raw, "config")
    repeats = int(root.get("repeats", 5))
    raw_skip_repeat_indices = root.get("skip_repeat_indices", [])
    if not isinstance(raw_skip_repeat_indices, list):
        raise ValueError("skip_repeat_indices must be a list of zero-based repeat indices.")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in raw_skip_repeat_indices):
        raise ValueError("skip_repeat_indices must contain only integers.")
    skip_repeat_indices = [int(value) for value in raw_skip_repeat_indices]
    iterations = int(root.get("iterations", 10))
    probes_per_batch = int(root.get("probes_per_batch", 5))
    if repeats <= 0:
        raise ValueError("repeats must be > 0.")
    if len(set(skip_repeat_indices)) != len(skip_repeat_indices):
        raise ValueError("skip_repeat_indices must not contain duplicates.")
    if any(index < 0 or index >= repeats for index in skip_repeat_indices):
        raise ValueError("skip_repeat_indices entries must be between 0 and repeats - 1.")
    if len(skip_repeat_indices) == repeats:
        raise ValueError("skip_repeat_indices cannot skip every configured repeat.")
    if iterations <= 0:
        raise ValueError("iterations must be > 0.")
    if not 1 <= probes_per_batch <= 8:
        raise ValueError("probes_per_batch must be between 1 and 8 inclusive.")

    hardware = _load_two_mirror_cavity_hardware_config(root)
    calibration_values, calibration_source_path, calibration_origin_definition = (
        _load_two_mirror_cavity_real_llm_calibration(root)
    )
    axes_raw = root.get("axes")
    if not isinstance(axes_raw, list) or len(axes_raw) != 2:
        raise ValueError("axes must contain exactly two entries, one x axis and one y axis.")
    axes: list[TwoMirrorCavityRealLLMAxisConfig] = []
    seen_names: set[str] = set()
    seen_roles: set[str] = set()
    for index, item in enumerate(axes_raw):
        axis_raw = _require_mapping(item, f"axes[{index}]")
        name = str(axis_raw.get("name", "")).strip()
        role = str(axis_raw.get("role", "")).strip().lower()
        controller_ip = str(axis_raw.get("controller_ip", "")).strip()
        direction_sign = int(axis_raw.get("direction_sign", 1))
        hidden_min = int(axis_raw.get("hidden_offset_min_steps", 0))
        hidden_max = int(axis_raw.get("hidden_offset_max_steps", 0))
        llm_min = int(axis_raw.get("llm_min_steps", 0))
        llm_max = int(axis_raw.get("llm_max_steps", 0))
        if not name or name in seen_names:
            raise ValueError(f"axes[{index}].name must be non-empty and unique.")
        if role not in {"x", "y"} or role in seen_roles:
            raise ValueError("axes must define unique roles x and y.")
        if _is_placeholder_hardware_address(controller_ip) and not hardware.dry_run:
            raise ValueError(f"axes[{index}].controller_ip is required when not in dry_run mode.")
        motor_number = int(axis_raw.get("motor_number", 0))
        if motor_number not in {1, 2, 3}:
            raise ValueError(f"axes[{index}].motor_number must be one of [1, 2, 3].")
        if direction_sign not in {-1, 1}:
            raise ValueError(f"axes[{index}].direction_sign must be -1 or 1.")
        if hidden_min > hidden_max:
            raise ValueError(f"axes[{index}] hidden offset minimum must be <= maximum.")
        if llm_min >= llm_max:
            raise ValueError(f"axes[{index}] LLM minimum must be < maximum.")
        command_value_count = llm_max - llm_min + 1
        if command_value_count <= 0:
            raise ValueError(f"axes[{index}] LLM command range must contain integers.")
        safety_steps = int(round(hardware.max_cumulative_revolutions * hardware.steps_per_revolution))
        max_physical_magnitude = max(
            abs(hidden_min + llm_min),
            abs(hidden_min + llm_max),
            abs(hidden_max + llm_min),
            abs(hidden_max + llm_max),
        )
        if max_physical_magnitude > safety_steps:
            raise ValueError(
                f"axes[{index}] hidden-plus-LLM range can exceed the hardware net displacement safety limit."
            )
        axis_calibration = calibration_values.get(role)
        axes.append(
            TwoMirrorCavityRealLLMAxisConfig(
                name=name,
                role=role,
                controller_ip=controller_ip,
                motor_number=motor_number,
                direction_sign=direction_sign,
                hidden_offset_min_steps=hidden_min,
                hidden_offset_max_steps=hidden_max,
                llm_min_steps=llm_min,
                llm_max_steps=llm_max,
                calibration_slope_normalized_per_step=(
                    axis_calibration[0]
                    if axis_calibration is not None
                    else float(
                        axis_raw.get("calibration_slope_normalized_per_step", 0.0)
                    )
                ),
                calibration_offset_normalized=(
                    axis_calibration[1]
                    if axis_calibration is not None
                    else float(axis_raw.get("calibration_offset_normalized", 0.0))
                ),
            )
        )
        seen_names.add(name)
        seen_roles.add(role)
    if seen_roles != {"x", "y"}:
        raise ValueError("axes must define exactly one x role and one y role.")
    command_capacity = 1
    for axis in axes:
        command_capacity *= axis.llm_max_steps - axis.llm_min_steps + 1
    if command_capacity < probes_per_batch:
        raise ValueError("The configured LLM command ranges cannot provide enough distinct probes.")

    llm_raw = _require_mapping(root.get("llm", {}), "llm")
    llm = LLMConfig(
        model=str(llm_raw.get("model", "openai/gpt-5.6-sol")),
        effort=str(llm_raw.get("effort", "high")),
        max_tokens=int(llm_raw.get("max_tokens", 16000)),
        reasoning_max_tokens=(
            None
            if llm_raw.get("reasoning_max_tokens") is None
            else int(llm_raw["reasoning_max_tokens"])
        ),
    )
    if llm.effort not in ALLOWED_LLM_EFFORTS:
        raise ValueError(f"llm.effort must be one of {sorted(ALLOWED_LLM_EFFORTS)}")
    if llm.max_tokens <= 0:
        raise ValueError("llm.max_tokens must be > 0.")
    if llm.reasoning_max_tokens is not None and llm.reasoning_max_tokens <= 0:
        raise ValueError("llm.reasoning_max_tokens must be > 0 when provided.")
    return TwoMirrorCavityRealLLMRunsConfig(
        experiment_name=str(root.get("experiment_name", "two_mirror_cavity_real_llm_runs")),
        output_root=str(root.get("output_root", "runs/two_mirror_cavity_real_llm")),
        repeats=repeats,
        skip_repeat_indices=sorted(skip_repeat_indices),
        iterations=iterations,
        probes_per_batch=probes_per_batch,
        base_seed=int(root.get("base_seed", 20260718)),
        calibration_source_path=calibration_source_path,
        calibration_origin_definition=calibration_origin_definition,
        camera=_load_two_mirror_cavity_camera_config(root),
        hardware=hardware,
        axes=axes,
        llm=llm,
    )


def load_real_mi_llm_runs_config(path: str | Path) -> RealMILLMRunsConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = _require_mapping(raw, "config")
    output_root = str(root.get("output_root", "runs"))
    experiment_name = str(root.get("experiment_name", "mi_real_llm_runs"))
    repeats = int(root.get("repeats", 1))
    iterations = int(root.get("iterations", 10))
    base_seed = int(root.get("base_seed", 0))
    max_parallel_jobs = int(root.get("max_parallel_jobs", 1))
    methods = [str(value) for value in root.get("methods", ["llm"])]
    if repeats <= 0:
        raise ValueError("repeats must be > 0.")
    if iterations <= 0:
        raise ValueError("iterations must be > 0.")
    if max_parallel_jobs <= 0:
        raise ValueError("max_parallel_jobs must be > 0.")
    invalid_methods = set(methods) - ALLOWED_REAL_MI_METHODS
    if invalid_methods:
        raise ValueError(f"Unsupported real MI methods: {sorted(invalid_methods)}")
    llm_raw = _require_mapping(root.get("llm", {}), "llm")
    llm = LLMConfig(
        model=str(llm_raw.get("model", "google/gemini-2.5-flash")),
        effort=str(llm_raw.get("effort", "high")),
        max_tokens=int(llm_raw.get("max_tokens", 6000)),
        reasoning_max_tokens=(
            None
            if llm_raw.get("reasoning_max_tokens", 2000) is None
            else int(llm_raw.get("reasoning_max_tokens", 2000))
        ),
    )
    if llm.effort not in ALLOWED_LLM_EFFORTS:
        raise ValueError(f"llm.effort must be one of {sorted(ALLOWED_LLM_EFFORTS)}")
    if llm.max_tokens <= 0:
        raise ValueError("llm.max_tokens must be > 0.")
    if llm.reasoning_max_tokens is not None and llm.reasoning_max_tokens <= 0:
        raise ValueError("llm.reasoning_max_tokens must be > 0 when provided.")
    return RealMILLMRunsConfig(
        experiment_name=experiment_name,
        output_root=output_root,
        repeats=repeats,
        iterations=iterations,
        base_seed=base_seed,
        max_parallel_jobs=max_parallel_jobs,
        methods=methods,
        camera=_load_real_mi_camera_config(root),
        hardware=_load_real_mi_hardware_config(root),
        axes=_load_real_mi_axes(root),
        llm=llm,
        initial_condition_mode=_load_mi_initial_condition_mode(root),
    )











def load_michelson_interferometer_compare_methods_config(
    path: str | Path,
) -> MichelsonInterferometerCompareMethodsConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = _require_mapping(raw, "config")
    repeats = int(root.get("repeats", 10))
    iterations = int(root.get("iterations", 60))
    max_parallel_jobs = int(root.get("max_parallel_jobs", 1))
    methods = [str(value) for value in root.get("methods", ["llm", "bayes", "random"])]
    if repeats <= 0 or iterations <= 0 or max_parallel_jobs <= 0:
        raise ValueError("repeats, iterations, and max_parallel_jobs must be > 0.")
    invalid_methods = set(methods) - ALLOWED_MICHELSON_INTERFEROMETER_COMPARISON_METHODS
    if invalid_methods:
        raise ValueError(f"Unsupported methods: {sorted(invalid_methods)}")

    initial_conditions_path = str(root.get("initial_conditions_path", "")).strip()
    if not initial_conditions_path:
        raise ValueError("initial_conditions_path is required.")
    resolved_initial_conditions_path = Path(initial_conditions_path)
    if not resolved_initial_conditions_path.is_absolute():
        resolved_initial_conditions_path = config_path.parent / resolved_initial_conditions_path
    initial_conditions = json.loads(resolved_initial_conditions_path.read_text(encoding="utf-8"))
    if not isinstance(initial_conditions, list) or len(initial_conditions) != repeats:
        raise ValueError("initial_conditions_path must contain one command object per repeat.")
    expected_axes = set(MICHELSON_INTERFEROMETER_AXIS_NAMES)
    for idx, condition in enumerate(initial_conditions):
        if not isinstance(condition, dict) or set(condition) != expected_axes:
            raise ValueError(f"initial_conditions[{idx}] must contain exactly {sorted(expected_axes)}.")

    models_raw = root.get("llm_models", [])
    if not isinstance(models_raw, list):
        raise ValueError("llm_models must be a list.")
    llm_models: list[MichelsonInterferometerModelConfig] = []
    seen_keys: set[str] = set()
    for idx, item in enumerate(models_raw):
        model_raw = _require_mapping(item, f"llm_models[{idx}]")
        key = str(model_raw.get("key", "")).strip()
        if not key or key in seen_keys:
            raise ValueError(f"llm_models[{idx}].key must be non-empty and unique.")
        seen_keys.add(key)
        llm = LLMConfig(
            model=str(model_raw.get("model", "")),
            effort=str(model_raw.get("effort", "high")),
            max_tokens=int(model_raw.get("max_tokens", 20_000)),
            reasoning_max_tokens=(
                None
                if model_raw.get("reasoning_max_tokens") is None
                else int(model_raw["reasoning_max_tokens"])
            ),
        )
        if not llm.model or llm.effort not in ALLOWED_LLM_EFFORTS or llm.max_tokens <= 0:
            raise ValueError(f"Invalid LLM configuration at llm_models[{idx}].")
        llm_models.append(MichelsonInterferometerModelConfig(key=key, llm=llm))
    if "llm" in methods and not llm_models:
        raise ValueError("At least one llm_models entry is required when methods includes llm.")

    return MichelsonInterferometerCompareMethodsConfig(
        experiment_name=str(root.get("experiment_name", "michelson_interferometer_compare_methods")),
        output_root=str(root.get("output_root", "runs")),
        repeats=repeats,
        iterations=iterations,
        base_seed=int(root.get("base_seed", 0)),
        max_parallel_jobs=max_parallel_jobs,
        methods=methods,
        initial_conditions_path=str(resolved_initial_conditions_path),
        simulation=_load_michelson_interferometer_simulation_config(root),
        axes=_load_michelson_interferometer_axes(root),
        llm_models=llm_models,
    )


def load_two_mirror_cavity_simulation_compare_methods_config(
    path: str | Path,
) -> TwoMirrorCavitySimulationCompareMethodsConfig:
    """Load the lab-matched comparison experiment for the empirical cavity simulator."""
    config_path = Path(path)
    root = _require_mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}, "config")
    repeats = int(root.get("repeats", 10))
    iterations = int(root.get("iterations", 50))
    probes_per_batch = int(root.get("probes_per_batch", 1))
    max_parallel_jobs = int(root.get("max_parallel_jobs", 1))
    methods = [str(value) for value in root.get("methods", ["llm", "bayes", "random"])]
    if repeats <= 0 or iterations <= 0 or max_parallel_jobs <= 0:
        raise ValueError("repeats, iterations, and max_parallel_jobs must be > 0.")
    if not 1 <= probes_per_batch <= 8:
        raise ValueError("probes_per_batch must be between 1 and 8 inclusive.")
    invalid_methods = set(methods) - ALLOWED_MICHELSON_INTERFEROMETER_COMPARISON_METHODS
    if invalid_methods:
        raise ValueError(f"Unsupported methods: {sorted(invalid_methods)}")

    initial_path_value = str(root.get("initial_conditions_path", "")).strip()
    if not initial_path_value:
        raise ValueError("initial_conditions_path is required.")
    initial_path = Path(initial_path_value)
    if not initial_path.is_absolute():
        initial_path = config_path.parent / initial_path
    initial_conditions = json.loads(initial_path.read_text(encoding="utf-8"))
    if not isinstance(initial_conditions, list) or len(initial_conditions) != repeats:
        raise ValueError("initial_conditions_path must contain one command object per repeat.")
    for idx, condition in enumerate(initial_conditions):
        if not isinstance(condition, dict) or set(condition) != {"x", "y"}:
            raise ValueError(f"initial_conditions[{idx}] must contain exactly ['x', 'y'].")

    simulation_raw = _require_mapping(root.get("simulation", {}), "simulation")
    profile_path_value = str(simulation_raw.get("profile_path", "")).strip()
    if not profile_path_value:
        raise ValueError("simulation.profile_path is required.")
    profile_path = Path(profile_path_value)
    if not profile_path.is_absolute():
        profile_path = config_path.parent / profile_path
    if not profile_path.is_file():
        raise ValueError(f"simulation.profile_path does not exist: {profile_path}")

    axes_raw = root.get("axes")
    if not isinstance(axes_raw, list) or len(axes_raw) != 2:
        raise ValueError("axes must contain exactly the x and y axes.")
    axes: list[TwoMirrorCavityRealLLMAxisConfig] = []
    for idx, item in enumerate(axes_raw):
        axis = _require_mapping(item, f"axes[{idx}]")
        role = str(axis.get("role", "")).strip().lower()
        resolved_axis = TwoMirrorCavityRealLLMAxisConfig(
            name=str(axis.get("name", role)), role=role, controller_ip="",
            motor_number=int(axis.get("motor_number", 0)), direction_sign=int(axis.get("direction_sign", 1)),
            hidden_offset_min_steps=int(axis.get("hidden_offset_min_steps", 0)),
            hidden_offset_max_steps=int(axis.get("hidden_offset_max_steps", 0)),
            llm_min_steps=int(axis.get("llm_min_steps", 0)), llm_max_steps=int(axis.get("llm_max_steps", 0)),
            calibration_slope_normalized_per_step=float(axis.get("calibration_slope_normalized_per_step", 0.0)),
            calibration_offset_normalized=float(axis.get("calibration_offset_normalized", 0.0)),
        )
        if resolved_axis.motor_number not in {1, 2, 3}:
            raise ValueError(f"axes[{idx}].motor_number must be one of [1, 2, 3].")
        if resolved_axis.llm_min_steps >= resolved_axis.llm_max_steps:
            raise ValueError(f"axes[{idx}] LLM minimum must be < maximum.")
        axes.append(resolved_axis)
    if {axis.role for axis in axes} != {"x", "y"}:
        raise ValueError("axes must define exactly one x role and one y role.")
    by_role = {axis.role: axis for axis in axes}
    for idx, condition in enumerate(initial_conditions):
        for role in ("x", "y"):
            value = condition[role]
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"initial_conditions[{idx}].{role} must be an integer.")
            axis = by_role[role]
            if not axis.hidden_offset_min_steps <= value <= axis.hidden_offset_max_steps:
                raise ValueError(f"initial_conditions[{idx}].{role} is outside the configured hidden range.")

    models: list[MichelsonInterferometerModelConfig] = []
    seen: set[str] = set()
    for idx, item in enumerate(root.get("llm_models", [])):
        model = _require_mapping(item, f"llm_models[{idx}]")
        key = str(model.get("key", "")).strip()
        llm = LLMConfig(
            model=str(model.get("model", "")), effort=str(model.get("effort", "high")),
            max_tokens=int(model.get("max_tokens", 20_000)),
            reasoning_max_tokens=None if model.get("reasoning_max_tokens") is None else int(model["reasoning_max_tokens"]),
        )
        if (not key or key in seen or not llm.model or llm.effort not in ALLOWED_LLM_EFFORTS
                or llm.max_tokens <= 0
                or (llm.reasoning_max_tokens is not None and llm.reasoning_max_tokens <= 0)):
            raise ValueError(f"Invalid LLM configuration at llm_models[{idx}].")
        seen.add(key)
        models.append(MichelsonInterferometerModelConfig(key=key, llm=llm))
    if "llm" in methods and not models:
        raise ValueError("At least one llm_models entry is required when methods includes llm.")
    return TwoMirrorCavitySimulationCompareMethodsConfig(
        experiment_name=str(root.get("experiment_name", "two_mirror_cavity_simulation_compare_methods")),
        output_root=str(root.get("output_root", "runs")), repeats=repeats, iterations=iterations,
        probes_per_batch=probes_per_batch, base_seed=int(root.get("base_seed", 0)),
        max_parallel_jobs=max_parallel_jobs, methods=methods,
        initial_conditions_path=str(initial_path.resolve()), profile_path=str(profile_path.resolve()),
        axes=axes, llm_models=models,
    )


def load_experiment_type(path: str | Path) -> str:
    """Read experiment type selector from config."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = _require_mapping(raw, "config")
    experiment_type = str(root.get("experiment_type", "")).strip()
    if not experiment_type:
        raise ValueError("config.experiment_type is required.")
    return experiment_type
