"""Hardware smoke test for the real two-mirror cavity setup."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import time
from typing import Any

from src.hardware.mindvision_camera import MindVisionCamera, MindVisionCameraConfig
from src.hardware.motors import AxisMotorConfig, StepperMotorRig
from src.io.artifacts import ensure_dir, write_json
from src.io.schemas import (
    TwoMirrorCavityAxisConfig,
    TwoMirrorCavityCameraConfig,
    TwoMirrorCavityHardwareConfig,
    load_two_mirror_cavity_hardware_smoke_config,
)


def perform_two_mirror_cavity_hardware_smoke(
    *,
    out_dir: str | Path,
    camera_config: TwoMirrorCavityCameraConfig,
    hardware_config: TwoMirrorCavityHardwareConfig,
    axis: TwoMirrorCavityAxisConfig,
) -> dict[str, Any]:
    """Capture before and after one safety-tracked logical motor move."""
    out_path = ensure_dir(Path(out_dir))
    camera = MindVisionCamera(MindVisionCameraConfig(**asdict(camera_config)))
    rig = StepperMotorRig(
        [
            AxisMotorConfig(
                name=axis.name,
                controller_ip=axis.controller_ip,
                motor_number=axis.motor_number,
                direction_sign=axis.direction_sign,
            )
        ],
        session_state_path=hardware_config.session_state_path,
        steps_per_revolution=hardware_config.steps_per_revolution,
        max_cumulative_revolutions=hardware_config.max_cumulative_revolutions,
        backlash_compensation=hardware_config.backlash_compensation,
        dry_run=hardware_config.dry_run,
    )
    rig.reset_session()
    manifest: dict[str, Any] = {
        "experiment_type": "two_mirror_cavity_hardware_smoke",
        "dry_run": hardware_config.dry_run,
        "axis": asdict(axis),
    }
    try:
        camera.open()
        before = camera.save_capture(
            out_path,
            "before",
            extra_metadata={"capture_kind": "before_move"},
        )
        motion = rig.move_axis(axis.name, axis.logical_steps)
        if hardware_config.settle_time_sec > 0 and not hardware_config.dry_run:
            time.sleep(hardware_config.settle_time_sec)
        after = camera.save_capture(
            out_path,
            "after",
            extra_metadata={
                "capture_kind": "after_move",
                "logical_position_steps": rig.state.current_positions[axis.name],
            },
        )
        manifest.update(
            {
                "before_capture": asdict(before),
                "motion": asdict(motion),
                "after_capture": asdict(after),
                "session_state_path": hardware_config.session_state_path,
            }
        )
        write_json(out_path / "smoke_manifest.json", manifest)
        return manifest
    finally:
        camera.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument(
        "--out",
        type=str,
        default=datetime.now().strftime(
            "experiments/two_mirror_cavity_hardware_smoke_%Y-%m-%d_%H-%M-%S"
        ),
    )
    args = parser.parse_args()
    config = load_two_mirror_cavity_hardware_smoke_config(args.config)
    perform_two_mirror_cavity_hardware_smoke(
        out_dir=args.out,
        camera_config=config.camera,
        hardware_config=config.hardware,
        axis=config.axis,
    )


if __name__ == "__main__":
    main()
