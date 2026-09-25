"""Shared real Michelson hardware workflow logic."""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

from src.agents.llm_agent import LLM
from src.hardware.camera import CameraCaptureConfig, HardwareCamera
from src.hardware.motors import (
    AxisMotorConfig,
    StepperMotorRig,
)
from src.io.artifacts import ensure_dir, write_json
from src.io.schemas import LLMConfig, RealMIAxisConfig, RealMICameraConfig, RealMIHardwareConfig
from src.tasks.mi_real_prompting import (
    build_real_mi_system_prompt,
    sample_hidden_offset_steps,
    verify_real_motor_command,
)
from src.tasks.timing import add_run_timing, llm_call_count, llm_request_wall_time_since, start_timer, step_timing_fields

logger = logging.getLogger(__name__)


def _camera_config(config: RealMICameraConfig) -> CameraCaptureConfig:
    return CameraCaptureConfig(
        device_index=config.device_index,
        frame_width=config.frame_width,
        frame_height=config.frame_height,
        average_frames=config.average_frames,
        warmup_frames=config.warmup_frames,
        llm_image_format=config.llm_image_format,
    )


def _axis_configs(axes: list[RealMIAxisConfig]) -> list[AxisMotorConfig]:
    return [AxisMotorConfig(**asdict(axis)) for axis in axes]


def _sleep_if_needed(seconds: float, *, dry_run: bool) -> None:
    if seconds > 0 and not dry_run:
        time.sleep(seconds)


def perform_real_mi_preflight(
    *,
    out_dir: str | Path,
    camera_config: RealMICameraConfig,
    hardware_config: RealMIHardwareConfig,
    axes: list[RealMIAxisConfig],
) -> dict[str, Any]:
    run_started_at, run_start_perf = start_timer()
    out_path = ensure_dir(Path(out_dir))
    camera = HardwareCamera(_camera_config(camera_config))
    rig = StepperMotorRig(
        _axis_configs(axes),
        session_state_path=hardware_config.session_state_path,
        steps_per_revolution=hardware_config.steps_per_revolution,
        max_cumulative_revolutions=hardware_config.max_cumulative_revolutions,
        backlash_compensation=hardware_config.backlash_compensation,
        dry_run=hardware_config.dry_run,
    )
    rig.reset_session()
    manifest: dict[str, Any] = {"baseline": {}, "axis_tests": [], "dry_run": hardware_config.dry_run}
    try:
        baseline_capture = camera.save_capture(
            out_path,
            "baseline",
            extra_metadata={"capture_kind": "baseline"},
            save_raw_array=False,
            save_metadata=False,
        )
        stable_baseline_image = Path(hardware_config.baseline_state_path).with_suffix(f".{camera_config.llm_image_format}")
        stable_baseline_image.parent.mkdir(parents=True, exist_ok=True)
        Path(stable_baseline_image).write_bytes(Path(baseline_capture.llm_image_path).read_bytes())
        rig.write_baseline_state(
            hardware_config.baseline_state_path,
            baseline_image_path=str(stable_baseline_image),
            baseline_capture_metadata_path=None,
        )
        manifest["baseline"] = {
            "capture": asdict(baseline_capture),
            "baseline_state_path": hardware_config.baseline_state_path,
            "session_state_path": hardware_config.session_state_path,
        }
        for axis in axes:
            requested_steps = axis.preflight_test_steps
            move_record = rig.move_axis(axis.name, requested_steps)
            _sleep_if_needed(hardware_config.settle_time_sec, dry_run=hardware_config.dry_run)
            moved_capture = camera.save_capture(
                out_path,
                axis.name,
                extra_metadata={
                    "capture_kind": "axis_test",
                    "axis": axis.name,
                    "requested_steps": requested_steps,
                },
                save_raw_array=False,
                save_metadata=False,
            )
            return_record = rig.move_axis(axis.name, -requested_steps)
            _sleep_if_needed(hardware_config.settle_time_sec, dry_run=hardware_config.dry_run)
            manifest["axis_tests"].append(
                {
                    "axis": axis.name,
                    "requested_steps": requested_steps,
                    "move_record": asdict(move_record),
                    "capture": asdict(moved_capture),
                    "return_record": asdict(return_record),
                    "baseline_restored": rig.state.current_positions[axis.name] == 0,
                }
            )
        final_return_capture = camera.save_capture(
            out_path,
            "final_return",
            extra_metadata={"capture_kind": "final_return"},
            save_raw_array=False,
            save_metadata=False,
        )
        manifest["final_return"] = asdict(final_return_capture)
    finally:
        camera.close()
    return manifest


def _command_from_answer(answer: str, axes: list[RealMIAxisConfig]) -> dict[str, int]:
    payload = json.loads(answer)
    return {axis.name: int(payload["command"][axis.name]) for axis in axes}


def _command_delta(
    current_visible_positions: dict[str, int],
    command: dict[str, int],
) -> dict[str, int]:
    return {
        axis_name: int(command[axis_name]) - int(current_visible_positions.get(axis_name, 0))
        for axis_name in command
    }


def _true_positions(
    hidden_offset: dict[str, int],
    visible_positions: dict[str, int],
) -> dict[str, int]:
    return {
        axis_name: int(hidden_offset.get(axis_name, 0)) + int(visible_positions.get(axis_name, 0))
        for axis_name in hidden_offset
    }


def run_real_llm_alignment(
    *,
    out_dir: str | Path,
    iterations: int,
    seed: int,
    repeat_index: int,
    restore_baseline_at_end: bool = False,
    llm_config: LLMConfig,
    camera_config: RealMICameraConfig,
    hardware_config: RealMIHardwareConfig,
    axes: list[RealMIAxisConfig],
    initial_condition_mode: str = "random",
) -> dict[str, Any]:
    run_started_at, run_start_perf = start_timer()
    out_path = ensure_dir(Path(out_dir))
    camera = HardwareCamera(_camera_config(camera_config))
    rig = StepperMotorRig(
        _axis_configs(axes),
        session_state_path=hardware_config.session_state_path,
        steps_per_revolution=hardware_config.steps_per_revolution,
        max_cumulative_revolutions=hardware_config.max_cumulative_revolutions,
        backlash_compensation=hardware_config.backlash_compensation,
        dry_run=hardware_config.dry_run,
    )
    baseline = rig.load_baseline_state(hardware_config.baseline_state_path)
    hidden_offset = sample_hidden_offset_steps(
        axes,
        seed,
        initial_condition_mode=initial_condition_mode,
    )
    visible_positions = {axis.name: 0 for axis in axes}
    llm = LLM(
        model=llm_config.model,
        save_messages_json_path=os.path.join(out_dir, "messages.json"),
        save_messages_markdown_path=os.path.join(out_dir, "messages.md"),
        save_usage_json_path=os.path.join(out_dir, "llm_usage.json"),
    )
    llm.add_system_message(
        build_real_mi_system_prompt(
            max_iterations=iterations,
            axes=axes,
            steps_per_revolution=hardware_config.steps_per_revolution,
        )
    )

    manifest: dict[str, Any] = {
        "repeat_index": repeat_index,
        "seed": seed,
        "max_iterations": iterations,
        "initial_condition_mode": initial_condition_mode,
        "hidden_offset_steps": hidden_offset,
        "steps": [],
        "dry_run": hardware_config.dry_run,
        "baseline_state_path": hardware_config.baseline_state_path,
        "session_state_path": hardware_config.session_state_path,
        "restore_baseline_at_end": restore_baseline_at_end,
        "completed_early": False,
        "completion_step": None,
    }
    try:
        if repeat_index == 0:
            rig.sync_current_state_to_baseline(baseline)
        return_records = [asdict(record) for record in rig.return_to_baseline(baseline)]
        if return_records:
            _sleep_if_needed(hardware_config.settle_time_sec, dry_run=hardware_config.dry_run)
        write_json(out_path / "baseline_restore.json", {"records": return_records, "positions": rig.state.current_positions})
        baseline_capture = camera.save_capture(
            out_path,
            "baseline",
            extra_metadata={"capture_kind": "baseline"},
            save_raw_array=False,
            save_metadata=False,
        )
        manifest["baseline_capture"] = asdict(baseline_capture)

        hidden_offset_records = [asdict(record) for record in rig.move_many(hidden_offset)]
        _sleep_if_needed(hardware_config.settle_time_sec, dry_run=hardware_config.dry_run)
        write_json(
            out_path / "hidden_offset.json",
            {
                "initial_condition_mode": initial_condition_mode,
                "hidden_offset_steps": hidden_offset,
                "records": hidden_offset_records,
            },
        )

        step0_capture = camera.save_capture(
            out_path,
            "step_0",
            extra_metadata={"step_index": 0, "capture_kind": "initial"},
            save_raw_array=False,
            save_metadata=False,
        )
        manifest["steps"].append(
            {
                "step_index": 0,
                "capture": asdict(step0_capture),
                "logical_positions": dict(visible_positions),
                "visible_positions_steps": dict(visible_positions),
                "true_positions_steps": _true_positions(hidden_offset, visible_positions),
            }
        )
        llm.add_user_image(step0_capture.llm_image_path)
        llm.add_user_message(
            "This is the initial real camera image at visible motor-step setting (0, 0, 0, 0), after hidden random mirror tilts were applied. "
            "Describe the visible beam pattern using normalized coordinates when possible, then propose the first absolute visible motor-step command. "
            'If the beams already appear overlapped, return {"visual_description": "...", "done": true, "command": {...best visible motor-step settings...}}.'
        )

        for step_index in range(1, iterations + 1):
            step_started_at, step_start_perf = start_timer()
            llm_call_count_before = llm_call_count(llm)
            answer = llm.query_llm(
                max_tokens=llm_config.max_tokens,
                effort=llm_config.effort,
                reasoning_max_tokens=llm_config.reasoning_max_tokens,
                require_json=True,
                num_tries=3,
                verify=lambda text: verify_real_motor_command(text, axes),
            )
            payload = json.loads(answer)
            command = _command_from_answer(answer, axes)
            done = bool(payload["done"])
            visual_description = str(payload["visual_description"])
            move_delta = _command_delta(visible_positions, command)
            move_records = [asdict(record) for record in rig.move_many(move_delta)]
            visible_positions = dict(command)
            _sleep_if_needed(hardware_config.settle_time_sec, dry_run=hardware_config.dry_run)
            capture = camera.save_capture(
                out_path,
                f"step_{step_index}",
                extra_metadata={
                    "step_index": step_index,
                    "capture_kind": "iteration",
                    "command": command,
                    "move_delta": move_delta,
                },
                save_raw_array=False,
                save_metadata=False,
            )
            step_record = {
                "step_index": step_index,
                "done": done,
                "visual_description": visual_description,
                "command": command,
                "move_delta": move_delta,
                "move_records": move_records,
                "capture": asdict(capture),
                "logical_positions": dict(visible_positions),
                "visible_positions_steps": dict(visible_positions),
                "true_positions_steps": _true_positions(hidden_offset, visible_positions),
            }
            step_record.update(
                step_timing_fields(
                    step_started_at,
                    step_start_perf,
                    llm_request_wall_time_sec=llm_request_wall_time_since(llm, llm_call_count_before),
                )
            )
            manifest["steps"].append(step_record)
            if done:
                manifest["completed_early"] = True
                manifest["completion_step"] = step_index
                break
            if step_index < iterations:
                llm.add_user_image(capture.llm_image_path)
                llm.add_user_message(
                    "Here is the updated real camera image. "
                    "Describe the current visual evidence quantitatively, compare it to the previous command, "
                    "then propose the next absolute visible motor-step command. "
                    'If you believe the beams are now overlapped well enough, return {"visual_description": "...", "done": true, "command": {...best visible motor-step settings...}}.'
                )

        final_capture = camera.save_capture(
            out_path,
            "step_final",
            extra_metadata={"capture_kind": "final", "step_index": iterations},
            save_raw_array=False,
            save_metadata=False,
        )
        manifest["final_capture"] = asdict(final_capture)
        llm.add_user_image(final_capture.llm_image_path)
        llm.add_user_message("Here is the final resulting real camera image.")
    finally:
        if restore_baseline_at_end:
            try:
                final_return_records = [asdict(record) for record in rig.return_to_baseline(baseline)]
                if final_return_records:
                    _sleep_if_needed(hardware_config.settle_time_sec, dry_run=hardware_config.dry_run)
                restore_payload = {
                    "records": final_return_records,
                    "positions": dict(rig.state.current_positions),
                }
                write_json(out_path / "final_baseline_restore.json", restore_payload)
                manifest["final_baseline_restore"] = restore_payload
            except Exception as exc:  # pragma: no cover - hardware cleanup failure path
                logger.exception("Failed to restore MI baseline at end of run.")
                manifest["final_baseline_restore_error"] = repr(exc)
        camera.close()
        manifest["llm_usage"] = llm.get_usage_summary()
        manifest["llm_usage_path"] = str(out_path / "llm_usage.json")
        add_run_timing(
            manifest,
            run_started_at=run_started_at,
            run_start_perf=run_start_perf,
            steps=manifest["steps"],
        )
        write_json(out_path / "run_manifest.json", manifest)
    return manifest
