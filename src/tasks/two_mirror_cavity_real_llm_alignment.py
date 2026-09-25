"""Real-hardware, batched LLM optimization for the two-mirror cavity."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Sequence

from src.agents.llm_agent import LLM
from src.hardware.mindvision_camera import MindVisionCamera, MindVisionCameraConfig
from src.hardware.motors import AxisMotorConfig, BaselineState, StepperMotorRig
from src.io.artifacts import ensure_dir, write_json
from src.io.schemas import (
    LLMConfig,
    TwoMirrorCavityCameraConfig,
    TwoMirrorCavityHardwareConfig,
    TwoMirrorCavityRealLLMAxisConfig,
)
from src.tasks.timing import add_run_timing, llm_call_count, start_timer
from src.tasks.two_mirror_cavity_real_llm_artifacts import organize_and_annotate_repeat
from src.tasks.two_mirror_cavity_real_llm_prompting import build_system_prompt, verify_response


def absolute_visible_to_physical(
    hidden_offset: dict[str, int], absolute_visible: dict[str, int]
) -> dict[str, int]:
    return {role: int(hidden_offset[role]) + int(absolute_visible[role]) for role in ("x", "y")}


def absolute_target_delta(
    current_physical: dict[str, int], target_physical: dict[str, int]
) -> dict[str, int]:
    return {role: int(target_physical[role]) - int(current_physical[role]) for role in ("x", "y")}


def _sleep_if_needed(config: TwoMirrorCavityHardwareConfig) -> None:
    if config.settle_time_sec > 0 and not config.dry_run:
        time.sleep(config.settle_time_sec)


def _position_by_role(
    rig: StepperMotorRig, axes: Sequence[TwoMirrorCavityRealLLMAxisConfig]
) -> dict[str, int]:
    return {axis.role: int(rig.state.current_positions[axis.name]) for axis in axes}


def _move_to_visible(
    rig: StepperMotorRig,
    axes: Sequence[TwoMirrorCavityRealLLMAxisConfig],
    hidden_offset: dict[str, int],
    visible_target: dict[str, int],
    hardware_config: TwoMirrorCavityHardwareConfig,
) -> tuple[dict[str, int], dict[str, int], list[dict[str, Any]]]:
    physical_target = absolute_visible_to_physical(hidden_offset, visible_target)
    current = _position_by_role(rig, axes)
    delta = absolute_target_delta(current, physical_target)
    records = []
    for axis in axes:
        if delta[axis.role] != 0:
            records.append(asdict(rig.move_axis(axis.name, delta[axis.role])))
    if records:
        _sleep_if_needed(hardware_config)
    return physical_target, delta, records


def _add_labeled_images(
    llm: LLM,
    captures: Sequence[dict[str, Any]],
    *,
    heading: str,
) -> None:
    llm.add_user_message(heading)
    for index, item in enumerate(captures, start=1):
        command = item["absolute_visible_command"]
        llm.add_user_message(
            f"Image {index}: absolute visible command x={command['x']}, y={command['y']}."
        )
        llm.add_user_image(item["capture"]["llm_image_path"])


def run_two_mirror_cavity_real_llm_alignment(
    *,
    out_dir: str | Path,
    repeat_index: int,
    seed: int,
    hidden_offset: dict[str, int],
    iterations: int,
    probes_per_batch: int,
    llm_config: LLMConfig,
    camera_config: TwoMirrorCavityCameraConfig,
    hardware_config: TwoMirrorCavityHardwareConfig,
    axes: list[TwoMirrorCavityRealLLMAxisConfig],
    reset_session: bool = False,
) -> dict[str, Any]:
    run_started_at, run_start_perf = start_timer()
    out_path = ensure_dir(Path(out_dir))
    original_image_path = ensure_dir(out_path / "images" / "original")
    camera = MindVisionCamera(MindVisionCameraConfig(**asdict(camera_config)))
    rig = StepperMotorRig(
        [AxisMotorConfig(axis.name, axis.controller_ip, axis.motor_number, axis.direction_sign) for axis in axes],
        session_state_path=hardware_config.session_state_path,
        steps_per_revolution=hardware_config.steps_per_revolution,
        max_cumulative_revolutions=hardware_config.max_cumulative_revolutions,
        backlash_compensation=hardware_config.backlash_compensation,
        dry_run=hardware_config.dry_run,
    )
    if reset_session:
        rig.reset_session()
    baseline = BaselineState(baseline_positions={axis.name: 0 for axis in axes})
    llm = LLM(
        model=llm_config.model,
        save_messages_json_path=os.path.join(out_dir, "messages.json"),
        save_messages_markdown_path=os.path.join(out_dir, "messages.md"),
        save_usage_json_path=os.path.join(out_dir, "llm_usage.json"),
    )
    llm.add_system_message(
        build_system_prompt(
            axes,
            max_probe_batches=iterations,
            probes_per_batch=probes_per_batch,
        )
    )
    manifest: dict[str, Any] = {
        "experiment_type": "two_mirror_cavity_real_llm_runs",
        "repeat_index": repeat_index,
        "seed": seed,
        "hidden_offset_steps": dict(hidden_offset),
        "max_probe_batches": iterations,
        "probes_per_batch": probes_per_batch,
        "dry_run": hardware_config.dry_run,
        "batches": [],
        "decisions": [],
        "completed_early": False,
        "completion_batch": None,
        "restoration": {"attempted": False, "succeeded": False},
    }
    camera_open = False
    final_visible: dict[str, int] | None = None
    final_physical: dict[str, int] | None = None
    current_input_commands: list[dict[str, int]] = [{"x": 0, "y": 0}]
    primary_error: BaseException | None = None
    try:
        camera.open()
        camera_open = True
        baseline_capture = camera.save_capture(
            original_image_path, "physical_origin_before", extra_metadata={"capture_kind": "physical_origin_before"},
            save_raw_array=False,
        )
        manifest["physical_origin_before_capture"] = asdict(baseline_capture)

        hidden_named = {axis.name: int(hidden_offset[axis.role]) for axis in axes}
        hidden_moves = [asdict(record) for record in rig.move_many(hidden_named)]
        _sleep_if_needed(hardware_config)
        write_json(out_path / "hidden_offset.json", {"hidden_offset_steps": hidden_offset, "moves": hidden_moves})

        initial_capture = camera.save_capture(
            original_image_path,
            "initial_visible_x_+00000_y_+00000",
            extra_metadata={
                "capture_kind": "initial",
                "absolute_visible_command": {"x": 0, "y": 0},
                "physical_position_steps": _position_by_role(rig, axes),
            },
            save_raw_array=False,
        )
        initial_item = {
            "absolute_visible_command": {"x": 0, "y": 0},
            "physical_position_steps": _position_by_role(rig, axes),
            "capture": asdict(initial_capture),
        }
        manifest["initial_capture"] = initial_item
        _add_labeled_images(
            llm,
            [initial_item],
            heading=(
                "This is the initial image at absolute visible command (0, 0). "
                f"Analyze it and either finish or propose the first {probes_per_batch} absolute probes."
            ),
        )

        for batch_index in range(1, iterations + 1):
            answer = llm.query_llm(
                max_tokens=llm_config.max_tokens,
                effort=llm_config.effort,
                reasoning_max_tokens=llm_config.reasoning_max_tokens,
                require_json=True,
                num_tries=3,
                verify=lambda text, expected=list(current_input_commands): verify_response(
                    text,
                    axes,
                    expected_observation_count=len(expected),
                    expected_commands=expected,
                    probes_per_batch=probes_per_batch,
                ),
            )
            payload = json.loads(answer)
            manifest["decisions"].append(
                {"input_batch": batch_index - 1, "forced_final": False, "response": payload}
            )
            if payload["done"]:
                final_visible = dict(payload["final_command"])
                manifest["completed_early"] = True
                manifest["completion_batch"] = batch_index - 1
                break

            batch_captures: list[dict[str, Any]] = []
            for probe_index, command in enumerate(payload["probe_commands"], start=1):
                visible_command = {"x": int(command["x"]), "y": int(command["y"])}
                physical_target, delta, move_records = _move_to_visible(
                    rig, axes, hidden_offset, visible_command, hardware_config
                )
                stem = (
                    f"batch_{batch_index:02d}_probe_{probe_index:02d}_"
                    f"x_{visible_command['x']:+06d}_y_{visible_command['y']:+06d}"
                )
                capture = camera.save_capture(
                    original_image_path,
                    stem,
                    extra_metadata={
                        "capture_kind": "probe",
                        "batch_index": batch_index,
                        "probe_index": probe_index,
                        "absolute_visible_command": visible_command,
                        "physical_target_steps": physical_target,
                        "motor_delta_steps": delta,
                    },
                    save_raw_array=False,
                )
                batch_captures.append(
                    {
                        "probe_index": probe_index,
                        "absolute_visible_command": visible_command,
                        "physical_target_steps": physical_target,
                        "motor_delta_steps": delta,
                        "move_records": move_records,
                        "capture": asdict(capture),
                    }
                )
            manifest["batches"].append(
                {
                    "batch_index": batch_index,
                    "llm_response": payload,
                    "captures": batch_captures,
                }
            )
            current_input_commands = [dict(item["absolute_visible_command"]) for item in batch_captures]
            _add_labeled_images(
                llm,
                batch_captures,
                heading=(
                    f"These are the {probes_per_batch} results from probe batch {batch_index}. "
                    + (
                        "This was the final allowed probe batch. You must now return done=true with your best final absolute command."
                        if batch_index == iterations
                        else (
                            f"Analyze all {probes_per_batch} in order, then either finish or propose "
                            f"{probes_per_batch} new absolute probes."
                        )
                    )
                ),
            )
            if batch_index == iterations:
                answer = llm.query_llm(
                    max_tokens=llm_config.max_tokens,
                    effort=llm_config.effort,
                    reasoning_max_tokens=llm_config.reasoning_max_tokens,
                    require_json=True,
                    num_tries=3,
                    verify=lambda text, expected=list(current_input_commands): verify_response(
                        text,
                        axes,
                        expected_observation_count=len(expected),
                        expected_commands=expected,
                        probes_per_batch=probes_per_batch,
                        force_final=True,
                    ),
                )
                final_payload = json.loads(answer)
                manifest["decisions"].append(
                    {"input_batch": iterations, "forced_final": True, "response": final_payload}
                )
                final_visible = dict(final_payload["final_command"])
                manifest["forced_final_response"] = final_payload
                manifest["completion_batch"] = iterations

        if final_visible is None:
            raise RuntimeError("LLM optimization ended without a final absolute command.")
        final_physical, final_delta, final_moves = _move_to_visible(
            rig, axes, hidden_offset, final_visible, hardware_config
        )
        final_capture = camera.save_capture(
            original_image_path,
            f"final_x_{final_visible['x']:+06d}_y_{final_visible['y']:+06d}",
            extra_metadata={
                "capture_kind": "final",
                "absolute_visible_command": final_visible,
                "physical_target_steps": final_physical,
                "motor_delta_steps": final_delta,
            },
            save_raw_array=False,
        )
        by_role = {axis.role: axis for axis in axes}
        predicted_x = by_role["x"].calibration_slope_normalized_per_step * final_physical["x"]
        predicted_y = by_role["y"].calibration_slope_normalized_per_step * final_physical["y"]
        manifest["final"] = {
            "absolute_visible_command": final_visible,
            "physical_position_steps": final_physical,
            "motor_delta_steps": final_delta,
            "move_records": final_moves,
            "capture": asdict(final_capture),
        }
        manifest["evaluation"] = {
            "final_physical_error_steps": final_physical,
            "euclidean_motor_error_steps": math.hypot(final_physical["x"], final_physical["y"]),
            "predicted_normalized_displacement": {"x": predicted_x, "y": predicted_y},
            "predicted_normalized_distance": math.hypot(predicted_x, predicted_y),
            "probe_batches": len(manifest["batches"]),
            "probe_images": sum(len(batch["captures"]) for batch in manifest["batches"]),
        }
    except BaseException as exc:
        primary_error = exc
        manifest["run_error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        restore_error: Exception | None = None
        manifest["restoration"]["attempted"] = True
        try:
            restore_records = [asdict(record) for record in rig.return_to_baseline(baseline)]
            if restore_records:
                _sleep_if_needed(hardware_config)
            manifest["restoration"].update(
                {
                    "succeeded": True,
                    "move_records": restore_records,
                    "physical_position_after": _position_by_role(rig, axes),
                }
            )
            if camera_open:
                restored_capture = camera.save_capture(
                    original_image_path,
                    "physical_origin_restored",
                    extra_metadata={"capture_kind": "physical_origin_restored"},
                    save_raw_array=False,
                )
                manifest["restoration"]["capture"] = asdict(restored_capture)
        except Exception as exc:
            restore_error = exc
            manifest["restoration"].update(
                {"succeeded": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
            )
        try:
            camera.close()
        except Exception as exc:
            manifest["camera_close_error"] = {"type": type(exc).__name__, "message": str(exc)}
            if restore_error is None:
                restore_error = exc
        manifest["llm_usage"] = llm.get_usage_summary()
        manifest["llm_call_count"] = llm_call_count(llm)
        manifest["camera_image_count"] = (
            int("physical_origin_before_capture" in manifest)
            + int("initial_capture" in manifest)
            + sum(len(batch["captures"]) for batch in manifest["batches"])
            + int("final" in manifest)
            + int("capture" in manifest["restoration"])
        )
        add_run_timing(manifest, run_started_at=run_started_at, run_start_perf=run_start_perf)
        artifact_error: Exception | None = None
        try:
            organize_and_annotate_repeat(out_path, manifest, axes, move_existing=False)
        except Exception as exc:
            artifact_error = exc
            manifest["artifact_processing_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        write_json(out_path / "run_manifest.json", manifest)
        if restore_error is not None and primary_error is None:
            raise RuntimeError(f"Failed to restore the cavity to physical origin: {restore_error}") from restore_error
        if artifact_error is not None and primary_error is None:
            raise RuntimeError(f"Failed to organize or annotate cavity images: {artifact_error}") from artifact_error
    return manifest
