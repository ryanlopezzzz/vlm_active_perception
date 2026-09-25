"""LLM alignment workflow for the motor-step Michelson simulation."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from src.agents.llm_agent import LLM
from src.io.artifacts import ensure_dir, write_json
from src.io.schemas import LLMConfig, MichelsonInterferometerAxisConfig, MichelsonInterferometerSimulationConfig
from src.sim.michelson_interferometer_simulation import AXIS_NAMES, BeamCenters, MichelsonInterferometerSimulation, MichelsonInterferometerConfig
from src.tasks.michelson_interferometer_prompting import (
    build_michelson_interferometer_system_prompt,
    hidden_offset_displacements_rot,
    sample_hidden_offset_steps,
    verify_michelson_interferometer_command,
)
from src.tasks.timing import add_run_timing, llm_call_count, llm_request_wall_time_since, start_timer, step_timing_fields


def _to_simulation_config(config: MichelsonInterferometerSimulationConfig) -> MichelsonInterferometerConfig:
    payload = asdict(config)
    payload.pop("initial_condition_mode", None)
    return MichelsonInterferometerConfig(**payload)


def _mock_llm_answer(axes: list[MichelsonInterferometerAxisConfig]) -> str:
    return json.dumps(
        {
            "visual_description": "Mock response: no optimization performed.",
            "done": True,
            "command": {axis.name: 0 for axis in axes},
        }
    )


def _command_from_answer(answer: str, axes: list[MichelsonInterferometerAxisConfig]) -> dict[str, int]:
    payload = json.loads(answer)
    return {axis.name: int(payload["command"][axis.name]) for axis in axes}


def _add_commands(
    first: Mapping[str, int | float],
    second: Mapping[str, int | float],
) -> dict[str, int]:
    return {axis: int(round(float(first.get(axis, 0)) + float(second.get(axis, 0)))) for axis in AXIS_NAMES}


def _normalized_center(config: MichelsonInterferometerConfig, x_px: float, y_px: float) -> dict[str, float]:
    return {
        "x": float((x_px - config.center_x_px) / (0.5 * config.frame_width_px)),
        "y": float((config.center_y_px - y_px) / (0.5 * config.frame_height_px)),
    }


def _alignment_metrics(config: MichelsonInterferometerConfig, centers: BeamCenters) -> dict[str, Any]:
    midpoint_x = 0.5 * (centers.first_x_px + centers.second_x_px)
    midpoint_y = 0.5 * (centers.first_y_px + centers.second_y_px)
    center_offset_px = float(((midpoint_x - config.center_x_px) ** 2 + (midpoint_y - config.center_y_px) ** 2) ** 0.5)
    separation_px = centers.separation_px
    simple_score_px = separation_px + center_offset_px
    diagonal_px = float((config.frame_width_px**2 + config.frame_height_px**2) ** 0.5)
    first_normalized = _normalized_center(config, centers.first_x_px, centers.first_y_px)
    second_normalized = _normalized_center(config, centers.second_x_px, centers.second_y_px)
    center_error_normalized = math.sqrt(
        (
            first_normalized["x"] ** 2
            + first_normalized["y"] ** 2
            + second_normalized["x"] ** 2
            + second_normalized["y"] ** 2
        )
        / 2.0
    )
    return {
        "beam_separation_px": separation_px,
        "beam_center_offset_px": center_offset_px,
        "simple_score_px": simple_score_px,
        "beam_separation_normalized": separation_px / diagonal_px,
        "beam_center_offset_normalized": center_offset_px / diagonal_px,
        "simple_score_normalized": simple_score_px / diagonal_px,
        "first_center_px": {
            "x": float(centers.first_x_px),
            "y": float(centers.first_y_px),
        },
        "second_center_px": {
            "x": float(centers.second_x_px),
            "y": float(centers.second_y_px),
        },
        "midpoint_px": {
            "x": float(midpoint_x),
            "y": float(midpoint_y),
        },
        "first_center_normalized": first_normalized,
        "second_center_normalized": second_normalized,
        "center_error_normalized": center_error_normalized,
        "midpoint_normalized": _normalized_center(config, midpoint_x, midpoint_y),
    }


def _step_record(
    *,
    simulation: MichelsonInterferometerSimulation,
    hidden_offset_steps: dict[str, int],
    visible_command_steps: dict[str, int],
    step_index: int,
    image_path: Path,
    done: bool | None = None,
    command: dict[str, int] | None = None,
    visual_description: str | None = None,
) -> dict[str, Any]:
    true_command_steps = _add_commands(hidden_offset_steps, visible_command_steps)
    rendered_true_command_steps = true_command_steps
    centers = simulation.beam_centers(rendered_true_command_steps, clip_command=False)
    record: dict[str, Any] = {
        "step_index": step_index,
        "image_path": str(image_path),
        "visible_command_steps": dict(visible_command_steps),
        "hidden_offset_steps": dict(hidden_offset_steps),
        "true_command_steps": true_command_steps,
        "rendered_true_command_steps": rendered_true_command_steps,
        "alignment_metrics": _alignment_metrics(simulation.config, centers),
    }
    if done is not None:
        record["done"] = done
    if command is not None:
        record["command"] = command
    if visual_description is not None:
        record["visual_description"] = visual_description
    return record


def run_michelson_interferometer_llm_alignment(
    *,
    out_dir: str | Path,
    iterations: int,
    seed: int,
    repeat_index: int,
    llm_config: LLMConfig,
    simulation_config: MichelsonInterferometerSimulationConfig,
    axes: list[MichelsonInterferometerAxisConfig],
    hidden_offset_steps_override: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    run_started_at, run_start_perf = start_timer()
    out_path = ensure_dir(Path(out_dir))
    simulation = MichelsonInterferometerSimulation(_to_simulation_config(simulation_config))
    max_steps = simulation.config.max_steps
    hidden_offset_steps = (
        {axis.name: int(hidden_offset_steps_override[axis.name]) for axis in axes}
        if hidden_offset_steps_override is not None
        else sample_hidden_offset_steps(
            axes,
            seed=seed,
            initial_condition_mode=simulation_config.initial_condition_mode,
            steps_per_revolution=simulation.config.steps_per_revolution,
        )
    )
    visible_command_steps = {axis.name: 0 for axis in axes}

    llm = LLM(
        model=llm_config.model,
        save_messages_json_path=os.path.join(out_dir, "messages.json"),
        save_messages_markdown_path=os.path.join(out_dir, "messages.md"),
        save_usage_json_path=os.path.join(out_dir, "llm_usage.json"),
    )
    llm.add_system_message(
        build_michelson_interferometer_system_prompt(
            max_iterations=iterations,
            axes=axes,
            max_steps=max_steps,
            steps_per_revolution=simulation.config.steps_per_revolution,
            initial_condition_mode=simulation_config.initial_condition_mode,
        )
    )

    manifest: dict[str, Any] = {
        "repeat_index": repeat_index,
        "seed": seed,
        "max_iterations": iterations,
        "initial_condition_mode": simulation_config.initial_condition_mode,
        "hidden_offset_steps": hidden_offset_steps,
        "hidden_offset_displacements_rot": hidden_offset_displacements_rot(
            hidden_offset_steps,
            steps_per_revolution=simulation.config.steps_per_revolution,
        ),
        "steps": [],
        "simulation": asdict(simulation_config),
        "completed_early": False,
        "completion_step": None,
    }
    write_json(
        out_path / "hidden_offset.json",
        {
            "hidden_offset_steps": hidden_offset_steps,
            "hidden_offset_displacements_rot": manifest["hidden_offset_displacements_rot"],
        },
    )

    step0_path = out_path / "step_0.png"
    simulation.save_image(
        _add_commands(hidden_offset_steps, visible_command_steps),
        step0_path,
        clip_command=False,
    )
    manifest["steps"].append(
        _step_record(
            simulation=simulation,
            hidden_offset_steps=hidden_offset_steps,
            visible_command_steps=visible_command_steps,
            step_index=0,
            image_path=step0_path,
        )
    )
    latest_image_path = step0_path
    llm.add_user_image(str(step0_path))
    llm.add_user_message(
        "This is the initial simulated camera image with all visible motor-step settings at 0. "
        "Describe the visible beam pattern using normalized coordinates when possible, then propose the first absolute command. "
        "Remember that the objective is to overlap both beams at the camera center, not to move them offscreen."
    )

    for step_index in range(1, iterations + 1):
        step_started_at, step_start_perf = start_timer()
        llm_call_count_before = llm_call_count(llm)
        if llm_config.model.startswith("mock"):
            answer = _mock_llm_answer(axes)
        else:
            answer = llm.query_llm(
                max_tokens=llm_config.max_tokens,
                effort=llm_config.effort,
                reasoning_max_tokens=llm_config.reasoning_max_tokens,
                require_json=True,
                num_tries=10,
                verify=lambda text: verify_michelson_interferometer_command(
                    text,
                    axes,
                    max_steps=max_steps,
                    require_visual_description=True,
                ),
            )
        payload = json.loads(answer)
        command = _command_from_answer(answer, axes)
        done = bool(payload["done"])
        visual_description = str(payload["visual_description"])
        visible_command_steps = {
            axis.name: max(-axis.visible_step_limit, min(axis.visible_step_limit, command[axis.name]))
            for axis in axes
        }

        image_path = out_path / f"step_{step_index}.png"
        simulation.save_image(
            _add_commands(hidden_offset_steps, visible_command_steps),
            image_path,
            clip_command=False,
        )
        step_record = _step_record(
            simulation=simulation,
            hidden_offset_steps=hidden_offset_steps,
            visible_command_steps=visible_command_steps,
            step_index=step_index,
            image_path=image_path,
            done=done,
            command=command,
            visual_description=visual_description,
        )
        step_record.update(
            step_timing_fields(
                step_started_at,
                step_start_perf,
                llm_request_wall_time_sec=llm_request_wall_time_since(llm, llm_call_count_before),
            )
        )
        manifest["steps"].append(step_record)
        latest_image_path = image_path
        if done:
            manifest["completed_early"] = True
            manifest["completion_step"] = step_index
            break
        if step_index < iterations:
            llm.add_user_image(str(image_path))
            llm.add_user_message(
                "Here is the updated simulated camera image. "
                "Describe the current visual evidence quantitatively, compare it to the previous command, "
                "then propose the next absolute motor-step command."
            )

    llm.add_user_image(str(latest_image_path))
    llm.add_user_message("Here is the final resulting simulated camera image for the final visible command you selected.")

    manifest["final_image_path"] = str(latest_image_path)
    manifest["final_visible_command_steps"] = dict(visible_command_steps)
    manifest["final_true_command_steps"] = _add_commands(hidden_offset_steps, visible_command_steps)
    manifest["final_rendered_true_command_steps"] = dict(manifest["final_true_command_steps"])
    manifest["final_alignment_metrics"] = manifest["steps"][-1]["alignment_metrics"]
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
