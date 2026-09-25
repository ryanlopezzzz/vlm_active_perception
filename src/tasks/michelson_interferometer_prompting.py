"""Prompt and command helpers for the primary Michelson interferometer simulation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, Sequence

from src.tasks.mi_initial_conditions import sample_initial_condition_steps


MI_INTERFEROMETER_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "michelson_interferometer.txt"


class MichelsonInterferometerAxisLike(Protocol):
    name: str


def build_michelson_interferometer_system_prompt(
    *,
    max_iterations: int,
    axes: Sequence[MichelsonInterferometerAxisLike],
    max_steps: int,
    steps_per_revolution: int,
    initial_condition_mode: str,
) -> str:
    axis_limits = "\n".join(
        f"- {axis.name}: integer visible motor-step setting within "
        f"[{-getattr(axis, 'visible_step_limit', max_steps)}, {getattr(axis, 'visible_step_limit', max_steps)}]"
        for axis in axes
    )
    command_shape = ",\n".join(f'        "{axis.name}": int' for axis in axes)
    template = MI_INTERFEROMETER_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template.replace("{{MAX_ITERATIONS}}", str(max_iterations))
        .replace("{{AXIS_LIMITS}}", axis_limits)
        .replace("{{COMMAND_SHAPE}}", command_shape)
        .replace("{{MAX_STEPS}}", str(max_steps))
        .replace("{{STEPS_PER_REVOLUTION}}", str(steps_per_revolution))
        .replace("{{INITIAL_CONDITION_MODE}}", initial_condition_mode)
        .strip()
    )


def verify_michelson_interferometer_command(
    answer: str,
    axes: Sequence[MichelsonInterferometerAxisLike],
    *,
    max_steps: int,
    require_visual_description: bool = True,
) -> bool:
    try:
        payload = json.loads(answer)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    if require_visual_description:
        description = payload.get("visual_description")
        if not isinstance(description, str) or not description.strip():
            return False
    if not isinstance(payload.get("done"), bool):
        return False
    command = payload.get("command")
    if not isinstance(command, dict):
        return False
    axis_names = {axis.name for axis in axes}
    if set(command.keys()) != axis_names:
        return False
    for axis in axes:
        value = command.get(axis.name)
        if not isinstance(value, int) or isinstance(value, bool):
            return False
        if abs(value) > getattr(axis, "visible_step_limit", max_steps):
            return False
    return True


def sample_hidden_offset_steps(
    axes: Sequence[MichelsonInterferometerAxisLike],
    *,
    seed: int,
    initial_condition_mode: str,
    steps_per_revolution: int,
) -> dict[str, int]:
    _ = steps_per_revolution
    return sample_initial_condition_steps(
        axes,
        seed=seed,
        initial_condition_mode=initial_condition_mode,
    )


def hidden_offset_displacements_rot(
    hidden_offset_steps: dict[str, int],
    *,
    steps_per_revolution: int,
) -> dict[str, dict[str, float]]:
    return {
        "mirror_1": {
            "x_rot": hidden_offset_steps["mirror_1_axis_1"] / steps_per_revolution,
            "y_rot": hidden_offset_steps["mirror_1_axis_2"] / steps_per_revolution,
        },
        "mirror_2": {
            "x_rot": hidden_offset_steps["mirror_2_axis_1"] / steps_per_revolution,
            "y_rot": hidden_offset_steps["mirror_2_axis_2"] / steps_per_revolution,
        },
    }
