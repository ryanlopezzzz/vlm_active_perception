"""Shared prompt and command helpers for real-style Michelson alignment flows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, Sequence

from src.tasks.mi_initial_conditions import sample_initial_condition_steps


class PromptAxisLike(Protocol):
    name: str
    llm_step_limit: int


class OffsetAxisLike(PromptAxisLike, Protocol):
    hidden_offset_max_steps: int


MI_REAL_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "michelson_interferometer.txt"


def build_real_mi_system_prompt(
    *,
    max_iterations: int,
    axes: Sequence[PromptAxisLike],
    steps_per_revolution: int = 4096,
) -> str:
    limits = "\n".join(
        f"- {axis.name}: integer visible motor-step setting within [{-axis.llm_step_limit}, {axis.llm_step_limit}]"
        for axis in axes
    )
    command_shape = ",\n".join(f'        "{axis.name}": int' for axis in axes)
    optimal_range_description = _optimal_range_description(axes)
    max_steps = max((int(axis.llm_step_limit) for axis in axes), default=0)

    template = MI_REAL_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template.replace("{{MAX_ITERATIONS}}", str(max_iterations))
        .replace("{{AXIS_LIMITS}}", limits)
        .replace("{{COMMAND_SHAPE}}", command_shape)
        .replace("{{MAX_STEPS}}", str(max_steps))
        .replace("{{STEP_GUIDANCE}}", "")
        .replace("{{OPTIMAL_RANGE_DESCRIPTION}}", optimal_range_description)
        .replace("{{STEPS_PER_REVOLUTION}}", str(steps_per_revolution))
        .strip()
    )


def _optimal_range_description(axes: Sequence[PromptAxisLike]) -> str:
    ranges = [
        f"- {axis.name}: optimal hidden offset lies within +/-{int(axis.hidden_offset_max_steps)} motor steps."
        for axis in axes
        if hasattr(axis, "hidden_offset_max_steps")
    ]
    if ranges:
        return "\n".join(ranges)
    return "The optimal hidden offset for each mirror axis lies within a bounded motor-step range."


def verify_real_motor_command(answer: str, axes: Sequence[PromptAxisLike]) -> bool:
    try:
        payload = json.loads(answer)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict) or "command" not in payload:
        return False
    description = payload.get("visual_description")
    if not isinstance(description, str) or not description.strip():
        return False
    if "done" not in payload or not isinstance(payload["done"], bool):
        return False
    command = payload["command"]
    if not isinstance(command, dict):
        return False
    axis_names = {axis.name for axis in axes}
    if set(command.keys()) != axis_names:
        return False
    for axis in axes:
        value = command.get(axis.name)
        if not isinstance(value, int) or isinstance(value, bool):
            return False
        if abs(value) > axis.llm_step_limit:
            return False
    return True


def sample_hidden_offset_steps(
    axes: Sequence[OffsetAxisLike],
    seed: int,
    initial_condition_mode: str = "random",
) -> dict[str, int]:
    return sample_initial_condition_steps(
        axes,
        seed=seed,
        initial_condition_mode=initial_condition_mode,
    )
