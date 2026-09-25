"""Prompt, response validation, and sampling for the real cavity optimizer."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Sequence

from src.io.schemas import TwoMirrorCavityRealLLMAxisConfig


VISIBILITY_VALUES = {"visible", "overlapped_or_hidden", "offscreen", "unknown"}
CONFIDENCE_VALUES = {"low", "medium", "high"}
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "two_mirror_cavity.txt"


def build_system_prompt(
    axes: Sequence[TwoMirrorCavityRealLLMAxisConfig],
    *,
    max_probe_batches: int,
    probes_per_batch: int,
) -> str:
    by_role = {axis.role: axis for axis in axes}
    x_axis = by_role["x"]
    y_axis = by_role["y"]
    example_commands = _example_probe_commands(x_axis, y_axis, probes_per_batch)
    template = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template.replace("{{X_MOTOR_NUMBER}}", str(x_axis.motor_number))
        .replace("{{X_MIN_STEPS}}", str(x_axis.llm_min_steps))
        .replace("{{X_MAX_STEPS}}", str(x_axis.llm_max_steps))
        .replace("{{Y_MOTOR_NUMBER}}", str(y_axis.motor_number))
        .replace("{{Y_MIN_STEPS}}", str(y_axis.llm_min_steps))
        .replace("{{Y_MAX_STEPS}}", str(y_axis.llm_max_steps))
        .replace("{{MAX_PROBE_BATCHES}}", str(max_probe_batches))
        .replace("{{PROBES_PER_BATCH}}", str(probes_per_batch))
        .replace("{{PROBE_COMMAND_EXAMPLE}}", json.dumps(example_commands, indent=4))
        .strip()
    )


def _example_probe_commands(
    x_axis: TwoMirrorCavityRealLLMAxisConfig,
    y_axis: TwoMirrorCavityRealLLMAxisConfig,
    count: int,
) -> list[dict[str, int]]:
    """Build a bounded, distinct JSON example with the configured batch size."""
    x_center = min(max(0, x_axis.llm_min_steps), x_axis.llm_max_steps)
    y_center = min(max(0, y_axis.llm_min_steps), y_axis.llm_max_steps)
    preferred = [
        (x_axis.llm_min_steps, y_center),
        (x_axis.llm_max_steps, y_center),
        (x_center, y_axis.llm_min_steps),
        (x_center, y_axis.llm_max_steps),
        (x_axis.llm_max_steps, y_axis.llm_max_steps),
        (x_axis.llm_min_steps, y_axis.llm_max_steps),
        (x_axis.llm_max_steps, y_axis.llm_min_steps),
        (x_axis.llm_min_steps, y_axis.llm_min_steps),
    ]
    commands: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()

    def add(x: int, y: int) -> None:
        if (x, y) not in seen and len(commands) < count:
            seen.add((x, y))
            commands.append({"x": x, "y": y})

    for x, y in preferred:
        add(x, y)
    if len(commands) < count:
        for x in range(x_axis.llm_min_steps, x_axis.llm_max_steps + 1):
            for y in range(y_axis.llm_min_steps, y_axis.llm_max_steps + 1):
                add(x, y)
                if len(commands) == count:
                    return commands
    return commands


def verify_response(
    answer: str,
    axes: Sequence[TwoMirrorCavityRealLLMAxisConfig],
    *,
    expected_observation_count: int,
    expected_commands: Sequence[dict[str, int]] | None = None,
    probes_per_batch: int,
    force_final: bool = False,
) -> bool:
    try:
        payload = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        return False
    required = {
        "visual_description",
        "image_observations",
        "strategy_summary",
        "done",
        "probe_commands",
        "final_command",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        return False
    if not isinstance(payload["done"], bool) or (force_final and not payload["done"]):
        return False

    observations = payload["image_observations"]
    if not isinstance(observations, list) or len(observations) != expected_observation_count:
        return False
    expected_indices = list(range(1, expected_observation_count + 1))
    if [item.get("image_index") if isinstance(item, dict) else None for item in observations] != expected_indices:
        return False
    for observation in observations:
        if not _valid_observation(observation, axes):
            return False
    if expected_commands is not None and [item["command"] for item in observations] != list(expected_commands):
        return False

    probes = payload["probe_commands"]
    if payload["done"]:
        return isinstance(probes, list) and not probes and _valid_command(payload["final_command"], axes)
    if payload["final_command"] is not None or not isinstance(probes, list) or len(probes) != probes_per_batch:
        return False
    if not all(_valid_command(command, axes) for command in probes):
        return False
    return len({(command["x"], command["y"]) for command in probes}) == probes_per_batch


def sample_distinct_hidden_offsets(
    axes: Sequence[TwoMirrorCavityRealLLMAxisConfig],
    *,
    repeats: int,
    base_seed: int,
) -> list[dict[str, int]]:
    by_role = {axis.role: axis for axis in axes}
    rng = random.Random(base_seed)
    offsets: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for _ in range(10_000):
        candidate = {
            "x": rng.randint(by_role["x"].hidden_offset_min_steps, by_role["x"].hidden_offset_max_steps),
            "y": rng.randint(by_role["y"].hidden_offset_min_steps, by_role["y"].hidden_offset_max_steps),
        }
        key = (candidate["x"], candidate["y"])
        if key not in seen:
            seen.add(key)
            offsets.append(candidate)
            if len(offsets) == repeats:
                return offsets
    raise RuntimeError("Unable to sample the requested number of distinct hidden offsets.")


def _valid_observation(value: object, axes: Sequence[TwoMirrorCavityRealLLMAxisConfig]) -> bool:
    required = {"image_index", "command", "visibility", "x", "y", "confidence", "evidence"}
    if not isinstance(value, dict) or set(value) != required or not isinstance(value["image_index"], int):
        return False
    if not _valid_command(value["command"], axes):
        return False
    visibility = value["visibility"]
    if visibility not in VISIBILITY_VALUES or value["confidence"] not in CONFIDENCE_VALUES:
        return False
    if not _nonempty_string(value["evidence"]):
        return False
    if visibility == "visible":
        return _normalized(value["x"]) and _normalized(value["y"])
    if visibility == "overlapped_or_hidden":
        coordinates_are_null = value["x"] is None and value["y"] is None
        coordinates_are_normalized = _normalized(value["x"]) and _normalized(value["y"])
        return coordinates_are_null or coordinates_are_normalized
    return value["x"] is None and value["y"] is None


def _valid_command(value: object, axes: Sequence[TwoMirrorCavityRealLLMAxisConfig]) -> bool:
    if not isinstance(value, dict) or set(value) != {"x", "y"}:
        return False
    by_role = {axis.role: axis for axis in axes}
    for role in ("x", "y"):
        coordinate = value[role]
        axis = by_role[role]
        if not isinstance(coordinate, int) or isinstance(coordinate, bool):
            return False
        if coordinate < axis.llm_min_steps or coordinate > axis.llm_max_steps:
            return False
    return True


def _normalized(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and -1.0 <= float(value) <= 1.0


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())
