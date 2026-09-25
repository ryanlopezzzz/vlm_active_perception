"""Shared hidden-offset sampling for Michelson motor-step experiments."""

from __future__ import annotations

import random
from typing import Protocol, Sequence


INITIAL_CONDITION_MODES = {
    "random",
    "both_beams_onscreen",
    "one_beam_onscreen",
    "no_beams_onscreen",
}

MI_BEAM_AXIS_NAMES = (
    "mirror_1_axis_1",
    "mirror_1_axis_2",
    "mirror_2_axis_1",
    "mirror_2_axis_2",
)

RANDOM_LIMIT_STEPS = 8192
ONSCREEN_X_LIMIT_STEPS = 4096
ONSCREEN_Y_LIMIT_STEPS = 2048
VISIBLE_X_LIMIT_STEPS = 5120
VISIBLE_Y_LIMIT_STEPS = 3072


class AxisNameLike(Protocol):
    name: str


def sample_initial_condition_steps(
    axes: Sequence[AxisNameLike],
    *,
    seed: int,
    initial_condition_mode: str,
) -> dict[str, int]:
    _validate_axes(axes)
    if initial_condition_mode not in INITIAL_CONDITION_MODES:
        raise ValueError(f"Unsupported initial_condition_mode: {initial_condition_mode}")

    rng = random.Random(seed)
    if initial_condition_mode == "random":
        return {axis: rng.randint(-RANDOM_LIMIT_STEPS, RANDOM_LIMIT_STEPS) for axis in MI_BEAM_AXIS_NAMES}

    if initial_condition_mode == "both_beams_onscreen":
        mirror_1 = _sample_onscreen_beam(rng)
        mirror_2 = _sample_onscreen_beam(rng)
    elif initial_condition_mode == "one_beam_onscreen":
        offscreen_mirror = rng.choice(("mirror_1", "mirror_2"))
        mirror_1 = _sample_offscreen_beam(rng) if offscreen_mirror == "mirror_1" else _sample_onscreen_beam(rng)
        mirror_2 = _sample_offscreen_beam(rng) if offscreen_mirror == "mirror_2" else _sample_onscreen_beam(rng)
    else:
        mirror_1 = _sample_offscreen_beam(rng)
        mirror_2 = _sample_offscreen_beam(rng)

    return {
        "mirror_1_axis_1": mirror_1[0],
        "mirror_1_axis_2": mirror_1[1],
        "mirror_2_axis_1": mirror_2[0],
        "mirror_2_axis_2": mirror_2[1],
    }


def _validate_axes(axes: Sequence[AxisNameLike]) -> None:
    names = {axis.name for axis in axes}
    missing = set(MI_BEAM_AXIS_NAMES) - names
    if missing:
        raise ValueError(f"Missing required MI axes for initial condition sampling: {sorted(missing)}")


def _sample_onscreen_beam(rng: random.Random) -> tuple[int, int]:
    return (
        rng.randint(-ONSCREEN_X_LIMIT_STEPS, ONSCREEN_X_LIMIT_STEPS),
        rng.randint(-ONSCREEN_Y_LIMIT_STEPS, ONSCREEN_Y_LIMIT_STEPS),
    )


def _sample_offscreen_beam(rng: random.Random) -> tuple[int, int]:
    for _ in range(10_000):
        x = rng.randint(-RANDOM_LIMIT_STEPS, RANDOM_LIMIT_STEPS)
        y = rng.randint(-RANDOM_LIMIT_STEPS, RANDOM_LIMIT_STEPS)
        if abs(x) > VISIBLE_X_LIMIT_STEPS or abs(y) > VISIBLE_Y_LIMIT_STEPS:
            return (x, y)
    raise RuntimeError("Failed to sample an offscreen beam after 10,000 attempts.")
