"""Oracle-objective baselines for the motor-step Michelson simulation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from bayes_opt import BayesianOptimization, acquisition

from src.io.artifacts import ensure_dir, write_json
from src.io.schemas import MichelsonInterferometerAxisConfig, MichelsonInterferometerSimulationConfig
from src.sim.michelson_interferometer_simulation import MichelsonInterferometerSimulation
from src.tasks.michelson_interferometer_alignment import _add_commands, _alignment_metrics, _to_simulation_config
from src.tasks.michelson_interferometer_prompting import hidden_offset_displacements_rot
from src.tasks.timing import add_run_timing, start_timer


OBJECTIVE_FRAME_WIDTH_PX = 1920
OBJECTIVE_FRAME_HEIGHT_PX = 1080


def plotted_center_error_px(metrics: dict[str, Any]) -> float:
    half_width = (OBJECTIVE_FRAME_WIDTH_PX - 1) / 2.0
    half_height = (OBJECTIVE_FRAME_HEIGHT_PX - 1) / 2.0

    def beam_distance(center: dict[str, float]) -> float:
        return float(np.hypot(
            float(center["x"]) * half_width,
            float(center["y"]) * half_height,
        ))

    return (
        beam_distance(metrics["first_center_normalized"])
        + beam_distance(metrics["second_center_normalized"])
    )


def run_michelson_interferometer_oracle_alignment(
    *,
    out_dir: str | Path,
    method: str,
    iterations: int,
    seed: int,
    repeat_index: int,
    hidden_offset_steps: dict[str, int],
    simulation_config: MichelsonInterferometerSimulationConfig,
    axes: list[MichelsonInterferometerAxisConfig],
) -> dict[str, Any]:
    if method not in {"bayes", "random"}:
        raise ValueError("method must be 'bayes' or 'random'.")
    run_started_at, run_start_perf = start_timer()
    out_path = ensure_dir(Path(out_dir))
    simulation = MichelsonInterferometerSimulation(_to_simulation_config(simulation_config))
    axis_names = [axis.name for axis in axes]
    hidden = {axis.name: int(hidden_offset_steps[axis.name]) for axis in axes}
    steps: list[dict[str, Any]] = []
    best_error = float("inf")
    best_command = {name: 0 for name in axis_names}

    def evaluate(command: dict[str, int]) -> float:
        nonlocal best_error, best_command
        command = {
            axis.name: int(np.clip(command[axis.name], -axis.visible_step_limit, axis.visible_step_limit))
            for axis in axes
        }
        true_command = _add_commands(hidden, command)
        centers = simulation.beam_centers(true_command, clip_command=False)
        metrics = _alignment_metrics(simulation.config, centers)
        error = plotted_center_error_px(metrics)
        if error < best_error:
            best_error = error
            best_command = dict(command)
        image_path = out_path / f"step_{len(steps)}.png"
        simulation.save_image(true_command, image_path, clip_command=False)
        steps.append(
            {
                "step_index": len(steps),
                "image_path": str(image_path),
                "visible_command_steps": command,
                "hidden_offset_steps": hidden,
                "true_command_steps": true_command,
                "alignment_metrics": metrics,
                "plotted_center_error_px": error,
                "objective": -error,
                "incumbent_visible_command_steps": dict(best_command),
                "incumbent_plotted_center_error_px": best_error,
            }
        )
        return -error

    zero = {name: 0 for name in axis_names}
    if method == "random":
        evaluate(zero)
        rng = np.random.default_rng(seed)
        for _ in range(iterations):
            evaluate(
                {
                    axis.name: int(rng.integers(-axis.visible_step_limit, axis.visible_step_limit + 1))
                    for axis in axes
                }
            )
    else:
        pbounds = {
            axis.name: (-float(axis.visible_step_limit), float(axis.visible_step_limit))
            for axis in axes
        }

        def objective(**kwargs: float) -> float:
            return evaluate({name: int(round(kwargs[name])) for name in axis_names})

        optimizer = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            acquisition_function=acquisition.ExpectedImprovement(xi=0.01),
            random_state=seed,
            verbose=0,
        )
        initial_target = objective(**{name: 0.0 for name in axis_names})
        optimizer.register({name: 0.0 for name in axis_names}, initial_target)
        optimizer.maximize(init_points=0, n_iter=iterations)

    final_true_command = _add_commands(hidden, best_command)
    final_image_path = out_path / "final_best.png"
    simulation.save_image(final_true_command, final_image_path, clip_command=False)
    final_metrics = _alignment_metrics(
        simulation.config,
        simulation.beam_centers(final_true_command, clip_command=False),
    )
    manifest: dict[str, Any] = {
        "method": method,
        "objective_name": "negative_sum_beam_distances_from_center_px",
        "objective_definition": (
            "Negative sum of both beam-center distances from image center, with normalized "
            f"simulation coordinates expressed at {OBJECTIVE_FRAME_WIDTH_PX} x "
            f"{OBJECTIVE_FRAME_HEIGHT_PX} pixels."
        ),
        "objective_frame_width_px": OBJECTIVE_FRAME_WIDTH_PX,
        "objective_frame_height_px": OBJECTIVE_FRAME_HEIGHT_PX,
        "repeat_index": repeat_index,
        "seed": seed,
        "max_iterations": iterations,
        "hidden_offset_steps": hidden,
        "hidden_offset_displacements_rot": hidden_offset_displacements_rot(
            hidden,
            steps_per_revolution=simulation.config.steps_per_revolution,
        ),
        "simulation": asdict(simulation_config),
        "steps": steps,
        "final_image_path": str(final_image_path),
        "final_visible_command_steps": best_command,
        "final_true_command_steps": final_true_command,
        "final_alignment_metrics": final_metrics,
    }
    add_run_timing(manifest, run_started_at=run_started_at, run_start_perf=run_start_perf, steps=steps)
    write_json(out_path / "run_manifest.json", manifest)
    return manifest
