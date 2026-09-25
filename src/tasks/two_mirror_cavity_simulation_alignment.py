"""Shared rendering, metrics, and optimizers for the empirical cavity simulation."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from bayes_opt import BayesianOptimization, acquisition
from PIL import Image

from src.agents.llm_agent import LLM
from src.io.artifacts import ensure_dir, write_json
from src.io.schemas import LLMConfig, TwoMirrorCavityRealLLMAxisConfig
from src.sim.two_mirror_cavity_simulation import TwoMirrorCavityConfig, TwoMirrorCavitySimulation
from src.tasks.timing import add_run_timing, start_timer
from src.tasks.two_mirror_cavity_real_llm_prompting import build_system_prompt, verify_response


def _physical_command(hidden: dict[str, int], visible: dict[str, int]) -> dict[str, int]:
    return {role: int(hidden[role]) + int(visible[role]) for role in ("x", "y")}


def motor_distance_steps(physical: dict[str, int]) -> float:
    return float(np.hypot(float(physical["x"]), float(physical["y"])))


def _metrics(simulation: TwoMirrorCavitySimulation, physical: dict[str, int]) -> dict[str, float]:
    centers = simulation.beam_centers(physical, clip_command=False)
    dx = centers.secondary_x_px - centers.main_x_px
    dy = centers.secondary_y_px - centers.main_y_px
    normalized = float(np.hypot(dx / (simulation.frame_width_px / 2), dy / (simulation.frame_height_px / 2)))
    return {
        "secondary_center_x_px": centers.secondary_x_px,
        "secondary_center_y_px": centers.secondary_y_px,
        "main_center_x_px": centers.main_x_px,
        "main_center_y_px": centers.main_y_px,
        "center_error_px": float(np.hypot(dx, dy)),
        "center_error_normalized": normalized,
    }


def _save_image(simulation: TwoMirrorCavitySimulation, physical: dict[str, int], path: Path) -> None:
    image = simulation.render(physical, clip_command=False)
    Image.fromarray(np.rint(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(path)


def _add_images(llm: LLM, captures: Sequence[dict[str, Any]], heading: str) -> None:
    llm.add_user_message(heading)
    for index, item in enumerate(captures, start=1):
        command = item["absolute_visible_command"]
        llm.add_user_message(f"Image {index}: absolute visible command x={command['x']}, y={command['y']}.")
        llm.add_user_image(item["image_path"])


def run_two_mirror_cavity_simulation_llm_alignment(
    *, out_dir: str | Path, repeat_index: int, seed: int, hidden_offset: dict[str, int],
    iterations: int, probes_per_batch: int, llm_config: LLMConfig,
    profile_path: str | Path, axes: list[TwoMirrorCavityRealLLMAxisConfig],
) -> dict[str, Any]:
    started_at, start_perf = start_timer()
    out_path = ensure_dir(Path(out_dir))
    image_dir = ensure_dir(out_path / "images" / "original")
    simulation = TwoMirrorCavitySimulation(TwoMirrorCavityConfig(profile_path=profile_path))
    hidden = {role: int(hidden_offset[role]) for role in ("x", "y")}
    llm = LLM(model=llm_config.model, save_messages_json_path=os.path.join(out_dir, "messages.json"),
              save_messages_markdown_path=os.path.join(out_dir, "messages.md"),
              save_usage_json_path=os.path.join(out_dir, "llm_usage.json"))
    llm.add_system_message(build_system_prompt(
        axes, max_probe_batches=iterations, probes_per_batch=probes_per_batch
    ))
    manifest: dict[str, Any] = {
        "method": "llm", "repeat_index": repeat_index, "seed": seed, "max_iterations": iterations,
        "probes_per_batch": probes_per_batch,
        "hidden_offset_steps": hidden, "model": asdict(llm_config), "decisions": [], "batches": [],
    }
    zero = {"x": 0, "y": 0}
    initial_path = image_dir / "initial_visible_x_+00000_y_+00000.png"
    _save_image(simulation, hidden, initial_path)
    initial = {"absolute_visible_command": zero, "physical_command_steps": hidden,
               "image_path": str(initial_path), "alignment_metrics": _metrics(simulation, hidden)}
    manifest["initial_capture"] = initial
    current_commands = [zero]
    _add_images(llm, [initial], "This is the initial image at absolute visible command (0, 0). Analyze it and either finish or propose the first probe.")
    final_visible: dict[str, int] | None = None
    for batch_index in range(1, iterations + 1):
        answer = llm.query_llm(
            max_tokens=llm_config.max_tokens, effort=llm_config.effort,
            reasoning_max_tokens=llm_config.reasoning_max_tokens, require_json=True, num_tries=10,
            verify=lambda text, expected=list(current_commands): verify_response(
                text, axes, expected_observation_count=len(expected), expected_commands=expected,
                probes_per_batch=probes_per_batch),
        )
        payload = json.loads(answer)
        manifest["decisions"].append({"input_batch": batch_index - 1, "forced_final": False, "response": payload})
        if payload["done"]:
            final_visible = {k: int(v) for k, v in payload["final_command"].items()}
            manifest["completed_early"] = True
            break
        captures = []
        for probe_index, command in enumerate(payload["probe_commands"], start=1):
            visible = {"x": int(command["x"]), "y": int(command["y"])}
            physical = _physical_command(hidden, visible)
            path = image_dir / (f"batch_{batch_index:02d}_probe_{probe_index:02d}_"
                                f"x_{visible['x']:+06d}_y_{visible['y']:+06d}.png")
            _save_image(simulation, physical, path)
            captures.append({"probe_index": probe_index, "absolute_visible_command": visible,
                             "physical_command_steps": physical, "image_path": str(path),
                             "alignment_metrics": _metrics(simulation, physical)})
        manifest["batches"].append({"batch_index": batch_index, "llm_response": payload, "captures": captures})
        current_commands = [dict(item["absolute_visible_command"]) for item in captures]
        _add_images(llm, captures, f"These are the results from probe batch {batch_index}. Analyze them and either finish or propose the next probe.")
        if batch_index == iterations:
            answer = llm.query_llm(
                max_tokens=llm_config.max_tokens, effort=llm_config.effort,
                reasoning_max_tokens=llm_config.reasoning_max_tokens, require_json=True, num_tries=10,
                verify=lambda text, expected=list(current_commands): verify_response(
                    text, axes, expected_observation_count=len(expected), expected_commands=expected,
                    probes_per_batch=probes_per_batch, force_final=True),
            )
            payload = json.loads(answer)
            manifest["decisions"].append({"input_batch": batch_index, "forced_final": True, "response": payload})
            final_visible = {k: int(v) for k, v in payload["final_command"].items()}
    if final_visible is None:
        raise RuntimeError("LLM optimization ended without a final command.")
    final_physical = _physical_command(hidden, final_visible)
    final_path = image_dir / f"final_x_{final_visible['x']:+06d}_y_{final_visible['y']:+06d}.png"
    _save_image(simulation, final_physical, final_path)
    manifest.update({"final_visible_command_steps": final_visible, "final_true_command_steps": final_physical,
                     "final_image_path": str(final_path), "final_alignment_metrics": _metrics(simulation, final_physical),
                     "llm_usage": llm.get_usage_summary(), "llm_usage_path": str(out_path / "llm_usage.json")})
    add_run_timing(manifest, run_started_at=started_at, run_start_perf=start_perf,
                   steps=[capture for batch in manifest["batches"] for capture in batch["captures"]])
    write_json(out_path / "run_manifest.json", manifest)
    return manifest


def run_two_mirror_cavity_simulation_oracle_alignment(
    *, out_dir: str | Path, method: str, iterations: int, seed: int, repeat_index: int,
    hidden_offset: dict[str, int], profile_path: str | Path,
    axes: list[TwoMirrorCavityRealLLMAxisConfig],
) -> dict[str, Any]:
    if method not in {"bayes", "random"}:
        raise ValueError("method must be 'bayes' or 'random'.")
    started_at, start_perf = start_timer()
    out_path = ensure_dir(Path(out_dir))
    simulation = TwoMirrorCavitySimulation(TwoMirrorCavityConfig(profile_path=profile_path))
    hidden = {role: int(hidden_offset[role]) for role in ("x", "y")}
    by_role = {axis.role: axis for axis in axes}
    steps: list[dict[str, Any]] = []
    best_error = float("inf")
    best_visible = {"x": 0, "y": 0}

    def evaluate(visible: dict[str, int]) -> float:
        nonlocal best_error, best_visible
        visible = {role: int(np.clip(visible[role], by_role[role].llm_min_steps, by_role[role].llm_max_steps)) for role in ("x", "y")}
        physical = _physical_command(hidden, visible)
        metrics = _metrics(simulation, physical)
        error = motor_distance_steps(physical)
        if error < best_error:
            best_error, best_visible = error, dict(visible)
        path = out_path / f"step_{len(steps):03d}.png"
        _save_image(simulation, physical, path)
        steps.append({"step_index": len(steps), "image_path": str(path), "visible_command_steps": visible,
                      "hidden_offset_steps": hidden, "true_command_steps": physical,
                      "alignment_metrics": metrics, "motor_distance_steps": error, "objective": -error,
                      "incumbent_visible_command_steps": dict(best_visible),
                      "incumbent_motor_distance_steps": best_error})
        return -error

    if method == "random":
        evaluate({"x": 0, "y": 0})
        rng = np.random.default_rng(seed)
        for _ in range(iterations):
            evaluate({role: int(rng.integers(by_role[role].llm_min_steps, by_role[role].llm_max_steps + 1)) for role in ("x", "y")})
    else:
        bounds = {role: (float(by_role[role].llm_min_steps), float(by_role[role].llm_max_steps)) for role in ("x", "y")}
        optimizer = BayesianOptimization(
            f=lambda **values: evaluate({role: int(round(values[role])) for role in ("x", "y")}),
            pbounds=bounds, acquisition_function=acquisition.ExpectedImprovement(xi=0.01),
            random_state=seed, verbose=0)
        initial_target = evaluate({"x": 0, "y": 0})
        optimizer.register({"x": 0.0, "y": 0.0}, initial_target)
        optimizer.maximize(init_points=0, n_iter=iterations)
    final_physical = _physical_command(hidden, best_visible)
    final_path = out_path / "final_best.png"
    _save_image(simulation, final_physical, final_path)
    manifest = {"method": method, "objective_name": "negative_motor_distance_steps",
                "objective_definition": "Negative Euclidean distance of the physical x/y motor command from the aligned zero position, in motor steps.",
                "repeat_index": repeat_index, "seed": seed, "max_iterations": iterations,
                "hidden_offset_steps": hidden, "steps": steps, "final_image_path": str(final_path),
                "final_visible_command_steps": best_visible, "final_true_command_steps": final_physical,
                "final_alignment_metrics": _metrics(simulation, final_physical)}
    add_run_timing(manifest, run_started_at=started_at, run_start_perf=start_perf, steps=steps)
    write_json(out_path / "run_manifest.json", manifest)
    return manifest
