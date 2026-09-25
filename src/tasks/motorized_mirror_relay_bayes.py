"""Bayesian optimization of the lab-matched relay using a sparse C8 objective."""

from dataclasses import asdict
from pathlib import Path

from bayes_opt import BayesianOptimization, acquisition
import numpy as np

from src.io.artifacts import write_json
from src.sim.motorized_mirror_relay import AXES, NO_SIGNAL_OBJECTIVE, RELAY_SEQUENCE, MotorizedMirrorRelay, RelayConfig
from src.tasks.motorized_mirror_relay_matched import camera_reachability, simulator_command
from src.tasks.timing import add_run_timing, start_timer, step_timing_fields


def center_distance_px(trace, config):
    if not trace["screen_hit"] or trace["mirror_hit_sequence"] != RELAY_SEQUENCE:
        return None
    x_px = float(trace["u_mm"]) * config.frame_width_px / config.camera_width_mm
    y_px = float(trace["v_mm"]) * config.frame_height_px / config.camera_height_mm
    return float(np.hypot(x_px, y_px))


def furthest_mirror_hit(sequence):
    return max((int(name[1:]) for name in sequence if name in RELAY_SEQUENCE), default=0)


def run_bayes_trial(*, out_dir, spec, simulation_config, dry_run=False):
    if dry_run:
        raise ValueError("Bayesian optimization does not support dry-run mode")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=False)
    started, start_perf = start_timer()
    config = RelayConfig(**simulation_config)
    sim = MotorizedMirrorRelay(
        config,
        seed=spec["simulation_seed"],
        hidden_offsets=simulator_command(spec["scramble_delta_steps"]),
    )
    steps = []
    best_objective = float("-inf")
    best_offsets = dict.fromkeys(AXES, 0)
    best_furthest_mirror = 0
    best_furthest_camera = 0
    c8_pose = sim.camera_locations["C8"]
    manifest = {
        "status": "running",
        "method": "bayes",
        "model": "bayes",
        "backend": "simulation",
        "protocol": "sparse_final_camera_bo_v1",
        "trial": spec["trial"],
        "iterations": spec["iterations"],
        "optimization_seed": spec["optimization_seed"],
        "simulation_seed": spec["simulation_seed"],
        "axis_bounds_steps": [-15000, 15000],
        "objective_name": "negative_c8_center_distance_px",
        "objective_definition": (
            "Negative C8 beam-center distance in pixels after the correct M1-M4 relay path; "
            "-1e6 when the beam does not hit the C8 aperture."
        ),
        "no_signal_objective": NO_SIGNAL_OBJECTIVE,
        "axis_mapping": spec.get("axis_mapping"),
        "simulation": asdict(config),
        "steps": steps,
    }

    def save():
        write_json(root / "run.json", manifest)

    def evaluate(values):
        nonlocal best_objective, best_offsets, best_furthest_mirror, best_furthest_camera
        step_started, step_perf = start_timer()
        offsets = {
            axis: int(np.clip(round(float(values[axis])), -15000, 15000))
            for axis in AXES
        }
        command = simulator_command(offsets)
        trace = sim.trace(command, c8_pose)
        distance_px = center_distance_px(trace, config)
        objective = -distance_px if distance_px is not None else NO_SIGNAL_OBJECTIVE
        reach = camera_reachability(sim, command)
        mirror = furthest_mirror_hit(trace["mirror_hit_sequence"])
        best_furthest_mirror = max(best_furthest_mirror, mirror)
        best_furthest_camera = max(best_furthest_camera, reach["furthest_camera_index"])
        if objective > best_objective:
            best_objective = float(objective)
            best_offsets = dict(offsets)
        record = {
            "iteration": len(steps),
            "offsets": offsets,
            "objective": float(objective),
            "c8_center_distance_px": distance_px,
            "c8_screen_hit": bool(distance_px is not None),
            "mirror_hit_sequence": trace["mirror_hit_sequence"],
            "furthest_mirror_hit": mirror,
            "best_furthest_mirror_hit": best_furthest_mirror,
            "best_furthest_camera_index": best_furthest_camera,
            "incumbent_objective": best_objective,
            "incumbent_offsets": dict(best_offsets),
            **reach,
        }
        record.update(step_timing_fields(step_started, step_perf, llm_request_wall_time_sec=0))
        steps.append(record)
        save()
        return float(objective)

    save()
    try:
        zero = dict.fromkeys(AXES, 0.0)
        initial_target = evaluate(zero)
        optimizer = BayesianOptimization(
            f=lambda **values: evaluate(values),
            pbounds={axis: (-15000.0, 15000.0) for axis in AXES},
            acquisition_function=acquisition.ExpectedImprovement(xi=0.01),
            random_state=spec["optimization_seed"],
            verbose=0,
        )
        optimizer.register(zero, initial_target)
        optimizer.maximize(init_points=0, n_iter=spec["iterations"])
        manifest["status"] = "completed"
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        best_command = simulator_command(best_offsets)
        best_trace = sim.trace(best_command, c8_pose)
        manifest.update(
            best_objective=best_objective,
            best_offsets=best_offsets,
            best_c8_center_distance_px=center_distance_px(best_trace, config),
            best_furthest_mirror_hit=best_furthest_mirror,
            best_furthest_camera_index=best_furthest_camera,
            best_furthest_camera_station=f"C{best_furthest_camera}" if best_furthest_camera else None,
            final_command=best_offsets,
            final_camera_station="C8",
            camera_image_count=0,
            camera_placement_count=0,
            final_exit_beam_visible=bool(center_distance_px(best_trace, config) is not None),
        )
        add_run_timing(manifest, run_started_at=started, run_start_perf=start_perf, steps=steps)
        save()
    return manifest
