"""Uniform random eight-axis guesses with offline camera-reach diagnostics."""

import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from src.io.artifacts import write_json
from src.sim.motorized_mirror_relay import AXES, RELAY_SEQUENCE, MotorizedMirrorRelay, RelayConfig
from src.tasks.motorized_mirror_relay_matched import (
    AXIS_MAPPING, camera_reachability, simulator_command,
)
from src.tasks.timing import add_run_timing, start_timer, step_timing_fields


def sample_offsets(rng):
    return dict(zip(AXES, map(int, rng.integers(-15000, 15001, size=len(AXES)))))


def run_random_trial(*, out_dir, spec, simulation_config, dry_run=False):
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=False)
    started, start_perf = start_timer()
    config = RelayConfig(**simulation_config)
    sim = MotorizedMirrorRelay(config, seed=spec['simulation_seed'],
                              hidden_offsets=simulator_command(spec['scramble_delta_steps']))
    rng = np.random.default_rng(spec['sampling_seed'])
    offsets = dict.fromkeys(AXES, 0)
    manifest = dict(status='running', method='random', model='random', backend='simulation',
                    protocol='uniform_random_offsets_v1', trial=spec['trial'], iterations=spec['iterations'],
                    sampling_seed=spec['sampling_seed'], simulation_seed=spec['simulation_seed'],
                    axis_mapping=AXIS_MAPPING, simulation=asdict(config),
                    sampling='Independent absolute post-scramble offsets on all eight axes; inclusive [-15000,15000].',
                    reach_definition='Furthest nominal C1-C8 aperture hit by the central ray after the correct mirror prefix; 0 means none.',
                    diagnostic_definition='Independent virtual station traces, not eight exposures or camera placements.',
                    initial_zero_command_reach=camera_reachability(sim, simulator_command(offsets)), steps=[])
    write_json(root / 'trial_spec.json', spec)
    write_json(root / 'initial_conditions.json', dict(
        lab_scramble_delta_steps=spec['scramble_delta_steps'],
        simulation=sim.initial_conditions(), baseline='true_alignment'))

    def save():
        write_json(root / 'run.json', manifest)

    save()
    best = 0
    try:
        with (root / 'reachability.csv').open('w', newline='') as handle:
            fields = ['iteration', 'furthest_camera_index', 'furthest_camera_station',
                      'furthest_plane_index', 'furthest_plane_station', 'best_furthest_camera_index',
                      'best_furthest_camera_station', *AXES]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for turn in range(1 if dry_run else spec['iterations']):
                step_started, step_perf = start_timer()
                offsets = dict.fromkeys(AXES, 0) if dry_run else sample_offsets(rng)
                command = simulator_command(offsets)
                image = root / 'images' / f'{turn:03d}_C8.png'
                capture = sim.capture(command, 'C8', image)
                with Image.open(image) as frame:
                    rgb = frame.convert('RGB')
                rgb.save(image)
                reach = camera_reachability(sim, command)
                best = max(best, reach['furthest_camera_index'])
                record = dict(turn=turn, iteration=turn+1, offsets=dict(offsets), station='C8', capture=capture,
                              best_furthest_camera_index=best,
                              best_furthest_camera_station=f'C{best}' if best else None, **reach)
                manifest['steps'].append(record)
                save()
                record.update(step_timing_fields(step_started, step_perf, llm_request_wall_time_sec=0))
                writer.writerow({**{key: record[key] for key in fields if key not in AXES}, **offsets})
                handle.flush()
                save()
        manifest['status'] = 'dry_run' if dry_run else 'completed'
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        command = simulator_command(offsets)
        exit_trace = sim.trace(command, sim.camera_locations['C8'])
        manifest.update(final_command=offsets, final_camera_station='C8',
                        camera_image_count=sim.camera_image_count, camera_placement_count=sim.camera_placements,
                        best_furthest_camera_index=best,
                        best_furthest_camera_station=f'C{best}' if best else None,
                        final_nominal_c8=exit_trace,
                        final_exit_beam_visible=bool(exit_trace['screen_hit'] and
                                                    exit_trace['mirror_hit_sequence'] == RELAY_SEQUENCE))
        add_run_timing(manifest, run_started_at=started, run_start_perf=start_perf, steps=manifest['steps'])
        save()
    return manifest
