"""Lab action protocol on the optical simulator; no hardware imports."""

from dataclasses import asdict
from pathlib import Path
import json

from PIL import Image

from src.agents.llm_agent import LLM
from src.io.artifacts import write_json
from src.sim.motorized_mirror_relay import AXES, RELAY_SEQUENCE, MotorizedMirrorRelay, RelayConfig
from src.tasks.motorized_mirror_relay_protocol import feedback_validator, parse_action
from src.tasks.timing import add_run_timing, start_timer, step_timing_fields, llm_call_count, llm_request_wall_time_since


AXIS_MAPPING = {
    f'mirror_{mirror}_{lab_axis}_steps': f'mirror_{mirror}_{sim_axis}_steps'
    for mirror in range(1, 5)
    for lab_axis, sim_axis in (('in_plane', 'out_of_plane'), ('out_of_plane', 'in_plane'))
}


def camera_reachability(sim, command):
    stations = {}
    furthest_hit = 0
    furthest_plane = 0
    for index, (name, pose) in enumerate(sim.camera_locations.items(), 1):
        trace = sim.trace(command, pose)
        correct_path = trace['mirror_hit_sequence'] == RELAY_SEQUENCE[:(index+1)//2]
        plane_reached = bool(correct_path and trace['plane_reached'])
        hit = bool(correct_path and trace['screen_hit'])
        stations[name] = dict(plane_reached=plane_reached, camera_hit=hit,
                              beam_offset_mm=trace['beam_offset_mm'],
                              mirror_hit_sequence=trace['mirror_hit_sequence'])
        if hit:
            furthest_hit = index
        if plane_reached:
            furthest_plane = index
    return dict(furthest_camera_index=furthest_hit,
                furthest_camera_station=f'C{furthest_hit}' if furthest_hit else None,
                furthest_plane_index=furthest_plane,
                furthest_plane_station=f'C{furthest_plane}' if furthest_plane else None,
                station_diagnostics=stations)


def simulator_command(lab_command):
    if set(lab_command) != set(AXES) or any(type(value) is not int for value in lab_command.values()):
        raise ValueError('Expected eight integer lab-axis settings')
    return {AXIS_MAPPING[axis]: value for axis, value in lab_command.items()}


def observation(station, offsets, turns_remaining):
    return dict(station=station, cumulative_commanded_steps={
        f'M{mirror}': {
            'vertical': offsets[f'mirror_{mirror}_in_plane_steps'],
            'horizontal': offsets[f'mirror_{mirror}_out_of_plane_steps'],
        } for mirror in range(1, 5)
    }, turns_remaining=turns_remaining)


def run_trial(*, out_dir, spec, simulation_config, llm=None, dry_run=False):
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=False)
    started, start_perf = start_timer()
    settings = spec['llm']
    iterations = spec['iterations']
    config = RelayConfig(**simulation_config)
    sim = MotorizedMirrorRelay(config, seed=spec['simulation_seed'],
                              hidden_offsets=simulator_command(spec['scramble_delta_steps']))
    offsets = dict.fromkeys(AXES, 0)
    station = 'C8'
    manifest = dict(status='running', protocol='active_perception_delta_v3', method='llm',
                    backend='simulation', per_move_limit=15000, cumulative_limit=30000,
                    trial=spec['trial'], model=spec['model'], iterations=iterations,
                    control_origin=dict(offsets), axis_mapping=AXIS_MAPPING,
                    simulation=asdict(config), steps=[])
    write_json(root / 'trial_spec.json', spec)
    write_json(root / 'initial_conditions.json', dict(
        lab_scramble_delta_steps=spec['scramble_delta_steps'],
        simulation=sim.initial_conditions(), baseline='true_alignment'))
    (root / 'system_prompt.txt').write_text(spec['prompt'], encoding='utf-8')

    def save():
        write_json(root / 'run.json', manifest)

    save()
    try:
        if not dry_run:
            llm = llm or LLM(model=spec['model'],
                            save_messages_json_path=str(root / 'messages.json'),
                            save_messages_markdown_path=str(root / 'messages.md'),
                            save_usage_json_path=str(root / 'usage.json'))
            llm.add_system_message(spec['prompt'])
        for turn in range(1 if dry_run else iterations):
            step_started, step_perf = start_timer()
            image = root / 'images' / f'{turn:03d}_{station}.png'
            command = simulator_command(offsets)
            capture = sim.capture(command, station, image)
            with Image.open(image) as frame:
                rgb = frame.convert('RGB')
            rgb.save(image)
            record = dict(turn=turn, station=station, offsets=dict(offsets), capture=capture,
                          **camera_reachability(sim, command))
            manifest['steps'].append(record)
            save()
            if dry_run:
                manifest['status'] = 'dry_run'
                break
            llm.add_user_message(json.dumps(observation(station, offsets, iterations-turn)))
            llm.add_user_image(str(image))
            calls_before = llm_call_count(llm)
            answer = llm.query_llm(
                max_tokens=settings['max_tokens'], effort=settings['effort'],
                reasoning_max_tokens=settings.get('reasoning_max_tokens'), require_json=True,
                num_tries=3, verify=feedback_validator(llm, station, offsets, record, save))
            action = parse_action(answer, station, offsets)
            record.update(action=action, execution='pending')
            save()
            if action['action'] == 'tilt':
                mirror = int(action['mirror'][1])
                offsets[f'mirror_{mirror}_in_plane_steps'] += action['vertical']
                offsets[f'mirror_{mirror}_out_of_plane_steps'] += action['horizontal']
            elif action['action'] == 'measure':
                station = action['station']
                record['camera_moved'] = sim.place_camera(station)
            else:
                manifest['status'] = 'llm_declared_done'
                manifest['final_verified_image'] = str(image)
            record['execution'] = 'completed'
            record.update(step_timing_fields(step_started, step_perf,
                          llm_request_wall_time_sec=llm_request_wall_time_since(llm, calls_before)))
            save()
            if action['action'] == 'done':
                break
        else:
            manifest['status'] = 'iteration_limit'
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        command = simulator_command(offsets)
        exit_trace = sim.trace(command, sim.camera_locations['C8'])
        manifest.update(final_command=dict(offsets), final_camera_station=station,
                        camera_image_count=sim.camera_image_count,
                        camera_placement_count=sim.camera_placements,
                        final_nominal_c8=exit_trace,
                        final_exit_beam_visible=(exit_trace['screen_hit'] and
                                                 exit_trace['mirror_hit_sequence'] == RELAY_SEQUENCE))
        last = manifest['steps'][-1] if manifest['steps'] else None
        manifest['final_state_measured'] = bool(last and last['offsets'] == offsets and last['station'] == station)
        if llm is not None and not dry_run:
            manifest['llm_usage'] = llm.get_usage_summary()
        add_run_timing(manifest, run_started_at=started, run_start_perf=start_perf, steps=manifest['steps'])
        save()
    return manifest
