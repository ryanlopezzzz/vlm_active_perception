"""LLM-selected measurements and mirror adjustments for real-lab relay alignment."""
import argparse
import copy
from datetime import datetime
import importlib
import json
from pathlib import Path
import random

import yaml

from src.hardware.camera_station_localization import CheckedArm
from src.hardware.station_repeatability import snapshot
from src.agents.llm_agent import LLM
from src.hardware.motorized_mirror_relay import RelayLabSession, STATION_IDS, validate_lab_config
from src.sim.motorized_mirror_relay import AXES
from src.tasks.motorized_mirror_relay_protocol import parse_action, valid_action, feedback_validator


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "four_mirror_relay.txt"


def run_alignment(session, llm, root, iterations, capture_fn=snapshot):
    origin = dict(session.current_command)
    station = session.current_camera_location
    if station not in STATION_IDS:
        raise ValueError('Declare the current camera station before alignment')
    manifest = {'status': 'running', 'protocol': 'active_perception_delta_v3',
                'per_move_limit': 15000, 'cumulative_limit': 30000, 'control_origin': origin, 'steps': []}
    def save():
        (root / 'run.json').write_text(json.dumps(manifest, indent=2))
    prompt = PROMPT_PATH.read_text(encoding='utf-8')
    (root / 'system_prompt.txt').write_text(prompt, encoding='utf-8')
    llm.add_system_message(prompt)
    save()
    try:
        for turn in range(iterations):
            session.place_camera(station)
            image = root / 'images' / f'{turn:03d}_{station}.png'
            capture = capture_fn(session.config['camera'], image)
            offsets = {axis: session.current_command[axis] - origin[axis] for axis in AXES}
            record = dict(turn=turn, station=station, offsets=offsets, capture=capture)
            manifest['steps'].append(record)
            save()
            named_offsets = {f'M{m}': {
                'vertical': offsets[f'mirror_{m}_in_plane_steps'],
                'horizontal': offsets[f'mirror_{m}_out_of_plane_steps']}
                for m in range(1, 5)}
            llm.add_user_message(json.dumps(dict(station=station,
                                                cumulative_commanded_steps=named_offsets, turns_remaining=iterations-turn)))
            llm.add_user_image(str(image))
            settings = session.config['llm']
            answer = llm.query_llm(max_tokens=settings.get('max_tokens', 12000),
                effort=settings.get('effort', 'high'),
                reasoning_max_tokens=settings.get('reasoning_max_tokens'), require_json=True,
                num_tries=3, verify=feedback_validator(llm, station, offsets, record, save))
            action = parse_action(answer, station, offsets)
            record['action'] = action
            record['execution'] = 'pending'
            save()
            print(f'{station}: {action}', flush=True)
            if action['action'] == 'tilt':
                mirror = int(action['mirror'][1])
                target = dict(session.current_command)
                for label, axis in (('vertical', 'in_plane'), ('horizontal', 'out_of_plane')):
                    key = f'mirror_{mirror}_{axis}_steps'
                    target[key] = session.current_command[key] + action[label]
                record['motor_moves'] = session.set_motor_steps(target)
            elif action['action'] == 'measure':
                session.place_camera(action['station'])
                station = action['station']
            else:
                manifest['status'] = 'llm_declared_done'
                manifest['final_verified_image'] = str(image)
            record['execution'] = 'completed'
            save()
            if action['action'] == 'done':
                break
        else:
            manifest['status'] = 'iteration_limit'
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = repr(exc)
        raise
    finally:
        manifest['final_command'] = dict(session.current_command)
        manifest['final_camera_station'] = session.current_camera_location
        save()
    return manifest


def snapshot_all(session, root, phase, capture_fn=snapshot):
    folder = root / phase
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for station in STATION_IDS:
        print(f'{phase}: {station}', flush=True)
        placement = session.place_camera(station)
        capture = capture_fn(session.config['camera'], folder / f'{station}.png')
        records.append(dict(station=station, placement=placement, capture=capture,
                            motor_command=dict(session.current_command)))
        (folder / 'captures.json').write_text(json.dumps(records, indent=2))
    return records


def prepare_run(session, root, scramble, before_after=False, capture_fn=snapshot,
                all_before=False, all_after=False):
    captures = {}
    if all_before:
        snapshot_all(session, root, 'before_scramble', capture_fn)
    if before_after:
        session.place_camera('C8')
        captures['before_scramble'] = capture_fn(session.config['camera'], root / 'C8_before_scramble.png')
        (root / 'c8_comparison.json').write_text(json.dumps(captures, indent=2))
    if scramble:
        moves = session.set_motor_steps({a: session.current_command[a] + scramble[a] for a in AXES})
        (root / 'scramble_moves.json').write_text(json.dumps(moves, indent=2))
    if before_after:
        captures['after_scramble'] = capture_fn(session.config['camera'], root / 'C8_after_scramble.png')
        (root / 'c8_comparison.json').write_text(json.dumps(captures, indent=2))
    if all_after:
        snapshot_all(session, root, 'after_scramble', capture_fn)


def run_trials(session, root, config, args, iterations, scrambles):
    summary = {'requested_runs': len(scrambles), 'trials': [], 'status': 'running'}
    def save():
        (root / 'batch.json').write_text(json.dumps(summary, indent=2))
    save()
    try:
        for index, scramble in enumerate(scrambles, 1):
            if index < getattr(args, 'start_trial', 1):
                continue
            snapshot_limit = getattr(args, 'snapshot_first_runs', None)
            snapshots_enabled = snapshot_limit is None or index <= snapshot_limit
            folder = root if len(scrambles) == 1 else root / f'trial_{index:02d}'
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'config_used.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
            (folder / 'startup.json').write_text(json.dumps(dict(
                start=session.current_camera_location, scramble=scramble, seed=args.seed,
                trial=index), indent=2))
            record = {'trial': index, 'folder': str(folder), 'status': 'running'}
            summary['trials'].append(record)
            save()
            print(f'Trial {index}/{len(scrambles)}: {folder}', flush=True)
            # Diagnostic at the current station, before any scramble or camera relocation.
            first = snapshot(config['camera'], folder / 'first_image.png')
            first['station'] = session.current_camera_location
            (folder / 'first_image.json').write_text(json.dumps(first, indent=2))
            print(f'Initial camera image: {folder / "first_image.png"}', flush=True)
            # Fresh conversation and usage accounting for every trial.
            llm = LLM(model=config['llm']['model'],
                save_messages_json_path=str(folder / 'messages.json'),
                save_messages_markdown_path=str(folder / 'messages.md'),
                save_usage_json_path=str(folder / 'usage.json'))
            prepare_run(session, folder, scramble, snapshots_enabled and args.snapshot_c8_before_after,
                        all_before=snapshots_enabled and args.snapshot_all_before_scramble,
                        all_after=snapshots_enabled and args.snapshot_all_after_scramble)
            result = run_alignment(session, llm, folder, iterations)
            record['status'] = result['status']
            save()
            if result['status'] != 'llm_declared_done':
                summary['status'] = 'stopped_without_alignment'
                print('Stopping batch: LLM did not declare alignment complete.')
                break
            if snapshots_enabled and args.snapshot_all_after_alignment:
                snapshot_all(session, folder, 'after_alignment')
        else:
            summary['status'] = 'completed'
    except BaseException as exc:
        summary['status'] = 'failed'
        summary['error'] = repr(exc)
        if summary['trials']:
            summary['trials'][-1]['status'] = 'failed'
        raise
    finally:
        save()
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('configs/motorized_mirror_relay/lab_runs.yaml'))
    p.add_argument('--start', choices=STATION_IDS, help='Actual current camera location')
    p.add_argument('--scramble', nargs='?', const=15000, type=int, metavar='LIMIT')
    p.add_argument('--seed', type=int)
    p.add_argument('--iterations', type=int)
    p.add_argument('--runs', type=int, default=1, help='Number of trials in one hardware session')
    p.add_argument('--start-trial', type=int, default=1, help='Start at this trial number in the full scramble sequence')
    p.add_argument('--scramble-plan', type=Path, help='Load a previously saved scramble_plan.json; its length sets the total trials')
    p.add_argument('--snapshot-first-runs', type=int,
                   help='Apply enabled diagnostic snapshot passes only to the first N trials')
    p.add_argument('--snapshot-c8-before-after', action='store_true',
                   help='Capture C8 before and after startup scrambling, then let the LLM choose measurements')
    p.add_argument('--snapshot-all-before-scramble', action='store_true', help='Capture C1-C8 before scrambling')
    p.add_argument('--snapshot-all-after-scramble', action='store_true', help='Capture C1-C8 after scrambling, before LLM control')
    p.add_argument('--snapshot-all-after-alignment', action='store_true', help='Capture C1-C8 after the LLM declares done')
    args = p.parse_args()
    plan = None
    if args.scramble_plan:
        plan = json.loads(args.scramble_plan.read_text())
        scrambles_from_file = plan.get('scrambles')
        if not isinstance(scrambles_from_file, list) or not scrambles_from_file:
            p.error('Scramble plan needs a nonempty scrambles list')
        for command in scrambles_from_file:
            if (not isinstance(command, dict) or set(command) != set(AXES)
                    or any(type(v) is not int or abs(v) > 15000 for v in command.values())):
                p.error('Each planned scramble must contain eight integer offsets within +/-15000')
        args.runs = len(scrambles_from_file)
    if args.runs < 1:
        p.error('--runs must be positive')
    if not 1 <= args.start_trial <= args.runs:
        p.error('--start-trial must be between 1 and the total number of trials')
    if args.snapshot_first_runs is not None and args.snapshot_first_runs < 0:
        p.error('--snapshot-first-runs must be nonnegative')
    if args.scramble is not None and not 0 <= args.scramble <= 15000:
        p.error('--scramble must be between 0 and 15000')
    if args.snapshot_c8_before_after and args.scramble is None and plan is None:
        p.error('--snapshot-c8-before-after requires --scramble')
    if (args.snapshot_all_before_scramble or args.snapshot_all_after_scramble) and args.scramble is None and plan is None:
        p.error('Before/after scramble snapshot passes require --scramble')
    config = yaml.safe_load(args.config.read_text())
    validate_lab_config(config)
    iterations = args.iterations if args.iterations is not None else config['iterations']
    if iterations < 1:
        p.error('--iterations must be positive')
    start = args.start
    while start not in STATION_IDS:
        start = input('Actual camera station C1-C8 (q to quit): ').strip().upper()
        if start == 'Q':
            return
    rng = random.Random(args.seed)
    scrambles = [{axis: rng.randint(-args.scramble, args.scramble) for axis in AXES}
                 if args.scramble is not None else {} for _ in range(args.runs)]
    if plan is not None:
        scrambles = scrambles_from_file
    print(f'Will run up to {args.runs} trials; stop on failure or no completion declaration.')
    print('Relative scrambles for each trial:', json.dumps(scrambles, indent=2))
    print('Close camera apps and other robot controllers. Uses default camera capture settings.')
    if args.snapshot_all_before_scramble:
        print('Capture all C1-C8 before scrambling.')
    if args.snapshot_c8_before_after:
        print('First: move to C8, capture before scrambling, scramble, capture again without moving the camera.')
    print('LLM uses delta moves: +/-15000 per move, +/-30000 cumulative commanded steps after scrambling.')
    if args.snapshot_all_after_scramble:
        print('Capture all C1-C8 immediately after scrambling, before LLM control.')
    if args.snapshot_all_after_alignment:
        print('Capture all C1-C8 after LLM-declared completion (skip on failure or turn limit).')
    if input('Connect, move robot, and execute this batch? [y/N] ').strip().lower() not in ('y', 'yes'):
        return
    root = Path('runs/motorized_mirror_relay_staged') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True)
    (root / 'scramble_plan.json').write_text(json.dumps(dict(
        seed=args.seed if plan is None else plan.get('seed'), scrambles=scrambles), indent=2))
    (root / 'config_used.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (root / 'batch_options.json').write_text(json.dumps(vars(args), default=str, indent=2))
    print(f'Output: {root.resolve()}', flush=True)
    s = importlib.import_module(config['hardware']['skeleton_module'])
    original = s.arm
    s.arm = CheckedArm(original)
    try:
        with RelayLabSession(copy.deepcopy(config), skeleton=s, tracked_camera=True,
                             enable_beam_camera=False, enforce_motor_limits=False) as session:
            session.declare_camera_station(start)
            run_trials(session, root, config, args, iterations, scrambles)
    except BaseException:
        original.set_state(4)
        raise
    finally:
        s.arm = original
        original.disconnect()


if __name__ == '__main__':
    main()
