"""Localize one station, or interactively calibrate and test all eight stations."""
import argparse
import importlib
import json
import math
from datetime import datetime
from pathlib import Path
import os
import tempfile

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


class CheckedArm:
    """Abort localization if any SDK command/readback fails."""
    def __init__(self, arm):
        self.arm = arm

    def __getattr__(self, name):
        method = getattr(self.arm, name)

        def checked(*args, **kwargs):
            result = method(*args, **kwargs)
            code = result[0] if isinstance(result, (tuple, list)) else result
            if code != 0:
                raise RuntimeError(f"Robot {name} failed: code={code}")
            return result
        return checked


def placement_values(pose, x_adjust, reference_angle):
    """Invert the skeleton's placement orientation, checking representability."""
    measured = Rotation.from_rotvec(np.radians(pose[3:]))
    yaw = measured.as_euler('xyz', degrees=True)[2]
    angle = reference_angle + (yaw - 90 - reference_angle + 180) % 360 - 180
    # The placement helper constrains orientation to roll=180, pitch=0.
    target = Rotation.from_euler('xyz', [180, 0, 90 + angle], degrees=True)
    error = np.degrees((target.inv() * measured).magnitude())
    if error > 2:
        raise RuntimeError(f"Tool orientation differs from placement convention by {error:.2f} degrees")
    return dict(x_position_f=round(float(pose[0] + x_adjust), 3),
                y_position_f=round(float(pose[1]), 3), added_angle=round(float(angle), 3))


def locate(skeleton, station, timeout):
    arm = CheckedArm(skeleton.arm)
    x = float(station.get('view_x', station['x_position_f'] - 74.2))
    y = float(station.get('view_y', station['y_position_f'] - 35.5))
    angle = math.degrees(math.atan2(y, x)) + 180
    if x < 0 and y >= 0:
        angle = 280
    # Same startup and approach as manual placement, with checked return codes.
    arm.set_tcp_load(weight=0.610, center_of_gravity=(0.06125, 0.0458, 0.0375))
    arm.set_servo_angle(angle=[180, 75, -180, 20, 0, 90, -60],
                        is_radian=False, speed=30, wait=True)
    pipeline = skeleton.initialize_realsense_only()[3]
    try:
        arm.set_servo_angle(servo_id=1, angle=angle, is_radian=False, speed=30, wait=True)
        arm.set_position_aa([x, y, 500, -180, 0, 0], is_radian=False,
                            speed=30, mvacc=100, wait=True)
        depth, rotation = skeleton.fine_adjust(arm, pipeline, skeleton.CAMERA_TAG,
                                               timeout_s=timeout)
        _, pose = arm.get_position_aa(is_radian=False)
        correction = skeleton._x_adjust(pose[0], pose[1])
        values = placement_values(pose, correction, float(station['added_angle']))
        return values, dict(localized_tcp_pose=[float(v) for v in pose], depth_mm=float(depth),
                            tag_rotation_degrees=float(rotation), x_correction_mm=float(correction))
    finally:
        pipeline.stop()


def save_station(path, name, values):
    """Replace only the three scalar values, preserving comments and other settings."""
    text = path.read_bytes().decode('utf-8')
    root = yaml.compose(text)
    def child(node, key):
        return next(value for field, value in node.value if field.value == key)
    node = child(child(root, 'camera_stations'), name)
    edits = [(child(node, key).start_mark.index, child(node, key).end_mark.index,
              str(float(value))) for key, value in values.items()]
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    yaml.safe_load(text)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def calibrate_all(s, config_path, timeout, folder, *, input_fn=input, output_fn=print):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'config_before.yaml').write_bytes(config_path.read_bytes())
    arm = CheckedArm(s.arm)
    for name in [f'C{i}' for i in range(1, 9)]:
        attempt = 0
        while True:
            answer = input_fn(f'Position camera at desired {name}, clear the arm path, then Enter to localize '
                              'and test pickup/replacement (s to skip, q to quit): ').strip().lower()
            if answer == 'q':
                return
            if answer in ('s', 'skip'):
                output_fn(f'Skipped {name}; saved coordinates unchanged.')
                break
            if answer:
                continue
            station = yaml.safe_load(config_path.read_text())['camera_stations'][name]
            attempt += 1
            arm.set_gripper_enable(True)
            arm.set_gripper_speed(2000)
            arm.set_gripper_position(850, wait=True)
            values, details = locate(s, station, timeout)
            depth = details['depth_mm']
            if not math.isfinite(depth) or depth <= 0:
                raise RuntimeError('Invalid RealSense depth; refusing pickup')
            record = dict(station=name, previous=station, suggested=values, **details,
                          accepted=False, placement_completed=False)
            record_path = folder / f'{name}_attempt_{attempt}.json'
            record_path.write_text(json.dumps(record, indent=2))
            s.grab_and_place_from_located_pose(
                details['localized_tcp_pose'], depth, details['x_correction_mm'],
                values['x_position_f'], values['y_position_f'], values['added_angle'], arm_api=arm)
            record['placement_completed'] = True
            record_path.write_text(json.dumps(record, indent=2))
            output_fn(yaml.safe_dump({name: values}, sort_keys=False))
            while True:
                answer = input_fn(f'Was the replacement at {name} good? [y=save / n=reposition and retry / q=quit]: ').strip().lower()
                if answer in ('y', 'yes', 'n', 'no', 'q'):
                    break
            if answer == 'q':
                return
            if answer in ('y', 'yes'):
                save_station(config_path, name, values)
                record['accepted'] = True
                record_path.write_text(json.dumps(record, indent=2))
                output_fn(f'Saved {name} to {config_path}.')
                break
    output_fn('Finished C1-C8. Approved stations saved; skipped stations unchanged.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('station', nargs='?', choices=[f'C{i}' for i in range(1, 9)])
    parser.add_argument('--config', type=Path,
                        default=Path('configs/motorized_mirror_relay/lab_runs.yaml'))
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    config = yaml.safe_load(args.config.read_text())
    if args.station:
        print(f"Manually position the camera near {args.station}, with tag 16 visible.")
    else:
        print('Guided C1-C8 calibration: localize, pick up and replace, then approve each YAML update.')
    print("Arm will move to startup, approach at z=500 mm, center on the tag, then rotate its wrist.")
    print('Leave the gripper empty and stop other robot controllers.')
    if args.station:
        print('Single-station mode: no pickup or config update.')
    if input('Connect and perform these movements? [y/N] ').strip().lower() not in ('y', 'yes'):
        return
    s = importlib.import_module(config['hardware'].get('skeleton_module',
                                                     'pick_and_place_and_capture_skeleton'))
    try:
        folder = Path('runs/station_localization') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        if args.station is None:
            calibrate_all(s, args.config, args.timeout, folder)
            return
        station = config['camera_stations'][args.station]
        values, details = locate(s, station, args.timeout)
        folder.mkdir(parents=True)
        snippet = yaml.safe_dump({args.station: values}, sort_keys=False)
        (folder / 'suggested_station.yaml').write_text(snippet)
        (folder / 'measurement.json').write_text(json.dumps(dict(
            station=args.station, previous=station, suggested=values, **details), indent=2))
        print('\nSuggested camera_stations entry:\n' + snippet)
        print(f'Saved to {folder}. Configuration has not been changed.')
        print('These use the existing gripper calibration; verify a subsequent placement before adopting them.')
    except KeyboardInterrupt:
        s.arm.set_state(4)
        print('\nInterrupted; robot stop requested. Previously approved stations remain saved.')
    except Exception:
        s.arm.set_state(4)
        raise
    finally:
        s.arm.disconnect()


if __name__ == '__main__':
    main()
