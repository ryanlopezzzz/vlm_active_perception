"""Visit C1-C8 twice and compare beam images; no mirror motor moves."""
import argparse
from datetime import datetime
import importlib
import json
from pathlib import Path
import cv2

from PIL import Image, ImageDraw, ImageOps
import yaml

from src.hardware.camera_station_localization import CheckedArm
from src.hardware.motorized_mirror_relay import RelayLabSession, STATION_IDS, validate_lab_config


def contact_sheet(records, root, passes):
    width, height, label = 400, 300, 28
    sheet = Image.new('RGB', (passes * width, 8 * (height + label)), '#222222')
    draw = ImageDraw.Draw(sheet)
    for row, station in enumerate(STATION_IDS):
        for column in range(passes):
            draw.text((column * width + 8, row * (height + label) + 7),
                      f'{station} - pass {column + 1}', fill='white')
    for record in records:
        with Image.open(root / record['image']) as source:
            preview = ImageOps.contain(source.convert('RGB'), (width, height))
        x = (record['pass'] - 1) * width + (width - preview.width) // 2
        y = STATION_IDS.index(record['station']) * (height + label) + label
        sheet.paste(preview, (x, y + (height - preview.height) // 2))
    sheet.save(root / 'contact_sheet.jpg', quality=95)


def snapshot(camera, destination):
    """Capture with camera defaults, eight reads, and the last successful frame."""
    cap = cv2.VideoCapture(camera['index'], getattr(cv2, camera['backend']))
    try:
        if not cap.isOpened():
            raise RuntimeError('Could not open beam camera')
        frame = None
        for _ in range(8):
            ok, candidate = cap.read()
            if ok:
                frame = candidate
        if frame is None:
            raise RuntimeError('No beam camera frame received')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), frame):
            raise RuntimeError(f'Failed to save {destination}')
        print(f'Saved {destination.name}: {frame.shape}', flush=True)
        return {'raw_image_path': str(destination), 'raw_shape': list(frame.shape),
                'capture_mode': 'snapshot_defaults', 'index': camera['index'],
                'backend': camera['backend']}
    finally:
        cap.release()


def run_sequence(session, root, passes=2):
    records = []
    # Ensure even the first C1 image follows a robot placement.
    if session.current_camera_location == 'C1':
        session.place_camera('C8')
    try:
        for cycle in range(1, passes + 1):
            for station in STATION_IDS:
                print(f'Pass {cycle}/{passes}: {station}', flush=True)
                placement = session.place_camera(station)
                destination = root / f'pass_{cycle}' / f'{station}.png'
                capture = snapshot(session.config['camera'], destination)
                records.append({'pass': cycle, 'station': station,
                                'image': str(destination.relative_to(root)),
                                'placement': placement, 'capture': capture})
                (root / 'measurements.json').write_text(json.dumps(records, indent=2))
                contact_sheet(records, root, passes)
    finally:
        if records:
            contact_sheet(records, root, passes)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/motorized_mirror_relay/lab_runs.yaml'))
    parser.add_argument('--start', choices=STATION_IDS, help='Actual current camera station')
    parser.add_argument('--passes', type=int, default=2)
    args = parser.parse_args()
    if args.passes < 1:
        parser.error('--passes must be positive')
    config = yaml.safe_load(args.config.read_text())
    validate_lab_config(config)
    start = args.start
    while start not in STATION_IDS:
        start = input('Where is the camera currently? C1-C8 (q to quit): ').strip().upper()
        if start == 'Q':
            return
    print(f'Will perform {args.passes} complete C1-C8 circuits and capture each placement.')
    print('Close the beam-camera app and stop other robot controllers. Mirror motors will not be commanded.')
    print('Beam snapshots use camera defaults; YAML resolution/format/exposure overrides are not applied.')
    if start == 'C1':
        print('First moves C1 to C8 so the first recorded C1 is also a fresh placement.')
    if input('Connect, move to startup, and run all placements? [y/N] ').strip().lower() not in ('y', 'yes'):
        return
    root = Path('runs/station_repeatability') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True)
    (root / 'config_used.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    print(f'Output: {root.resolve()}', flush=True)
    s = importlib.import_module(config['hardware']['skeleton_module'])
    original_arm = s.arm
    s.arm = CheckedArm(original_arm)
    try:
        with RelayLabSession(config, skeleton=s, tracked_camera=True, enable_beam_camera=False) as session:
            session.declare_camera_station(start)
            run_sequence(session, root, args.passes)
        print(f'Done. Contact sheet: {root / "contact_sheet.jpg"}')
    except BaseException:
        original_arm.set_state(4)
        print(f'Stopped; completed images remain in {root}.')
        raise
    finally:
        s.arm = original_arm
        original_arm.disconnect()


if __name__ == '__main__':
    main()
