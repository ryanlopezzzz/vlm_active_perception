import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from PIL import Image

from src.hardware.station_repeatability import run_sequence, snapshot, STATION_IDS


class RepeatabilityTests(unittest.TestCase):
    def test_snapshot_defaults_and_release(self):
        cap = Mock()
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        cap.read.return_value = (True, frame)
        with tempfile.TemporaryDirectory() as directory:
            with patch('src.hardware.station_repeatability.cv2.VideoCapture', return_value=cap):
                result = snapshot({'index': 1, 'backend': 'CAP_DSHOW'}, Path(directory) / 'image.png')
        cap.set.assert_not_called()
        self.assertEqual(cap.read.call_count, 8)
        cap.release.assert_called_once()
        self.assertEqual(result['raw_shape'], [48, 64, 3])

    def test_two_full_circuits_and_sheet_without_motor_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = Mock(current_camera_location='C1')
            session.place_camera.return_value = {'camera_moved': True}
            session.config = {'camera': {'index': 1, 'backend': 'CAP_DSHOW'}}
            def capture(camera, path):
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (64, 48), 'red').save(path)
                return {'raw_image_path': str(path)}
            with patch('src.hardware.station_repeatability.snapshot', side_effect=capture):
                records = run_sequence(session, root)
            session.capture.assert_not_called()
            self.assertEqual([r['station'] for r in records], list(STATION_IDS) * 2)
            self.assertEqual(len(list(root.glob('pass_*/C*.png'))), 16)
            self.assertEqual([c.args[0] for c in session.place_camera.call_args_list],
                             ['C8'] + list(STATION_IDS) * 2)
            session.set_motor_steps.assert_not_called()
            with Image.open(root / 'contact_sheet.jpg') as sheet:
                self.assertEqual(sheet.size, (800, 2624))


if __name__ == '__main__':
    unittest.main()
