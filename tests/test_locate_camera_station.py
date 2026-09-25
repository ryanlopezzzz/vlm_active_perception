import unittest
import json
import yaml
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

from src.hardware.camera_station_localization import CheckedArm, locate, placement_values, calibrate_all


class LocateStationTests(unittest.TestCase):
    def test_guided_retry_save_and_quit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'lab.yaml'
            original = '# preserve me\ncamera_stations:\n' + ''.join(
                f'  C{i}:\n    x_position_f: 1 # x\n    y_position_f: 2\n    added_angle: 0\n'
                for i in range(1, 9))
            path.write_text(original)
            s = Mock()
            for name in ('set_gripper_enable', 'set_gripper_speed', 'set_gripper_position'):
                getattr(s.arm, name).return_value = 0
            values = dict(x_position_f=10., y_position_f=20., added_angle=30.)
            details = dict(depth_mm=282., localized_tcp_pose=[1, 2, 500, -180, 0, 0],
                           x_correction_mm=1.)
            answers = iter(['s', '', 'n', '', 'y', 'q'])
            def answer(prompt):
                value = next(answers)
                if value == 'y':
                    self.assertEqual(path.read_text(), original)
                return value
            with patch('src.hardware.camera_station_localization.locate', return_value=(values, details)):
                calibrate_all(s, path, 30, Path(directory) / 'out', input_fn=answer,
                              output_fn=lambda _: None)
            self.assertEqual(s.grab_and_place_from_located_pose.call_count, 2)
            data = yaml.safe_load(path.read_text())['camera_stations']
            self.assertEqual(data['C2'], values)
            self.assertEqual(data['C1']['x_position_f'], 1)
            self.assertEqual(data['C3']['x_position_f'], 1)
            self.assertIn('# preserve me', path.read_text())
            self.assertIn('# x', path.read_text())
            self.assertEqual((Path(directory) / 'out/config_before.yaml').read_text(), original)

    def test_round_trip_placement_angles_and_correction(self):
        for angle in (-90, 0, 90, 94, 94.163, -87):
            # Reproduce the actual skeleton's placement rotation construction.
            inverted = -Rotation.from_euler('xyz', [180, 0, 90 + angle],
                                           degrees=True).as_rotvec() * 180 / np.pi
            pose = [100, 200, 500, -inverted[0], -inverted[1], 0]
            result = placement_values(pose, 1.2, angle)
            self.assertAlmostEqual(result['added_angle'], angle)
            self.assertEqual(result['x_position_f'], 101.2)
            self.assertEqual(result['y_position_f'], 200)
            self.assertEqual(yaml.safe_load(yaml.safe_dump({'C1': result})), {'C1': result})
            self.assertEqual(json.loads(json.dumps(result)), result)

    def test_sdk_failure_aborts(self):
        robot = Mock()
        robot.set_position_aa.return_value = 9
        with self.assertRaisesRegex(RuntimeError, 'code=9'):
            CheckedArm(robot).set_position_aa([0] * 6)

    def test_locate_does_not_grasp_and_releases_pipeline(self):
        s = Mock()
        s.arm.set_tcp_load.return_value = 0
        s.arm.set_servo_angle.return_value = 0
        s.arm.set_position_aa.return_value = 0
        s.arm.get_position_aa.return_value = (0, [100, 200, 500, -180, 0, 0])
        pipeline = Mock()
        s.initialize_realsense_only.return_value = (None, None, None, pipeline)
        s.fine_adjust.return_value = (300, 0)
        s._x_adjust.return_value = 1
        values, _ = locate(s, dict(x_position_f=200, y_position_f=100, added_angle=-90), 30)
        self.assertEqual(values['x_position_f'], 101)
        s.arm.set_gripper_position.assert_not_called()
        s.grab_and_place_from_located_pose.assert_not_called()
        pipeline.stop.assert_called_once()
        s.fine_adjust.side_effect = RuntimeError('timeout')
        with self.assertRaisesRegex(RuntimeError, 'timeout'):
            locate(s, dict(x_position_f=200, y_position_f=100, added_angle=-90), 30)
        self.assertEqual(pipeline.stop.call_count, 2)


if __name__ == '__main__':
    unittest.main()
