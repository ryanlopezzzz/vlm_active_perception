import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import yaml

from src.hardware.motorized_mirror_relay import RelayLabSession, prepare_llm_image, validate_lab_config
from src.hardware.motors import StepperMotorRig
from src.sim.motorized_mirror_relay import AXES


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/motorized_mirror_relay/lab_runs.yaml"


def filled_config(directory):
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config.update(iterations=4, repeats=2, output_root=str(Path(directory) / "runs"))
    config["hardware"].update(session_state_path=str(Path(directory) / "state.json"), settle_time_sec=0.)
    config["camera"].update(settle_s=0., llm_width=64, llm_height=48)
    for i, controller in enumerate(config["mirror_controllers"].values()):
        controller["controller_ip"] = f"192.0.2.{i+1}"
    for i, station in enumerate(config["camera_stations"].values()):
        station.update(x_position_f=100.+i*50, y_position_f=200., added_angle=90.)
    return config


class FakeSkeleton:
    cv2 = SimpleNamespace(CAP_DSHOW=700)
    CaptureStep = SimpleNamespace

    def __init__(self):
        self.events = []
        self.devices = [SimpleNamespace(release=lambda: self.events.append("release")) for _ in range(4)]
        self.pipeline = SimpleNamespace(stop=lambda: self.events.append("stop_pipeline"))

    def start(self):
        self.events.append("start")

    def initialize_cams(self):
        self.events.append("initialize_cams")
        return (*self.devices[:3], self.pipeline)

    def initialize_realsense_only(self):
        self.events.append("initialize_realsense_only")
        return (None, None, None, self.pipeline)

    def place_camera_from_known_station(self, cameras, source, step):
        self.events.append(("source", dict(source)))
        self.place_camera_at_xy(cameras, step)

    def open_elp_camera(self):
        self.events.append("open_elp")
        return self.devices[3]

    def place_camera_at_xy(self, cameras, step):
        self.events.append(("place", step.x_position_f, step.y_position_f, step.added_angle))

    def capture_beam_image(self, camera, step):
        self.events.append(("capture", step.name))
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        frame[20:35, 30:45] = 200
        Image.fromarray(frame).save(self.SNAPSHOT_DIR / f"{step.name}.png")
        return frame


def dry_rig(*args, **kwargs):
    return StepperMotorRig(*args, **kwargs, dry_run=True)


class LabTests(unittest.TestCase):
    def test_placeholders_fail_before_hardware_import(self):
        config = yaml.safe_load(CONFIG_PATH.read_text())
        config["mirror_controllers"]["M4"]["controller_ip"] = None
        config["camera_stations"]["C8"]["added_angle"] = None
        with patch("src.hardware.motorized_mirror_relay.importlib.import_module") as imported:
            with self.assertRaises(ValueError) as raised:
                RelayLabSession(config)
            self.assertIn("M4.controller_ip", str(raised.exception))
            self.assertIn("C8.added_angle", str(raised.exception))
            imported.assert_not_called()

    def test_invalid_parallelism_and_duplicate_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            config = filled_config(directory)
            config["max_parallel_jobs"] = 2
            with self.assertRaisesRegex(ValueError, "max_parallel_jobs"):
                validate_lab_config(config)
            config["max_parallel_jobs"] = 1
            config["mirror_controllers"]["M1"]["out_of_plane"]["motor_number"] = 1
            with self.assertRaisesRegex(ValueError, "multiple axes"):
                validate_lab_config(config)

    def test_real_open_imports_skeleton_once_and_configures_persistent_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            config = filled_config(directory)
            calibration = Path(directory) / "calibration.npz"
            calibration.touch()
            config["hardware"]["calibration_path"] = str(calibration)
            skeleton = FakeSkeleton()
            with patch("src.hardware.motorized_mirror_relay.importlib.import_module", return_value=skeleton) as imported:
                with RelayLabSession(config, rig_factory=dry_rig) as session:
                    session.open()
                    imported.assert_called_once_with("pick_and_place_and_capture_skeleton")
                    self.assertEqual(skeleton.ELP_WIDTH, 8000)
                    self.assertEqual(skeleton.CALIBRATION_PATH, calibration.resolve())
                    self.assertEqual(skeleton.events.count("initialize_cams"), 1)
            self.assertEqual(skeleton.events.count("release"), 4)
            self.assertEqual(skeleton.events.count("stop_pipeline"), 1)

    def test_absolute_steps_convert_to_deltas_and_persist(self):
        with tempfile.TemporaryDirectory() as directory:
            config = filled_config(directory)
            config["mirror_controllers"]["M1"]["in_plane"]["direction_sign"] = -1
            skeleton = FakeSkeleton()
            with RelayLabSession(config, skeleton=skeleton, rig_factory=dry_rig) as session:
                command = dict.fromkeys(AXES, 0)
                command[AXES[0]] = 100
                first = session.set_motor_steps(command)
                self.assertEqual(first[0]["requested_logical_steps"], 100)
                self.assertEqual(first[0]["requested_hardware_steps"], -100)
                self.assertEqual(session.set_motor_steps(command), [])
                command[AXES[0]] = 130
                self.assertEqual(session.set_motor_steps(command)[0]["requested_logical_steps"], 30)
            # Reopening uses the persisted motor position as this batch's baseline.
            with RelayLabSession(config, skeleton=FakeSkeleton(), rig_factory=dry_rig) as session:
                self.assertEqual(session.baseline_positions[AXES[0]], 130)
                command = dict.fromkeys(AXES, 0)
                command[AXES[0]] = -10
                self.assertEqual(session.set_motor_steps(command)[0]["requested_logical_steps"], -10)
                self.assertEqual(session.rig.state.current_positions[AXES[0]], 120)

    def test_capture_keep_and_return_use_real_camera_only(self):
        with tempfile.TemporaryDirectory() as directory:
            skeleton = FakeSkeleton()
            with RelayLabSession(filled_config(directory), skeleton=skeleton, rig_factory=dry_rig) as session:
                records = [session.capture(choice, Path(directory), i)
                           for i, choice in enumerate(["C1", "keep", "C1", "C2", "C1"])]
                self.assertEqual([r["camera_moved"] for r in records], [True, False, False, True, True])
                self.assertEqual(session.camera_image_count, 5)
                self.assertEqual(session.camera_placements, 3)
                self.assertEqual(len([e for e in skeleton.events if isinstance(e, tuple) and e[0] == "place"]), 3)
                self.assertEqual(len(list((Path(directory) / "raw").glob("*.png"))), 5)
                with Image.open(records[-1]["image_path"]) as image:
                    self.assertEqual(image.size, (64, 48))
                self.assertNotIn("actual_camera", records[-1])



    def test_image_preserves_aspect_ratio_and_records_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = prepare_llm_image(np.full((100, 200, 3), 200, dtype=np.uint8),
                                         Path(directory) / "image.png", 64, 48)
            self.assertEqual(metadata["image_content_box_px"], [0, 8, 64, 32])
            with Image.open(Path(directory) / "image.png") as png:
                image = np.asarray(png)
            self.assertTrue(np.all(image[:8] == 0))
            self.assertTrue(np.all(image[8:40] == 200))


if __name__ == "__main__":
    unittest.main()
