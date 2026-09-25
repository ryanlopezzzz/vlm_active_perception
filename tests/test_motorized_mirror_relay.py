from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.sim.motorized_mirror_relay import AXES, CameraPose, MotorizedMirrorRelay, RelayConfig


class RelayTests(unittest.TestCase):
    def clean(self, **kwargs):
        return MotorizedMirrorRelay(RelayConfig(**kwargs), hidden_offsets=dict.fromkeys(AXES, 0))

    def test_geometry_and_exact_motor_cancellation(self):
        sim = self.clean()
        np.testing.assert_allclose(sim.nominal_centers[2] - sim.nominal_centers[0], [279.4, 203.2, 0])
        self.assertEqual(sim.config.mirror_diameter_mm, 25.4)
        self.assertAlmostEqual(4096 * sim.config.radians_per_step, 0.027)
        sim.hidden_offsets = {axis: 8192 if i % 2 else -8192 for i, axis in enumerate(AXES)}
        command = {axis: -value for axis, value in sim.hidden_offsets.items()}
        result = sim.trace(command, sim.camera_locations["C8"])
        self.assertTrue(result["screen_hit"])
        self.assertEqual(result["mirror_hit_sequence"], ["M1", "M2", "M3", "M4"])
        self.assertLess(result["beam_offset_mm"], 1e-10)

    def test_motor_normal_uses_mirror_calibration(self):
        sim = self.clean()
        command = sim.zero_command()
        command[AXES[0]] = 4096
        angle = np.arccos(sim.normals(command)[0] @ sim.ideal_normals[0])
        self.assertAlmostEqual(angle, np.arctan(0.027))

    def test_eight_stations_have_clearance_and_face_each_ideal_leg(self):
        sim = self.clean()
        self.assertEqual(list(sim.camera_locations), [f"C{i}" for i in range(1, 9)])
        self.assertEqual(len({p.center_mm for p in sim.camera_locations.values()}), 8)
        for index, pose in enumerate(sim.camera_locations.values()):
            self.assertGreaterEqual(np.min(np.linalg.norm(sim.nominal_centers - pose.center_mm, axis=1)), 40.0)
            trace = sim.trace(sim.zero_command(), pose)
            self.assertTrue(trace["screen_hit"])
            self.assertEqual(trace["mirror_hit_sequence"], [f"M{i}" for i in range(1, index // 2 + 2)])
            self.assertAlmostEqual(np.dot(pose.basis()[0], trace["final_direction"]), -1.)
        np.testing.assert_allclose(sim.camera_locations["C1"].center_mm, [120, 42.5, 0])
        np.testing.assert_allclose(sim.camera_locations["C2"].center_mm, [120, 160.7, 0])
        np.testing.assert_allclose(sim.camera_locations["C7"].center_mm, [441.9, 0, 0])
        np.testing.assert_allclose(sim.camera_locations["C8"].center_mm, [481.9, 0, 0])
        with self.assertRaises(ValueError):
            self.clean(height_mm=80.)

    def test_missed_camera_plane_does_not_block_later_camera_hit(self):
        sim = self.clean()
        # The infinite camera plane crosses the input ray before M1, far outside
        # its aperture; the finite camera actually receives the beam after M2.
        pose = CameraPose((200., 203.2, 0.), 148.5, 0.)
        trace = sim.trace(sim.zero_command(), pose)
        self.assertTrue(trace["screen_hit"])
        self.assertEqual(trace["mirror_hit_sequence"], ["M1", "M2"])
        self.assertLess(trace["beam_offset_mm"], 1e-10)

    def test_disk_rejects_square_corner(self):
        sim = self.clean()
        # Move M1 in its tangent plane: each coordinate lies inside a 25.4 mm
        # square, but radius sqrt(10^2+10^2) lies outside the 12.7 mm disk.
        tangent = np.array([1., 1., 0.]) / np.sqrt(2)
        sim.centers[0] += 10 * tangent + np.array([0, 0, 10.])
        trace = sim.trace(sim.zero_command(), CameraPose((120., 40., 0.), -90., 0.))
        self.assertNotIn("M1", trace["mirror_hit_sequence"])
        sim.centers[0] -= np.array([0, 0, 10.])
        trace = sim.trace(sim.zero_command(), CameraPose((120., 40., 0.), -90., 0.))
        self.assertEqual(trace["mirror_hit_sequence"], ["M1"])

    def test_rectangular_camera_and_gaussian_radius(self):
        sim = self.clean()
        for y, z, expected in [(3.1, 0, True), (3.3, 0, False), (0, 2.3, True), (0, 2.5, False)]:
            pose = CameraPose((sim.exit_pose().center_mm[0], y, z))
            self.assertEqual(sim.trace(sim.zero_command(), pose)["screen_hit"], expected)
        trace = sim.trace(sim.zero_command(), sim.exit_pose())
        image = sim.image(trace)
        self.assertEqual(image.shape, (480, 640))
        self.assertEqual(image.dtype, np.uint8)
        # Pixel centers near r=0.8 mm should be near exp(-2) of peak.
        self.assertAlmostEqual(float(image[239, 399]) / 255, np.exp(-2), delta=0.006)
        self.assertEqual(int(sim.image({"plane_reached": False}).max()), 0)

    def test_geometry_is_exact_and_hidden_offsets_are_seeded(self):
        sim, twin = MotorizedMirrorRelay(seed=72), MotorizedMirrorRelay(seed=72)
        np.testing.assert_array_equal(sim.centers, twin.centers)
        np.testing.assert_array_equal(sim.centers, sim.nominal_centers)
        self.assertEqual(sim.hidden_offsets, twin.hidden_offsets)
        with tempfile.TemporaryDirectory() as directory:
            captures = [sim.capture(sim.zero_command(), location, Path(directory) / f"{i}.png")
                        for i, location in enumerate(("C7", "C8"))]
            first_twin = twin.capture(twin.zero_command(), "C7", Path(directory) / "twin.png")
            self.assertEqual(captures[0]["actual_camera"], first_twin["actual_camera"])
            self.assertEqual(captures[0]["actual_camera"], captures[0]["nominal_camera"])
            self.assertEqual(captures[1]["actual_camera"], captures[1]["nominal_camera"])
        self.assertEqual(sim.camera_placements, 2)

    def test_keep_preserves_actual_pose_and_rng_but_applies_motor_changes(self):
        sim = MotorizedMirrorRelay(seed=72, hidden_offsets=dict.fromkeys(AXES, 0))
        twin = MotorizedMirrorRelay(seed=72, hidden_offsets=dict.fromkeys(AXES, 0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.png"
            with self.assertRaises(ValueError):
                sim.capture(sim.zero_command(), "keep", path)
            first = sim.capture(sim.zero_command(), "C1", path)
            first_image = np.asarray(Image.open(path)).copy()
            twin.capture(twin.zero_command(), "C1", path)
            command = sim.zero_command()
            command[AXES[0]] = 300
            kept = sim.capture(command, "keep", path)
            self.assertFalse(np.array_equal(first_image, np.asarray(Image.open(path))))
            same_id = sim.capture(command, "C1", path)
            self.assertEqual(first["actual_camera"], kept["actual_camera"])
            self.assertEqual(kept["actual_camera"], same_id["actual_camera"])
            self.assertFalse(kept["camera_moved"])
            self.assertFalse(same_id["camera_moved"])
            self.assertEqual(sim.camera_placements, 1)
            moved = sim.capture(command, "C2", path)
            twin_moved = twin.capture(command, "C2", path)
            self.assertEqual(moved["actual_camera"], twin_moved["actual_camera"])
            returned = sim.capture(command, "C1", path)
            self.assertEqual(returned["actual_camera"], first["actual_camera"])
            self.assertEqual(sim.camera_placements, 3)
            self.assertEqual(sim.camera_image_count, 5)






if __name__ == "__main__":
    unittest.main()
