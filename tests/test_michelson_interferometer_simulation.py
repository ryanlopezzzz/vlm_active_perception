import math
import unittest

import numpy as np

from src.sim.michelson_interferometer_simulation import MichelsonInterferometerSimulation, MichelsonInterferometerConfig


class MichelsonInterferometerTest(unittest.TestCase):
    def test_default_dimensions_and_calibration(self) -> None:
        config = MichelsonInterferometerConfig()
        self.assertEqual(config.frame_width_px, 960)
        self.assertEqual(config.frame_height_px, 540)
        self.assertEqual(config.beam_diameter_px, 240)
        self.assertEqual(config.beam_radius_px, 120)
        self.assertEqual(config.gaussian_sigma_px, 60)
        self.assertAlmostEqual(config.pixels_per_step, 480 / 4096)
        self.assertEqual(config.max_steps, 8192)
        self.assertEqual(config.fringe_period_px, 30)
        self.assertAlmostEqual(config.fringe_radians_per_px, 2 * math.pi / 30)

    def test_axis_directions(self) -> None:
        simulation = MichelsonInterferometerSimulation()
        scale = simulation.config.pixels_per_step
        centers = simulation.beam_centers(
            {
                "mirror_1_axis_1": 100,
                "mirror_1_axis_2": 200,
                "mirror_2_axis_1": -300,
                "mirror_2_axis_2": -400,
            }
        )
        self.assertAlmostEqual(centers.first_x_px, simulation.config.center_x_px - 100 * scale)
        self.assertAlmostEqual(centers.first_y_px, simulation.config.center_y_px - 200 * scale)
        self.assertAlmostEqual(centers.second_x_px, simulation.config.center_x_px + 300 * scale)
        self.assertAlmostEqual(centers.second_y_px, simulation.config.center_y_px + 400 * scale)

    def test_render_shape_and_range(self) -> None:
        simulation = MichelsonInterferometerSimulation()
        image = simulation.render(simulation.zero_command())
        self.assertEqual(image.shape, (540, 960))
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)

    def test_command_clipping_supports_two_rotations(self) -> None:
        simulation = MichelsonInterferometerSimulation()
        command = {axis: 999_999 for axis in simulation.zero_command()}
        clipped = simulation.clip_command(command)
        self.assertEqual(set(clipped.values()), {8192})

    def test_phase_ramp_has_eight_fringes_across_beam_diameter(self) -> None:
        simulation = MichelsonInterferometerSimulation()
        phase = simulation._build_phase_ramp()
        cx = int(simulation.config.center_x_px)
        cy = int(simulation.config.center_y_px)
        half_diagonal = int(round(simulation.config.beam_diameter_px / (2 * math.sqrt(2))))
        phase_delta = phase[cy - half_diagonal, cx + half_diagonal] - phase[cy + half_diagonal, cx - half_diagonal]
        cycles = phase_delta / (2 * math.pi)
        self.assertAlmostEqual(cycles, 8.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
