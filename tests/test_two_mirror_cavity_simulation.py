"""Tests for the lab-derived two-mirror cavity simulation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.analysis.extract_two_mirror_cavity_profiles import extract_profiles
from src.sim.two_mirror_cavity_simulation import (
    DEFAULT_PROFILE_PATH,
    TwoMirrorCavityConfig,
    TwoMirrorCavitySimulation,
)


class TwoMirrorCavitySimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.simulation = TwoMirrorCavitySimulation()

    def test_shipped_profiles_are_valid_and_preserve_relative_scale(self) -> None:
        with np.load(DEFAULT_PROFILE_PATH) as profiles:
            main = profiles["main_intensity"]
            secondary = profiles["secondary_intensity"]
            support = profiles["secondary_support"]
        self.assertEqual(main.shape, (540, 960))
        self.assertEqual(secondary.shape, main.shape)
        self.assertTrue(np.all(np.isfinite(main)))
        self.assertTrue(np.all(np.isfinite(secondary)))
        self.assertGreaterEqual(float(np.min(main)), 0.0)
        self.assertGreaterEqual(float(np.min(secondary)), 0.0)
        self.assertGreater(float(np.max(secondary)), 0.0)
        self.assertLess(float(np.max(secondary)), float(np.max(main)))
        self.assertTrue(np.all(secondary[support == 0.0] == 0.0))

        provenance = json.loads(DEFAULT_PROFILE_PATH.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertIn("source_id", provenance["main_source"])
        self.assertEqual(len(provenance["main_source"]["sha256"]), 64)
        stats = provenance["profile_statistics"]
        self.assertAlmostEqual(float(np.max(main)), stats["main_peak"], places=6)
        self.assertAlmostEqual(float(np.max(secondary)), stats["secondary_peak"], places=6)

    def test_zero_command_aligns_centers(self) -> None:
        centers = self.simulation.beam_centers(self.simulation.zero_command())
        self.assertAlmostEqual(centers.main_x_px, centers.secondary_x_px)
        self.assertAlmostEqual(centers.main_y_px, centers.secondary_y_px)
        self.assertAlmostEqual(centers.separation_px, 0.0)
        secondary = self.simulation.render_components({"x": 0, "y": 0})["secondary_intensity"]
        y, x = np.indices(secondary.shape, dtype=np.float64)
        self.assertAlmostEqual(float(np.sum(x * secondary) / np.sum(secondary)), centers.main_x_px, places=4)
        self.assertAlmostEqual(float(np.sum(y * secondary) / np.sum(secondary)), centers.main_y_px, places=4)

    def test_lab_axis_signs_and_scales(self) -> None:
        config = self.simulation.config
        centers = self.simulation.beam_centers({"x": 1000, "y": 1000})
        self.assertAlmostEqual(
            centers.secondary_x_px - centers.main_x_px,
            1000 * config.x_image_pixels_per_step,
        )
        self.assertAlmostEqual(
            centers.secondary_y_px - centers.main_y_px,
            1000 * config.y_image_pixels_per_step,
        )
        self.assertLess(centers.secondary_x_px, centers.main_x_px)
        self.assertLess(centers.secondary_y_px, centers.main_y_px)

    def test_command_clipping_and_offscreen_translation(self) -> None:
        self.assertEqual(
            self.simulation.clip_command({"x": 99999, "y": -99999}),
            {"x": 8000, "y": -5000},
        )
        components = self.simulation.render_components({"x": 8000, "y": 5000})
        centered_sum = float(
            np.sum(self.simulation.render_components({"x": 0, "y": 0})["secondary_intensity"])
        )
        shifted_sum = float(np.sum(components["secondary_intensity"]))
        self.assertLess(shifted_sum, centered_sum)

    def test_intensity_addition_and_sensor_clipping(self) -> None:
        components = self.simulation.render_components({"x": 0, "y": 0})
        expected = components["main_intensity"] + components["secondary_intensity"]
        np.testing.assert_allclose(components["combined_intensity"], expected)
        rendered = self.simulation.render({"x": 0, "y": 0})
        self.assertGreaterEqual(float(np.min(rendered)), 0.0)
        self.assertLessEqual(float(np.max(rendered)), self.simulation.config.sensor_saturation)
        self.assertTrue(np.any(components["combined_intensity"] > rendered))

    def test_extractor_localizes_nonnegative_secondary(self) -> None:
        height, width = 540, 960
        y, x = np.indices((height, width), dtype=np.float64)
        main = 20.0 + 180.0 * np.exp(-((x - 470.0) ** 2 + (y - 276.0) ** 2) / 5000.0)
        secondary = 35.0 * np.exp(-((x - 110.0) ** 2 + (y - 292.0) ** 2) / 800.0)
        combined = np.clip(main + secondary, 0.0, 255.0)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            main_path = root / "main.png"
            combined_path = root / "combined.png"
            Image.fromarray(main.astype(np.uint8)).save(main_path)
            Image.fromarray(combined.astype(np.uint8)).save(combined_path)
            metadata = {
                "llm_image_shape": [540, 960],
                "average_frames": 2,
                "exposure_ms": 300.0,
                "analog_gain": 10,
            }
            main_path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
            combined_path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
            arrays, _ = extract_profiles(main_path, combined_path)
        extracted = arrays["secondary_intensity"]
        support = arrays["secondary_support"]
        self.assertGreater(float(np.max(extracted)), 0.0)
        self.assertGreaterEqual(float(np.min(extracted)), 0.0)
        self.assertTrue(np.all(extracted[support == 0.0] == 0.0))


if __name__ == "__main__":
    unittest.main()
