from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.io.schemas import load_michelson_interferometer_compare_methods_config
from src.runners.run_experiment import run_michelson_interferometer_compare_methods
from src.sim.michelson_interferometer_simulation import MichelsonInterferometerSimulation
from src.tasks.michelson_interferometer_oracle_alignment import plotted_center_error_px


AXIS_NAMES = (
    "mirror_1_axis_1",
    "mirror_1_axis_2",
    "mirror_2_axis_1",
    "mirror_2_axis_2",
)


class MichelsonInterferometerCompareMethodsTest(unittest.TestCase):
    def test_oracle_objective_matches_plotted_pixel_metric(self) -> None:
        metrics = {
            "first_center_normalized": {"x": 1.0, "y": 0.0},
            "second_center_normalized": {"x": 0.0, "y": 1.0},
        }
        self.assertAlmostEqual(plotted_center_error_px(metrics), 959.5 + 539.5)

    def test_true_position_can_render_beyond_visible_command_limit(self) -> None:
        simulation = MichelsonInterferometerSimulation()
        command = {name: 10_000 for name in AXIS_NAMES}
        clipped = simulation.beam_centers(command)
        true = simulation.beam_centers(command, clip_command=False)
        self.assertNotEqual(clipped.first_x_px, true.first_x_px)

    def test_config_and_mock_comparison_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial_path = root / "initial.json"
            initial = [{name: (100 if index % 2 == 0 else -100) for index, name in enumerate(AXIS_NAMES)}]
            initial_path.write_text(json.dumps(initial), encoding="utf-8")
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "experiment_type": "michelson_interferometer_compare_methods",
                        "experiment_name": "comparison",
                        "output_root": str(root / "runs"),
                        "repeats": 1,
                        "iterations": 1,
                        "base_seed": 7,
                        "max_parallel_jobs": 1,
                        "methods": ["llm", "bayes", "random"],
                        "initial_conditions_path": str(initial_path),
                        "simulation": {"frame_width_px": 160, "frame_height_px": 90},
                        "axes": [
                            {
                                "name": name,
                                "visible_step_limit": 8192 if index % 2 == 0 else 6144,
                                "hidden_offset_max_steps": 6144 if index % 2 == 0 else 4096,
                            }
                            for index, name in enumerate(AXIS_NAMES)
                        ],
                        "llm_models": [
                            {
                                "key": "mock_model",
                                "model": "mock/test",
                                "effort": "none",
                                "max_tokens": 100,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = load_michelson_interferometer_compare_methods_config(config_path)
            self.assertEqual([axis.visible_step_limit for axis in config.axes], [8192, 6144, 8192, 6144])
            run_root = run_michelson_interferometer_compare_methods(config_path)
            for method in ("mock_model", "bayes", "random"):
                manifest = json.loads(
                    (run_root / method / "repeat_00" / "run_manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(manifest["hidden_offset_steps"], initial[0])
                self.assertIn("center_error_normalized", manifest["final_alignment_metrics"])
                if method in {"bayes", "random"}:
                    self.assertEqual(
                        manifest["objective_name"],
                        "negative_sum_beam_distances_from_center_px",
                    )
                    self.assertIn("plotted_center_error_px", manifest["steps"][0])
            resumed_root = run_michelson_interferometer_compare_methods(
                config_path,
                resume_run_root=run_root,
            )
            self.assertEqual(resumed_root, run_root.resolve())
            root_manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(
                all(
                    method["resumed_existing"]
                    for repeat in root_manifest["repeats"]
                    for method in repeat["methods"]
                )
            )
            incomplete_dir = run_root / "random" / "repeat_00"
            (incomplete_dir / "run_manifest.json").unlink()
            stale_image = incomplete_dir / "step_999.png"
            stale_image.write_bytes(b"stale")
            run_michelson_interferometer_compare_methods(
                config_path,
                resume_run_root=run_root,
            )
            self.assertFalse(stale_image.exists())
            self.assertTrue((incomplete_dir / "run_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
