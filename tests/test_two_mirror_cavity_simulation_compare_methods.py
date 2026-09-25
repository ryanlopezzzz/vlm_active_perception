from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.io.schemas import load_two_mirror_cavity_simulation_compare_methods_config
from src.sim.two_mirror_cavity_simulation import TwoMirrorCavityConfig, TwoMirrorCavitySimulation
from src.tasks.two_mirror_cavity_simulation_alignment import (
    _metrics,
    _physical_command,
    motor_distance_steps,
    run_two_mirror_cavity_simulation_oracle_alignment,
)


CONFIG_PATH = Path("configs/two_mirror_cavity_simulation/compare_methods.yaml")


class TwoMirrorCavitySimulationComparisonTests(unittest.TestCase):
    def test_comparison_config_matches_final_lab_experiment(self) -> None:
        config = load_two_mirror_cavity_simulation_compare_methods_config(CONFIG_PATH)
        conditions = json.loads(Path(config.initial_conditions_path).read_text(encoding="utf-8"))
        self.assertEqual(config.repeats, 10)
        self.assertEqual(config.iterations, 50)
        self.assertEqual(config.probes_per_batch, 1)
        self.assertEqual(
            [model.key for model in config.llm_models],
            ["gpt_5_6_sol", "gemini_3_5_flash", "qwen3_vl_32b_instruct"],
        )
        self.assertEqual(conditions[0], {"x": -1313, "y": -1748})
        self.assertEqual(conditions[-1], {"x": -3688, "y": 1780})

    def test_visible_command_cancels_hidden_lab_offset(self) -> None:
        config = load_two_mirror_cavity_simulation_compare_methods_config(CONFIG_PATH)
        simulation = TwoMirrorCavitySimulation(TwoMirrorCavityConfig(profile_path=config.profile_path))
        hidden = {"x": -1313, "y": -1748}
        physical = _physical_command(hidden, {"x": 1313, "y": 1748})
        self.assertEqual(physical, {"x": 0, "y": 0})
        self.assertAlmostEqual(_metrics(simulation, physical)["center_error_normalized"], 0.0)
        self.assertAlmostEqual(motor_distance_steps(physical), 0.0)

    def test_motor_distance_matches_combined_plot_metric(self) -> None:
        self.assertAlmostEqual(motor_distance_steps({"x": 300, "y": -400}), 500.0)

    def test_oracle_methods_write_complete_manifests(self) -> None:
        config = load_two_mirror_cavity_simulation_compare_methods_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temp_dir:
            for method in ("bayes", "random"):
                manifest = run_two_mirror_cavity_simulation_oracle_alignment(
                    out_dir=Path(temp_dir) / method, method=method, iterations=2, seed=7,
                    repeat_index=0, hidden_offset={"x": -1313, "y": -1748},
                    profile_path=config.profile_path, axes=config.axes,
                )
                self.assertEqual(manifest["objective_name"], "negative_motor_distance_steps")
                self.assertEqual(len(manifest["steps"]), 3)
                self.assertTrue(Path(manifest["final_image_path"]).is_file())
                self.assertTrue((Path(temp_dir) / method / "run_manifest.json").is_file())
                self.assertTrue(all(step["objective"] <= 0 for step in manifest["steps"]))
                self.assertTrue(all("motor_distance_steps" in step for step in manifest["steps"]))


if __name__ == "__main__":
    unittest.main()
