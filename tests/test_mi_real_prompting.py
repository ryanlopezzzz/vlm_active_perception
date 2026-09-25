from __future__ import annotations

from pathlib import Path
import unittest

from src.io.schemas import RealMIAxisConfig
from src.tasks.mi_real_prompting import (
    MI_REAL_PROMPT_PATH,
    build_real_mi_system_prompt,
    verify_real_motor_command,
)
from src.tasks.michelson_interferometer_prompting import MI_INTERFEROMETER_PROMPT_PATH


class RealMIPromptingTests(unittest.TestCase):
    def test_repository_has_exactly_three_canonical_prompts(self) -> None:
        prompt_dir = Path(__file__).resolve().parents[1] / "src/prompts"
        self.assertEqual(
            sorted(path.name for path in prompt_dir.glob("*.txt")),
            ["four_mirror_relay.txt", "michelson_interferometer.txt", "two_mirror_cavity.txt"],
        )
        self.assertEqual(MI_REAL_PROMPT_PATH, MI_INTERFEROMETER_PROMPT_PATH)

    def test_real_prompt_is_standard_motor_step_prompt(self) -> None:
        axes = [
            RealMIAxisConfig(
                name="mirror_1_axis_1",
                controller_ip="x",
                motor_number=1,
                hidden_offset_max_steps=4096,
                llm_step_limit=4096,
            ),
            RealMIAxisConfig(
                name="mirror_1_axis_2",
                controller_ip="x",
                motor_number=2,
                hidden_offset_max_steps=4096,
                llm_step_limit=4096,
            ),
        ]
        prompt = build_real_mi_system_prompt(max_iterations=12, axes=axes)
        self.assertIn("12", prompt)
        self.assertIn("4096", prompt)
        self.assertIn("absolute visible motor-step settings", prompt)
        self.assertIn("visual_description", prompt)
        self.assertIn("mirror_1_axis_1", prompt)
        self.assertNotIn("{{MAX_ITERATIONS}}", prompt)
        self.assertNotIn("{{AXIS_LIMITS}}", prompt)
        self.assertNotIn("mrad", prompt)
        self.assertNotIn("simulated", prompt.lower())
        self.assertNotIn("coherent sum", prompt.lower())

    def test_verify_real_motor_command(self) -> None:
        axes = [
            RealMIAxisConfig(name="mirror_1_axis_1", controller_ip="x", motor_number=1, llm_step_limit=10),
            RealMIAxisConfig(name="mirror_1_axis_2", controller_ip="x", motor_number=2, llm_step_limit=10),
        ]
        valid = '{"visual_description": "two beams visible", "done": false, "command": {"mirror_1_axis_1": 3, "mirror_1_axis_2": -2}}'
        done_nonzero = '{"visual_description": "best overlap is at nonzero absolute settings", "done": true, "command": {"mirror_1_axis_1": 3, "mirror_1_axis_2": 0}}'
        invalid = '{"visual_description": "too far", "done": false, "command": {"mirror_1_axis_1": 11, "mirror_1_axis_2": 0}}'
        missing_description = '{"done": false, "command": {"mirror_1_axis_1": 3, "mirror_1_axis_2": -2}}'
        self.assertTrue(verify_real_motor_command(valid, axes))
        self.assertTrue(verify_real_motor_command(done_nonzero, axes))
        self.assertFalse(verify_real_motor_command(invalid, axes))
        self.assertFalse(verify_real_motor_command(missing_description, axes))

if __name__ == "__main__":
    unittest.main()
