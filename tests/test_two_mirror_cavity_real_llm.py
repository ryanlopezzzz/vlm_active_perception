from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.cli.main import main as cli_main
from src.hardware.camera import CaptureArtifacts
from src.io.schemas import (
    LLMConfig,
    TwoMirrorCavityCameraConfig,
    TwoMirrorCavityHardwareConfig,
    TwoMirrorCavityRealLLMAxisConfig,
    load_two_mirror_cavity_real_llm_runs_config,
)
from src.tasks.two_mirror_cavity_real_llm_alignment import (
    absolute_target_delta,
    absolute_visible_to_physical,
    run_two_mirror_cavity_real_llm_alignment,
)
from src.tasks.two_mirror_cavity_real_llm_prompting import (
    build_system_prompt,
    sample_distinct_hidden_offsets,
    verify_response,
)
from src.runners.run_experiment import run_two_mirror_cavity_real_llm_runs


def axes() -> list[TwoMirrorCavityRealLLMAxisConfig]:
    return [
        TwoMirrorCavityRealLLMAxisConfig(
            "cavity_beam_x", "x", "ip", 3, 1, -3000, 4000, -7000, 7000, -0.0002
        ),
        TwoMirrorCavityRealLLMAxisConfig(
            "cavity_beam_y", "y", "ip", 1, 1, -2000, 2000, -5000, 5000, 0.0003
        ),
    ]


def load_checked_release_config(path: Path):
    source = path.read_text(encoding="utf-8").replace("REPLACE_ME", "192.0.2.1")
    with tempfile.TemporaryDirectory() as directory:
        resolved = Path(directory) / "config.yaml"
        resolved.write_text(source, encoding="utf-8")
        return load_two_mirror_cavity_real_llm_runs_config(resolved)


def observation(index: int, command: dict[str, int]) -> dict[str, object]:
    return {
        "image_index": index,
        "command": command,
        "visibility": "visible",
        "x": 0.1,
        "y": -0.2,
        "confidence": "high",
        "evidence": "A separate faint spot is visible.",
    }


def response(
    commands: list[dict[str, int]], *, done: bool, final=None,
    probe_commands: list[dict[str, int]] | None = None,
) -> str:
    return json.dumps(
        {
            "visual_description": "The faint beam follows a consistent trajectory.",
            "image_observations": [observation(i, command) for i, command in enumerate(commands, 1)],
            "strategy_summary": "Use the inferred linear response to approach the main beam.",
            "done": done,
            "probe_commands": [] if done else (probe_commands or PROBES[:5]),
            "final_command": final if done else None,
        }
    )


PROBES = [
    {"x": -1000, "y": 500},
    {"x": 1000, "y": 500},
    {"x": 0, "y": -500},
    {"x": 0, "y": 500},
    {"x": 500, "y": 0},
    {"x": -500, "y": 0},
    {"x": 500, "y": 500},
    {"x": -500, "y": -500},
]


class FakeCamera:
    instances: list["FakeCamera"] = []

    def __init__(self, config) -> None:
        self.closed = False
        self.__class__.instances.append(self)

    def open(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    def save_capture(self, out_dir, stem, **kwargs) -> CaptureArtifacts:
        path = Path(out_dir) / f"{stem}.png"
        cv2.imwrite(str(path), np.zeros((40, 60, 3), dtype=np.uint8))
        return CaptureArtifacts(None, str(path), None, (2, 2), "uint8")


class FakeLLM:
    responses: list[object] = []
    instances: list["FakeLLM"] = []

    def __init__(self, **kwargs) -> None:
        self.messages = []
        self.api_calls = []
        self.__class__.instances.append(self)

    def add_system_message(self, value) -> None:
        self.messages.append(("system", value))

    def add_user_message(self, value) -> None:
        self.messages.append(("text", value))

    def add_user_image(self, value) -> None:
        self.messages.append(("image", value))

    def query_llm(self, *, verify, **kwargs) -> str:
        value = self.__class__.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        assert isinstance(value, str) and verify(value)
        self.api_calls.append({"duration_sec": 0.01})
        return value

    def get_usage_summary(self):
        return {"request_attempts": len(self.api_calls), "total_tokens": 10 * len(self.api_calls)}


class TwoMirrorCavityRealLLMTests(unittest.TestCase):
    def test_absolute_coordinate_math(self) -> None:
        self.assertEqual(
            absolute_visible_to_physical({"x": 100, "y": -50}, {"x": -1500, "y": 200}),
            {"x": -1400, "y": 150},
        )
        self.assertEqual(
            absolute_target_delta(
                {"x": -1000, "y": 500}, {"x": -1500, "y": 200}
            ),
            {"x": -500, "y": -300},
        )

    def test_sampling_is_unique_reproducible_and_allows_origin(self) -> None:
        first = sample_distinct_hidden_offsets(axes(), repeats=5, base_seed=22)
        second = sample_distinct_hidden_offsets(axes(), repeats=5, base_seed=22)
        self.assertEqual(first, second)
        self.assertEqual(len({(item["x"], item["y"]) for item in first}), 5)
        origin_axes = [
            TwoMirrorCavityRealLLMAxisConfig("x", "x", "ip", 3, 1, 0, 0, -1, 1, 0),
            TwoMirrorCavityRealLLMAxisConfig("y", "y", "ip", 1, 1, 0, 0, -1, 1, 0),
        ]
        self.assertEqual(
            sample_distinct_hidden_offsets(origin_axes, repeats=1, base_seed=1),
            [{"x": 0, "y": 0}],
        )

    def test_response_validation(self) -> None:
        initial = [{"x": 0, "y": 0}]
        self.assertTrue(
            verify_response(
                response(initial, done=False), axes(), expected_observation_count=1,
                expected_commands=initial, probes_per_batch=5,
            )
        )
        self.assertFalse(
            verify_response(
                response(initial, done=False), axes(), expected_observation_count=1,
                expected_commands=initial, probes_per_batch=5, force_final=True,
            )
        )
        self.assertTrue(
            verify_response(
                response(PROBES[:5], done=True, final={"x": -1500, "y": 200}), axes(),
                expected_observation_count=5, expected_commands=PROBES[:5],
                probes_per_batch=5, force_final=True,
            )
        )

    def test_response_validation_supports_every_batch_size(self) -> None:
        initial = [{"x": 0, "y": 0}]
        for count in range(1, 9):
            with self.subTest(count=count):
                answer = response(
                    initial,
                    done=False,
                    probe_commands=PROBES[:count],
                )
                self.assertTrue(
                    verify_response(
                        answer,
                        axes(),
                        expected_observation_count=1,
                        expected_commands=initial,
                        probes_per_batch=count,
                    )
                )

    def test_response_validation_allows_relaxed_descriptions_and_overlap_coordinates(self) -> None:
        command = {"x": 0, "y": 0}
        payload = json.loads(
            response([command], done=False, probe_commands=PROBES[:1])
        )
        payload["visual_description"] = None
        payload["strategy_summary"] = {"fit": "still accepted by validation"}
        payload["image_observations"][0].update(
            {
                "visibility": "overlapped_or_hidden",
                "x": 0.05,
                "y": -0.1,
            }
        )

        self.assertTrue(
            verify_response(
                json.dumps(payload),
                axes(),
                expected_observation_count=1,
                expected_commands=[command],
                probes_per_batch=1,
            )
        )

        payload["image_observations"][0]["visibility"] = "offscreen"
        self.assertFalse(
            verify_response(
                json.dumps(payload),
                axes(),
                expected_observation_count=1,
                expected_commands=[command],
                probes_per_batch=1,
            )
        )

    def test_system_prompt_example_matches_configured_batch_size(self) -> None:
        for count in (1, 3, 8):
            with self.subTest(count=count):
                prompt = build_system_prompt(axes(), max_probe_batches=10, probes_per_batch=count)
                encoded = prompt.split('"probe_commands": ', 1)[1].split(',\n  "final_command"', 1)[0]
                self.assertEqual(len(json.loads(encoded)), count)
                self.assertIn(f"exactly {count} distinct", prompt)

    def test_accuracy_first_prompt_preserves_contract_and_adds_validation_rules(self) -> None:
        prompt = build_system_prompt(
            axes(),
            max_probe_batches=10,
            probes_per_batch=5,
        )
        self.assertIn("Accuracy is more important than minimizing batches", prompt)
        self.assertIn("center of the smaller", prompt)
        self.assertIn("Require genuine two-sided validation", prompt)
        self.assertIn("Reject stationary optical artifacts", prompt)
        self.assertIn("Use a quantitative independent-axis fit", prompt)
        self.assertIn("Treat disappearance conservatively", prompt)
        self.assertIn("Resolve contradictions before finishing", prompt)
        self.assertIn("Verify the final candidate with a local cross", prompt)
        self.assertNotIn("0.000231", prompt)
        self.assertIn("exactly 5 distinct", prompt)

    def test_workflow_batches_absolute_commands_and_restores(self) -> None:
        FakeCamera.instances.clear()
        FakeLLM.instances.clear()
        FakeLLM.responses = [
            response([{"x": 0, "y": 0}], done=False),
            response(PROBES[:5], done=True, final={"x": -1500, "y": 200}),
        ]
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.tasks.two_mirror_cavity_real_llm_alignment.MindVisionCamera", FakeCamera
        ), patch("src.tasks.two_mirror_cavity_real_llm_alignment.LLM", FakeLLM):
            base = Path(tmp)
            manifest = run_two_mirror_cavity_real_llm_alignment(
                out_dir=base / "out",
                repeat_index=0,
                seed=1,
                hidden_offset={"x": 100, "y": -50},
                iterations=10,
                probes_per_batch=5,
                llm_config=LLMConfig(model="mock"),
                camera_config=TwoMirrorCavityCameraConfig(warmup_frames=0),
                hardware_config=TwoMirrorCavityHardwareConfig(
                    session_state_path=str(base / "session.json"), settle_time_sec=0, dry_run=True
                ),
                axes=axes(),
                reset_session=True,
            )
            state = json.loads((base / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(state["current_positions"], {"cavity_beam_x": 0, "cavity_beam_y": 0})
            self.assertEqual(manifest["final"]["physical_position_steps"], {"x": -1400, "y": 150})
            self.assertTrue(manifest["restoration"]["succeeded"])
            self.assertEqual(len(manifest["batches"][0]["captures"]), 5)
            self.assertEqual(manifest["image_artifacts"]["image_count"], 9)
            self.assertEqual(manifest["image_artifacts"]["red_x_count"], 6)
            self.assertTrue((base / "out/images/original").is_dir())
            self.assertTrue((base / "out/images/annotated").is_dir())
            labels = [value for kind, value in FakeLLM.instances[-1].messages if kind == "text"]
            self.assertTrue(any("absolute visible command x=-1000, y=500" in label for label in labels))
        self.assertTrue(FakeCamera.instances[-1].closed)

    def test_keyboard_interrupt_still_restores_and_closes(self) -> None:
        FakeCamera.instances.clear()
        FakeLLM.responses = [KeyboardInterrupt()]
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.tasks.two_mirror_cavity_real_llm_alignment.MindVisionCamera", FakeCamera
        ), patch("src.tasks.two_mirror_cavity_real_llm_alignment.LLM", FakeLLM):
            base = Path(tmp)
            with self.assertRaises(KeyboardInterrupt):
                run_two_mirror_cavity_real_llm_alignment(
                    out_dir=base / "out", repeat_index=0, seed=1,
                    hidden_offset={"x": 100, "y": -50}, iterations=10, probes_per_batch=5,
                    llm_config=LLMConfig(model="mock"),
                    camera_config=TwoMirrorCavityCameraConfig(warmup_frames=0),
                    hardware_config=TwoMirrorCavityHardwareConfig(
                        session_state_path=str(base / "session.json"), settle_time_sec=0, dry_run=True
                    ),
                    axes=axes(), reset_session=True,
                )
            state = json.loads((base / "session.json").read_text(encoding="utf-8"))
            manifest = json.loads((base / "out" / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(state["current_positions"], {"cavity_beam_x": 0, "cavity_beam_y": 0})
            self.assertTrue(manifest["restoration"]["succeeded"])
        self.assertTrue(FakeCamera.instances[-1].closed)

    def test_tenth_batch_forces_final_response(self) -> None:
        FakeCamera.instances.clear()
        FakeLLM.responses = [
            response([{"x": 0, "y": 0}], done=False),
            response(PROBES[:5], done=True, final={"x": 10, "y": -20}),
        ]
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.tasks.two_mirror_cavity_real_llm_alignment.MindVisionCamera", FakeCamera
        ), patch("src.tasks.two_mirror_cavity_real_llm_alignment.LLM", FakeLLM):
            base = Path(tmp)
            manifest = run_two_mirror_cavity_real_llm_alignment(
                out_dir=base / "out", repeat_index=0, seed=1,
                hidden_offset={"x": 0, "y": 0}, iterations=1, probes_per_batch=5,
                llm_config=LLMConfig(model="mock"),
                camera_config=TwoMirrorCavityCameraConfig(warmup_frames=0),
                hardware_config=TwoMirrorCavityHardwareConfig(
                    session_state_path=str(base / "session.json"), settle_time_sec=0, dry_run=True
                ),
                axes=axes(), reset_session=True,
            )
        self.assertEqual(manifest["completion_batch"], 1)
        self.assertTrue(manifest["decisions"][-1]["forced_final"])

    def test_restoration_failure_is_fatal_and_recorded(self) -> None:
        FakeCamera.instances.clear()
        FakeLLM.responses = [
            response([{"x": 0, "y": 0}], done=True, final={"x": 0, "y": 0})
        ]
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.tasks.two_mirror_cavity_real_llm_alignment.MindVisionCamera", FakeCamera
        ), patch("src.tasks.two_mirror_cavity_real_llm_alignment.LLM", FakeLLM), patch(
            "src.tasks.two_mirror_cavity_real_llm_alignment.StepperMotorRig.return_to_baseline",
            side_effect=RuntimeError("restore failed"),
        ):
            base = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "Failed to restore"):
                run_two_mirror_cavity_real_llm_alignment(
                    out_dir=base / "out", repeat_index=0, seed=1,
                    hidden_offset={"x": 0, "y": 0}, iterations=1, probes_per_batch=5,
                    llm_config=LLMConfig(model="mock"),
                    camera_config=TwoMirrorCavityCameraConfig(warmup_frames=0),
                    hardware_config=TwoMirrorCavityHardwareConfig(
                        session_state_path=str(base / "session.json"), settle_time_sec=0, dry_run=True
                    ),
                    axes=axes(), reset_session=True,
                )
            manifest = json.loads((base / "out" / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["restoration"]["succeeded"])
            self.assertIn("restore failed", manifest["restoration"]["error"]["message"])
        self.assertTrue(FakeCamera.instances[-1].closed)

    def test_schema_defaults_and_validation(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs/two_mirror_cavity/real_llm_runs.yaml"
        with self.assertRaisesRegex(ValueError, "controller_ip"):
            load_two_mirror_cavity_real_llm_runs_config(config_path)
        config = load_checked_release_config(config_path)
        self.assertEqual((config.iterations, config.probes_per_batch), (50, 1))
        self.assertEqual(config.skip_repeat_indices, [])
        self.assertEqual(config.llm.model, "openai/gpt-5.6-sol")
        self.assertEqual({axis.role: axis.motor_number for axis in config.axes}, {"x": 3, "y": 1})
        source = config_path.read_text(encoding="utf-8").replace("REPLACE_ME", "192.0.2.1")
        skip_line = next(line for line in source.splitlines() if line.startswith("skip_repeat_indices:"))
        with tempfile.TemporaryDirectory() as tmp:
            for count in (1, 8):
                path = Path(tmp) / f"valid_{count}.yaml"
                path.write_text(source.replace("probes_per_batch: 1", f"probes_per_batch: {count}"), encoding="utf-8")
                self.assertEqual(load_two_mirror_cavity_real_llm_runs_config(path).probes_per_batch, count)
            for count in (0, 9):
                path = Path(tmp) / f"invalid_{count}.yaml"
                path.write_text(source.replace("probes_per_batch: 1", f"probes_per_batch: {count}"), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "between 1 and 8"):
                    load_two_mirror_cavity_real_llm_runs_config(path)
            no_skip_path = Path(tmp) / "no_skip.yaml"
            no_skip_path.write_text(
                source.replace(skip_line, "skip_repeat_indices: []"), encoding="utf-8"
            )
            self.assertEqual(
                load_two_mirror_cavity_real_llm_runs_config(no_skip_path).skip_repeat_indices,
                [],
            )
            invalid_skips = {
                "duplicate": ("skip_repeat_indices: [3, 3]", "duplicates"),
                "out_of_range": ("skip_repeat_indices: [10]", "between 0 and repeats - 1"),
                "all": (f"skip_repeat_indices: {list(range(10))}", "cannot skip every"),
            }
            for name, (replacement, message) in invalid_skips.items():
                path = Path(tmp) / f"invalid_skip_{name}.yaml"
                path.write_text(source.replace(skip_line, replacement), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_two_mirror_cavity_real_llm_runs_config(path)

    def test_cli_dispatch_and_runner_continues_after_safely_restored_failure(self) -> None:
        with patch(
            "src.cli.main.load_experiment_type", return_value="two_mirror_cavity_real_llm_runs"
        ), patch(
            "src.cli.main.run_two_mirror_cavity_real_llm_runs", return_value=Path("run")
        ) as run_real, patch("sys.argv", ["robotics", "run", "--config=config.yaml"]):
            cli_main()
        run_real.assert_called_once_with(Path("config.yaml"))

        config_path = Path(__file__).resolve().parents[1] / "configs/two_mirror_cavity/real_llm_runs.yaml"
        config = load_checked_release_config(config_path)
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.runners.run_experiment.load_two_mirror_cavity_real_llm_runs_config",
            return_value=config,
        ), patch(
            "src.runners.run_experiment.make_experiment_root", return_value=Path(tmp)
        ), patch(
            "src.runners.run_experiment.setup_run_logger"
        ), patch(
            "src.runners.run_experiment._run_method_task",
            side_effect=[
                {"status": "failed", "error": "JSON validation failed"},
                *[
                    {"status": "succeeded"}
                    for _ in range(config.repeats - len(config.skip_repeat_indices) - 1)
                ],
            ],
        ) as run_task:
            with patch(
                "src.runners.run_experiment._repeat_restored_to_physical_origin",
                return_value=(True, "restoration succeeded at physical x=0, y=0"),
            ):
                run_two_mirror_cavity_real_llm_runs(config_path)
        self.assertEqual(
            run_task.call_count,
            config.repeats - len(config.skip_repeat_indices),
        )
        task = run_task.call_args.args[0]
        self.assertEqual(task.repeat_index, config.repeats - 1)

    def test_runner_stops_after_failure_when_restoration_is_not_verified(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs/two_mirror_cavity/real_llm_runs.yaml"
        config = load_checked_release_config(config_path)
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.runners.run_experiment.load_two_mirror_cavity_real_llm_runs_config",
            return_value=config,
        ), patch(
            "src.runners.run_experiment.make_experiment_root", return_value=Path(tmp)
        ), patch(
            "src.runners.run_experiment.setup_run_logger"
        ), patch(
            "src.runners.run_experiment._run_method_task",
            return_value={"status": "failed", "error": "JSON validation failed"},
        ) as run_task, patch(
            "src.runners.run_experiment._repeat_restored_to_physical_origin",
            return_value=(False, "restoration did not succeed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "restoration to physical origin was not verified"):
                run_two_mirror_cavity_real_llm_runs(config_path)
        run_task.assert_called_once()


if __name__ == "__main__":
    unittest.main()
