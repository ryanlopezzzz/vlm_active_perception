from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.hardware.camera import CameraCaptureConfig, HardwareCamera
from src.hardware.motors import AxisMotorConfig, MotionSafetyLimitError, StepperMotorRig
from src.io.schemas import (
    LLMConfig,
    RealMIAxisConfig,
    RealMICameraConfig,
    RealMIHardwareConfig,
    load_real_mi_llm_runs_config,
)
from src.runners.run_experiment import run_mi_real_llm_runs
from src.tasks.mi_initial_conditions import (
    ONSCREEN_X_LIMIT_STEPS,
    ONSCREEN_Y_LIMIT_STEPS,
    RANDOM_LIMIT_STEPS,
    VISIBLE_X_LIMIT_STEPS,
    VISIBLE_Y_LIMIT_STEPS,
)
from src.tasks.mi_real_alignment import (
    perform_real_mi_preflight,
    run_real_llm_alignment,
    sample_hidden_offset_steps,
)
from src.tasks.mi_real_prompting import build_real_mi_system_prompt, verify_real_motor_command


class FakeResponse:
    def raise_for_status(self) -> None:
        return None


class FakeHTTPSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, int]]] = []

    def get(self, url: str, params: dict[str, int], timeout: int) -> FakeResponse:
        self.calls.append((url, params))
        return FakeResponse()


class FakeHardwareCamera:
    def __init__(self, config: object) -> None:
        self.config = config
        self.counter = 0

    def save_capture(
        self,
        out_dir: str | Path,
        stem: str,
        *,
        average_frames=None,
        extra_metadata=None,
        save_raw_array=True,
        save_metadata=True,
    ):
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        llm_path = out_path / f"{stem}.png"
        raw_path = out_path / f"{stem}.npy"
        meta_path = out_path / f"{stem}.json"
        array = np.full((4, 4), self.counter, dtype=np.uint8)
        if save_raw_array:
            np.save(raw_path, array)
        llm_path.write_bytes(b"fake-image")
        metadata = {"stem": stem, "extra_metadata": extra_metadata or {}, "counter": self.counter}
        if save_metadata:
            meta_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.counter += 1
        return FakeCaptureArtifacts(
            raw_array_path=str(raw_path) if save_raw_array else None,
            llm_image_path=str(llm_path),
            metadata_path=str(meta_path) if save_metadata else None,
            shape=array.shape,
            dtype=str(array.dtype),
        )

    def close(self) -> None:
        return None


class FakeLLM:
    instances: list["FakeLLM"] = []

    def __init__(self, *args, **kwargs) -> None:
        self.model = str(kwargs.get("model", "fake-model"))
        self.messages: list[dict[str, object]] = []
        self._answers = [
            json.dumps(
                {
                    "visual_description": "two beams are separated",
                    "done": False,
                    "command": {
                        "mirror_1_axis_1": -10,
                        "mirror_1_axis_2": 5,
                        "mirror_2_axis_1": 0,
                        "mirror_2_axis_2": 0,
                    },
                }
            ),
            json.dumps(
                {
                    "visual_description": "beams are overlapped near center",
                    "done": True,
                    "command": {
                        "mirror_1_axis_1": 0,
                        "mirror_1_axis_2": 0,
                        "mirror_2_axis_1": 0,
                        "mirror_2_axis_2": 0,
                    },
                }
            ),
        ]
        self.api_calls: list[dict[str, object]] = []
        FakeLLM.instances.append(self)

    def add_system_message(self, text: str) -> None:
        self.messages.append({"role": "system", "content": text})

    def add_user_image(self, image_path: str) -> None:
        self.messages.append({"role": "user_image", "content": image_path})

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user_text", "content": text})

    def query_llm(self, **kwargs) -> str:
        return self._answers.pop(0)

    def get_usage_summary(self) -> dict[str, object]:
        return {
            "model_requested": self.model,
            "request_attempts": 0,
            "api_responses": 0,
            "accepted_responses": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_credits": 0.0,
            "responses_with_cost": 0,
            "cost_complete": False,
            "request_wall_time_sec": 0.0,
            "avg_request_wall_time_sec": 0.0,
            "min_request_wall_time_sec": 0.0,
            "max_request_wall_time_sec": 0.0,
        }


@dataclass(frozen=True)
class FakeCaptureArtifacts:
    raw_array_path: str
    llm_image_path: str
    metadata_path: str
    shape: tuple[int, ...]
    dtype: str


def _real_beam_steps(hidden: dict[str, int], mirror: str) -> tuple[int, int]:
    return hidden[f"{mirror}_axis_1"], hidden[f"{mirror}_axis_2"]


def _real_beam_is_onscreen(hidden: dict[str, int], mirror: str) -> bool:
    x, y = _real_beam_steps(hidden, mirror)
    return abs(x) <= ONSCREEN_X_LIMIT_STEPS and abs(y) <= ONSCREEN_Y_LIMIT_STEPS


def _real_beam_is_offscreen(hidden: dict[str, int], mirror: str) -> bool:
    x, y = _real_beam_steps(hidden, mirror)
    return (
        abs(x) <= RANDOM_LIMIT_STEPS
        and abs(y) <= RANDOM_LIMIT_STEPS
        and (abs(x) > VISIBLE_X_LIMIT_STEPS or abs(y) > VISIBLE_Y_LIMIT_STEPS)
    )


def _real_mi_axes(
    *,
    hidden_offset_max_steps: int = 8192,
    llm_step_limit: int = 8192,
    preflight_test_steps: int = 20,
) -> list[RealMIAxisConfig]:
    return [
        RealMIAxisConfig(
            name="mirror_1_axis_1",
            controller_ip="ip",
            motor_number=1,
            hidden_offset_max_steps=hidden_offset_max_steps,
            llm_step_limit=llm_step_limit,
            preflight_test_steps=preflight_test_steps,
        ),
        RealMIAxisConfig(
            name="mirror_1_axis_2",
            controller_ip="ip",
            motor_number=3,
            hidden_offset_max_steps=hidden_offset_max_steps,
            llm_step_limit=llm_step_limit,
            preflight_test_steps=preflight_test_steps,
        ),
        RealMIAxisConfig(
            name="mirror_2_axis_1",
            controller_ip="ip",
            motor_number=1,
            hidden_offset_max_steps=hidden_offset_max_steps,
            llm_step_limit=llm_step_limit,
            preflight_test_steps=preflight_test_steps,
        ),
        RealMIAxisConfig(
            name="mirror_2_axis_2",
            controller_ip="ip",
            motor_number=3,
            hidden_offset_max_steps=hidden_offset_max_steps,
            llm_step_limit=llm_step_limit,
            preflight_test_steps=preflight_test_steps,
        ),
    ]


class RealMIWorkflowTests(unittest.TestCase):
    def test_compensated_move_sequence_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            http = FakeHTTPSession()
            rig = StepperMotorRig(
                [AxisMotorConfig(name="a", controller_ip="127.0.0.1", motor_number=1)],
                session_state_path=Path(tmp) / "session.json",
                steps_per_revolution=4096,
                max_cumulative_revolutions=5,
                backlash_compensation=True,
                dry_run=False,
                http_session=http,
            )
            record = rig.move_axis("a", 50)
            self.assertEqual(record.executed_submoves, [-1, 51])
            self.assertEqual(rig.state.current_positions["a"], 50)
            self.assertEqual(http.calls[0][1]["steps"], -1)
            self.assertEqual(http.calls[1][1]["steps"], 51)

            record = rig.move_axis("a", -50)
            self.assertEqual(record.executed_submoves, [1, -51])
            self.assertEqual(rig.state.current_positions["a"], 0)

    def test_net_displacement_limit_raises_before_move(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rig = StepperMotorRig(
                [AxisMotorConfig(name="a", controller_ip="127.0.0.1", motor_number=1)],
                session_state_path=Path(tmp) / "session.json",
                steps_per_revolution=100,
                max_cumulative_revolutions=1,
                backlash_compensation=True,
                dry_run=True,
            )
            rig.move_axis("a", 100)
            rig.move_axis("a", -100)
            rig.move_axis("a", 100)
            with self.assertRaises(MotionSafetyLimitError):
                rig.move_axis("a", 1)

    def test_net_displacement_limit_can_be_explicitly_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rig = StepperMotorRig(
                [AxisMotorConfig(name="a", controller_ip="127.0.0.1", motor_number=1)],
                session_state_path=Path(tmp) / "session.json",
                steps_per_revolution=100,
                max_cumulative_revolutions=1,
                backlash_compensation=True,
                enforce_position_limit=False,
                dry_run=True,
            )
            record = rig.move_axis("a", 10_000)
            self.assertEqual(record.logical_position_after, 10_000)

    def test_hidden_offset_sampler_regions(self) -> None:
        axes = [
            RealMIAxisConfig(
                name="mirror_1_axis_1",
                controller_ip="ip",
                motor_number=1,
                hidden_offset_max_steps=8192,
                llm_step_limit=8192,
                preflight_test_steps=7,
            ),
            RealMIAxisConfig(
                name="mirror_1_axis_2",
                controller_ip="ip",
                motor_number=3,
                hidden_offset_max_steps=8192,
                llm_step_limit=8192,
                preflight_test_steps=7,
            ),
            RealMIAxisConfig(
                name="mirror_2_axis_1",
                controller_ip="ip",
                motor_number=1,
                hidden_offset_max_steps=8192,
                llm_step_limit=8192,
                preflight_test_steps=7,
            ),
            RealMIAxisConfig(
                name="mirror_2_axis_2",
                controller_ip="ip",
                motor_number=3,
                hidden_offset_max_steps=8192,
                llm_step_limit=8192,
                preflight_test_steps=7,
            ),
        ]
        for seed in range(100):
            sampled = sample_hidden_offset_steps(axes, seed=seed, initial_condition_mode="random")
            self.assertTrue(all(abs(value) <= RANDOM_LIMIT_STEPS for value in sampled.values()))

            sampled = sample_hidden_offset_steps(axes, seed=seed, initial_condition_mode="both_beams_onscreen")
            self.assertTrue(_real_beam_is_onscreen(sampled, "mirror_1"))
            self.assertTrue(_real_beam_is_onscreen(sampled, "mirror_2"))

            sampled = sample_hidden_offset_steps(axes, seed=seed, initial_condition_mode="one_beam_onscreen")
            onscreen_count = sum(
                _real_beam_is_onscreen(sampled, mirror) for mirror in ("mirror_1", "mirror_2")
            )
            offscreen_count = sum(
                _real_beam_is_offscreen(sampled, mirror) for mirror in ("mirror_1", "mirror_2")
            )
            self.assertEqual(onscreen_count, 1)
            self.assertEqual(offscreen_count, 1)

            sampled = sample_hidden_offset_steps(axes, seed=seed, initial_condition_mode="no_beams_onscreen")
            self.assertTrue(_real_beam_is_offscreen(sampled, "mirror_1"))
            self.assertTrue(_real_beam_is_offscreen(sampled, "mirror_2"))

    def test_real_mi_config_loads_initial_condition_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_path = base / "real_mi.yaml"
            common_config = """
experiment_type: mi_real_llm_runs
experiment_name: mi_real_llm_runs
output_root: {output_root}
repeats: 1
iterations: 1
base_seed: 123
max_parallel_jobs: 1
methods:
  - llm
camera:
  device_index: 0
hardware:
  baseline_state_path: {baseline_state_path}
  session_state_path: {session_state_path}
axes:
  - name: mirror_1_axis_1
    controller_ip: ip
    motor_number: 1
    hidden_offset_max_steps: 8192
    llm_step_limit: 8192
  - name: mirror_1_axis_2
    controller_ip: ip
    motor_number: 3
    hidden_offset_max_steps: 8192
    llm_step_limit: 8192
  - name: mirror_2_axis_1
    controller_ip: ip
    motor_number: 1
    hidden_offset_max_steps: 8192
    llm_step_limit: 8192
  - name: mirror_2_axis_2
    controller_ip: ip
    motor_number: 3
    hidden_offset_max_steps: 8192
    llm_step_limit: 8192
llm:
  model: mock
  effort: low
  max_tokens: 100
  reasoning_max_tokens: null
""".format(
                output_root=base / "runs",
                baseline_state_path=base / "session" / "baseline_state.json",
                session_state_path=base / "session" / "session_state.json",
            )
            config_path.write_text(common_config, encoding="utf-8")
            self.assertEqual(load_real_mi_llm_runs_config(config_path).initial_condition_mode, "random")

            config_path.write_text(
                common_config.replace("camera:\n", "initial_condition_mode: no_beams_onscreen\ncamera:\n"),
                encoding="utf-8",
            )
            self.assertEqual(load_real_mi_llm_runs_config(config_path).initial_condition_mode, "no_beams_onscreen")

    def test_absolute_visible_step_command_validation_allows_done_with_nonzero_best_setting(self) -> None:
        axes = [
            RealMIAxisConfig(
                name="mirror_1_axis_1",
                controller_ip="ip",
                motor_number=1,
                hidden_offset_max_steps=5,
                llm_step_limit=20,
                preflight_test_steps=7,
            ),
            RealMIAxisConfig(
                name="mirror_1_axis_2",
                controller_ip="ip",
                motor_number=3,
                hidden_offset_max_steps=5,
                llm_step_limit=20,
                preflight_test_steps=7,
            ),
        ]
        answer = json.dumps(
            {
                "visual_description": "overlap looks best at this nonzero command",
                "done": True,
                "command": {
                    "mirror_1_axis_1": 12,
                    "mirror_1_axis_2": -4,
                },
            }
        )
        self.assertTrue(verify_real_motor_command(answer, axes))

    def test_standard_prompt_uses_absolute_motor_steps_not_mrad(self) -> None:
        axes = [
            RealMIAxisConfig(
                name="mirror_1_axis_1",
                controller_ip="ip",
                motor_number=1,
                hidden_offset_max_steps=4096,
                llm_step_limit=4096,
                preflight_test_steps=7,
            ),
            RealMIAxisConfig(
                name="mirror_1_axis_2",
                controller_ip="ip",
                motor_number=3,
                hidden_offset_max_steps=4096,
                llm_step_limit=4096,
                preflight_test_steps=7,
            ),
        ]
        prompt = build_real_mi_system_prompt(max_iterations=5, axes=axes)
        self.assertIn("absolute motor-step", prompt)
        self.assertIn("[-4096, 4096]", prompt)
        self.assertIn("visual_description", prompt)
        self.assertNotIn("mrad", prompt)

    def test_camera_capture_saves_raw_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            camera = HardwareCamera(
                CameraCaptureConfig(average_frames=2, llm_image_format="png"),
                frame_provider=lambda: np.full((3, 3, 3), 7, dtype=np.uint8),
            )
            written_frames: list[np.ndarray] = []

            def fake_write(path: Path, frame: np.ndarray) -> None:
                written_frames.append(frame.copy())
                path.write_bytes(b"img")

            with patch.object(HardwareCamera, "_prepare_llm_image", side_effect=lambda frame: frame), patch.object(
                HardwareCamera,
                "_write_image",
                side_effect=fake_write,
            ):
                artifacts = camera.save_capture(tmp, "frame")
            self.assertTrue(Path(artifacts.raw_array_path).exists())
            self.assertTrue(Path(artifacts.llm_image_path).exists())
            metadata = json.loads(Path(artifacts.metadata_path).read_text(encoding="utf-8"))
            self.assertEqual(metadata["shape"], [3, 3, 3])
            self.assertEqual(metadata["llm_image_shape"], [3, 3, 3])
            self.assertEqual(written_frames[0].shape, (3, 3, 3))

    def test_preflight_and_llm_run_write_artifacts_and_hide_offset_from_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            camera_config = RealMICameraConfig(device_index=0, average_frames=1, llm_image_format="png")
            hardware_config = RealMIHardwareConfig(
                baseline_state_path=str(base / "session" / "baseline_state.json"),
                session_state_path=str(base / "session" / "session_state.json"),
                steps_per_revolution=4096,
                max_cumulative_revolutions=5.0,
                backlash_compensation=True,
                settle_time_sec=0.0,
                dry_run=True,
                use_baseline_reference=True,
            )
            axes = [
                RealMIAxisConfig(
                    name="mirror_1_axis_1",
                    controller_ip="ip",
                    motor_number=1,
                    hidden_offset_max_steps=10,
                    llm_step_limit=10,
                    preflight_test_steps=20,
                ),
                RealMIAxisConfig(
                    name="mirror_1_axis_2",
                    controller_ip="ip",
                    motor_number=3,
                    hidden_offset_max_steps=10,
                    llm_step_limit=10,
                    preflight_test_steps=20,
                ),
            ]
            with patch("src.tasks.mi_real_alignment.HardwareCamera", FakeHardwareCamera):
                preflight = perform_real_mi_preflight(
                    out_dir=base / "preflight",
                    camera_config=camera_config,
                    hardware_config=hardware_config,
                    axes=axes,
                )
            self.assertEqual(len(preflight["axis_tests"]), 2)
            self.assertIn("final_return", preflight)
            pngs = sorted(path.name for path in (base / "preflight").glob("*.png"))
            self.assertEqual(pngs, ["baseline.png", "final_return.png", "mirror_1_axis_1.png", "mirror_1_axis_2.png"])
            self.assertFalse(any((base / "preflight").glob("*.npy")))
            self.assertFalse(any((base / "preflight").glob("*.json")))

            alignment_axes = _real_mi_axes(llm_step_limit=20)
            FakeLLM.instances.clear()
            llm_config = LLMConfig(model="fake-model", effort="low", max_tokens=100, reasoning_max_tokens=50)
            with patch("src.tasks.mi_real_alignment.HardwareCamera", FakeHardwareCamera), patch(
                "src.tasks.mi_real_alignment.LLM",
                FakeLLM,
            ):
                manifest = run_real_llm_alignment(
                    out_dir=base / "run_00",
                    iterations=2,
                    seed=7,
                    repeat_index=0,
                    llm_config=llm_config,
                    camera_config=camera_config,
                    hardware_config=hardware_config,
                    axes=alignment_axes,
                )
            self.assertTrue((base / "run_00" / "run_manifest.json").exists())
            self.assertIn("hidden_offset_steps", manifest)
            messages_text = json.dumps(FakeLLM.instances[0].messages)
            self.assertNotIn("hidden_offset_steps", messages_text)
            self.assertNotIn("baseline reference image", messages_text)
            self.assertTrue(manifest["completed_early"])
            self.assertEqual(manifest["completion_step"], 2)
            zero_positions = {axis.name: 0 for axis in alignment_axes}
            self.assertEqual(manifest["steps"][0]["logical_positions"], zero_positions)
            self.assertEqual(
                manifest["steps"][1]["logical_positions"],
                {"mirror_1_axis_1": -10, "mirror_1_axis_2": 5, "mirror_2_axis_1": 0, "mirror_2_axis_2": 0},
            )
            self.assertEqual(manifest["steps"][1]["visual_description"], "two beams are separated")
            self.assertIn("step_wall_time_sec", manifest["steps"][1])
            self.assertIn("llm_request_wall_time_sec", manifest["steps"][1])
            self.assertIn("run_wall_time_sec", manifest)
            self.assertIn("total_step_wall_time_sec", manifest)
            self.assertTrue((base / "run_00" / "baseline.png").exists())
            self.assertTrue((base / "run_00" / "step_final.png").exists())
            self.assertFalse(any((base / "run_00").glob("step_*.npy")))
            self.assertFalse(any((base / "run_00").glob("step_*.json")))

    def test_first_repeat_syncs_manual_alignment_before_baseline_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            camera_config = RealMICameraConfig(device_index=0, average_frames=1, llm_image_format="png")
            hardware_config = RealMIHardwareConfig(
                baseline_state_path=str(base / "session" / "baseline_state.json"),
                session_state_path=str(base / "session" / "session_state.json"),
                steps_per_revolution=4096,
                max_cumulative_revolutions=5.0,
                backlash_compensation=True,
                settle_time_sec=0.0,
                dry_run=True,
                use_baseline_reference=True,
            )
            axes = _real_mi_axes(llm_step_limit=20)
            rig = StepperMotorRig(
                [
                    AxisMotorConfig(name="mirror_1_axis_1", controller_ip="ip", motor_number=1),
                    AxisMotorConfig(name="mirror_1_axis_2", controller_ip="ip", motor_number=3),
                    AxisMotorConfig(name="mirror_2_axis_1", controller_ip="ip", motor_number=1),
                    AxisMotorConfig(name="mirror_2_axis_2", controller_ip="ip", motor_number=3),
                ],
                session_state_path=hardware_config.session_state_path,
                steps_per_revolution=4096,
                max_cumulative_revolutions=5.0,
                backlash_compensation=True,
                dry_run=True,
            )
            rig.write_baseline_state(hardware_config.baseline_state_path)
            rig.move_axis("mirror_1_axis_1", 25)

            FakeLLM.instances.clear()
            llm_config = LLMConfig(model="fake-model", effort="low", max_tokens=100, reasoning_max_tokens=50)
            with patch("src.tasks.mi_real_alignment.HardwareCamera", FakeHardwareCamera), patch(
                "src.tasks.mi_real_alignment.LLM",
                FakeLLM,
            ):
                run_real_llm_alignment(
                    out_dir=base / "run_00",
                    iterations=1,
                    seed=7,
                    repeat_index=0,
                    llm_config=llm_config,
                    camera_config=camera_config,
                    hardware_config=hardware_config,
                    axes=axes,
                )

            restore = json.loads((base / "run_00" / "baseline_restore.json").read_text(encoding="utf-8"))
            self.assertTrue(all(record["requested_logical_steps"] == 0 for record in restore["records"]))
            self.assertTrue(all(record["executed_submoves"] == [] for record in restore["records"]))
            self.assertEqual(
                restore["positions"],
                {"mirror_1_axis_1": 0, "mirror_1_axis_2": 0, "mirror_2_axis_1": 0, "mirror_2_axis_2": 0},
            )

    def test_final_repeat_can_restore_baseline_at_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            camera_config = RealMICameraConfig(device_index=0, average_frames=1, llm_image_format="png")
            hardware_config = RealMIHardwareConfig(
                baseline_state_path=str(base / "session" / "baseline_state.json"),
                session_state_path=str(base / "session" / "session_state.json"),
                steps_per_revolution=4096,
                max_cumulative_revolutions=5.0,
                backlash_compensation=True,
                settle_time_sec=0.0,
                dry_run=True,
                use_baseline_reference=True,
            )
            axes = _real_mi_axes(llm_step_limit=20)
            rig = StepperMotorRig(
                [
                    AxisMotorConfig(name="mirror_1_axis_1", controller_ip="ip", motor_number=1),
                    AxisMotorConfig(name="mirror_1_axis_2", controller_ip="ip", motor_number=3),
                    AxisMotorConfig(name="mirror_2_axis_1", controller_ip="ip", motor_number=1),
                    AxisMotorConfig(name="mirror_2_axis_2", controller_ip="ip", motor_number=3),
                ],
                session_state_path=hardware_config.session_state_path,
                steps_per_revolution=4096,
                max_cumulative_revolutions=5.0,
                backlash_compensation=True,
                dry_run=True,
            )
            rig.write_baseline_state(hardware_config.baseline_state_path)
            FakeLLM.instances.clear()
            llm_config = LLMConfig(model="fake-model", effort="low", max_tokens=100, reasoning_max_tokens=50)
            with patch("src.tasks.mi_real_alignment.HardwareCamera", FakeHardwareCamera), patch(
                "src.tasks.mi_real_alignment.LLM",
                FakeLLM,
            ):
                manifest = run_real_llm_alignment(
                    out_dir=base / "run_00",
                    iterations=2,
                    seed=7,
                    repeat_index=0,
                    restore_baseline_at_end=True,
                    llm_config=llm_config,
                    camera_config=camera_config,
                    hardware_config=hardware_config,
                    axes=axes,
                )

            self.assertTrue((base / "run_00" / "final_baseline_restore.json").exists())
            restore = json.loads((base / "run_00" / "final_baseline_restore.json").read_text(encoding="utf-8"))
            self.assertEqual(
                restore["positions"],
                {"mirror_1_axis_1": 0, "mirror_1_axis_2": 0, "mirror_2_axis_1": 0, "mirror_2_axis_2": 0},
            )
            self.assertIn("final_baseline_restore", manifest)

    def test_real_mi_runner_requests_restore_after_every_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_path = base / "real_mi.yaml"
            config_path.write_text(
                """
experiment_type: mi_real_llm_runs
experiment_name: mi_real_llm_runs
output_root: {output_root}
repeats: 2
iterations: 1
base_seed: 123
max_parallel_jobs: 1
methods:
  - llm
camera:
  device_index: 0
  average_frames: 1
  warmup_frames: 0
  llm_image_format: png
hardware:
  baseline_state_path: {baseline_state_path}
  session_state_path: {session_state_path}
  steps_per_revolution: 4096
  max_cumulative_revolutions: 5.0
  backlash_compensation: true
  settle_time_sec: 0.0
  dry_run: true
  use_baseline_reference: true
axes:
  - name: mirror_1_axis_1
    controller_ip: ip
    motor_number: 1
    direction_sign: 1
    hidden_offset_max_steps: 4096
    llm_step_limit: 4096
    preflight_test_steps: 20
llm:
  model: mock
  effort: low
  max_tokens: 100
  reasoning_max_tokens: null
""".format(
                    output_root=base / "runs",
                    baseline_state_path=base / "session" / "baseline_state.json",
                    session_state_path=base / "session" / "session_state.json",
                ),
                encoding="utf-8",
            )

            seen_commands: list[list[str]] = []

            def fake_run_method_task(task, repo_root):
                seen_commands.append(task.command)
                return {
                    "repeat_index": task.repeat_index,
                    "name": task.method,
                    "status": "succeeded",
                    "command": task.command,
                    "output_dir": str(task.output_dir),
                    "log_path": str(task.log_path),
                }

            with patch("src.runners.run_experiment._run_method_task", side_effect=fake_run_method_task):
                run_mi_real_llm_runs(config_path)

            self.assertEqual(len(seen_commands), 2)
            self.assertTrue(all("--restore-baseline-at-end" in command for command in seen_commands))


if __name__ == "__main__":
    unittest.main()
