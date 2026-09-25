from __future__ import annotations

import ctypes
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.cli.main import main as cli_main
from src.hardware.camera import CaptureArtifacts
from src.hardware.mindvision_camera import MindVisionCamera, MindVisionCameraConfig
from src.hardware.motors import AxisMotorConfig, StepperMotorRig
from src.io.schemas import (
    TwoMirrorCavityAxisConfig,
    TwoMirrorCavityCameraConfig,
    TwoMirrorCavityHardwareConfig,
    load_two_mirror_cavity_hardware_smoke_config,
)
from src.tasks.two_mirror_cavity_hardware_smoke import (
    perform_two_mirror_cavity_hardware_smoke,
)


class FakeMindVisionSDK:
    CAMERA_MEDIA_TYPE_MONO8 = 1
    CAMERA_MEDIA_TYPE_BGR8 = 3
    c_ubyte = ctypes.c_ubyte

    def __init__(self, *, mono: bool = True, failures: int = 0) -> None:
        self.mono = mono
        self.failures = failures
        self.calls: list[str] = []
        self.buffer = None
        self.exposure_us = None
        self.gain = None

    def CameraEnumerateDevice(self):
        return [object()]

    def CameraInit(self, device, mode, parameter):
        self.calls.append("init")
        return 7

    def CameraGetCapability(self, handle):
        return SimpleNamespace(
            sIspCapacity=SimpleNamespace(bMonoSensor=1 if self.mono else 0),
            sResolutionRange=SimpleNamespace(iWidthMax=2, iHeightMax=2),
        )

    def CameraSetIspOutFormat(self, handle, media_type):
        self.media_type = media_type

    def CameraAlignMalloc(self, size, alignment):
        self.buffer = (ctypes.c_ubyte * size)()
        return ctypes.addressof(self.buffer)

    def CameraSetAeState(self, handle, state):
        return None

    def CameraSetExposureTime(self, handle, exposure_us):
        self.exposure_us = exposure_us

    def CameraSetAnalogGain(self, handle, gain):
        self.gain = gain

    def CameraSetTriggerMode(self, handle, mode):
        return None

    def CameraPlay(self, handle):
        self.calls.append("play")

    def CameraGetImageBuffer(self, handle, timeout_ms):
        self.calls.append("grab")
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("timeout")
        channels = 1 if self.mono else 3
        return 99, SimpleNamespace(
            uBytes=2 * 2 * channels,
            iHeight=2,
            iWidth=2,
            uiMediaType=self.CAMERA_MEDIA_TYPE_MONO8 if self.mono else self.CAMERA_MEDIA_TYPE_BGR8,
        )

    def CameraImageProcess(self, handle, raw_data, frame_buffer, frame_head):
        values = bytes(range(1, frame_head.uBytes + 1))
        ctypes.memmove(frame_buffer, values, frame_head.uBytes)

    def CameraFlipFrameBuffer(self, frame_buffer, frame_head, direction):
        self.calls.append("flip")

    def CameraReleaseImageBuffer(self, handle, raw_data):
        self.calls.append("release")

    def CameraPause(self, handle):
        self.calls.append("pause")

    def CameraUnInit(self, handle):
        self.calls.append("uninit")

    def CameraAlignFree(self, frame_buffer):
        self.calls.append("free")


class FakeResponse:
    def raise_for_status(self) -> None:
        return None


class RecordingHTTPSession:
    def __init__(self) -> None:
        self.urls: list[str] = []
        self.params: list[dict[str, int]] = []

    def get(self, url, params, timeout):
        self.urls.append(url)
        self.params.append(params)
        return FakeResponse()


class FakeSmokeCamera:
    instances: list["FakeSmokeCamera"] = []
    fail_after_first = False

    def __init__(self, config) -> None:
        self.config = config
        self.closed = False
        self.captures = 0
        self.__class__.instances.append(self)

    def open(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    def save_capture(self, out_dir, stem, **kwargs) -> CaptureArtifacts:
        self.captures += 1
        if self.fail_after_first and self.captures > 1:
            raise RuntimeError("capture failed")
        path = Path(out_dir) / f"{stem}.png"
        path.write_bytes(b"image")
        return CaptureArtifacts(None, str(path), None, (2, 2), "uint8")


class TwoMirrorCavityHardwareTests(unittest.TestCase):
    def test_mindvision_mono_and_color_frames_and_cleanup(self) -> None:
        for mono, expected_shape in ((True, (2, 2, 1)), (False, (2, 2, 3))):
            with self.subTest(mono=mono):
                sdk = FakeMindVisionSDK(mono=mono)
                camera = MindVisionCamera(
                    MindVisionCameraConfig(
                        exposure_ms=100.0,
                        analog_gain=2,
                        warmup_frames=0,
                    ),
                    sdk=sdk,
                )
                camera.open()
                frame = camera.capture_array()
                camera.close()
                self.assertEqual(frame.shape, expected_shape)
                self.assertEqual(sdk.exposure_us, 100_000)
                self.assertEqual(sdk.gain, 2)
                self.assertIn("release", sdk.calls)
                self.assertEqual(sdk.calls[-3:], ["pause", "uninit", "free"])

    def test_mindvision_retries_are_bounded(self) -> None:
        sdk = FakeMindVisionSDK(failures=3)
        camera = MindVisionCamera(
            MindVisionCameraConfig(warmup_frames=0, frame_retries=3), sdk=sdk
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "after 3 attempt"):
                camera.capture_array()
        finally:
            camera.close()
        self.assertEqual(sdk.calls.count("grab"), 3)
        self.assertIn("free", sdk.calls)

    def test_mindvision_capture_metadata_includes_camera_settings(self) -> None:
        sdk = FakeMindVisionSDK()
        camera = MindVisionCamera(
            MindVisionCameraConfig(
                exposure_ms=12.5,
                analog_gain=4,
                warmup_frames=0,
            ),
            sdk=sdk,
        )
        with tempfile.TemporaryDirectory() as tmp:
            try:
                artifacts = camera.save_capture(tmp, "capture")
            finally:
                camera.close()
            metadata = json.loads(Path(artifacts.metadata_path).read_text(encoding="utf-8"))
            self.assertEqual(metadata["camera_backend"], "mindvision")
            self.assertEqual(metadata["exposure_ms"], 12.5)
            self.assertEqual(metadata["analog_gain"], 4)

    def test_logical_move_uses_only_move_endpoint_and_compensation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            http = RecordingHTTPSession()
            rig = StepperMotorRig(
                [AxisMotorConfig("cavity", "192.0.2.1", 3)],
                session_state_path=Path(tmp) / "session.json",
                steps_per_revolution=4096,
                max_cumulative_revolutions=5.0,
                backlash_compensation=True,
                http_session=http,
            )
            record = rig.move_axis("cavity", -2000)
        self.assertEqual(record.executed_submoves, [1, -2001])
        self.assertEqual(http.params, [{"motor": 3, "steps": 1}, {"motor": 3, "steps": -2001}])
        self.assertTrue(all(url.endswith("/move") for url in http.urls))
        self.assertFalse(any("status" in url or "power" in url for url in http.urls))

    def test_smoke_workflow_captures_moves_and_closes(self) -> None:
        FakeSmokeCamera.instances.clear()
        FakeSmokeCamera.fail_after_first = False
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.tasks.two_mirror_cavity_hardware_smoke.MindVisionCamera",
            FakeSmokeCamera,
        ):
            base = Path(tmp)
            manifest = perform_two_mirror_cavity_hardware_smoke(
                out_dir=base / "out",
                camera_config=TwoMirrorCavityCameraConfig(warmup_frames=0),
                hardware_config=TwoMirrorCavityHardwareConfig(
                    session_state_path=str(base / "session.json"),
                    settle_time_sec=0,
                    dry_run=True,
                ),
                axis=TwoMirrorCavityAxisConfig("cavity", "ip", 3, logical_steps=-2000),
            )
            self.assertEqual(manifest["motion"]["executed_submoves"], [1, -2001])
            self.assertTrue((base / "out" / "smoke_manifest.json").exists())
        self.assertTrue(FakeSmokeCamera.instances[-1].closed)

    def test_smoke_workflow_closes_camera_on_capture_failure(self) -> None:
        FakeSmokeCamera.instances.clear()
        FakeSmokeCamera.fail_after_first = True
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.tasks.two_mirror_cavity_hardware_smoke.MindVisionCamera",
            FakeSmokeCamera,
        ):
            base = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                perform_two_mirror_cavity_hardware_smoke(
                    out_dir=base / "out",
                    camera_config=TwoMirrorCavityCameraConfig(warmup_frames=0),
                    hardware_config=TwoMirrorCavityHardwareConfig(
                        session_state_path=str(base / "session.json"),
                        settle_time_sec=0,
                        dry_run=True,
                    ),
                    axis=TwoMirrorCavityAxisConfig("cavity", "ip", 3, logical_steps=-2000),
                )
        self.assertTrue(FakeSmokeCamera.instances[-1].closed)
        FakeSmokeCamera.fail_after_first = False

    def test_schema_validation_and_cli_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                """
camera:
  exposure_ms: 0
hardware:
  session_state_path: session.json
  dry_run: true
axis:
  name: cavity
  motor_number: 3
  logical_steps: -2000
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exposure_ms"):
                load_two_mirror_cavity_hardware_smoke_config(path)

        with patch("src.cli.main.load_experiment_type", return_value="two_mirror_cavity_hardware_smoke"), patch(
            "src.cli.main.run_two_mirror_cavity_hardware_smoke", return_value=Path("run")
        ) as run_smoke, patch("sys.argv", ["robotics", "run", "--config=config.yaml"]):
            cli_main()
        run_smoke.assert_called_once_with(Path("config.yaml"))


if __name__ == "__main__":
    unittest.main()
