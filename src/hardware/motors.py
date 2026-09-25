"""Motor control helpers with session-state persistence and safety tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import requests


@dataclass(frozen=True)
class AxisMotorConfig:
    name: str
    controller_ip: str
    motor_number: int
    direction_sign: int = 1
    hidden_offset_max_steps: int = 0
    llm_step_limit: int = 0
    preflight_test_steps: int = 0


@dataclass(frozen=True)
class MotionRecord:
    axis_name: str
    requested_logical_steps: int
    requested_hardware_steps: int
    executed_submoves: list[int]
    cumulative_executed_steps: int
    cumulative_executed_revolutions: float
    logical_position_after: int
    dry_run: bool


@dataclass(frozen=True)
class BaselineState:
    baseline_positions: dict[str, int]
    baseline_image_path: str | None = None
    baseline_capture_metadata_path: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class SessionMotionState:
    current_positions: dict[str, int]
    cumulative_executed_steps: dict[str, int]
    cumulative_net_steps: dict[str, int]
    move_records: list[dict[str, Any]] = field(default_factory=list)


class MotionSafetyLimitError(RuntimeError):
    """Raised when a move would exceed the configured per-axis net displacement cap."""


class StepperMotorRig:
    """Logical-axis motor rig with backlash compensation and net-displacement safety."""

    def __init__(
        self,
        axes: list[AxisMotorConfig],
        *,
        session_state_path: str | Path,
        steps_per_revolution: int,
        max_cumulative_revolutions: float,
        backlash_compensation: bool = True,
        enforce_position_limit: bool = True,
        dry_run: bool = False,
        http_session: requests.Session | None = None,
    ) -> None:
        if steps_per_revolution <= 0:
            raise ValueError("steps_per_revolution must be > 0.")
        self.axes = {axis.name: axis for axis in axes}
        self.session_state_path = Path(session_state_path)
        self.steps_per_revolution = int(steps_per_revolution)
        self.max_cumulative_revolutions = float(max_cumulative_revolutions)
        self.backlash_compensation = bool(backlash_compensation)
        self.enforce_position_limit = bool(enforce_position_limit)
        self.dry_run = bool(dry_run)
        self.http_session = http_session or requests.Session()
        self.state = self._load_or_init_state()

    def reset_session(self) -> SessionMotionState:
        self.state = SessionMotionState(
            current_positions={name: 0 for name in self.axes},
            cumulative_executed_steps={name: 0 for name in self.axes},
            cumulative_net_steps={name: 0 for name in self.axes},
            move_records=[],
        )
        self._save_state()
        return self.state

    def _load_or_init_state(self) -> SessionMotionState:
        if not self.session_state_path.exists():
            return self.reset_session()
        payload = json.loads(self.session_state_path.read_text(encoding="utf-8"))
        return SessionMotionState(
            current_positions={name: int(payload["current_positions"].get(name, 0)) for name in self.axes},
            cumulative_executed_steps={
                name: int(payload["cumulative_executed_steps"].get(name, 0)) for name in self.axes
            },
            cumulative_net_steps={name: int(payload["cumulative_net_steps"].get(name, 0)) for name in self.axes},
            move_records=list(payload.get("move_records", [])),
        )

    def _save_state(self) -> None:
        self.session_state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "current_positions": self.state.current_positions,
            "cumulative_executed_steps": self.state.cumulative_executed_steps,
            "cumulative_net_steps": self.state.cumulative_net_steps,
            "move_records": self.state.move_records,
            "steps_per_revolution": self.steps_per_revolution,
            "max_cumulative_revolutions": self.max_cumulative_revolutions,
            "enforce_position_limit": self.enforce_position_limit,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.session_state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_baseline_state(
        self,
        baseline_state_path: str | Path,
        *,
        baseline_image_path: str | None = None,
        baseline_capture_metadata_path: str | None = None,
    ) -> BaselineState:
        baseline = BaselineState(
            baseline_positions={name: 0 for name in self.axes},
            baseline_image_path=baseline_image_path,
            baseline_capture_metadata_path=baseline_capture_metadata_path,
        )
        path = Path(baseline_state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(baseline), indent=2), encoding="utf-8")
        return baseline

    def load_baseline_state(self, baseline_state_path: str | Path) -> BaselineState:
        payload = json.loads(Path(baseline_state_path).read_text(encoding="utf-8"))
        return BaselineState(
            baseline_positions={key: int(value) for key, value in payload["baseline_positions"].items()},
            baseline_image_path=payload.get("baseline_image_path"),
            baseline_capture_metadata_path=payload.get("baseline_capture_metadata_path"),
            created_at=str(payload.get("created_at", "")),
        )

    def sync_current_state_to_baseline(self, baseline: BaselineState) -> SessionMotionState:
        self.state.current_positions = {
            name: int(baseline.baseline_positions.get(name, 0)) for name in self.axes
        }
        self.state.cumulative_net_steps = {name: 0 for name in self.axes}
        self._save_state()
        return self.state

    def compute_compensated_submoves(self, hardware_steps: int) -> list[int]:
        if hardware_steps == 0:
            return []
        if not self.backlash_compensation:
            return [int(hardware_steps)]
        sign = 1 if hardware_steps > 0 else -1
        return [-sign, int(hardware_steps + sign)]

    def move_axis(self, axis_name: str, logical_steps: int) -> MotionRecord:
        if axis_name not in self.axes:
            raise KeyError(f"Unknown axis: {axis_name}")
        axis = self.axes[axis_name]
        logical_steps = int(logical_steps)
        hardware_steps = logical_steps * int(axis.direction_sign)
        submoves = self.compute_compensated_submoves(hardware_steps)
        executed_abs_steps = sum(abs(step) for step in submoves)
        new_cumulative = self.state.cumulative_executed_steps[axis_name] + executed_abs_steps
        max_steps = int(round(self.max_cumulative_revolutions * self.steps_per_revolution))
        projected_position = self.state.current_positions[axis_name] + logical_steps
        if self.enforce_position_limit and abs(projected_position) > max_steps:
            raise MotionSafetyLimitError(
                f"Axis {axis_name} would exceed net displacement limit: "
                f"{projected_position} steps exceeds +/-{max_steps}."
            )
        for step in submoves:
            self._send_move(axis.controller_ip, axis.motor_number, step)
        self.state.current_positions[axis_name] = projected_position
        self.state.cumulative_executed_steps[axis_name] = new_cumulative
        self.state.cumulative_net_steps[axis_name] += abs(logical_steps)
        record = MotionRecord(
            axis_name=axis_name,
            requested_logical_steps=logical_steps,
            requested_hardware_steps=hardware_steps,
            executed_submoves=submoves,
            cumulative_executed_steps=new_cumulative,
            cumulative_executed_revolutions=new_cumulative / self.steps_per_revolution,
            logical_position_after=self.state.current_positions[axis_name],
            dry_run=self.dry_run,
        )
        self.state.move_records.append(asdict(record))
        self._save_state()
        return record

    def move_many(self, deltas: dict[str, int]) -> list[MotionRecord]:
        records: list[MotionRecord] = []
        for axis_name, logical_steps in deltas.items():
            records.append(self.move_axis(axis_name, logical_steps))
        return records

    def return_to_baseline(self, baseline: BaselineState) -> list[MotionRecord]:
        deltas = {}
        for axis_name, baseline_position in baseline.baseline_positions.items():
            current = self.state.current_positions.get(axis_name, 0)
            deltas[axis_name] = int(baseline_position) - int(current)
        return self.move_many(deltas)

    def _send_move(self, controller_ip: str, motor_number: int, steps: int) -> None:
        if steps == 0:
            return
        if self.dry_run:
            return
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.http_session.get(
                    f"http://{controller_ip}/move",
                    params={"motor": int(motor_number), "steps": int(steps)},
                    timeout=(5, 30),
                )
                response.raise_for_status()
                return
            except requests.RequestException as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.0)
        if last_error is not None:
            raise last_error
