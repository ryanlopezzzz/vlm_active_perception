from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def start_timer() -> tuple[str, float]:
    return utc_now_iso(), time.perf_counter()


def finish_timer(started_at: str, start_perf: float) -> dict[str, Any]:
    return {
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "wall_time_sec": time.perf_counter() - start_perf,
    }


def step_timing_fields(started_at: str, start_perf: float, *, llm_request_wall_time_sec: float) -> dict[str, Any]:
    finished = finish_timer(started_at, start_perf)
    return {
        "step_started_at": finished["started_at"],
        "step_finished_at": finished["finished_at"],
        "step_wall_time_sec": finished["wall_time_sec"],
        "llm_request_wall_time_sec": float(llm_request_wall_time_sec),
    }


def llm_request_wall_time_since(llm: Any, call_count: int) -> float:
    calls = getattr(llm, "api_calls", [])[call_count:]
    return float(
        sum(
            call.get("duration_sec", 0.0)
            for call in calls
            if isinstance(call.get("duration_sec"), (int, float)) and not isinstance(call.get("duration_sec"), bool)
        )
    )


def llm_call_count(llm: Any) -> int:
    return len(getattr(llm, "api_calls", []))


def timing_totals_from_steps(steps: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "total_step_wall_time_sec": float(
            sum(
                step.get("step_wall_time_sec", 0.0)
                for step in steps
                if isinstance(step.get("step_wall_time_sec"), (int, float))
                and not isinstance(step.get("step_wall_time_sec"), bool)
            )
        ),
        "total_llm_request_wall_time_sec": float(
            sum(
                step.get("llm_request_wall_time_sec", 0.0)
                for step in steps
                if isinstance(step.get("llm_request_wall_time_sec"), (int, float))
                and not isinstance(step.get("llm_request_wall_time_sec"), bool)
            )
        ),
    }


def add_run_timing(
    manifest: dict[str, Any],
    *,
    run_started_at: str,
    run_start_perf: float,
    steps: list[dict[str, Any]] | None = None,
) -> None:
    finished = finish_timer(run_started_at, run_start_perf)
    manifest["run_started_at"] = finished["started_at"]
    manifest["run_finished_at"] = finished["finished_at"]
    manifest["run_wall_time_sec"] = finished["wall_time_sec"]
    if steps is not None:
        manifest.update(timing_totals_from_steps(steps))


def write_timing_json(
    out_dir: str | Path,
    *,
    run_started_at: str,
    run_start_perf: float,
    steps: list[dict[str, Any]],
    llm: Any | None = None,
) -> None:
    payload: dict[str, Any] = {
        "steps": steps,
    }
    add_run_timing(payload, run_started_at=run_started_at, run_start_perf=run_start_perf, steps=steps)
    if llm is not None:
        payload["llm_usage"] = llm.get_usage_summary()
    Path(out_dir, "timing.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
