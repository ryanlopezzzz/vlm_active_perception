"""Run experiments from validated configuration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.io.artifacts import ensure_dir, make_experiment_root, write_json, write_resolved_config_yaml
from src.io.logging import setup_run_logger
from src.io.schemas import (
    MichelsonInterferometerCompareMethodsConfig,
    RealMILLMRunsConfig,
    RealMIPreflightConfig,
    TwoMirrorCavityRealLLMRunsConfig,
    TwoMirrorCavitySimulationCompareMethodsConfig,
    load_michelson_interferometer_compare_methods_config,
    load_real_mi_llm_runs_config,
    load_real_mi_preflight_config,
    load_two_mirror_cavity_hardware_smoke_config,
    load_two_mirror_cavity_real_llm_runs_config,
    load_two_mirror_cavity_simulation_compare_methods_config,
)
from src.io.seed import repeat_seed
from src.tasks.two_mirror_cavity_real_llm_prompting import sample_distinct_hidden_offsets


@dataclass(frozen=True)
class MethodTask:
    repeat_index: int
    method: str
    command: list[str]
    output_dir: Path
    log_path: Path


def _clear_incomplete_task_directory(task_dir: Path, run_root: Path) -> None:
    """Remove artifacts from one incomplete method/repeat before restarting it."""
    resolved_root = run_root.resolve()
    resolved_task = task_dir.resolve()
    try:
        relative = resolved_task.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to clear task directory outside run root: {resolved_task}") from exc
    if len(relative.parts) != 2 or not relative.parts[1].startswith("repeat_"):
        raise ValueError(f"Refusing to clear unexpected task directory: {resolved_task}")
    for child in resolved_task.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()










def _run_subprocess(cmd: list[str], cwd: Path, log_path: Path) -> None:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(cwd) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        combined = []
        if completed.stdout:
            combined.append("=== STDOUT ===\n" + completed.stdout)
        if completed.stderr:
            combined.append("=== STDERR ===\n" + completed.stderr)
        log_path.write_text("\n\n".join(combined).strip() + "\n", encoding="utf-8")
    except subprocess.CalledProcessError as exc:
        combined = [f"Command failed with exit code {exc.returncode}", f"Command: {cmd}"]
        if exc.stdout:
            combined.append("=== STDOUT ===\n" + exc.stdout)
        if exc.stderr:
            combined.append("=== STDERR ===\n" + exc.stderr)
        log_path.write_text("\n\n".join(combined).strip() + "\n", encoding="utf-8")
        raise RuntimeError(f"Subprocess failed. See log: {log_path}") from exc






def _build_real_mi_preflight_command(config_path: str | Path, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        "src/tasks/mi_real_preflight.py",
        f"--config={config_path}",
        f"--out={out_dir}",
    ]


def _build_two_mirror_cavity_hardware_smoke_command(
    config_path: str | Path, out_dir: Path
) -> list[str]:
    return [
        sys.executable,
        "src/tasks/two_mirror_cavity_hardware_smoke.py",
        f"--config={config_path}",
        f"--out={out_dir}",
    ]


def _build_two_mirror_cavity_real_llm_command(
    config: TwoMirrorCavityRealLLMRunsConfig,
    config_path: str | Path,
    out_dir: Path,
    repeat_index: int,
    seed: int,
    hidden_offset: dict[str, int],
    *,
    reset_session: bool,
) -> list[str]:
    command = [
        sys.executable,
        "src/tasks/two_mirror_cavity_real_llm_optimize.py",
        f"--config={config_path}",
        f"--out={out_dir}",
        f"--repeat-index={repeat_index}",
        f"--seed={seed}",
        f"--hidden-x={hidden_offset['x']}",
        f"--hidden-y={hidden_offset['y']}",
    ]
    if reset_session:
        command.append("--reset-session")
    return command


def _build_real_mi_llm_command(
    config: RealMILLMRunsConfig,
    config_path: str | Path,
    out_dir: Path,
    repeat_index: int,
    seed: int,
    restore_baseline_at_end: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "src/tasks/mi_real_llm_optimize.py",
        f"--config={config_path}",
        f"--out={out_dir}",
        f"--repeat-index={repeat_index}",
        f"--seed={seed}",
    ]
    if restore_baseline_at_end:
        command.append("--restore-baseline-at-end")
    return command





def _build_michelson_interferometer_compare_command(
    config_path: str | Path,
    out_dir: Path,
    repeat_index: int,
    seed: int,
    method: str,
    model_key: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "src/tasks/michelson_interferometer_compare_optimize.py",
        f"--config={config_path}",
        f"--out={out_dir}",
        f"--repeat-index={repeat_index}",
        f"--seed={seed}",
        f"--method={method}",
    ]
    if model_key is not None:
        command.append(f"--model-key={model_key}")
    return command


def _build_two_mirror_cavity_simulation_compare_command(
    config_path: str | Path, out_dir: Path, repeat_index: int, seed: int,
    method: str, model_key: str | None = None,
) -> list[str]:
    command = [
        sys.executable, "src/tasks/two_mirror_cavity_simulation_compare_optimize.py",
        f"--config={config_path}", f"--out={out_dir}", f"--repeat-index={repeat_index}",
        f"--seed={seed}", f"--method={method}",
    ]
    if model_key is not None:
        command.append(f"--model-key={model_key}")
    return command










def _run_method_task(task: MethodTask, repo_root: Path) -> dict[str, Any]:
    try:
        _run_subprocess(task.command, cwd=repo_root, log_path=task.log_path)
        result = {
            "repeat_index": task.repeat_index,
            "name": task.method,
            "status": "succeeded",
            "command": task.command,
            "output_dir": str(task.output_dir),
            "log_path": str(task.log_path),
        }
        method_manifest_path = task.output_dir / "run_manifest.json"
        if method_manifest_path.exists():
            method_manifest = json.loads(method_manifest_path.read_text(encoding="utf-8"))
            if isinstance(method_manifest.get("llm_usage"), dict):
                result["llm_usage"] = method_manifest["llm_usage"]
        return result
    except Exception as exc:
        return {
            "repeat_index": task.repeat_index,
            "name": task.method,
            "status": "failed",
            "error": str(exc),
            "command": task.command,
            "output_dir": str(task.output_dir),
            "log_path": str(task.log_path),
        }


def _aggregate_llm_usage(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    models = sorted(
        {
            str(summary["model_requested"])
            for summary in summaries
            if summary.get("model_requested")
        }
    )
    request_wall_time_sec = sum(float(summary.get("request_wall_time_sec", 0.0)) for summary in summaries)
    request_attempts = sum(int(summary.get("request_attempts", 0)) for summary in summaries)
    return {
        "run_count": len(summaries),
        "models_requested": models,
        "request_attempts": request_attempts,
        "api_responses": sum(int(summary.get("api_responses", 0)) for summary in summaries),
        "accepted_responses": sum(int(summary.get("accepted_responses", 0)) for summary in summaries),
        "prompt_tokens": sum(int(summary.get("prompt_tokens", 0)) for summary in summaries),
        "completion_tokens": sum(int(summary.get("completion_tokens", 0)) for summary in summaries),
        "total_tokens": sum(int(summary.get("total_tokens", 0)) for summary in summaries),
        "cost_credits": sum(float(summary.get("cost_credits", 0.0)) for summary in summaries),
        "responses_with_cost": sum(int(summary.get("responses_with_cost", 0)) for summary in summaries),
        "cost_complete": bool(summaries) and all(bool(summary.get("cost_complete")) for summary in summaries),
        "request_wall_time_sec": request_wall_time_sec,
        "avg_request_wall_time_sec": (request_wall_time_sec / request_attempts) if request_attempts else 0.0,
        "min_request_wall_time_sec": min(
            (float(summary.get("min_request_wall_time_sec", 0.0)) for summary in summaries),
            default=0.0,
        ),
        "max_request_wall_time_sec": max(
            (float(summary.get("max_request_wall_time_sec", 0.0)) for summary in summaries),
            default=0.0,
        ),
    }




def run_mi_real_preflight(config_path: str | Path) -> Path:
    config = load_real_mi_preflight_config(config_path)
    run_root = make_experiment_root(config.output_root, config.experiment_name)
    logger = setup_run_logger(run_root, name="mi_real_preflight")
    repo_root = Path(__file__).resolve().parents[2]

    write_resolved_config_yaml(run_root / "config_resolved.yaml", config)
    logger.info("Starting real MI preflight in %s", run_root)
    command = _build_real_mi_preflight_command(config_path, run_root)
    result = _run_method_task(
        MethodTask(
            repeat_index=0,
            method="preflight",
            command=command,
            output_dir=run_root,
            log_path=run_root / "preflight_subprocess.log",
        ),
        repo_root,
    )
    write_json(
        run_root / "run_manifest.json",
        {
            "experiment_name": config.experiment_name,
            "config_path": str(config_path),
            "run": result,
        },
    )
    if result["status"] == "failed":
        logger.error("Real MI preflight failed in %s", run_root)
        raise RuntimeError(f"Preflight failed. See {run_root / 'preflight_subprocess.log'}")
    logger.info("Finished real MI preflight in %s", run_root)
    return run_root


def run_two_mirror_cavity_hardware_smoke(config_path: str | Path) -> Path:
    config = load_two_mirror_cavity_hardware_smoke_config(config_path)
    run_root = make_experiment_root(config.output_root, config.experiment_name)
    logger = setup_run_logger(run_root, name="two_mirror_cavity_hardware_smoke")
    repo_root = Path(__file__).resolve().parents[2]

    write_resolved_config_yaml(run_root / "config_resolved.yaml", config)
    logger.info("Starting two-mirror cavity hardware smoke test in %s", run_root)
    result = _run_method_task(
        MethodTask(
            repeat_index=0,
            method="hardware_smoke",
            command=_build_two_mirror_cavity_hardware_smoke_command(config_path, run_root),
            output_dir=run_root,
            log_path=run_root / "hardware_smoke_subprocess.log",
        ),
        repo_root,
    )
    write_json(
        run_root / "run_manifest.json",
        {
            "experiment_name": config.experiment_name,
            "config_path": str(config_path),
            "run": result,
        },
    )
    if result["status"] == "failed":
        logger.error("Two-mirror cavity hardware smoke test failed in %s", run_root)
        raise RuntimeError(
            f"Hardware smoke test failed. See {run_root / 'hardware_smoke_subprocess.log'}"
        )
    logger.info("Finished two-mirror cavity hardware smoke test in %s", run_root)
    return run_root


def run_two_mirror_cavity_real_llm_runs(config_path: str | Path) -> Path:
    config = load_two_mirror_cavity_real_llm_runs_config(config_path)
    run_root = make_experiment_root(config.output_root, config.experiment_name)
    logger = setup_run_logger(run_root, name="two_mirror_cavity_real_llm_runs")
    if config.calibration_source_path is not None:
        calibration_payload = json.loads(
            Path(config.calibration_source_path).read_text(encoding="utf-8")
        )
        write_json(run_root / "calibration_used.json", calibration_payload)
        logger.info("Using fitted cavity calibration %s", config.calibration_source_path)
    repo_root = Path(__file__).resolve().parents[2]
    offsets = sample_distinct_hidden_offsets(
        config.axes, repeats=config.repeats, base_seed=config.base_seed
    )
    skipped = set(config.skip_repeat_indices)
    scheduled = [
        (repeat_index, hidden_offset)
        for repeat_index, hidden_offset in enumerate(offsets)
        if repeat_index not in skipped
    ]
    resolved_config_path = run_root / "config_resolved.yaml"
    write_resolved_config_yaml(resolved_config_path, config)
    manifest: dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "config_path": str(config_path),
        "calibration_source_path": config.calibration_source_path,
        "calibration_snapshot_path": (
            str(run_root / "calibration_used.json")
            if config.calibration_source_path is not None
            else None
        ),
        "execution": "sequential_continue_after_restored_repeat_failure",
        "configured_repeat_count": config.repeats,
        "skip_repeat_indices": config.skip_repeat_indices,
        "scheduled_repeat_indices": [repeat_index for repeat_index, _ in scheduled],
        "repeats": [],
        "failed_repeat_indices": [],
    }
    logger.info("Starting real two-mirror cavity LLM runs in %s", run_root)
    for execution_index, (repeat_index, hidden_offset) in enumerate(scheduled):
        seed = repeat_seed(config.base_seed, repeat_index)
        repeat_dir = ensure_dir(run_root / f"repeat_{repeat_index:02d}")
        command = _build_two_mirror_cavity_real_llm_command(
            config,
            resolved_config_path,
            repeat_dir,
            repeat_index,
            seed,
            hidden_offset,
            reset_session=execution_index == 0,
        )
        result = _run_method_task(
            MethodTask(
                repeat_index=repeat_index,
                method="llm",
                command=command,
                output_dir=repeat_dir,
                log_path=repeat_dir / "llm_subprocess.log",
            ),
            repo_root,
        )
        manifest["repeats"].append(
            {
                "repeat_index": repeat_index,
                "seed": seed,
                "hidden_offset_steps": hidden_offset,
                "run": result,
            }
        )
        write_json(run_root / "run_manifest.json", manifest)
        logger.info(
            "Cavity run complete repeat=%d seed=%d status=%s",
            repeat_index,
            seed,
            result["status"],
        )
        if result["status"] == "failed":
            manifest["failed_repeat_indices"].append(repeat_index)
            restored, restoration_reason = _repeat_restored_to_physical_origin(repeat_dir)
            manifest["repeats"][-1]["safe_to_continue"] = restored
            manifest["repeats"][-1]["restoration_check"] = restoration_reason
            write_json(run_root / "run_manifest.json", manifest)
            if not restored:
                logger.error(
                    "Aborting remaining repeats after repeat %d failed without verified restoration: %s",
                    repeat_index,
                    restoration_reason,
                )
                raise RuntimeError(
                    f"Two-mirror cavity repeat {repeat_index} failed and restoration to physical "
                    f"origin was not verified ({restoration_reason}); remaining repeats were not "
                    f"started. See {repeat_dir / 'llm_subprocess.log'} and "
                    f"{repeat_dir / 'run_manifest.json'}."
                )
            logger.warning(
                "Repeat %d failed, but restoration to physical origin was verified; "
                "continuing with the next scheduled repeat",
                repeat_index,
            )
    usage = [item["run"]["llm_usage"] for item in manifest["repeats"] if "llm_usage" in item["run"]]
    if usage:
        manifest["llm_usage"] = _aggregate_llm_usage(usage)
    write_json(run_root / "run_manifest.json", manifest)
    logger.info("Finished real two-mirror cavity LLM runs in %s", run_root)
    return run_root


def _repeat_restored_to_physical_origin(repeat_dir: Path) -> tuple[bool, str]:
    """Verify that a failed hardware repeat left both tracked axes at physical zero."""
    manifest_path = repeat_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return False, f"missing repeat manifest: {manifest_path}"
    try:
        repeat_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"unreadable repeat manifest: {exc}"
    restoration = repeat_manifest.get("restoration")
    if not isinstance(restoration, dict):
        return False, "repeat manifest has no restoration record"
    if restoration.get("attempted") is not True:
        return False, "restoration was not attempted"
    if restoration.get("succeeded") is not True:
        error = restoration.get("error")
        return False, f"restoration did not succeed: {error}"
    position = restoration.get("physical_position_after")
    if not isinstance(position, dict) or set(position) < {"x", "y"}:
        return False, "restoration has no complete physical_position_after"
    try:
        x_position = int(position["x"])
        y_position = int(position["y"])
    except (TypeError, ValueError):
        return False, f"restoration position is not integral: {position}"
    if x_position != 0 or y_position != 0:
        return False, f"restoration ended at x={x_position}, y={y_position}, not x=0, y=0"
    return True, "restoration succeeded at physical x=0, y=0"


def run_mi_real_llm_runs(config_path: str | Path) -> Path:
    config = load_real_mi_llm_runs_config(config_path)
    run_root = make_experiment_root(config.output_root, config.experiment_name)
    logger = setup_run_logger(run_root, name="mi_real_llm_runs")
    repo_root = Path(__file__).resolve().parents[2]

    write_resolved_config_yaml(run_root / "config_resolved.yaml", config)
    logger.info("Starting real MI LLM runs in %s", run_root)
    run_manifest: dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "config_path": str(config_path),
        "initial_condition_mode": config.initial_condition_mode,
        "methods": config.methods,
        "repeats": [],
    }
    failures: list[dict[str, Any]] = []
    for repeat_idx in range(config.repeats):
        seed = repeat_seed(config.base_seed, repeat_idx)
        repeat_dir = ensure_dir(run_root / f"repeat_{repeat_idx:02d}")
        method_dir = ensure_dir(repeat_dir / "llm")
        command = _build_real_mi_llm_command(
            config,
            config_path,
            method_dir,
            repeat_idx,
            seed,
            restore_baseline_at_end=True,
        )
        result = _run_method_task(
            MethodTask(
                repeat_index=repeat_idx,
                method="llm",
                command=command,
                output_dir=method_dir,
                log_path=repeat_dir / "llm_subprocess.log",
            ),
            repo_root,
        )
        repeat_record = {
            "repeat_index": repeat_idx,
            "seed": seed,
            "run": result,
        }
        run_manifest["repeats"].append(repeat_record)
        logger.info(
            "Real MI run complete repeat=%d seed=%d status=%s",
            repeat_idx,
            seed,
            result["status"],
        )
        if result["status"] == "failed":
            failures.append(result)
    write_json(run_root / "run_manifest.json", run_manifest)
    if failures:
        logger.error("Finished real MI runs with %d failure(s) in %s", len(failures), run_root)
        raise RuntimeError(
            f"Real MI runs finished with {len(failures)} failed task(s). "
            f"See run_manifest.json and per-repeat subprocess logs in {run_root}."
        )
    logger.info("Finished real MI runs in %s", run_root)
    return run_root





def run_michelson_interferometer_compare_methods(
    config_path: str | Path,
    *,
    resume_run_root: str | Path | None = None,
) -> Path:
    config: MichelsonInterferometerCompareMethodsConfig = (
        load_michelson_interferometer_compare_methods_config(config_path)
    )
    if resume_run_root is None:
        run_root = make_experiment_root(config.output_root, config.experiment_name)
    else:
        run_root = Path(resume_run_root).resolve()
        if not run_root.is_dir():
            raise ValueError(f"Resume run directory does not exist: {run_root}")
    logger = setup_run_logger(run_root, name="michelson_interferometer_compare_methods")
    repo_root = Path(__file__).resolve().parents[2]
    if resume_run_root is None:
        write_resolved_config_yaml(run_root / "config_resolved.yaml", config)

    method_names = (
        ([model.key for model in config.llm_models] if "llm" in config.methods else [])
        + [method for method in ("bayes", "random") if method in config.methods]
    )
    run_manifest: dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "config_path": str(config_path),
        "objective_name": "sum_beam_distances_from_center_px",
        "method_names": method_names,
        "repeats": [],
    }
    tasks: list[MethodTask] = []
    repeat_records: dict[int, dict[str, Any]] = {}
    for repeat_idx in range(config.repeats):
        seed = repeat_seed(config.base_seed, repeat_idx)
        repeat_records[repeat_idx] = {"repeat_index": repeat_idx, "seed": seed, "methods": []}
        for name in method_names:
            is_llm = name not in {"bayes", "random"}
            method = "llm" if is_llm else name
            method_dir = ensure_dir(run_root / name / f"repeat_{repeat_idx:02d}")
            task = MethodTask(
                repeat_index=repeat_idx,
                method=name,
                command=_build_michelson_interferometer_compare_command(
                    Path(config_path).resolve(),
                    method_dir,
                    repeat_idx,
                    seed,
                    method,
                    model_key=name if is_llm else None,
                ),
                output_dir=method_dir,
                log_path=method_dir / "subprocess.log",
            )
            completed_manifest_path = method_dir / "run_manifest.json"
            if resume_run_root is not None and completed_manifest_path.exists():
                try:
                    completed_manifest = json.loads(completed_manifest_path.read_text(encoding="utf-8"))
                    if (
                        completed_manifest.get("repeat_index") != repeat_idx
                        or not isinstance(completed_manifest.get("final_alignment_metrics"), dict)
                        or not completed_manifest.get("final_image_path")
                    ):
                        raise ValueError("manifest is missing required completion fields")
                    result: dict[str, Any] = {
                        "repeat_index": repeat_idx,
                        "name": name,
                        "status": "succeeded",
                        "resumed_existing": True,
                        "command": task.command,
                        "output_dir": str(method_dir),
                        "log_path": str(task.log_path),
                    }
                    if isinstance(completed_manifest.get("llm_usage"), dict):
                        result["llm_usage"] = completed_manifest["llm_usage"]
                    repeat_records[repeat_idx]["methods"].append(result)
                    logger.info("Skipping completed task repeat=%d method=%s", repeat_idx, name)
                    continue
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning(
                        "Restarting task with incomplete manifest repeat=%d method=%s: %s",
                        repeat_idx,
                        name,
                        exc,
                    )
            if resume_run_root is not None:
                logger.info(
                    "Clearing incomplete task directory before restart repeat=%d method=%s",
                    repeat_idx,
                    name,
                )
                _clear_incomplete_task_directory(method_dir, run_root)
            tasks.append(task)

    logger.info(
        "Executing %d Michelson comparison tasks (%d already complete)",
        len(tasks),
        config.repeats * len(method_names) - len(tasks),
    )
    with ThreadPoolExecutor(max_workers=config.max_parallel_jobs) as executor:
        futures = {executor.submit(_run_method_task, task, repo_root): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            repeat_records[task.repeat_index]["methods"].append(result)
            logger.info(
                "Michelson comparison complete repeat=%d method=%s status=%s",
                task.repeat_index,
                task.method,
                result["status"],
            )

    method_order = {name: idx for idx, name in enumerate(method_names)}
    failures: list[dict[str, Any]] = []
    for repeat_idx in range(config.repeats):
        record = repeat_records[repeat_idx]
        record["methods"].sort(key=lambda item: method_order[item["name"]])
        failures.extend(item for item in record["methods"] if item["status"] == "failed")
        run_manifest["repeats"].append(record)
    usage = [
        item["llm_usage"]
        for record in run_manifest["repeats"]
        for item in record["methods"]
        if isinstance(item.get("llm_usage"), dict)
    ]
    run_manifest["llm_usage"] = _aggregate_llm_usage(usage)
    write_json(run_root / "run_manifest.json", run_manifest)
    if failures:
        raise RuntimeError(
            f"Michelson comparison finished with {len(failures)} failed task(s). See {run_root}."
        )
    logger.info("Finished Michelson comparison in %s", run_root)
    return run_root


def run_two_mirror_cavity_simulation_compare_methods(
    config_path: str | Path, *, resume_run_root: str | Path | None = None,
) -> Path:
    """Run or resume the lab-matched LLM/Bayesian/random cavity comparison."""
    config: TwoMirrorCavitySimulationCompareMethodsConfig = (
        load_two_mirror_cavity_simulation_compare_methods_config(config_path)
    )
    run_root = (make_experiment_root(config.output_root, config.experiment_name)
                if resume_run_root is None else Path(resume_run_root).resolve())
    if resume_run_root is not None and not run_root.is_dir():
        raise ValueError(f"Resume run directory does not exist: {run_root}")
    logger = setup_run_logger(run_root, name="two_mirror_cavity_simulation_compare_methods")
    repo_root = Path(__file__).resolve().parents[2]
    if resume_run_root is None:
        write_resolved_config_yaml(run_root / "config_resolved.yaml", config)
    method_names = (([model.key for model in config.llm_models] if "llm" in config.methods else [])
                    + [method for method in ("bayes", "random") if method in config.methods])
    run_manifest: dict[str, Any] = {
        "experiment_name": config.experiment_name, "config_path": str(config_path),
        "objective_name": "motor_distance_steps", "method_names": method_names, "repeats": [],
    }
    tasks: list[MethodTask] = []
    repeat_records: dict[int, dict[str, Any]] = {}
    for repeat_idx in range(config.repeats):
        seed = repeat_seed(config.base_seed, repeat_idx)
        repeat_records[repeat_idx] = {"repeat_index": repeat_idx, "seed": seed, "methods": []}
        for name in method_names:
            is_llm = name not in {"bayes", "random"}
            method_dir = ensure_dir(run_root / name / f"repeat_{repeat_idx:02d}")
            task = MethodTask(
                repeat_index=repeat_idx, method=name,
                command=_build_two_mirror_cavity_simulation_compare_command(
                    Path(config_path).resolve(), method_dir, repeat_idx, seed,
                    "llm" if is_llm else name, name if is_llm else None),
                output_dir=method_dir, log_path=method_dir / "subprocess.log")
            manifest_path = method_dir / "run_manifest.json"
            if resume_run_root is not None and manifest_path.exists():
                try:
                    completed = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if (completed.get("repeat_index") != repeat_idx
                            or not isinstance(completed.get("final_alignment_metrics"), dict)
                            or not completed.get("final_image_path")):
                        raise ValueError("manifest is missing required completion fields")
                    result: dict[str, Any] = {
                        "repeat_index": repeat_idx, "name": name, "status": "succeeded",
                        "resumed_existing": True, "command": task.command,
                        "output_dir": str(method_dir), "log_path": str(task.log_path)}
                    if isinstance(completed.get("llm_usage"), dict):
                        result["llm_usage"] = completed["llm_usage"]
                    repeat_records[repeat_idx]["methods"].append(result)
                    logger.info("Skipping completed task repeat=%d method=%s", repeat_idx, name)
                    continue
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning("Restarting incomplete repeat=%d method=%s: %s", repeat_idx, name, exc)
            if resume_run_root is not None:
                _clear_incomplete_task_directory(method_dir, run_root)
            tasks.append(task)
    logger.info("Executing %d cavity simulation comparison tasks", len(tasks))
    with ThreadPoolExecutor(max_workers=config.max_parallel_jobs) as executor:
        futures = {executor.submit(_run_method_task, task, repo_root): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            repeat_records[task.repeat_index]["methods"].append(future.result())
    order = {name: idx for idx, name in enumerate(method_names)}
    failures = []
    for repeat_idx in range(config.repeats):
        record = repeat_records[repeat_idx]
        record["methods"].sort(key=lambda item: order[item["name"]])
        failures.extend(item for item in record["methods"] if item["status"] == "failed")
        run_manifest["repeats"].append(record)
    usage = [item["llm_usage"] for record in run_manifest["repeats"] for item in record["methods"]
             if isinstance(item.get("llm_usage"), dict)]
    run_manifest["llm_usage"] = _aggregate_llm_usage(usage)
    write_json(run_root / "run_manifest.json", run_manifest)
    if failures:
        raise RuntimeError(f"Cavity simulation comparison finished with {len(failures)} failed task(s). See {run_root}.")
    return run_root
