"""CLI entrypoint for running and managing experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.io.schemas import load_experiment_type
from src.runners.run_experiment import (
    run_mi_real_llm_runs,
    run_mi_real_preflight,
    run_michelson_interferometer_compare_methods,
    run_two_mirror_cavity_hardware_smoke,
    run_two_mirror_cavity_real_llm_runs,
    run_two_mirror_cavity_simulation_compare_methods,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robotics alignment experiment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment from config")
    run_parser.add_argument("--config", required=True, help="Path to YAML config")
    run_parser.add_argument(
        "--resume-run",
        help="Existing run directory to resume; currently supported by the primary Michelson comparison.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        config_path = Path(args.config)
        experiment_type = load_experiment_type(config_path)
        resumable = {"michelson_interferometer_compare_methods", "two_mirror_cavity_simulation_compare_methods"}
        if args.resume_run and experiment_type not in resumable:
            raise ValueError(f"--resume-run is currently supported only for {sorted(resumable)}.")
        if experiment_type == "motorized_mirror_relay_matched":
            from src.runners.run_motorized_mirror_relay_matched import run
            run_root = run(config_path)
        elif experiment_type == "mi_real_preflight":
            run_root = run_mi_real_preflight(config_path)
        elif experiment_type == "mi_real_llm_runs":
            run_root = run_mi_real_llm_runs(config_path)
        elif experiment_type == "michelson_interferometer_compare_methods":
            run_root = run_michelson_interferometer_compare_methods(
                config_path,
                resume_run_root=args.resume_run,
            )
        elif experiment_type == "two_mirror_cavity_hardware_smoke":
            run_root = run_two_mirror_cavity_hardware_smoke(config_path)
        elif experiment_type == "two_mirror_cavity_real_llm_runs":
            run_root = run_two_mirror_cavity_real_llm_runs(config_path)
        elif experiment_type == "two_mirror_cavity_simulation_compare_methods":
            run_root = run_two_mirror_cavity_simulation_compare_methods(
                config_path, resume_run_root=args.resume_run)
        else:
            raise ValueError(
                f"Unsupported experiment_type: {experiment_type}. "
                "Expected 'motorized_mirror_relay_matched', 'mi_real_preflight', 'mi_real_llm_runs', "
                "'michelson_interferometer_compare_methods', "
                "'two_mirror_cavity_hardware_smoke', "
                "'two_mirror_cavity_real_llm_runs', "
                "or 'two_mirror_cavity_simulation_compare_methods'."
            )
        print(f"Run complete: {run_root}")


if __name__ == "__main__":
    main()
