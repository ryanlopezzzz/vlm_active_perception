"""Run one method/repeat of the two-mirror cavity simulation comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.io.schemas import load_two_mirror_cavity_simulation_compare_methods_config
from src.tasks.two_mirror_cavity_simulation_alignment import (
    run_two_mirror_cavity_simulation_llm_alignment,
    run_two_mirror_cavity_simulation_oracle_alignment,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeat-index", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--method", required=True, choices=("llm", "bayes", "random"))
    parser.add_argument("--model-key")
    args = parser.parse_args()
    config = load_two_mirror_cavity_simulation_compare_methods_config(args.config)
    initial_conditions = json.loads(Path(config.initial_conditions_path).read_text(encoding="utf-8"))
    hidden_offset = initial_conditions[args.repeat_index]
    if args.method == "llm":
        models = {model.key: model.llm for model in config.llm_models}
        if args.model_key not in models:
            raise ValueError(f"Unknown model key: {args.model_key}")
        run_two_mirror_cavity_simulation_llm_alignment(
            out_dir=args.out, repeat_index=args.repeat_index, seed=args.seed,
            hidden_offset=hidden_offset, iterations=config.iterations,
            probes_per_batch=config.probes_per_batch,
            llm_config=models[args.model_key], profile_path=config.profile_path, axes=config.axes)
    else:
        run_two_mirror_cavity_simulation_oracle_alignment(
            out_dir=args.out, method=args.method, iterations=config.iterations, seed=args.seed,
            repeat_index=args.repeat_index, hidden_offset=hidden_offset,
            profile_path=config.profile_path, axes=config.axes)


if __name__ == "__main__":
    main()
