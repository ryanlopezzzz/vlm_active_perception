from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.io.schemas import load_michelson_interferometer_compare_methods_config
from src.tasks.michelson_interferometer_alignment import run_michelson_interferometer_llm_alignment
from src.tasks.michelson_interferometer_oracle_alignment import run_michelson_interferometer_oracle_alignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeat-index", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--method", required=True, choices=("llm", "bayes", "random"))
    parser.add_argument("--model-key")
    args = parser.parse_args()
    config = load_michelson_interferometer_compare_methods_config(args.config)
    initial_conditions = json.loads(Path(config.initial_conditions_path).read_text(encoding="utf-8"))
    hidden_offset_steps = initial_conditions[args.repeat_index]
    if args.method == "llm":
        models = {model.key: model.llm for model in config.llm_models}
        if args.model_key not in models:
            raise ValueError(f"Unknown model key: {args.model_key}")
        run_michelson_interferometer_llm_alignment(
            out_dir=args.out,
            iterations=config.iterations,
            seed=args.seed,
            repeat_index=args.repeat_index,
            llm_config=models[args.model_key],
            simulation_config=config.simulation,
            axes=config.axes,
            hidden_offset_steps_override=hidden_offset_steps,
        )
    else:
        run_michelson_interferometer_oracle_alignment(
            out_dir=args.out,
            method=args.method,
            iterations=config.iterations,
            seed=args.seed,
            repeat_index=args.repeat_index,
            hidden_offset_steps=hidden_offset_steps,
            simulation_config=config.simulation,
            axes=config.axes,
        )


if __name__ == "__main__":
    main()
