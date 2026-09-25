from __future__ import annotations

import argparse
from datetime import datetime

from src.io.schemas import load_real_mi_llm_runs_config
from src.tasks.mi_real_alignment import run_real_llm_alignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--repeat-index", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--restore-baseline-at-end", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default=datetime.now().strftime("experiments/mi_real_llm_%Y-%m-%d_%H-%M-%S"),
    )
    args = parser.parse_args()
    config = load_real_mi_llm_runs_config(args.config)
    run_real_llm_alignment(
        out_dir=args.out,
        iterations=config.iterations,
        seed=args.seed,
        repeat_index=args.repeat_index,
        restore_baseline_at_end=args.restore_baseline_at_end,
        llm_config=config.llm,
        camera_config=config.camera,
        hardware_config=config.hardware,
        axes=config.axes,
        initial_condition_mode=config.initial_condition_mode,
    )


if __name__ == "__main__":
    main()
