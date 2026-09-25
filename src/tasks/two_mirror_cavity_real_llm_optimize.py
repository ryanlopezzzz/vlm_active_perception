"""CLI entrypoint for one real two-mirror cavity LLM optimization repeat."""

from __future__ import annotations

import argparse
from datetime import datetime

from src.io.schemas import load_two_mirror_cavity_real_llm_runs_config
from src.tasks.two_mirror_cavity_real_llm_alignment import run_two_mirror_cavity_real_llm_alignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=datetime.now().strftime("experiments/two_mirror_cavity_real_llm_%Y-%m-%d_%H-%M-%S"))
    parser.add_argument("--repeat-index", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--hidden-x", required=True, type=int)
    parser.add_argument("--hidden-y", required=True, type=int)
    parser.add_argument("--reset-session", action="store_true")
    args = parser.parse_args()
    config = load_two_mirror_cavity_real_llm_runs_config(args.config)
    run_two_mirror_cavity_real_llm_alignment(
        out_dir=args.out,
        repeat_index=args.repeat_index,
        seed=args.seed,
        hidden_offset={"x": args.hidden_x, "y": args.hidden_y},
        iterations=config.iterations,
        probes_per_batch=config.probes_per_batch,
        llm_config=config.llm,
        camera_config=config.camera,
        hardware_config=config.hardware,
        axes=config.axes,
        reset_session=args.reset_session,
    )


if __name__ == "__main__":
    main()
