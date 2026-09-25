from __future__ import annotations

import argparse
from datetime import datetime

from src.io.schemas import load_real_mi_preflight_config
from src.tasks.mi_real_alignment import perform_real_mi_preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument(
        "--out",
        type=str,
        default=datetime.now().strftime("experiments/mi_real_preflight_%Y-%m-%d_%H-%M-%S"),
    )
    args = parser.parse_args()
    config = load_real_mi_preflight_config(args.config)
    perform_real_mi_preflight(
        out_dir=args.out,
        camera_config=config.camera,
        hardware_config=config.hardware,
        axes=config.axes,
    )


if __name__ == "__main__":
    main()
