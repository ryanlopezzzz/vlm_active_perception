"""Artifact path and persistence helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def make_experiment_root(output_root: str, experiment_name: str) -> Path:
    """Create and return a timestamped root folder for one execution."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = Path(output_root) / experiment_name / f"run_{ts}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_resolved_config_yaml(path: Path, config_obj: Any) -> None:
    payload = asdict(config_obj) if is_dataclass(config_obj) else config_obj
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
