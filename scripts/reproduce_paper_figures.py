"""Regenerate all paper-result figures using only compact paper artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.plot_combined_alignment_comparison import plot as plot_combined


MODEL_COLORS = {
    "gpt_5_6_sol": ("#1769aa", "#8ec7e8"),
    "gemini_3_5_flash": ("#d95f02", "#fdbb84"),
    "claude_sonnet_5": ("#238b45", "#a1d99b"),
    "qwen3_vl_32b_instruct": ("#756bb1", "#bcbddc"),
    "bayes": ("#8c564b", "#c9aaa4"),
    "random": ("#636363", "#bdbdbd"),
}
MODEL_LABELS = {
    "gpt_5_6_sol": "GPT-5.6 SOL",
    "gemini_3_5_flash": "Gemini 3.5 Flash",
    "claude_sonnet_5": "Claude Sonnet 5",
    "qwen3_vl_32b_instruct": "Qwen3-VL-32B-Instruct",
    "bayes": "Bayesian optimization",
    "random": "Random search",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value: str) -> float:
    return float(value)


def save_trajectory(rows, groups, key_field, iteration_field, mean_field, std_field, ylabel, title, output, threshold=None, figsize=(9.5, 5.6)):
    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    for key in groups:
        selected = sorted((row for row in rows if row[key_field] == key), key=lambda row: int(row[iteration_field]))
        x = np.asarray([int(row[iteration_field]) for row in selected])
        mean = np.asarray([number(row[mean_field]) for row in selected])
        std = np.asarray([number(row[std_field]) for row in selected])
        color, fill = MODEL_COLORS[key]
        axis.plot(x, mean, color=color, linewidth=2.2, label=MODEL_LABELS[key])
        axis.fill_between(x, np.maximum(0.0, mean - std), mean + std, color=fill, alpha=0.30)
    if threshold is not None:
        axis.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.4)
    axis.set_xlabel("Optimization iteration")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_trajectory_with_success(rows, repeat_rows, groups, key_field, mean_field, std_field, success_field, output, title):
    figure, (axis, success_axis) = plt.subplots(
        2, 1, figsize=(9, 7.6), constrained_layout=True, gridspec_kw={"height_ratios": [4, 1]}
    )
    for key in groups:
        selected = sorted((row for row in rows if row[key_field] == key), key=lambda row: int(row["iteration"]))
        x = np.asarray([int(row["iteration"]) for row in selected])
        mean = np.asarray([number(row[mean_field]) for row in selected])
        std = np.asarray([number(row[std_field]) for row in selected])
        color, fill = MODEL_COLORS[key]
        axis.plot(x, mean, color=color, linewidth=2.2, label=MODEL_LABELS[key])
        axis.fill_between(x, np.maximum(0.0, mean - std), mean + std, color=fill, alpha=0.30)
    axis.axhline(500.0, color="#d62728", linestyle="--", linewidth=1.4)
    axis.set(xlabel="Optimization iteration", ylabel="Motor distance (steps)", title=title, ylim=(0, None))
    axis.grid(True, alpha=0.3)
    axis.legend()
    percentages = []
    labels = []
    for key in groups:
        selected = [row for row in repeat_rows if row[key_field] == key]
        successes = sum(row[success_field].lower() == "true" for row in selected)
        percentages.append(100.0 * successes / len(selected))
        labels.append(f"{successes}/{len(selected)}")
    bars = success_axis.bar(np.arange(len(groups)), percentages, width=0.45, color=[MODEL_COLORS[key][0] for key in groups])
    for bar, value, label in zip(bars, percentages, labels):
        success_axis.text(bar.get_x() + bar.get_width() / 2, value + 3, f"{value:.0f}% ({label})", ha="center")
    success_axis.set_xticks(np.arange(len(groups)), [MODEL_LABELS[key] for key in groups])
    success_axis.set_ylabel("Success (%)")
    success_axis.set_ylim(0, 125)
    success_axis.grid(True, axis="y", alpha=0.25)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def reproduce_michelson(data_root: Path, output_root: Path) -> None:
    lab_data = data_root / "michelson_lab"
    lab_out = output_root / "michelson_lab"
    lab_out.mkdir(parents=True)
    lab_groups = ("gpt_5_6_sol", "gemini_3_5_flash", "claude_sonnet_5")
    iteration = read_csv(lab_data / "iteration_summary.csv")
    save_trajectory(iteration, lab_groups, "model_key", "iteration", "center_error_mean", "center_error_std", "Sum of beam distances from center (px)", "Lab Michelson alignment over iterations", lab_out / "01_center_error_over_iterations.png", figsize=(9, 5.4))


def reproduce_cavity(data_root: Path, output_root: Path) -> None:
    lab_data = data_root / "two_mirror_cavity_lab"
    lab_out = output_root / "two_mirror_cavity_lab"
    lab_out.mkdir(parents=True)
    lab_groups = ("gpt_5_6_sol", "gemini_3_5_flash", "qwen3_vl_32b_instruct")
    iteration = read_csv(lab_data / "iteration_summary.csv")
    repeats = read_csv(lab_data / "repeat_metrics.csv")
    save_trajectory_with_success(iteration, repeats, lab_groups, "model", "motor_distance_mean", "motor_distance_std", "success_within_500_steps", lab_out / "07_motor_distance_over_iterations.png", "Lab cavity alignment over iterations")


def reproduce_relay(data_root: Path, output_root: Path) -> None:
    lab_rows = read_csv(data_root / "four_mirror_relay_lab/milestones.csv")
    lab_out = output_root / "four_mirror_relay_lab"
    lab_out.mkdir(parents=True)
    figure, axis = plt.subplots(figsize=(9, 5.8))
    positions = np.arange(4)
    width = 0.34
    for index, model in enumerate(("GPT-5.6 Sol", "Gemini 3.5 Flash")):
        selected = np.asarray([[number(row[key]) for key in ("M1", "M2", "M3", "completion")] for row in lab_rows if row["model"] == model])
        mean, std = selected.mean(axis=0), selected.std(axis=0, ddof=1)
        offsets = positions + (index - 0.5) * width
        axis.bar(offsets, mean, width, color=("#1769aa", "#d95f02")[index], yerr=std, capsize=5, label=model)
    axis.set(xticks=positions, xticklabels=["M1 last tilt", "M2 last tilt", "M3 last tilt", "C8 completion"], ylabel="Cumulative image/action turns", title="Four-mirror relay laboratory alignment")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(lab_out / f"mirror_alignment_iterations.{suffix}", dpi=200)
    plt.close(figure)



def validate_iteration_table(metrics_path, summary_path, group_field, value_field, mean_field, std_field):
    metrics = read_csv(metrics_path)
    summary = read_csv(summary_path)
    for row in summary:
        selected = [
            number(item[value_field])
            for item in metrics
            if item[group_field] == row[group_field] and item["iteration"] == row["iteration"]
        ]
        if not np.isclose(np.mean(selected), number(row[mean_field]), rtol=0, atol=1e-9):
            raise ValueError(f"Mean mismatch in {summary_path}: {row}")
        if not np.isclose(np.std(selected, ddof=1), number(row[std_field]), rtol=0, atol=1e-9):
            raise ValueError(f"Standard deviation mismatch in {summary_path}: {row}")


def validate_relay_tables(data_root: Path) -> None:
    for relative, group_field in (("four_mirror_relay_lab/milestones.csv", "model"), ("four_mirror_relay_simulation_comparison/paired_iterations.csv", "environment")):
        rows = read_csv(data_root / relative)
        if len({row[group_field] for row in rows}) != 2:
            raise ValueError(f"Expected two relay groups in {relative}")
    evaluations = read_csv(data_root / "four_mirror_relay_baselines/evaluations.csv")
    summaries = read_csv(data_root / "four_mirror_relay_baselines/trial_summary.csv")
    for row in summaries:
        selected = [item for item in evaluations if item["method"] == row["method"] and item["trial"] == row["trial"]]
        if max(int(item["furthest_mirror"]) for item in selected) != int(row["furthest_mirror"]):
            raise ValueError(f"Relay furthest-mirror mismatch: {row}")
        if max(int(item["furthest_camera"]) for item in selected) != int(row["furthest_camera"]):
            raise ValueError(f"Relay furthest-camera mismatch: {row}")


def validate_manifest(artifact_root: Path) -> None:
    manifest = json.loads((artifact_root / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["sha256"].items():
        if not relative.startswith("data/"):
            continue
        actual = hashlib.sha256((artifact_root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Artifact checksum mismatch: {relative}")


def validate_numeric_artifacts(artifact_root: Path) -> dict[str, Any]:
    data = artifact_root / "data"
    validate_manifest(artifact_root)
    validate_iteration_table(data / "michelson_lab/iteration_metrics.csv", data / "michelson_lab/iteration_summary.csv", "model_key", "center_error", "center_error_mean", "center_error_std")
    validate_iteration_table(data / "michelson_simulation/iteration_metrics.csv", data / "michelson_simulation/iteration_summary.csv", "method_key", "center_error", "center_error_mean", "center_error_std")
    validate_iteration_table(data / "two_mirror_cavity_lab/iteration_metrics.csv", data / "two_mirror_cavity_lab/iteration_summary.csv", "model", "motor_distance_steps", "motor_distance_mean", "motor_distance_std")
    validate_iteration_table(data / "two_mirror_cavity_simulation/iteration_metrics.csv", data / "two_mirror_cavity_simulation/iteration_summary.csv", "method_key", "motor_step_error", "motor_step_error_mean", "motor_step_error_std")
    validate_relay_tables(data)
    return {"status": "passed", "validated_data_root": str(data), "checks": 7}


def reproduce(artifact_root: Path, output: Path, overwrite: bool) -> None:
    verification = validate_numeric_artifacts(artifact_root)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    data = artifact_root / "data"
    reproduce_michelson(data, output)
    reproduce_cavity(data, output)
    reproduce_relay(data, output)
    plot_combined(data, output / "combined_alignment_comparison")
    verification["figure_count"] = sum(1 for path in output.rglob("*") if path.is_file())
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("paper_artifacts"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    reproduce(args.artifact_root.resolve(), args.output.resolve(), args.overwrite)
    print(f"Figures reproduced in {args.output.resolve()}")


if __name__ == "__main__":
    main()
