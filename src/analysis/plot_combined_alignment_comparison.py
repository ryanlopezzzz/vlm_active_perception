"""Plot the paper's combined comparison from compact paper artifacts only."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SERIES = (
    {"key": "gpt_simulation", "label": "GPT-5.6 SOL (sim.)", "color": "#1769aa", "linestyle": "-"},
    {"key": "gpt_laboratory", "label": "GPT-5.6 SOL (lab.)", "color": "#1769aa", "linestyle": "--"},
    {"key": "bayes_simulation", "label": "BO (sim.)", "color": "#8c564b", "linestyle": "-"},
    {"key": "random_simulation", "label": "Random search (sim.)", "color": "#636363", "linestyle": "-"},
)


def load_curves(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for spec in SERIES:
        selected = [row for row in rows if row["series"] == spec["key"]]
        repeats = sorted({int(row["repeat_index"]) for row in selected})
        evaluations = sorted({int(row["evaluation"]) for row in selected})
        values = np.empty((len(repeats), len(evaluations)), dtype=float)
        for row in selected:
            values[int(row["repeat_index"]), int(row["evaluation"])] = float(row["best_error"])
        output[spec["key"]] = values
    return output


def load_relay_statistics(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    statistics = {}
    for environment in ("Lab", "Simulation"):
        values = np.asarray(
            [
                [float(row[key]) for key in ("M1", "M2", "M3", "completion")]
                for row in rows
                if row["environment"] == environment
            ],
            dtype=float,
        )
        statistics[environment] = {
            "mean": values.mean(axis=0),
            "sample_sd": values.std(axis=0, ddof=1),
        }
    return statistics


def plot_continuous(axis, curves, *, title, ylabel, threshold=None):
    for spec in SERIES:
        values = curves[spec["key"]]
        mean = values.mean(axis=0)
        sample_std = values.std(axis=0, ddof=1)
        evaluations = np.arange(values.shape[1])
        axis.plot(evaluations, mean, color=spec["color"], linestyle=spec["linestyle"], linewidth=2.15)
        axis.fill_between(
            evaluations,
            np.maximum(0.0, mean - sample_std),
            mean + sample_std,
            color=spec["color"],
            alpha=0.075,
            linewidth=0,
        )
    if threshold is not None:
        axis.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.4)
    axis.set_title(title, fontsize=16)
    axis.set_xlabel("Evaluations")
    axis.set_ylabel(ylabel)
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.grid(True, alpha=0.25)


def plot_relay(axis, statistics):
    positions = np.arange(4)
    width = 0.36
    maximum = 0.0
    styles = {
        "Lab": dict(facecolor="#1769aa", edgecolor="#1769aa", linewidth=1.0, linestyle="-", hatch=None),
        "Simulation": dict(facecolor="#d7e8f5", edgecolor="#1769aa", linewidth=1.8, linestyle="--", hatch="///"),
    }
    for index, environment in enumerate(("Lab", "Simulation")):
        means = np.asarray(statistics[environment]["mean"], dtype=float)
        deviations = np.asarray(statistics[environment]["sample_sd"], dtype=float)
        offsets = positions + (index - 0.5) * width
        axis.bar(offsets, means, width, yerr=deviations, capsize=5, **styles[environment])
        maximum = max(maximum, float(np.max(means + deviations)))
        for x_position, mean, deviation in zip(offsets, means, deviations):
            axis.text(x_position, mean + deviation + 1.2, f"{mean:.1f}", ha="center", va="bottom", fontsize=11)
    axis.set_title("Four-mirror relay", fontsize=16)
    axis.set_ylabel("Cumulative image/action turns")
    axis.set_xticks(positions, ("M1 last tilt", "M2 last tilt", "M3 last tilt", "C8 completion"))
    axis.set_ylim(0, maximum + 9)
    axis.grid(True, axis="y", alpha=0.25)


def plot(data_root: Path, output_dir: Path) -> None:
    michelson = load_curves(data_root / "combined_alignment_comparison/michelson_curves.csv")
    cavity = load_curves(data_root / "combined_alignment_comparison/two_mirror_cavity_curves.csv")
    relay = load_relay_statistics(
        data_root / "four_mirror_relay_simulation_comparison/paired_iterations.csv"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(12.0, 9.0))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(1.0, 1.12),
        left=0.085,
        right=0.98,
        top=0.96,
        bottom=0.155,
        hspace=0.38,
        wspace=0.28,
    )
    plot_continuous(
        figure.add_subplot(grid[0, 0]),
        michelson,
        title="Michelson interferometer",
        ylabel="Best-so-far beam center error (px)",
    )
    plot_continuous(
        figure.add_subplot(grid[0, 1]),
        cavity,
        title="Two-mirror cavity",
        ylabel="Best-so-far distance (motor steps)",
        threshold=500.0,
    )
    plot_relay(figure.add_subplot(grid[1, :]), relay)
    handles = [
        Line2D(
            [0],
            [0],
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=2.4,
            marker="s" if spec["key"] == "bayes_simulation" else "o" if spec["key"] == "random_simulation" else None,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=spec["label"],
        )
        for spec in SERIES
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        fontsize=11.5,
        columnspacing=1.5,
        handlelength=3.0,
    )
    figure.savefig(output_dir / "01_combined_alignment_comparison.pdf")
    figure.savefig(output_dir / "01_combined_alignment_comparison_preview.png", dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(args.data_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
