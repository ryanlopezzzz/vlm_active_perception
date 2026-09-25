"""Verify paper-figure reproduction without raw source directories."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def files_below(root: Path, *, exclude: set[str] | None = None) -> list[str]:
    excluded = exclude or set()
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in excluded
    )


def main() -> None:
    reference = REPO_ROOT / "paper_artifacts"
    with tempfile.TemporaryDirectory(prefix="paper-artifact-isolation-") as directory:
        isolated = Path(directory)
        (isolated / "scripts").mkdir()
        (isolated / "src/analysis").mkdir(parents=True)
        shutil.copy2(REPO_ROOT / "scripts/reproduce_paper_figures.py", isolated / "scripts")
        shutil.copy2(
            REPO_ROOT / "src/analysis/plot_combined_alignment_comparison.py",
            isolated / "src/analysis",
        )
        shutil.copytree(reference, isolated / "paper_artifacts")
        output = isolated / "reproduced"
        subprocess.run(
            [
                sys.executable,
                str(isolated / "scripts/reproduce_paper_figures.py"),
                "--artifact-root",
                str(isolated / "paper_artifacts"),
                "--output",
                str(output),
            ],
            cwd=isolated,
            check=True,
        )
        raw_present = [name for name in ("runs", "final_results", "development") if (isolated / name).exists()]
        if raw_present:
            raise AssertionError(f"Raw directories unexpectedly present: {raw_present}")
        expected = files_below(reference / "figures")
        actual = files_below(output, exclude={"verification.json"})
        if expected != actual:
            raise AssertionError("Regenerated figure filenames differ from paper_artifacts/figures")
        dimension_mismatches = []
        for relative in expected:
            if not relative.endswith(".png"):
                continue
            with Image.open(reference / "figures" / relative) as left, Image.open(output / relative) as right:
                if left.size != right.size:
                    dimension_mismatches.append((relative, left.size, right.size))
        if dimension_mismatches:
            raise AssertionError(f"PNG dimension mismatches: {dimension_mismatches}")
        verification = json.loads((output / "verification.json").read_text(encoding="utf-8"))
        if verification.get("status") != "passed":
            raise AssertionError(f"Numerical verification failed: {verification}")
        print(
            json.dumps(
                {
                    "status": "passed",
                    "raw_directories_present": raw_present,
                    "figure_count": len(expected),
                    "png_dimension_mismatches": len(dimension_mismatches),
                    "numeric_checks": verification["checks"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
