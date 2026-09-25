# Paper artifacts

This directory contains compact numeric data, supporting calibration assets, and the quantitative
plots used in the paper for the Michelson interferometer, two-mirror cavity, four-mirror relay, and
combined comparison. Raw source-run directories are not required for figure reproduction.

- `data/`: numerical CSV tables and essential calibration/simulation assets.
- `figures/`: the six regenerated quantitative paper outputs.
- `manifest.json`: provenance notes and SHA-256 checksums.

Regenerate the quantitative paper figures using only this bundle:

```bash
mamba run -n llm-optics-alignment python scripts/reproduce_paper_figures.py \
  --artifact-root paper_artifacts \
  --output /tmp/reproduced-paper-figures
```

Verify that reproduction works in an isolated temporary copy with no `runs/`, `final_results/`,
or `development/` directories:

```bash
mamba run -n llm-optics-alignment python scripts/test_paper_artifact_isolation.py
```
