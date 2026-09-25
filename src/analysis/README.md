# Run artifacts and plots

[Repository overview](../../README.md) · [Config catalog](../../configs/README.md)

These utilities consume saved experiment outputs. They do not run an optimizer,
but they may write plots or derived metrics into the supplied directories.
Check a script's `--help` before using historical or curated data.

## Read the artifacts first

Config-driven runs generally use `runs/<experiment_name>/run_<timestamp>/`.
Direct task commands can write to any `--out` directory. Method/repeat nesting
varies by family: use the root manifest rather than assuming a single layout.

| Artifact | What to inspect |
|---|---|
| `config_resolved.yaml` | Methods, model settings, geometry, and budget used for the batch |
| Root `run_manifest.json` | Trial paths and success/failure of execution |
| Trial `run_manifest.json` | Final state, metrics, counts, timing, and available usage |
| `global_metrics.json` | Per-turn records in workflows that produce it |
| `messages.json` / `messages.md` | LLM conversation and submitted observations |
| `llm_usage.json` | Recorded token usage, costs, and API request timings where available |
| `subprocess.log` | Captured task stdout/stderr for subprocess-based runners |

Successful execution does not imply successful optical alignment. Metrics also
differ between families: beam-center pixel error, motor-step distance, relay
milestones, and interference measurements are not interchangeable.

## Motorized relay

The matched relay simulation stores camera observations under each trial's
`images/` directory:

```text
gpt_5_6_sol/trial_01/
  images/000_C8.png
  run.json
```

The observation image is sent to the optimizer. Ground-truth station
reachability is recorded in the trial manifest but is never sent to the LLM.

Lab trials save full camera frames as `raw/step_N.png` and resized grayscale
observations as `step_N.png`. They do not generate simulated alignment scores.
Manual checkout writes `actions.jsonl` with no beam images.

## Plotting commands

Run from the repository root inside the project environment. Replace example
paths with your saved run folders.

Regenerate every checked-in quantitative paper figure from compact artifacts only:

```bash
python scripts/reproduce_paper_figures.py \
  --artifact-root paper_artifacts \
  --output /tmp/reproduced-paper-figures
```

The checked-in reproduction workflow intentionally generates only quantitative
plots used by the paper. Schematics, apparatus photographs, camera-image
trajectories, annotations, movies, and historical comparison dashboards are not
part of the reproducible artifact bundle.
