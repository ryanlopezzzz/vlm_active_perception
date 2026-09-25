# Motorized four-mirror relay

[Repository overview](../../README.md) · [Lab guide](LAB_README.md) · [Matched simulation](MATCHED_SIMULATION.md)

This directory documents the four-mirror relay workflow used in the paper. The
laboratory agent controls eight mirror-motor axes and actively selects among eight
camera stations. The matched simulator reuses the same prompts, scrambles, action
schema, and evaluation budgets.

## Laboratory workflow

```bash
python -m src.tasks.motorized_mirror_relay_staged \
  --config configs/motorized_mirror_relay/lab_runs.yaml \
  --scramble-plan configs/motorized_mirror_relay/scramble_plan_seed123.json \
  --start C8
```

The lab action protocol allows one action per turn: tilt one mirror, move the camera
to C1-C8, or declare completion at C8. Tilt commands are relative integer steps,
limited to +/-15,000 per action and +/-30,000 accumulated steps per axis after the
startup scramble.

## Matched simulation and baselines

```bash
python -m src.runners.run_motorized_mirror_relay_matched \
  --config configs/motorized_mirror_relay/matched_lab_simulation.yaml
```

Run one method only with `--method llm`, `--method random`, or `--method bayes`.
The tracked simulation-input archive contains the paired scrambles, prompts, and
model settings used for the matched trials.

The nominal motor calibration is 4096 steps per revolution and 0.027 radians of
mirror tilt per revolution, or approximately 6.59 microradians per step.
