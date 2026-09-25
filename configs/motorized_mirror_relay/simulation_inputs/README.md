# Compact inputs for lab/simulation comparison

This archive contains the 10 selected GPT-5.6 Sol and 10 selected Gemini 3.5 Flash trials, including GPT trial 2 (user-confirmed unsuccessful despite the model declaring done). No camera images are included. Replaced attempts are not included.

## Files

- `scramble_plan_seed123.json`: ten exact eight-axis relative scramble commands, shared by both models.
- `trials.json`: sampling definition, per-trial commands, nominal angle conversions, original motor records, baseline counters, control limits, source paths, and configuration references.
- `configs/`: deduplicated original per-run configurations, including all C1-C8 placement coordinates, controller direction signs, nominal motor calibration, and LLM settings.
- The canonical paper prompt is `src/prompts/four_mirror_relay.txt` and is shared by lab and simulation.

The export was checked against every archived startup and scramble motion record. Regenerating the ten draws with Python `random.Random(123)` also matched exactly. The seed applies to the entire sequence, not a new generator for each trial.

## Sampling and units

Each of eight axes was sampled independently and uniformly from the integers [-15000, 15000], inclusive. Axis order is M1 in-plane, M1 out-of-plane, then the same for M2-M4. Nominal mirror tilt in radians is `steps * 0.027 / 4096`; the bound corresponds to about 0.09888 radians (5.665 degrees). These are inferred motor-to-mirror changes, not measured angles. Reflected beam deflection is not the same quantity.

LLM actions were relative steps, limited to +/-15000 per action and +/-30000 cumulative steps relative to the post-scramble origin. The archived hardware config's older `command_limit_steps: 8192` was not enforced by the staged lab runner. Use the recorded runtime limits in trials.json.

## What can and cannot be reproduced

These scrambles are relative to each trial's pre-scramble physical state. These historical batches did NOT perform the later-added motor baseline return after every trial: consecutive trials generally started from the previous LLM alignment, and restarted batches could follow manual realignment. Tracked counters are preserved but do not measure absolute optical alignment, stalls, backlash, or manual mirror changes. Replaying the scramble from an ideal simulated relay reproduces the commanded perturbation, not the exact physical initial state.

C1-C8 `x_position_f` and `y_position_f` are robot placement targets in millimetres. `added_angle` is a robot placement orientation parameter in degrees. They are NOT sensor-center XYZ coordinates or camera-normal yaw angles. The pickup/drop code uses Euler [180, 0, 90 + added_angle], converts it to the arm representation, and uses a depth-derived placement height. A calibrated robot-to-optical transform, camera/tool offset, sensor height and normal, and actual mirror centers are needed to place optical planes accurately. The station description strings describe intended optical paths, not robot-coordinate axes.

The approximate box size was described as 8 by 11 inches; exact mirror centers were not recorded here. The motorized simulation uses that 8-by-11-inch nominal geometry, which should not be treated as a precise measurement of the lab mirror centers. Sensor physical size, beam radius, placement error distribution, and motor stall behavior are not established by these configs.

The staged lab capture used scanner-style default capture settings, not the requested width/exposure overrides in the archived camera config. First-capture metadata is preserved per trial. Actual images and large logs remain in the original archive.

## Connecting to the existing simulation

`src.sim.motorized_mirror_relay.MotorizedMirrorRelay` accepts explicit `hidden_offsets`; use the saved commands rather than its NumPy random sampler (the same numeric seed will NOT produce the Python lab sequence). Set `hidden_offset_max_steps=15000` and `command_limit_steps=30000` for the corresponding bounds. Accumulate LLM deltas before passing absolute visible commands to this simulator.

Axis names currently have different physical meanings: the simulator's first/in-plane axis rotates about world Z, whereas the lab prompt maps in-plane to vertical image motion and out-of-plane to horizontal. Verify/remap axes and signs before comparison; do not directly interpret matching names as matching physical rotations. Likewise, the simulation generates idealized C1-C8 poses rather than reading these robot station targets. This archive is an input/provenance dataset, not a claim that the current simulator already matches the lab geometry.

Example loading (no hardware or API calls):

```python
import json
from pathlib import Path
root = Path('configs/motorized_mirror_relay/simulation_inputs')
data = json.loads((root / 'trials.json').read_text())
trial = next(t for t in data['trials']
             if t['model'] == 'openai/gpt-5.6-sol' and t['trial'] == 1)
scramble = trial['scramble_delta_steps']
config_path = root / trial['config_file']
```
