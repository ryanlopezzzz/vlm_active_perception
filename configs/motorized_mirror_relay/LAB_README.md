# Four-mirror relay laboratory workflow

The paper relay uses `src.tasks.motorized_mirror_relay_staged`. It controls four
motorized mirrors through eight axes and uses the robot arm to move a beam camera
among calibrated stations C1-C8.

## Configuration

Edit `lab_runs.yaml` for the target laboratory:

- `hardware.skeleton_module` and calibration path
- motor-controller addresses, motor numbers, and direction signs
- camera-station robot coordinates and placement angles
- camera index and capture settings
- OpenRouter model settings and turn budget

The stereo calibration file referenced by the default config,
`stereo_calibration2.npz`, is laboratory-specific and is not included in the
repository. Place the calibrated file in the working directory or update
`hardware.calibration_path` before opening a real session.

Set the robot address in the environment before opening a real session:

```bash
export ROBOT_ARM_IP=<robot-address>
```

Validate the YAML without connecting to hardware:

```bash
python - <<'PY'
import yaml
from src.hardware.motorized_mirror_relay import validate_lab_config
validate_lab_config(yaml.safe_load(open('configs/motorized_mirror_relay/lab_runs.yaml')))
print('configuration valid')
PY
```

This validation checks structure and limits only. It cannot verify robot poses,
wiring, motor directions, optical calibration, or collision clearance.

## Run the paper protocol

```bash
python -m src.tasks.motorized_mirror_relay_staged \
  --config configs/motorized_mirror_relay/lab_runs.yaml \
  --scramble-plan configs/motorized_mirror_relay/scramble_plan_seed123.json \
  --start C8
```

Useful options include `--iterations`, `--start-trial`, `--snapshot-c8-before-after`,
`--snapshot-all-before-scramble`, `--snapshot-all-after-scramble`, and
`--snapshot-all-after-alignment`.

One startup confirmation authorizes the complete autonomous sequence. Close other
robot and camera controllers before confirming. The hardware skeleton connects to
and enables the robot when the real session is opened.

## Action and safety model

Each accepted turn performs exactly one action:

- `tilt`: move one mirror using relative vertical/horizontal motor steps
- `measure`: move to or recapture at one camera station
- `done`: stop while observing C8

Each tilt component is limited to +/-15,000 steps. Cumulative commanded motion after
the scramble is limited to +/-30,000 steps per axis. Invalid actions are rejected
without movement or a new image. Command history is not encoder feedback; backlash,
slip, stalls, and mechanical binding remain possible.

## Outputs

Runs are written under `runs/motorized_mirror_relay_staged/`. Each batch records the
resolved configuration, scramble plan, exact prompt, camera images, conversation,
usage information, executed actions, timing, and final status. `llm_declared_done`
is the model's judgment and is not independent optical validation.
