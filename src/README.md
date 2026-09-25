# Source guide

[Repository overview](../README.md) · [Config catalog](../configs/README.md) · [Tests](../tests/README.md)

The repository separates experiment orchestration from the optical model and
hardware access. Most entry points use `python -m ...` from the repository root.

## Where to look

| Directory | Responsibility |
|---|---|
| [cli/](cli/) | Config-driven command dispatch in `main.py` |
| [runners/](runners/) | Create run directories, schedule trials, save root manifests |
| [tasks/](tasks/) | Execute an optimization loop or a hardware procedure |
| [sim/](sim/) | Optical state, propagation/rendering, and simulation measurements |
| [hardware/](hardware/) | Cameras, motor controllers, and robot-session adapters |
| [agents/](agents/) | OpenRouter LLM transport and conversation logging |
| [prompts/](prompts/) | LLM instructions and response templates |
| [io/](io/) | Config schemas, seeds, artifact paths, and logging helpers |
| [analysis/](analysis/README.md) | Paper plots and run-artifact inspection |

Many older families share the large [run_experiment.py](runners/run_experiment.py).
Newer workflows may use dedicated runner modules. Follow the dispatch in
[cli/main.py](cli/main.py) to find the actual implementation for a config.

## Motorized relay walkthrough

The archived lab-matched simulation uses
`runners/run_motorized_mirror_relay_matched.py` and
`tasks/motorized_mirror_relay_matched.py`, sharing action validation with the lab
through `tasks/motorized_mirror_relay_protocol.py`. See the
[matched simulation guide](../configs/motorized_mirror_relay/MATCHED_SIMULATION.md)
for offline preflight and paid-run commands.

The paper relay uses one laboratory task and one matched-simulation runner:

| Part | File | Key behavior |
|---|---|---|
| Optical simulation | [sim/motorized_mirror_relay.py](sim/motorized_mirror_relay.py) | Eight motor axes, mirror intersections, camera stations, and rendered observations |
| Lab action protocol | [tasks/motorized_mirror_relay_staged.py](tasks/motorized_mirror_relay_staged.py) | Relative mirror tilts, active camera selection, and autonomous staged trials |
| Shared validation | [tasks/motorized_mirror_relay_protocol.py](tasks/motorized_mirror_relay_protocol.py) | One-action-per-turn schema and movement limits |
| Matched LLM simulation | [tasks/motorized_mirror_relay_matched.py](tasks/motorized_mirror_relay_matched.py) | Replays the paper prompt/interface on the simulator |
| Numerical baselines | [tasks/motorized_mirror_relay_random.py](tasks/motorized_mirror_relay_random.py), [tasks/motorized_mirror_relay_bayes.py](tasks/motorized_mirror_relay_bayes.py) | Random and sparse final-camera BO baselines |
| Matched orchestration | [runners/run_motorized_mirror_relay_matched.py](runners/run_motorized_mirror_relay_matched.py) | Paired scrambles, model settings, and trial execution |

### Command and observation boundaries

Each relay turn begins with one camera image and ends with exactly one accepted
action: tilt one mirror, move or recapture the camera at C1-C8, or declare done
while observing C8. Tilt values are relative integer motor steps. Shared protocol
validation enforces +/-15,000 steps per action and +/-30,000 accumulated steps per
axis after the startup scramble. Invalid actions do not move hardware or consume a
new observation.

The matched simulation records ray paths and reachability only as offline
diagnostics; these values are never supplied to the LLM. The laboratory workflow
records commanded positions rather than encoder measurements and includes physical
backlash, slip, stalls, and camera-placement repeatability.

### Motor reference and hardware imports

The lab session converts absolute offsets from the batch baseline into relative
movements through [StepperMotorRig](hardware/motors.py). That rig maintains a
persistent position file and applies direction signs, displacement limits, and
the existing backlash routine. Its positions are command history, not encoder
measurements. Consult the [lab guide](../configs/motorized_mirror_relay/LAB_README.md)
before changing zero/reference behavior.

The root-level [camera skeleton](../pick_and_place_and_capture_skeleton.py) has an
intentional robot connection on import. The adapter loads it only when opening
a real session. Do not import it to inspect constants or validate a configuration.
Offline tests inject a fake skeleton and a dry-run motor rig instead.

## Working on the code

For a change to the relay, update the matching config guide and prompt alongside
the implementation. Keep the simulation and lab's different observation
capabilities explicit; real hardware cannot provide a simulated ray trace.

For a new config-driven workflow, add its task/runner and validation, then wire
the `experiment_type` into the common CLI.

Use injected devices/agents for workflow tests. Do not substitute a real LLM model
for a fake merely to check plumbing: that makes paid API calls. The
[testing guide](../tests/README.md) lists focused commands and existing examples.
