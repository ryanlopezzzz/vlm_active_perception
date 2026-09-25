# Tests and offline checks

[Repository overview](../README.md) · [Source guide](../src/README.md)

Tests use Python's `unittest`. Run commands from the repository root with the
project environment activated, or prefix them with
`mamba run -n llm-optics-alignment`.

## Focused motorized-relay suite

```bash
python -m unittest tests.test_motorized_mirror_relay tests.test_motorized_mirror_relay_lab tests.test_motorized_mirror_relay_staged tests.test_motorized_mirror_relay_matched tests.test_motorized_mirror_relay_random
```

| Module | Coverage |
|---|---|
| [test_motorized_mirror_relay.py](test_motorized_mirror_relay.py) | Exact geometry, disks, motor calibration, station clearance/orientation, and camera semantics |
| [test_motorized_mirror_relay_lab.py](test_motorized_mirror_relay_lab.py) | Mocked skeleton initialization, tracked motor deltas, persistent devices, and image preparation |
| [test_motorized_mirror_relay_staged.py](test_motorized_mirror_relay_staged.py) | Paper lab action protocol and staged batch behavior |
| [test_motorized_mirror_relay_matched.py](test_motorized_mirror_relay_matched.py) | Matched simulation protocol and paired trials |
| [test_motorized_mirror_relay_random.py](test_motorized_mirror_relay_random.py) | Uniform random relay baseline |

These tests inject fake LLM and hardware objects. Motor transport uses a dry-run
rig; the real camera skeleton is not imported. They can exercise integration
logic without robot connections or paid API calls, but cannot establish that
the physical station coordinates, calibration, or motor wiring are correct.

## Other useful suites

```bash
python -m unittest tests.test_michelson_interferometer_simulation tests.test_michelson_interferometer_compare_methods
python -m unittest tests.test_two_mirror_cavity_simulation tests.test_two_mirror_cavity_simulation_compare_methods
python -m unittest tests.test_llm_usage
```

For all discoverable tests:

```bash
python -m unittest discover -s tests -t .
```

Tests with `real` or `hardware` in their names can still use fakes—inspect their
fixtures rather than treating a test name as permission to access lab equipment.

## Syntax and CLI checks

```bash
python -m compileall -q src
python -m src.cli.main --help
python -m src.tasks.motorized_mirror_relay_staged --help
```

`compileall` compiles source without executing its top-level code. Do not replace
this with a blanket import of every Python file: the root camera skeleton
deliberately connects to the robot on import.

Relay lab config checking is also non-connecting:

```bash
python - <<'PY'
import yaml
from src.hardware.motorized_mirror_relay import validate_lab_config
validate_lab_config(yaml.safe_load(open('configs/motorized_mirror_relay/lab_runs.yaml')))
PY
```

The checked-in placeholders are expected to fail until filled. This is a config
check, not a robot/camera/motor functional test.

## Writing a workflow test

Use temporary directories for state and artifacts. Inject a fake agent instead
of constructing a real-model LLM client, and inject device/session factories
instead of importing the lab skeleton. For motion changes, verify the requested
delta and resulting tracked state, including cancelled actions. For camera
changes, verify placement counts independently of image counts. Preserve the
ordering between an observation and the command proposed from that observation.

For headless plot tests, `MPLBACKEND=Agg` selects a noninteractive backend. If a
machine's Matplotlib cache is unwritable, set `MPLCONFIGDIR` to a writable local
directory before launching Python.
