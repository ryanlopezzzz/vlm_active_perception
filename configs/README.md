# Experiment configurations

[Repository overview](../README.md) · [Source guide](../src/README.md)

Run commands below from the repository root, inside the environment from
`environment.yml`. A config's `experiment_type` chooses its runner;
`experiment_name` chooses the output subdirectory. The filename alone does not
determine which methods run.

## Simulation catalog

Unless noted otherwise, use:

```bash
python -m src.cli.main run --config <config-path>
```

The configurations below enable LLM calls by default. Inspect `methods`, `llm`,
or `llm_models` before execution; Bayesian/random methods do not require OpenRouter.

| Experiment | Config | What it controls |
|---|---|---|
| Michelson method comparison | [michelson_interferometer/compare_methods.yaml](michelson_interferometer/compare_methods.yaml) | Multiple LLM models, Bayesian optimization, and random search |
| Michelson BO pixel-objective rerun | [michelson_interferometer/bayes_pixel_objective.yaml](michelson_interferometer/bayes_pixel_objective.yaml) | Bayesian optimization only; optimizes the plotted 1920 x 1080 sum-of-beam-distances metric |
| Two-mirror cavity simulation | [two_mirror_cavity_simulation/compare_methods.yaml](two_mirror_cavity_simulation/compare_methods.yaml) | Lab-derived beam response and method comparison |
| Two-mirror cavity BO motor-distance rerun | [two_mirror_cavity_simulation/bayes_motor_distance.yaml](two_mirror_cavity_simulation/bayes_motor_distance.yaml) | Bayesian optimization only; minimizes the plotted physical motor-step distance using the same ten initial conditions |
| Four-mirror relay matched simulation | [motorized_mirror_relay/matched_lab_simulation.yaml](motorized_mirror_relay/matched_lab_simulation.yaml) | Matched LLM, random, and BO relay trials; use the dedicated matched runner |
### Resuming a comparison

The common CLI supports `--resume-run` for the **primary Michelson comparison**
and **two-mirror cavity simulation comparison**:

```bash
python -m src.cli.main run --config configs/michelson_interferometer/compare_methods.yaml --resume-run runs/michelson_interferometer_compare_methods/run_<timestamp>
```

Replace the final path with an existing run. Other experiment types reject this
option; in particular it is not implemented for the motorized relay.

## Hardware catalog

These workflows access physical equipment. A preflight or smoke run can include
motor motion; it is not merely configuration validation.

| Workflow | Config / guide | Entry point |
|---|---|---|
| Robot-placed relay camera and four motorized mirrors | [lab_runs.yaml](motorized_mirror_relay/lab_runs.yaml), [lab guide](motorized_mirror_relay/LAB_README.md) | `python -m src.tasks.motorized_mirror_relay_staged` |
| Real Michelson LLM alignment | [mi_real/llm_runs.yaml](mi_real/llm_runs.yaml) | Common CLI, `mi_real_llm_runs` |
| Michelson preflight | [mi_real/preflight.yaml](mi_real/preflight.yaml) | Common CLI, `mi_real_preflight` |
| Two-mirror cavity hardware smoke | [two_mirror_cavity/hardware_smoke.yaml](two_mirror_cavity/hardware_smoke.yaml) | Common CLI, `two_mirror_cavity_hardware_smoke` |
| Two-mirror cavity LLM alignment | [two_mirror_cavity/real_llm_runs.yaml](two_mirror_cavity/real_llm_runs.yaml) | Common CLI, `two_mirror_cavity_real_llm_runs` |

The relay lab guide documents a non-connecting configuration-validation snippet.
Checked-in hardware addresses and poses are lab-specific settings, not automatically
the correct values for a new installation.

## Editing a config

| Field | Meaning |
|---|---|
| `experiment_type` | Runner selector; choose an implemented workflow |
| `experiment_name` | Name under `output_root` |
| `output_root` | Usually `runs`; relative paths use the command's working directory |
| `repeats` | Number of trials/initial conditions |
| `iterations` | Per-trial optimization budget; initial/final evaluation counting varies by workflow |
| `base_seed` | Reproducible simulation initial conditions where supported |
| `max_parallel_jobs` | Simulation job concurrency; relay lab runs require 1 |
| `methods` | Supported method subset for comparison runners; not present in every schema |
| `llm` / `llm_models` | One model's settings or a comparison's model list |

Copy an existing config from the same family rather than mixing fields across
families. Most schemas live in [src/io/schemas.py](../src/io/schemas.py); motorized
relay configuration is handled by its simulator, runner, and lab adapter.

The shared LLM client's usual reasoning behavior is a fixed reasoning-token
budget when `reasoning_max_tokens` is an integer, or the configured `effort` when
it is null. `effort: none` suppresses the reasoning payload; there is also a
model-specific branch in [llm_agent.py](../src/agents/llm_agent.py). Treat provider
acceptance of model names and settings separately from config syntax validation.

Keep OpenRouter credentials out of YAML. Non-mocked clients normally read the
ignored `openrouter_api.txt` file in the working directory.
