# Lab-matched relay simulation

This runner pairs the ten archived GPT-5.6 SOL trials with the ten Gemini 3.5
Flash trials, using their exact eight-axis scrambles, archived model settings,
and lab action protocol. Every trial begins from true optical alignment before
its scramble; previous trials' residual errors are not carried forward.
All ten trials are retained, including the lab's unsuccessful GPT trial 2.

## Run safely

From the repository root in the `llm-optics-alignment` environment:

```bash
python -m src.runners.run_motorized_mirror_relay_matched --config configs/motorized_mirror_relay/matched_lab_simulation.yaml --check-config
python -m src.runners.run_motorized_mirror_relay_matched --config configs/motorized_mirror_relay/matched_lab_simulation.yaml --dry-run
```

Neither command uses hardware or calls an LLM. The check only prints the plan.
The dry run saves one initial C8 image per trial and all input/source provenance
in a new `runs/motorized_mirror_relay_matched_lab_dry_run/` batch directory.
Matching trial numbers produce identical initial images for both models.

The following command **makes paid OpenRouter calls** for 20 LLM trials, with up to
100 turns each and up to three request attempts per turn, and also runs ten
random-baseline trials:

```bash
python -m src.cli.main run --config configs/motorized_mirror_relay/matched_lab_simulation.yaml
```

## Random baseline

To run only the random baseline, with **no API calls**:

```bash
python -m src.runners.run_motorized_mirror_relay_matched --config configs/motorized_mirror_relay/matched_lab_simulation.yaml --method random
```

`methods: [llm, random]` selects both by default; `--method llm` or
`--method random` overrides this for the dedicated runner. Each random trial
uses the corresponding archived scramble and 100 independent guesses. Each
guess samples all eight integer absolute offsets uniformly from [-15000, 15000],
inclusive, relative to the post-scramble origin. These are not accumulating
random deltas or physical tilts sampled about true alignment. The same axis
mapping applies to the scramble and guesses. There is no feedback-driven
selection or early stopping. `random_seed` controls guesses independently of
the simulation seed used for the paired trial.

Each iteration evaluates nominal C1-C8 separately, without placing cameras.
`furthest_camera_station` / `furthest_camera_index`
report the most downstream sensor aperture hit by the central ray after the
correct mirror sequence; null / 0 means no station was hit. A ray can hit a later
station without hitting an earlier aperture. `furthest_plane_station` separately
reports intersection with the infinite station plane, even when the sensor is
missed. Gaussian tails do not count as central-ray hits. Random trials record
current and best-so-far camera reach in `run.json` and `reachability.csv`.
The zero-command initial state is diagnostic only, not one of the random guesses.
Dry runs instead render that initial state without making any guesses.

One random iteration replaces all eight offsets and evaluates eight virtual
stations. This is not the same action/sensing budget as one LLM turn, which
changes one mirror or one camera station. Random guesses can differ by up to
30000 steps between iterations; the lab's per-action delta cap is not applied
to these independently sampled absolute configurations. The random method is
a blind configuration-search baseline, not a random implementation of the lab
action policy. Offline reachability is also logged for LLM observations but
never supplied to the agent.

The API key is read through the existing LLM client. No key is copied into run
artifacts. Model settings come from each archived config: both models use high
effort, 12,000 maximum output tokens per request, and no explicit reasoning-token
budget. `trial_ids` can select a smaller pilot, and `max_parallel_jobs` controls
concurrency. Failed trials are recorded without aborting the other trials; a
batch with execution failures raises after all scheduled trials finish. A turn
limit is a completed execution, not successful alignment. There is no automatic
retry/resume of whole trials, and each invocation creates a new batch directory.

## Matched interface

- Start at C8; the agent sees zero cumulative commanded offsets after scrambling.
- One image and one accepted action per turn: tilt one mirror, measure one
  station, or declare done. Tilts and measurements cannot be combined.
- Tilts are relative, limited to +/-15,000 per action and +/-30,000 accumulated
  steps per axis. Shared lab/simulation validation returns rejection feedback
  without moving hardware or taking an extra image. Retries use the same state.
- `vertical` controls simulated out-of-plane tilt (vertical spot displacement);
  `horizontal` controls simulated in-plane tilt (horizontal displacement).
  Archived scrambles are mapped through exactly the same axis swap. Sign choices
  follow the simulator; the prompt asks the agent to learn response signs.
- The archived prompts are identical after standardizing their image-format
  sentence. All simulated observations are full-field 640 x 480 RGB PNGs.
  The rest of the lab prompt, including its description of real hardware, is
  preserved. Only station, cumulative commands, remaining turns, and images are
  sent as observations; actual offsets and ray diagnostics remain offline.
- `done` is accepted only at C8 and causes no movement or additional exposure.
  A last-turn tilt or camera move is executed, but has no extra exposure; the
  manifest marks whether the final state was measured. A `done` declaration does
  not certify a visible beam or quantitative alignment.

## Geometry and explicit limitations

The configuration uses nominal 8 x 11 inch mirror-center spacing, 1-inch disk
mirrors, and the existing 4096 steps / 0.027 radians nominal tilt calibration.
Mirror centers and camera stations use the exact nominal geometry, ensuring that
zero actual tilt is truly aligned. Moving between stations changes only the
selected nominal camera pose; keeping a station preserves that pose.

The following are modeling assumptions, not measured lab calibration: input
distance 120 mm, ideal station geometry (42.5 mm from the adjacent mirror and
C8 at 82.5 mm after M4), 6.4 x 4.8 mm field of view, and 0.8 mm Gaussian beam
radius. Robot station targets cannot be used directly as optical sensor poses.
The simulator traces a central ray and uses its existing normalized tangent-plane
normal model, not exact finite-angle motor rotations. It does not reproduce
speckles, diffraction, saturation/background variation, finite-beam mirror-edge
clipping, mechanical stalls, or backlash. The comparison therefore matches
perturbations and agent interaction, not exact historical lab states or images.

## Artifacts and evaluation

Each batch saves the resolved config, original archived configs/prompts/trials,
input hashes, effective prompts, source snapshots/hashes, Git revision and dirty
status. Each `model/trial_NN/` has a `trial_spec.json`, `initial_conditions.json`,
`run.json`, images, and (for live runs) the conversation and usage records.

`run.json` separates model-declared completion from `final_exit_beam_visible`,
which checks central-ray visibility through M1-M4 at nominal C8. It also records
C8 spot offset, the full ray path, and noise-free output diagnostics at 100 and
180 mm after M4. These diagnostics take no camera exposures and consume no
placement random numbers. C8 visibility alone does not establish centered
position or correct output direction; no arbitrary success tolerance is imposed.

Offline regression tests:

```bash
python -m unittest tests.test_motorized_mirror_relay_matched tests.test_motorized_mirror_relay tests.test_motorized_mirror_relay_staged
python -m unittest tests.test_motorized_mirror_relay_random
```
