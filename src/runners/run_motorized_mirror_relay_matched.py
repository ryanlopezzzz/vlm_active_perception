"""Run the archived lab relay trials against an ideal, protocol-matched simulator."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from src.io.artifacts import make_experiment_root, write_json
from src.sim.motorized_mirror_relay import AXES, RelayConfig
from src.tasks.motorized_mirror_relay_matched import run_trial
from src.tasks.motorized_mirror_relay_bayes import run_bayes_trial
from src.tasks.motorized_mirror_relay_random import run_random_trial


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = REPO_ROOT / 'src/prompts/four_mirror_relay.txt'


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def prepare_experiment(config_path, *, method=None):
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    methods = [method] if method is not None else config.get('methods', ['llm'])
    if not methods or len(set(methods)) != len(methods) or set(methods) - {'llm', 'random', 'bayes'}:
        raise ValueError('methods must select llm, random, and/or bayes')
    config['methods'] = methods
    archive = (REPO_ROOT / config['lab_inputs']).resolve()
    trials_bytes = (archive / 'trials.json').read_bytes()
    trials = json.loads(trials_bytes)['trials']
    simulation = RelayConfig(**config['simulation'])
    if (simulation.frame_width_px, simulation.frame_height_px) != (640, 480):
        raise ValueError('The standardized prompt requires 640 x 480 images')
    if (simulation.hidden_offset_max_steps, simulation.command_limit_steps) != (15000, 30000):
        raise ValueError('Matched simulator bounds must be 15000 hidden / 30000 commanded steps')
    if (simulation.steps_per_revolution, simulation.mirror_radians_per_revolution) != (4096, 0.027):
        raise ValueError('Matched motor calibration must be 4096 steps / 0.027 radians')
    parallel = config.get('max_parallel_jobs', 1)
    if type(parallel) is not int or parallel < 1:
        raise ValueError('max_parallel_jobs must be a positive integer')
    seed = config.get('base_seed', 123)
    if type(seed) is not int or seed < 0:
        raise ValueError('base_seed must be a nonnegative integer')
    trial_ids = config.get('trial_ids', list(range(1, 11)))
    if (not trial_ids or any(type(number) is not int or not 1 <= number <= 10 for number in trial_ids)
            or len(set(trial_ids)) != len(trial_ids)):
        raise ValueError('trial_ids must be unique integers from 1 through 10')
    models = config['models']
    if not models or len(set(models.values())) != len(models):
        raise ValueError('Select distinct models')
    prompt = PROMPT_PATH.read_text(encoding='utf-8')
    specs, inputs, paired_scrambles = [], {
        'trials.json': trials_bytes,
        'system_prompt.txt': prompt.encode('utf-8'),
    }, {}
    for key, model in models.items():
        if not key.isidentifier():
            raise ValueError('Model output keys must be identifiers')
        for number in trial_ids:
            matching = [trial for trial in trials if trial['model'] == model and trial['trial'] == number]
            if len(matching) != 1:
                raise ValueError(f'Expected one archived trial for {model}, trial {number}')
            trial = matching[0]
            relative = trial['config_file']
            inputs[relative] = (archive / relative).read_bytes()
            archived_config = yaml.safe_load(inputs[trial['config_file']])
            settings = archived_config['llm']
            if settings['model'] != model:
                raise ValueError('Archived config model mismatch')
            iterations = archived_config['iterations']
            if type(iterations) is not int or iterations <= 0:
                raise ValueError('Archived iterations must be a positive integer')
            if type(settings['max_tokens']) is not int or settings['max_tokens'] <= 0:
                raise ValueError('Archived max_tokens must be a positive integer')
            if (trial['protocol'] != 'active_perception_delta_v3' or trial['startup']['start'] != 'C8'
                    or trial['per_move_limit'] != 15000 or trial['cumulative_limit'] != 30000):
                raise ValueError('Unsupported archived control protocol')
            scramble = trial['scramble_delta_steps']
            if (set(scramble) != set(AXES) or
                    any(type(value) is not int or abs(value) > 15000 for value in scramble.values())):
                raise ValueError('Invalid eight-axis archived scramble')
            if paired_scrambles.setdefault(number, scramble) != scramble:
                raise ValueError(f'Models have different scrambles for trial {number}')
            specs.append(dict(method='llm', model_key=key, model=model, trial=number, iterations=iterations,
                              llm=dict(settings), scramble_delta_steps=scramble,
                              simulation_seed=seed+number-1, prompt=prompt,
                              prompt_sha256=sha256(prompt.encode('utf-8')),
                              lab_source=trial['source'], lab_config_file=trial['config_file'],
                              lab_known_unsuccessful=trial['known_unsuccessful']))
    random_specs = []
    if 'random' in methods:
        random_seed = config.get('random_seed', 456)
        if type(random_seed) is not int or random_seed < 0:
            raise ValueError('random_seed must be a nonnegative integer')
        for number in trial_ids:
            paired = [spec for spec in specs if spec['trial'] == number]
            if len({spec['iterations'] for spec in paired}) != 1:
                raise ValueError('Random baseline requires matching archived turn budgets')
            reference = paired[0]
            random_specs.append(dict(method='random', model='random', model_key='random', trial=number,
                                     iterations=reference['iterations'],
                                     scramble_delta_steps=reference['scramble_delta_steps'],
                                     simulation_seed=reference['simulation_seed'], sampling_seed=random_seed+number-1))
    bayes_specs = []
    if 'bayes' in methods:
        bayes_seed = config.get('bayes_seed', 789)
        if type(bayes_seed) is not int or bayes_seed < 0:
            raise ValueError('bayes_seed must be a nonnegative integer')
        for number in trial_ids:
            paired = [spec for spec in specs if spec['trial'] == number]
            if len({spec['iterations'] for spec in paired}) != 1:
                raise ValueError('Bayesian baseline requires matching archived turn budgets')
            reference = paired[0]
            bayes_specs.append(dict(method='bayes', model='bayes', model_key='bayes', trial=number,
                                    iterations=reference['iterations'], axis_mapping=None,
                                    scramble_delta_steps=reference['scramble_delta_steps'],
                                    simulation_seed=reference['simulation_seed'],
                                    optimization_seed=bayes_seed+number-1))
    specs = (specs if 'llm' in methods else []) + random_specs + bayes_specs
    return config, asdict(simulation), specs, inputs


def source_provenance():
    paths = ['src/runners/run_motorized_mirror_relay_matched.py',
             'src/tasks/motorized_mirror_relay_matched.py',
             'src/tasks/motorized_mirror_relay_bayes.py',
             'src/tasks/motorized_mirror_relay_random.py',
             'src/tasks/motorized_mirror_relay_protocol.py',
             'src/sim/motorized_mirror_relay.py', 'src/agents/llm_agent.py',
             'src/tasks/timing.py', 'src/io/artifacts.py']
    result = {'source_sha256': {path: sha256((REPO_ROOT / path).read_bytes()) for path in paths}}
    for name, args in (('git_commit', ['rev-parse', 'HEAD']), ('git_status', ['status', '--porcelain'])):
        try:
            result[name] = subprocess.check_output(['git', *args], cwd=REPO_ROOT,
                                                   text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            result[name] = None
    return result


def run(config_path, *, dry_run=False, method=None):
    config, simulation, specs, inputs = prepare_experiment(config_path, method=method)
    suffix = '_random' if config['methods'] == ['random'] else '_bayes' if config['methods'] == ['bayes'] else ''
    name = config['experiment_name'] + suffix + ('_dry_run' if dry_run else '')
    root = make_experiment_root(config.get('output_root', 'runs'), name).resolve()
    (root / 'config_resolved.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    for relative, content in inputs.items():
        target = root / 'lab_inputs' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    provenance = source_provenance()
    for relative in provenance['source_sha256']:
        target = root / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO_ROOT / relative).read_bytes())
    manifest = dict(experiment_type='motorized_mirror_relay_matched', dry_run=dry_run,
                    status='running', simulation=simulation, trials=specs, results=[],
                    provenance=provenance,
                    input_sha256={path: sha256(content) for path, content in inputs.items()},
                    baseline='Independent true alignment before each archived scramble',
                    comparison='Matched commands and protocol, not a physical lab-state replay')
    write_json(root / 'run_manifest.json', manifest)

    def execute(spec):
        output = root / spec['model_key'] / f"trial_{spec['trial']:02d}"
        record = dict(model=spec['model'], trial=spec['trial'], output_dir=str(output))
        try:
            execute_trial = ({'random': run_random_trial, 'bayes': run_bayes_trial}.get(spec['method'], run_trial))
            result = execute_trial(out_dir=output, spec=spec, simulation_config=simulation, dry_run=dry_run)
            record.update(status=result['status'], camera_image_count=result['camera_image_count'],
                          final_exit_beam_visible=result['final_exit_beam_visible'])
            if spec['method'] in {'random', 'bayes'}:
                record['best_furthest_camera_index'] = result['best_furthest_camera_index']
                record['best_furthest_camera_station'] = result['best_furthest_camera_station']
            if spec['method'] == 'bayes':
                record['best_furthest_mirror_hit'] = result['best_furthest_mirror_hit']
                record['best_objective'] = result['best_objective']
        except Exception as exc:
            record.update(status='failed', error=repr(exc))
        return record

    with ThreadPoolExecutor(max_workers=config.get('max_parallel_jobs', 1)) as executor:
        futures = [executor.submit(execute, spec) for spec in specs]
        for future in as_completed(futures):
            manifest['results'].append(future.result())
            manifest['results'].sort(key=lambda record: (record['model'], record['trial']))
            write_json(root / 'run_manifest.json', manifest)
    failed = any(record['status'] == 'failed' for record in manifest['results'])
    manifest['status'] = 'failed' if failed else 'completed'
    write_json(root / 'run_manifest.json', manifest)
    if failed:
        raise RuntimeError(f'Some matched trials failed; see {root}')
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', choices=['llm', 'random', 'bayes'], help='Run only this method, overriding config methods')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-config', action='store_true', help='Validate and print the plan; no writes or API calls')
    modes.add_argument('--dry-run', action='store_true', help='Save all initial simulated images; no API calls')
    args = parser.parse_args()
    if args.check_config:
        _, simulation, specs, _ = prepare_experiment(args.config, method=args.method)
        print(json.dumps(dict(simulation=simulation, trials=[{
            key: spec[key] for key in ('method', 'model', 'trial', 'iterations', 'llm', 'prompt_sha256', 'sampling_seed', 'optimization_seed') if key in spec
        } for spec in specs]), indent=2))
    else:
        print(f'Run complete: {run(args.config, dry_run=args.dry_run, method=args.method)}')


if __name__ == '__main__':
    main()
