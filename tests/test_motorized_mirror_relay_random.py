import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from src.runners.run_motorized_mirror_relay_matched import prepare_experiment, run
from src.sim.motorized_mirror_relay import AXES, MotorizedMirrorRelay, RelayConfig
from src.tasks.motorized_mirror_relay_matched import camera_reachability, simulator_command
from src.tasks.motorized_mirror_relay_random import run_random_trial, sample_offsets
from tests.test_motorized_mirror_relay_matched import CONFIG


class RandomRelayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.simulation, cls.specs, _ = prepare_experiment(CONFIG, method='random')

    def test_random_specs_reuse_exact_scrambles_and_budgets(self):
        _, _, mixed, _ = prepare_experiment(CONFIG)
        self.assertEqual(len(mixed), 30)
        self.assertEqual(len(self.specs), 10)
        for baseline, llm in zip(self.specs, mixed[:10]):
            self.assertEqual(baseline['scramble_delta_steps'], llm['scramble_delta_steps'])
            self.assertEqual(baseline['iterations'], llm['iterations'])
            self.assertEqual(baseline['simulation_seed'], llm['simulation_seed'])
            self.assertEqual(baseline['method'], 'random')

    def test_uniform_sampler_bounds_and_repeatability(self):
        rng = Mock()
        rng.integers.return_value = np.array([-15000, 15000, 0, -1, 1, 123, -42, 7])
        draw = sample_offsets(rng)
        rng.integers.assert_called_once_with(-15000, 15001, size=8)
        self.assertEqual(list(draw), list(AXES))
        self.assertEqual(draw[AXES[0]], -15000)
        self.assertEqual(draw[AXES[1]], 15000)
        first, second = np.random.default_rng(456), np.random.default_rng(456)
        for _ in range(20):
            self.assertEqual(sample_offsets(first), sample_offsets(second))

    def test_reach_checks_apertures_and_prefixes_without_placements(self):
        sim = MotorizedMirrorRelay(RelayConfig(**self.simulation), hidden_offsets=dict.fromkeys(AXES, 0))
        reach = camera_reachability(sim, sim.zero_command())
        self.assertEqual(reach['furthest_camera_station'], 'C8')
        self.assertTrue(all(value['camera_hit'] for value in reach['station_diagnostics'].values()))
        self.assertEqual(sim.camera_placements, 0)
        self.assertEqual(sim.camera_image_count, 0)
        traces = [dict(screen_hit=False, plane_reached=True, beam_offset_mm=10,
                       mirror_hit_sequence=[f'M{mirror}' for mirror in range(1, (number+1)//2+1)])
                  for number in range(1, 9)]
        traces[0]['screen_hit'] = True
        traces[7].update(screen_hit=True, mirror_hit_sequence=['M1'])
        with patch.object(sim, 'trace', side_effect=traces):
            reach = camera_reachability(sim, sim.zero_command())
        self.assertEqual(reach['furthest_camera_station'], 'C1')
        self.assertEqual(reach['furthest_plane_station'], 'C7')
        self.assertFalse(reach['station_diagnostics']['C8']['camera_hit'])

    def test_random_draws_do_not_accumulate_or_stop_at_success(self):
        spec = copy.deepcopy(self.specs[0])
        spec['iterations'] = 3
        aligned = {axis: -value for axis, value in spec['scramble_delta_steps'].items()}
        guesses = [aligned, dict.fromkeys(AXES, 15000), dict.fromkeys(AXES, -15000)]
        with tempfile.TemporaryDirectory() as directory:
            with patch('src.tasks.motorized_mirror_relay_random.sample_offsets', side_effect=guesses):
                result = run_random_trial(out_dir=Path(directory) / 'trial', spec=spec,
                                          simulation_config=self.simulation)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual([step['offsets'] for step in result['steps']], guesses)
            self.assertEqual(result['steps'][0]['furthest_camera_index'], 8)
            self.assertEqual([step['best_furthest_camera_index'] for step in result['steps']], [8, 8, 8])
            self.assertEqual(result['camera_image_count'], 3)
            self.assertEqual(result['camera_placement_count'], 1)
            self.assertEqual(result['final_command'], guesses[-1])
            with (Path(directory) / 'trial/reachability.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]['iteration'], '1')
            self.assertEqual(rows[0]['furthest_camera_station'], 'C8')

    def test_random_only_runner_never_constructs_an_llm(self):
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(self.config)
            config.update(output_root=directory, trial_ids=[1])
            path = Path(directory) / 'config.yaml'
            path.write_text(yaml.safe_dump(config))
            with patch('src.tasks.motorized_mirror_relay_matched.LLM', side_effect=AssertionError('No API')):
                root = run(path, method='random')
            manifest = json.loads((root / 'run_manifest.json').read_text())
            self.assertEqual(manifest['results'][0]['model'], 'random')
            self.assertEqual(manifest['results'][0]['camera_image_count'], 100)
            self.assertEqual(manifest['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
