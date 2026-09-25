import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import yaml

from src.runners.run_motorized_mirror_relay_matched import PROMPT_PATH, prepare_experiment, run
from src.sim.motorized_mirror_relay import AXES, MotorizedMirrorRelay, RelayConfig
from src.tasks.motorized_mirror_relay_bayes import center_distance_px, furthest_mirror_hit
from src.tasks.motorized_mirror_relay_matched import run_trial, simulator_command
from src.tasks.motorized_mirror_relay_protocol import parse_action


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/motorized_mirror_relay/matched_lab_simulation.yaml'


class FakeLLM:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.messages = []
        self.images = []
        self.requests = []
        self.api_calls = []

    def add_system_message(self, text):
        self.prompt = text

    def add_user_message(self, text):
        self.messages.append(json.loads(text))

    def add_user_image(self, path):
        self.images.append(path)

    def query_llm(self, **kwargs):
        self.requests.append(kwargs)
        for _ in range(kwargs['num_tries']):
            answer = json.dumps(next(self.actions))
            self.api_calls.append({'duration_sec': 0.01})
            if kwargs['verify'](answer):
                return answer
        raise RuntimeError('All attempts rejected')

    def get_usage_summary(self):
        return {'request_attempts': len(self.api_calls)}


def tilt(vertical=0, horizontal=0, mirror='M1'):
    return dict(action='tilt', mirror=mirror, vertical=vertical, horizontal=horizontal, reason='Probe')


def measure(station):
    return dict(action='measure', station=station, reason='Inspect')


DONE = dict(action='done', reason='Localized beam at C8')


class MatchedRelayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.simulation, cls.specs, _ = prepare_experiment(CONFIG, method='llm')

    def trial(self, directory, actions, iterations=None):
        spec = copy.deepcopy(self.specs[0])
        spec['scramble_delta_steps'] = dict.fromkeys(AXES, 0)
        spec['iterations'] = iterations or len(actions)
        agent = FakeLLM(actions)
        result = run_trial(out_dir=Path(directory) / 'trial', spec=spec,
                           simulation_config=self.simulation, llm=agent)
        return result, agent

    def test_archive_pairing_and_settings(self):
        self.assertEqual(len(self.specs), 20)
        expected_prompt = PROMPT_PATH.read_text(encoding='utf-8')
        self.assertTrue(all(spec['prompt'] == expected_prompt for spec in self.specs))

    def test_bayes_plan_uses_paired_scrambles_and_budgets(self):
        _, _, bayes_specs, _ = prepare_experiment(CONFIG, method='bayes')
        self.assertEqual(len(bayes_specs), 10)
        self.assertTrue(all(spec['method'] == 'bayes' for spec in bayes_specs))
        self.assertTrue(all(spec['iterations'] == 100 for spec in bayes_specs))
        self.assertEqual(bayes_specs[0]['optimization_seed'], 789)
        self.assertEqual(bayes_specs[-1]['optimization_seed'], 798)

    def test_sparse_c8_objective_uses_pixel_distance_only_on_hit(self):
        config = RelayConfig(camera_width_mm=6.4, camera_height_mm=4.8,
                             frame_width_px=640, frame_height_px=480)
        hit = {'screen_hit': True, 'mirror_hit_sequence': ['M1', 'M2', 'M3', 'M4'],
               'u_mm': 1.0, 'v_mm': -1.0}
        miss = {**hit, 'screen_hit': False}
        self.assertAlmostEqual(center_distance_px(hit, config), np.hypot(100, 100))
        self.assertIsNone(center_distance_px(miss, config))
        self.assertEqual(furthest_mirror_hit(['M1', 'M2', 'M3']), 3)
        self.assertEqual({spec['model'] for spec in self.specs},
                         {'openai/gpt-5.6-sol', 'google/gemini-3.5-flash'})
        self.assertEqual(len({spec['prompt_sha256'] for spec in self.specs}), 1)
        for first, second in zip(self.specs[:10], self.specs[10:]):
            self.assertEqual(first['scramble_delta_steps'], second['scramble_delta_steps'])
            self.assertEqual(first['simulation_seed'], second['simulation_seed'])
            self.assertEqual(first['trial'], second['trial'])
        for spec in self.specs:
            self.assertEqual(spec['iterations'], 100)
            self.assertEqual(spec['llm']['effort'], 'high')
            self.assertEqual(spec['llm']['max_tokens'], 12000)
            self.assertIsNone(spec['llm']['reasoning_max_tokens'])
            self.assertIn('Images are full unmodified captures', spec['prompt'])
        self.assertTrue(self.specs[1]['lab_known_unsuccessful'])

    def test_aligned_origin_and_exact_scramble_cancellation(self):
        for spec in self.specs[:10]:
            hidden = simulator_command(spec['scramble_delta_steps'])
            sim = MotorizedMirrorRelay(RelayConfig(**self.simulation), hidden_offsets=hidden)
            corrected = {axis: -value for axis, value in hidden.items()}
            for station in sim.camera_locations.values():
                trace = sim.trace(corrected, station)
                self.assertTrue(trace['screen_hit'])
                self.assertLess(trace['beam_offset_mm'], 1e-9)
            self.assertEqual(sim.config.height_mm, 203.2)
            np.testing.assert_array_equal(sim.centers, sim.nominal_centers)

    def test_vertical_horizontal_axis_mapping_for_all_mirrors(self):
        sim = MotorizedMirrorRelay(RelayConfig(**self.simulation), hidden_offsets=dict.fromkeys(AXES, 0))
        for mirror in range(1, 5):
            station = sim.camera_locations[f'C{2*mirror-1}']
            vertical = sim.zero_command()
            vertical[f'mirror_{mirror}_in_plane_steps'] = 100
            trace = sim.trace(simulator_command(vertical), station)
            self.assertTrue(trace['screen_hit'])
            self.assertGreater(abs(trace['v_mm']), 0.01)
            self.assertLess(abs(trace['u_mm']), 0.001 * abs(trace['v_mm']))
            horizontal = sim.zero_command()
            horizontal[f'mirror_{mirror}_out_of_plane_steps'] = 100
            trace = sim.trace(simulator_command(horizontal), station)
            self.assertTrue(trace['screen_hit'])
            self.assertGreater(abs(trace['u_mm']), 0.01)
            self.assertLess(abs(trace['v_mm']), 1e-9)

    def test_protocol_turn_order_and_relative_commands(self):
        actions = [measure('C1'), tilt(2000), tilt(2000), measure('C8'), DONE]
        with tempfile.TemporaryDirectory() as directory:
            result, agent = self.trial(directory, actions)
            self.assertEqual([step['station'] for step in result['steps']], ['C8', 'C1', 'C1', 'C1', 'C8'])
            self.assertEqual([step['offsets'][AXES[0]] for step in result['steps']], [0, 0, 2000, 4000, 4000])
            self.assertEqual(result['final_command'][AXES[0]], 4000)
            self.assertEqual(result['camera_image_count'], 5)
            self.assertEqual(result['camera_placement_count'], 3)
            self.assertEqual(result['status'], 'llm_declared_done')
            self.assertTrue(result['final_state_measured'])
            self.assertEqual([message['turns_remaining'] for message in agent.messages], [5, 4, 3, 2, 1])
            for message in agent.messages:
                self.assertEqual(set(message), {'station', 'cumulative_commanded_steps', 'turns_remaining'})
            for request in agent.requests:
                self.assertEqual(request['effort'], 'high')
                self.assertEqual(request['max_tokens'], 12000)
                self.assertIsNone(request['reasoning_max_tokens'])
            for path in agent.images:
                with Image.open(path) as image:
                    self.assertEqual(image.size, (640, 480))
                    self.assertEqual(image.mode, 'RGB')

    def test_rejected_action_has_feedback_without_movement_or_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            result, agent = self.trial(directory, [tilt(15001), tilt(100), DONE], iterations=2)
            self.assertEqual(result['camera_image_count'], 2)
            self.assertEqual(result['final_command'][AXES[0]], 100)
            self.assertEqual(len(result['steps'][0]['rejections']), 1)
            self.assertIn('No movement was executed', agent.messages[1]['instruction'])
            self.assertEqual(agent.get_usage_summary()['request_attempts'], 3)

    def test_cumulative_and_station_limits(self):
        offsets = dict.fromkeys(AXES, 0)
        offsets[AXES[0]] = 29999
        for action, station in [(tilt(2), 'C8'), (tilt(True), 'C8'), (DONE, 'C1'),
                                (measure('C9'), 'C8')]:
            with self.assertRaises((ValueError, KeyError, TypeError)):
                parse_action(json.dumps(action), station, offsets)
        self.assertEqual(parse_action(json.dumps(tilt(1)), 'C8', offsets)['vertical'], 1)

    def test_final_tilt_is_applied_but_not_measured(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self.trial(directory, [tilt(100)])
            self.assertEqual(result['status'], 'iteration_limit')
            self.assertEqual(result['final_command'][AXES[0]], 100)
            self.assertFalse(result['final_state_measured'])
            self.assertEqual(result['camera_image_count'], 1)

    def test_final_measure_places_camera_without_extra_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self.trial(directory, [measure('C1')])
            self.assertEqual(result['final_camera_station'], 'C1')
            self.assertEqual(result['camera_image_count'], 1)
            self.assertEqual(result['camera_placement_count'], 2)
            self.assertFalse(result['final_state_measured'])

    def test_same_station_keeps_camera_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self.trial(directory, [measure('C8'), DONE])
            self.assertEqual(result['camera_placement_count'], 1)
            self.assertEqual(result['camera_image_count'], 2)

    def test_failed_requests_preserve_partial_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                self.trial(directory, [tilt(15001)]*3, iterations=1)
            result = json.loads((Path(directory) / 'trial/run.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['camera_image_count'], 1)
            self.assertEqual(result['final_command'], dict.fromkeys(AXES, 0))

    def test_dry_run_paired_images_and_provenance_without_llm(self):
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(self.config)
            config['output_root'] = directory
            config_path = Path(directory) / 'config.yaml'
            config_path.write_text(yaml.safe_dump(config))
            with patch('src.tasks.motorized_mirror_relay_matched.LLM', side_effect=AssertionError('No API calls')):
                root = run(config_path, dry_run=True)
            manifest = json.loads((root / 'run_manifest.json').read_text())
            self.assertEqual(len(manifest['results']), 20)
            self.assertEqual(manifest['status'], 'completed')
            self.assertTrue(all(record['status'] == 'dry_run' for record in manifest['results']))
            for number in range(1, 11):
                relative = f'trial_{number:02d}/images/000_C8.png'
                self.assertEqual((root / 'gpt_5_6_sol' / relative).read_bytes(),
                                 (root / 'gemini_3_5_flash' / relative).read_bytes())
            self.assertTrue((root / 'lab_inputs/trials.json').exists())
            self.assertTrue((root / 'source_snapshot/src/tasks/motorized_mirror_relay_matched.py').exists())

    def test_invalid_dimensions_or_command_limit_fail_before_running(self):
        with tempfile.TemporaryDirectory() as directory:
            for field, value in [('frame_width_px', 800), ('command_limit_steps', 8192)]:
                config = copy.deepcopy(self.config)
                config['simulation'][field] = value
                path = Path(directory) / 'config.yaml'
                path.write_text(yaml.safe_dump(config))
                with self.assertRaises(ValueError):
                    prepare_experiment(path)

    def test_runner_import_has_no_hardware_dependencies(self):
        code = ('import sys; import src.runners.run_motorized_mirror_relay_matched; '
                'assert not any(name.startswith("src.hardware") for name in sys.modules); '
                'assert "pick_and_place_and_capture_skeleton" not in sys.modules')
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)


if __name__ == '__main__':
    unittest.main()
