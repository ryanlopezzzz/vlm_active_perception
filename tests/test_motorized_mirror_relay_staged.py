import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from src.sim.motorized_mirror_relay import AXES
from src.tasks.motorized_mirror_relay_staged import PROMPT_PATH, parse_action, run_alignment, prepare_run, run_trials


class StagedTests(unittest.TestCase):
    def test_uses_canonical_relay_prompt(self):
        self.assertEqual(PROMPT_PATH.name, 'four_mirror_relay.txt')

    def test_resume_preserves_trial_numbers_and_scrambles(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(seed=123, start_trial=3, snapshot_first_runs=3,
                snapshot_c8_before_after=False, snapshot_all_before_scramble=False,
                snapshot_all_after_scramble=True, snapshot_all_after_alignment=False)
            commands = [dict.fromkeys(AXES, i) for i in range(10)]
            with patch('src.tasks.motorized_mirror_relay_staged.LLM'), \
                 patch('src.tasks.motorized_mirror_relay_staged.snapshot', return_value={}), \
                 patch('src.tasks.motorized_mirror_relay_staged.prepare_run') as prepare, \
                 patch('src.tasks.motorized_mirror_relay_staged.run_alignment',
                       return_value={'status': 'llm_declared_done'}):
                result = run_trials(Mock(current_camera_location='C8'), Path(directory),
                    {'camera': {}, 'llm': {'model': 'mock'}}, args, 100, commands)
            self.assertEqual([r['trial'] for r in result['trials']], list(range(3,11)))
            self.assertEqual([c.args[2] for c in prepare.call_args_list], commands[2:])
            self.assertEqual([c.kwargs['all_after'] for c in prepare.call_args_list], [True]+[False]*7)

    def test_snapshot_first_three_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(seed=123, snapshot_first_runs=3,
                snapshot_c8_before_after=False, snapshot_all_before_scramble=False,
                snapshot_all_after_scramble=True, snapshot_all_after_alignment=True)
            session = Mock(current_camera_location='C8')
            with patch('src.tasks.motorized_mirror_relay_staged.LLM'), \
                 patch('src.tasks.motorized_mirror_relay_staged.snapshot', return_value={}), \
                 patch('src.tasks.motorized_mirror_relay_staged.prepare_run') as prepare, \
                 patch('src.tasks.motorized_mirror_relay_staged.snapshot_all') as after, \
                 patch('src.tasks.motorized_mirror_relay_staged.run_alignment',
                       return_value={'status': 'llm_declared_done'}):
                run_trials(session, Path(directory), {'camera': {}, 'llm': {'model': 'mock'}},
                           args, 100, [{} for _ in range(10)])
            self.assertEqual([c.kwargs['all_after'] for c in prepare.call_args_list],
                             [True]*3 + [False]*7)
            self.assertEqual(after.call_count, 3)

    def test_repeated_delta_and_zero_axis(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Mock(config={'camera': {}, 'llm': {}},
                           current_command=dict.fromkeys(AXES, 9000), current_camera_location='C8')
            def move(target):
                session.current_command = target
                return []
            session.set_motor_steps.side_effect = move
            llm = Mock()
            llm.query_llm.side_effect = [json.dumps(dict(action='tilt', mirror='M1',
                vertical=v, horizontal=0, reason='probe')) for v in (2000, 2000, -1000)]
            run_alignment(session, llm, Path(directory), 3, capture_fn=lambda *a: {})
            self.assertEqual([c.args[0]['mirror_1_in_plane_steps']
                              for c in session.set_motor_steps.call_args_list], [11000, 13000, 12000])
            self.assertEqual(session.current_command['mirror_1_out_of_plane_steps'], 9000)
            offsets = dict.fromkeys(AXES, 0)
            offsets['mirror_1_in_plane_steps'] = 29000
            with self.assertRaises(ValueError):
                parse_action(json.dumps(dict(action='tilt', mirror='M1', vertical=2000,
                                             horizontal=0, reason='too far')), 'C8', offsets)

    def test_batch_stops_and_creates_fresh_conversations(self):
        for statuses in (['llm_declared_done'] * 3, ['llm_declared_done', 'iteration_limit']):
            with tempfile.TemporaryDirectory() as directory:
                args = SimpleNamespace(seed=1, snapshot_c8_before_after=False,
                    snapshot_all_before_scramble=False, snapshot_all_after_scramble=False,
                    snapshot_all_after_alignment=False)
                session = Mock(current_camera_location='C8')
                with patch('src.tasks.motorized_mirror_relay_staged.LLM') as model, \
                     patch('src.tasks.motorized_mirror_relay_staged.snapshot', return_value={}) as first, \
                     patch('src.tasks.motorized_mirror_relay_staged.prepare_run') as prepare, \
                     patch('src.tasks.motorized_mirror_relay_staged.run_alignment',
                           side_effect=[{'status': s} for s in statuses]):
                    result = run_trials(session, Path(directory), {'camera': {}, 'llm': {'model': 'mock'}},
                                        args, 100, [{}, {}, {}])
                self.assertEqual(model.call_count, len(statuses))
                self.assertEqual(first.call_count, len(statuses))
                self.assertEqual(prepare.call_count, len(statuses))
                self.assertEqual(result['status'], 'completed' if len(statuses) == 3
                                 else 'stopped_without_alignment')

    def test_all_snapshot_passes_bracket_scramble(self):
        with tempfile.TemporaryDirectory() as directory:
            events = []
            session = Mock(config={'camera': {}}, current_command=dict.fromkeys(AXES, 0))
            session.place_camera.side_effect = lambda station: events.append(station)
            session.set_motor_steps.side_effect = lambda target: events.append('scramble') or []
            def capture(camera, path):
                events.append(f'{path.parent.name}/{path.name}')
                return {}
            prepare_run(session, Path(directory), dict.fromkeys(AXES, 100),
                        capture_fn=capture, all_before=True, all_after=True)
            expected = []
            for phase in ('before_scramble', 'after_scramble'):
                if phase == 'after_scramble':
                    expected.append('scramble')
                for i in range(1, 9):
                    expected.extend([f'C{i}', f'{phase}/C{i}.png'])
            self.assertEqual(events, expected)
            session.set_motor_steps.assert_called_once()

    def test_before_after_order_then_c1(self):
        with tempfile.TemporaryDirectory() as directory:
            events = []
            session = Mock(config={'camera': {}, 'llm': {}}, current_command=dict.fromkeys(AXES, 100))
            def place(station):
                events.append(station)
                session.current_camera_location = station
            def move(target):
                events.append('scramble')
                session.current_command = target
                return []
            def capture(camera, path):
                events.append(path.name)
                return {'path': str(path)}
            session.place_camera.side_effect = place
            session.set_motor_steps.side_effect = move
            root = Path(directory)
            prepare_run(session, root, dict.fromkeys(AXES, -200), True, capture)
            llm = Mock()
            llm.query_llm.return_value = json.dumps(dict(action='measure', station='C1', reason='ready'))
            result = run_alignment(session, llm, root, 1, capture_fn=capture)
            self.assertEqual(events, ['C8', 'C8_before_scramble.png', 'scramble',
                                      'C8_after_scramble.png', 'C8', '000_C8.png', 'C1'])
            self.assertEqual(result['control_origin'], dict.fromkeys(AXES, -100))

    def test_invalid_actions(self):
        for station, action in [('C1', {'action': 'done'}), ('C8', {'action': 'next'}),
                                ('C1', {'action': 'tilt', 'mirror': 'M1', 'vertical': 15001, 'horizontal': 0}),
                                ('C1', {'action': 'tilt', 'mirror': 'M1', 'vertical': True, 'horizontal': 0}),
                                ('C1', {'action': 'next', 'mirror': 4})]:
            with self.assertRaises(ValueError):
                parse_action(json.dumps(dict(reason='test', **action)), station)

    def test_station_choice_revisits_and_selected_mirror(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Mock(config={'camera': {}, 'llm': {}},
                           current_command=dict.fromkeys(AXES, 12000), current_camera_location='C8')
            def place(station):
                session.current_camera_location = station
            session.place_camera.side_effect = place
            def move(target):
                session.current_command = target
                return []
            session.set_motor_steps.side_effect = move
            actions = [dict(action='measure', station='C3', reason='inspect'),
                       dict(action='tilt', mirror='M1', vertical=2000, horizontal=-2000, reason='probe upstream'),
                       dict(action='measure', station='C1', reason='revisit'),
                       dict(action='measure', station='C8', reason='verify'),
                       dict(action='done', reason='centered')]
            llm = Mock()
            llm.query_llm.side_effect = list(map(json.dumps, actions))
            result = run_alignment(session, llm, Path(directory), 5, capture_fn=lambda *a: {})
            self.assertEqual(result['status'], 'llm_declared_done')
            self.assertEqual([r['station'] for r in result['steps']], ['C8','C3','C3','C1','C8'])
            session.set_motor_steps.assert_called_once()
            target = session.set_motor_steps.call_args.args[0]
            expected = dict.fromkeys(AXES, 12000)
            expected['mirror_1_in_plane_steps'] = 14000
            expected['mirror_1_out_of_plane_steps'] = 10000
            self.assertEqual(target, expected)

    def test_limit_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Mock(config={'camera': {}, 'llm': {}}, current_command=dict.fromkeys(AXES, 0),
                           current_camera_location='C1')
            llm = Mock()
            llm.query_llm.return_value = json.dumps(dict(action='measure', station='C1', reason='ready'))
            result = run_alignment(session, llm, Path(directory), 1, capture_fn=lambda *a: {})
            self.assertEqual(result['status'], 'iteration_limit')
            session.set_motor_steps.assert_not_called()


if __name__ == '__main__':
    unittest.main()
