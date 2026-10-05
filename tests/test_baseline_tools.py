"""Portable control helpers are tested without inference, display, or service changes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.quiet_baseline_controller import acknowledge
from scripts.run_native_baseline import supervise, worker_command


class BaselineToolTests(unittest.TestCase):
    def test_worker_flags_preserve_full_protocol(self):
        command = worker_command(Path('model'), Path('verify.json'), Path('fresh'), 'paging-aware')
        for flag in ('--validate-reference', '--paired-aa', '--controlled', '--rehash-checkpoint',
                     '--stability-screen', '--per-request-control'):
            self.assertIn(flag, command)
        self.assertEqual(command[command.index('--resident-retries') + 1], '2')
        self.assertEqual(command[command.index('--swap-policy') + 1], 'paging-aware')
        self.assertNotIn('--human-review', command)
        self.assertNotIn('--dense-diagnostic', command)

    def test_dense_diagnostic_is_explicit(self):
        command = worker_command('model', 'verify', 'out', 'strict', Path('failed.json'))
        self.assertEqual(command[-2:], ['--dense-diagnostic', 'failed.json'])

    def test_nonpaused_gate_cannot_acknowledge(self):
        with self.assertRaises(ValueError):
            acknowledge({'status': 'running', 'token': 'old'}, Path('out'), 123)

    def test_equal_hold_and_display_control_requires_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'out'
            with patch('scripts.quiet_baseline_controller.time.sleep') as sleep, \
                 patch('scripts.quiet_baseline_controller.run') as run:
                self.assertEqual(acknowledge({'status': 'paused', 'token': 'fresh'}, output, 123),
                                 ['--resume', 'fresh'])
                sleep.assert_called_once_with(12)
                run.assert_not_called()
                acknowledge({'status': 'paused', 'token': 'next'}, output, 123, True)
                run.assert_called_once_with(['/usr/bin/pmset', 'displaysleepnow'])
            rows = [json.loads(line) for line in (Path(temporary)/'out-local-controller.jsonl').read_text().splitlines()]
            self.assertEqual([row['token'] for row in rows], ['fresh', 'next'])
            self.assertEqual([row['display_sleep'] for row in rows], [False, True])

    def test_controller_failure_stops_uncontrolled_worker(self):
        worker, controller = Mock(), Mock()
        worker.poll.return_value = None
        controller.poll.return_value = 1
        with patch('evaluator.baseline_control.current_gate', return_value=(Path('out'), {'status': 'running'})):
            with self.assertRaisesRegex(RuntimeError, 'Controller exited'):
                supervise(worker, controller, Path('out'))

    def test_terminal_gate_allows_controller_exit(self):
        worker, controller = Mock(), Mock()
        worker.poll.side_effect = [None, 0]
        worker.returncode = controller.poll.return_value = 0
        with patch('evaluator.baseline_control.current_gate', return_value=(Path('out'), {'status': 'completed_pending_review'})), \
             patch('scripts.run_native_baseline.time.sleep'):
            self.assertEqual(supervise(worker, controller, Path('out')), 0)

    def test_session_deadline_does_not_reroll(self):
        worker, controller = Mock(), Mock()
        worker.poll.return_value = controller.poll.return_value = None
        with patch('evaluator.baseline_control.current_gate', return_value=(Path('out'), {'status': 'running'})), \
             patch('scripts.run_native_baseline.time.monotonic', side_effect=[0, 10801]):
            with self.assertRaisesRegex(RuntimeError, 'Session time limit'):
                supervise(worker, controller, Path('out'))

    def test_idle_deadline_does_not_reload(self):
        worker, controller = Mock(), Mock()
        worker.poll.return_value = controller.poll.return_value = None
        with patch('evaluator.baseline_control.current_gate', return_value=(Path('out'), {'status': 'paused', 'token': 'fresh'})), \
             patch('scripts.run_native_baseline.time.monotonic', side_effect=[0, 1, 2, 3, 902]), \
             patch('scripts.run_native_baseline.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'Checkpoint idle'):
                supervise(worker, controller, Path('out'))


if __name__ == '__main__':
    unittest.main()
