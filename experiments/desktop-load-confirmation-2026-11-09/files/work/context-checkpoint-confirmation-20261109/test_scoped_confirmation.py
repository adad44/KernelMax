"""Regressions for the scoped schedule and whole-workload evidence gate."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evidence import GUARDS
from load_confirmation_protocol import load_protocol
from per_context_audit import clears_threshold
from test_schedule import FakeNative, SPEC
from verify_exit import finalize_exit

WORK = Path(__file__).resolve().parent


class ScopedConfirmationTests(unittest.TestCase):
    def test_scoped_schedule_and_exit_keep_full_pass_verdict_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            (work / 'confirmation-scope.json').write_bytes((WORK / 'confirmation-scope.json').read_bytes())
            protocol = load_protocol(SPEC, work)
            backend = FakeNative(work, protocol=protocol)
            self.addCleanup(backend.cleanup)
            backend.execute()
            self.assertEqual(backend.calls, 206)
            state = backend.controller.state()
            self.assertEqual(state['starts_reserved'], 206)
            self.assertEqual(list(state['seals']), ['4096'])
            measurements = {row['coordinates']['context'] for row in state['collections'].values()
                            if row['coordinates'].get('phase') in ('aa', 'comparison')}
            self.assertEqual(measurements, {4096})
            backend.controller.close('complete', dict.fromkeys(GUARDS, True) | {
                'worker_process': {'pid': 123, 'birth': 'synthetic CPU fixture'},
            })
            with patch('verify_exit.process_identity', return_value=None), patch('verify_exit.boot_identity', return_value=None), \
                    patch('verify_exit.verify_context', return_value=backend.retained[0]):
                post = finalize_exit(work, 'one')
            self.assertEqual(post['completed_confirmation_scope'], [4096])
            self.assertFalse(post['complete_locked_workload'])
            self.assertFalse(post['five_percent_improvement_verified'])

    def test_aggregate_requires_all_four_blocks_and_passing_controls(self):
        def block(lower):
            return {'pairs': 20, 'median_ratio': 1.13, 'bootstrap_95pct_ratio': [lower, 1.14],
                    'repeatable_gain_verified': True}
        contexts = {context: {'all_controls_passed': True, 'aa_noise': 0.004,
                              'blocks': {'1': block(1.02)}} for context in (512, 2048, 4096)}
        self.assertFalse(clears_threshold(contexts, 1.01))
        contexts[512]['blocks']['2'] = block(1.02)
        self.assertTrue(clears_threshold(contexts, 1.01))
        self.assertFalse(clears_threshold(contexts, 1.05))
        contexts[4096]['all_controls_passed'] = False
        self.assertFalse(clears_threshold(contexts, 1.01))


if __name__ == '__main__':
    unittest.main()
