import unittest
import json
import tempfile
from pathlib import Path

from evaluator.native_baseline import (EnvironmentInvalid, digest, environment_issues, fixed_inputs,
    materialize_parameters, metrics, quiet_preflight, resident_precondition, resident_retry_loop,
    swap_observation, verify_resident_inputs)
from evaluator.baseline_control import current_gate
from evaluator.schemas import load_contract


class ResidentWorkerTests(unittest.TestCase):
    def test_environment_failure_retained_before_verify_cooldown_and_fresh_attempt(self):
        events = []
        def run(index):
            events.append(('run', index))
            if index == 1:
                raise EnvironmentInvalid('swapins')
            return 'completed'
        result = resident_retry_loop(run, lambda i: events.append(('fresh', i)),
            lambda e, i: events.append(('retained', i)), lambda: events.append(('verify',)),
            lambda: events.append(('cooldown',)), 2)
        self.assertEqual(result, 'completed')
        self.assertEqual(events, [('run', 1), ('retained', 1), ('verify',),
                                  ('cooldown',), ('fresh', 2), ('run', 2)])

    def test_gpu_correctness_and_statistical_failures_never_retry(self):
        for message in ('GPU command buffer timeout', 'Stock-generation comparison failed',
                        'A/A noise calibration did not pass'):
            def run(index):
                raise RuntimeError(message)
            def forbidden(*args):
                self.fail('Unsafe failure reached retry path')
            with self.assertRaisesRegex(RuntimeError, message):
                resident_retry_loop(run, forbidden, forbidden, forbidden, forbidden, 2)

    def test_changed_protected_inputs_stop_before_cooldown_or_new_attempt(self):
        retained = []
        def run(index):
            raise EnvironmentInvalid('swapins')
        def changed():
            raise ValueError('Protected input changed')
        def forbidden(*args):
            self.fail('Changed inputs reached retry path')
        with self.assertRaisesRegex(ValueError, 'Protected input changed'):
            resident_retry_loop(run, forbidden, lambda e, i: retained.append(i), changed, forbidden, 2)
        self.assertEqual(retained, [1])

    def test_retry_cap_and_zero_retry_compatibility(self):
        for retries in (0, 2):
            calls, retained, fresh = [], [], []
            def run(index):
                calls.append(index)
                raise EnvironmentInvalid('swapins')
            with self.assertRaises(EnvironmentInvalid):
                resident_retry_loop(run, fresh.append, lambda e, i: retained.append(i),
                                    lambda: None, lambda: None, retries)
            self.assertEqual(calls, list(range(1, retries + 2)))
            self.assertEqual(retained, calls)
            self.assertEqual(fresh, calls[1:])

    def test_support_hash_and_checkpoint_stat_are_checked(self):
        contract = Path(__file__).resolve().parents[1] / 'kernelmaxxing.yaml'
        with tempfile.TemporaryDirectory() as temporary:
            support = Path(temporary) / 'config.json'
            support.write_text('{}')
            stat = support.stat()
            state = dict(contract_sha256=load_contract(contract).sha256,
                runtime_source_sha256={}, runtime_binary_sha256={}, evaluator_source_sha256={},
                model_support_sha256={str(support): digest(support)},
                checkpoint_stat_at_verification={str(support): [stat.st_dev, stat.st_ino, stat.st_size,
                                                                stat.st_mtime_ns, stat.st_ctime_ns]})
            verify_resident_inputs(state, contract)
            support.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'Protected input changed'):
                verify_resident_inputs(state, contract)
            state['model_support_sha256'][str(support)] = digest(support)
            with self.assertRaisesRegex(ValueError, 'Checkpoint changed'):
                verify_resident_inputs(state, contract)

    def test_controller_follows_retry_and_waits_during_cooldown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'session'
            root.mkdir()
            pointer = root / 'resident-session.json'
            session = dict(current_attempt=str(root), status='running', phase='resident_retry_cooldown')
            pointer.write_text(json.dumps(session))
            (root / 'gate.json').write_text(json.dumps(dict(status='failed')))
            self.assertEqual(current_gate(root)[1]['status'], 'waiting_for_worker')
            retry = root.with_name('session-retry02')
            retry.mkdir()
            (retry / 'gate.json').write_text(json.dumps(dict(status='paused', token='new')))
            session['current_attempt'] = str(retry)
            pointer.write_text(json.dumps(session))
            output, gate = current_gate(root)
            self.assertEqual((output, gate['token']), (retry, 'new'))
            session['status'] = 'failed'
            pointer.write_text(json.dumps(session))
            self.assertEqual(current_gate(root)[1]['status'], 'failed')


class BaselineTests(unittest.TestCase):
    def paging_environment(self):
        return dict(power="Now drawing from 'AC Power'", swap_used_mib=100,
                    thermal="No thermal warning level has been recorded\nNo performance warning level has been recorded",
                    power_settings="lowpowermode         0", thermal_state=0, swapins=0, swapouts=0)

    def test_paging_aware_allows_only_logged_swapins(self):
        before = self.paging_environment()
        after = dict(before, swapins=113)
        self.assertIn('swapins increased during request', environment_issues(before, after))
        self.assertEqual(environment_issues(before, after, 'paging-aware'), [])
        self.assertEqual(swap_observation(before, after, 'paging-aware')['swapin_pages'], 113)
        for key, value, expected in (
                ('swapouts', 1, 'swapouts increased during request'),
                ('swap_used_mib', 101, 'swap grew during the request'),
                ('thermal_state', 1, 'native thermal state is not nominal'),
                ('swapins', -1, 'swapins counter decreased unexpectedly')):
            self.assertIn(expected, environment_issues(before, dict(after, **{key: value}), 'paging-aware'))
        del after['swapouts']
        self.assertIn('swapouts observation missing', environment_issues(before, after, 'paging-aware'))
        with self.assertRaises(ValueError):
            environment_issues(before, before, 'unknown')

    def test_paging_aware_quiet_window_retains_allowed_activity(self):
        ticks = [0]
        before = self.paging_environment()
        result = quiet_preflight(lambda: dict(before, swapins=ticks[0]), lambda: ticks[0],
            lambda n: ticks.__setitem__(0, ticks[0]+n), quiet_seconds=6, timeout_seconds=10,
            swap_policy='paging-aware')
        self.assertEqual(result['waited_seconds'], 6)
        self.assertEqual(result['swap_policy'], 'paging-aware')
        self.assertTrue(all(x['swap_activity']['swapin_pages'] == 2 and not x['issues']
                            for x in result['observations']))

    def test_failed_quiet_window_preserves_observations(self):
        ticks = [0]
        before = self.paging_environment()
        with self.assertRaises(EnvironmentInvalid) as raised:
            quiet_preflight(lambda: dict(before, swapouts=ticks[0]), lambda: ticks[0],
                lambda n: ticks.__setitem__(0, ticks[0]+n), quiet_seconds=6,
                timeout_seconds=10, swap_policy='paging-aware')
        evidence = raised.exception.evidence
        self.assertEqual(evidence['waited_seconds'], 10)
        self.assertEqual(len(evidence['observations']), 5)
        self.assertTrue(all(x['issues'] for x in evidence['observations']))

    def test_paging_aware_priming_keeps_allowed_swapins(self):
        base = self.paging_environment()
        calls = [0]
        def observe():
            calls[0] += 1
            return dict(base, swapins=calls[0])
        result = resident_precondition([512], lambda n: {}, observe, swap_policy='paging-aware')
        self.assertEqual(len(result), 2)
        self.assertTrue(all(x['swap_activity']['swapin_pages'] == 1 and
                            not x['environment_events'] and x['excluded_from_baseline'] for x in result))

    def test_resident_priming_retains_cold_fault_and_needs_two_clean_rounds(self):
        base = dict(power="Now drawing from 'AC Power'", swap_used_mib=100,
                    thermal="No thermal warning level has been recorded\nNo performance warning level has been recorded",
                    power_settings="lowpowermode         0", thermal_state=0,
                    swapins=0, swapouts=0)
        runs, retained = [], []
        def run(length):
            runs.append(length)
            return {'output_token_ids': [1]*128}
        def observe():
            return dict(base, swapins=1 if len(runs) >= 2 else 0)
        result = resident_precondition([512,2048,4096], run, observe, retained.append)
        self.assertEqual(len(result), 9)
        self.assertEqual(result, retained)
        self.assertIn('swapins increased during request', result[1]['environment_events'])
        self.assertTrue(all(x['excluded_from_baseline'] for x in result))

    def test_resident_priming_fails_instead_of_accepting_paging(self):
        base = dict(power="Now drawing from 'AC Power'", swap_used_mib=100,
                    thermal="No thermal warning level has been recorded\nNo performance warning level has been recorded",
                    power_settings="lowpowermode         0", thermal_state=0,
                    swapins=0, swapouts=0)
        calls = [0]
        def observe():
            calls[0] += 1
            return dict(base, swapins=calls[0])
        with self.assertRaisesRegex(RuntimeError, 'no official warmups started'):
            resident_precondition([512], lambda n: {}, observe, max_rounds=3)

    def test_quiet_preflight_waits_without_weakening_request_invalidation(self):
        ticks = [0]
        base = dict(power="Now drawing from 'AC Power'", swap_used_mib=100,
                    thermal="No thermal warning level has been recorded\nNo performance warning level has been recorded",
                    power_settings="lowpowermode         0", thermal_state=0,
                    swapins=12, swapouts=4)
        def sleep(seconds):
            ticks[0] += seconds
        def observe():
            return dict(base, swapins=12 if ticks[0] < 4 else 13)
        result = quiet_preflight(observe, lambda: ticks[0], sleep,
                                 quiet_seconds=6, timeout_seconds=20)
        self.assertEqual(result['waited_seconds'], 10)
        self.assertIn('swapins increased during request', result['observations'][1]['issues'])
        self.assertIn('swapins increased during request', environment_issues(base, observe()))

    def test_quiet_preflight_timeout_does_not_start_request(self):
        ticks = [0]
        base = dict(power="Now drawing from 'AC Power'", swap_used_mib=100,
                    thermal="No thermal warning level has been recorded\nNo performance warning level has been recorded",
                    power_settings="lowpowermode         0", thermal_state=0,
                    swapins=0, swapouts=4)
        def sleep(seconds):
            ticks[0] += seconds
        with self.assertRaisesRegex(RuntimeError, 'no request started'):
            quiet_preflight(lambda: dict(base, swapins=ticks[0]), lambda: ticks[0], sleep,
                            quiet_seconds=6, timeout_seconds=10)

    def test_materialization_is_one_tensor_then_sync_without_modification(self):
        from types import SimpleNamespace
        tensors = [SimpleNamespace(nbytes=8), SimpleNamespace(nbytes=16)]
        calls, progress = [], []
        runtime = SimpleNamespace(eval=lambda tensor: calls.append(('eval', tensor)),
                                  synchronize=lambda: calls.append(('sync',)))
        materialize_parameters(zip(('a', 'b'), tensors), runtime,
                               lambda *args: progress.append(args))
        self.assertEqual(calls, [('eval', tensors[0]), ('sync',),
                                 ('eval', tensors[1]), ('sync',)])
        self.assertEqual(progress, [(0, 2, 'a', 8), (1, 2, 'b', 16), (2, 2, None, 0)])

    def test_materialization_failure_does_not_skip_a_tensor(self):
        from types import SimpleNamespace
        progress = []
        def fail(tensor):
            raise RuntimeError('GPU failure')
        runtime = SimpleNamespace(eval=fail, synchronize=lambda: None)
        with self.assertRaisesRegex(RuntimeError, 'GPU failure'):
            materialize_parameters([('a', SimpleNamespace(nbytes=8)),
                                    ('b', SimpleNamespace(nbytes=16))], runtime,
                                   lambda *args: progress.append(args))
        self.assertEqual(progress, [(0, 2, 'a', 8)])

    def test_inputs_are_reproducible_and_vocabulary_bounded(self):
        a = fixed_inputs([512, 2048, 4096], 201088, 42)
        self.assertEqual(a, fixed_inputs([512, 2048, 4096], 201088, 42))
        self.assertNotEqual(a, fixed_inputs([512, 2048, 4096], 201088, 43))
        for n, ids in a.items():
            self.assertEqual(len(ids), int(n))
            self.assertTrue(all(0 <= t < 201088 for t in ids))

    def test_exact_decode_intervals_and_timing(self):
        tokens = [2 + i * .05 for i in range(128)]
        result, intervals = metrics(512, 0, 1.9, tokens, 1024)
        self.assertAlmostEqual(result['decode_tokens_per_second'], 20)
        self.assertEqual(result['time_to_first_token_ms'], 2000)
        self.assertAlmostEqual(result['inter_token_latency_ms'], 50)
        self.assertEqual(len(intervals), 127)
        with self.assertRaises(ValueError):
            metrics(512, 0, 1.9, tokens[:-1], 1024)

    def test_environment_invalidity_is_retained(self):
        a = dict(power="Now drawing from 'AC Power'", swap_used_mib=100,
                 thermal="No thermal warning level has been recorded\nNo performance warning level has been recorded", power_settings="lowpowermode         0")
        self.assertEqual(environment_issues(a, dict(a)), [])
        b = dict(a, swap_used_mib=101, power="Battery Power", thermal="warning")
        issues = environment_issues(a, b)
        self.assertIn('swap grew during the request', issues)
        self.assertIn('AC power unavailable', issues)
        self.assertIn('thermal/performance observation changed', issues)

    def test_nominal_thermal_and_no_swap_activity_required(self):
        a = dict(power="Now drawing from 'AC Power'", swap_used_mib=100,
                 thermal="No thermal warning level has been recorded\nNo performance warning level has been recorded", power_settings="lowpowermode         0",
                 thermal_state=0, swapins=12, swapouts=4)
        self.assertEqual(environment_issues(a, dict(a)), [])
        issues=environment_issues(a, dict(a, thermal_state=1, swapins=13, swapouts=5))
        self.assertIn('native thermal state is not nominal', issues)
        self.assertIn('swapins increased during request', issues)
        self.assertIn('swapouts increased during request', issues)

    def test_aa_calibration_requires_complete_unbiased_pairs(self):
        from evaluator.baseline_validation import aa_noise
        rows=[dict(prompt_tokens=512,trial=i,lane=k,metrics={'decode_tokens_per_second':20})
              for i in range(1,21) for k in ('A','B')]
        self.assertTrue(aa_noise(rows,[512])['512']['passed'])
        with self.assertRaises(ValueError):
            aa_noise(rows[:-1],[512])
        for row in rows:
            if row['lane']=='B':
                row['metrics']['decode_tokens_per_second']=22
        self.assertFalse(aa_noise(rows,[512])['512']['passed'])


if __name__ == '__main__':
    unittest.main()
