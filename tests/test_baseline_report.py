import copy
import unittest
import json
import hashlib
import tempfile
from pathlib import Path

from evaluator.baseline_report import validate_baseline, validate_policy_review, validate_resident_chain, validate_session_completion
from evaluator.native_baseline import swap_observation
from evaluator.baseline_validation import aa_noise
from evaluator.schemas import load_contract


class ReportGateTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_contract(Path(__file__).resolve().parents[1] / 'kernelmaxxing.yaml')
        thermal='No thermal warning level has been recorded\nNo performance warning level has been recorded'
        env=dict(power="'AC Power'",swap_used_mib=100,thermal=thermal,power_settings='lowpowermode         0',thermal_state=0,swapins=0,swapouts=0)
        metrics={k:20 for k in self.contract.to_dict()['benchmark']['required_metrics']}
        base=dict(environment_before=env,environment_after=env,invalid_reasons=[],thermal_states_during_request=[0],
                  output_token_ids=[1]*128,inter_token_intervals_ms=[50]*127,metrics=metrics)
        samples=[]
        warmups=[]
        for n in (512,2048,4096):
            for i in range(1,21):
                for lane in (('A','B') if i%2 else ('B','A')):
                    samples.append(dict(base,prompt_tokens=n,trial=i,lane=lane))
            warmups.extend(dict(base,prompt_tokens=n,trial=i,lane='A') for i in range(1,6))
        self.state=dict(status='completed_pending_review',errors=[],contract_sha256=self.contract.sha256,
                        mode=dict(validate_reference=True,paired_aa=True,controlled=True,rehash_checkpoint=True),
                        primary_baseline_lane='A',checkpoint_files={str(i):dict(sha256='a',observed_sha256='a') for i in range(3)},
                        driver_correctness=[dict(passed=True,actual_token_ids=[1]*128,expected_token_ids=[1]*128) for _ in range(5)],
                        expert_correctness=[dict(passed=True,layer=layer,case=case) for layer in (0,12,23)
                                            for case in self.contract.to_dict()['correctness']['required_cases']],
                        controller_receipts=[{}]*75,
                        quiet_preflights=[dict(quiet_seconds_required=10,waited_seconds=10,
                                               observations=[dict(issues=[])]*5)]*75,
                        resident_preconditioning=[dict(prompt_tokens=n,excluded_from_baseline=True,
                                                        environment_events=[],environment_before=env,
                                                        environment_after=env) for n in [512,2048,4096]*2],
                        warmups=warmups,samples=samples,
                        aa_calibration=aa_noise(samples,[512,2048,4096]))

    def test_full_structural_fixture_passes(self):
        self.assertEqual(validate_baseline(self.state,self.contract),[])

    def test_per_request_control_requires_all_135_windows(self):
        self.state['mode']['per_request_control'] = True
        self.assertTrue(validate_baseline(self.state,self.contract))
        self.state['controller_receipts'] = [{}]*135
        self.state['quiet_preflights'] = self.state['quiet_preflights'] + self.state['quiet_preflights'][:60]
        self.assertEqual(validate_baseline(self.state,self.contract),[])
        self.assertTrue(validate_policy_review(self.state, {}))
        self.assertEqual(validate_policy_review(self.state, dict(per_request_control_approved=True)), [])

    def test_legacy_screen_attempt_cannot_silently_drop_diagnostic_evidence(self):
        self.state['mode']['stability_screen'] = True
        self.assertIn('Legacy stability-screen attempts require their original frozen writer',
                      validate_baseline(self.state, self.contract))

    def paging_fixture(self):
        self.state = copy.deepcopy(self.state)
        self.state['swap_policy'] = 'paging-aware'
        for x in self.state['quiet_preflights']:
            x['swap_policy'] = 'paging-aware'
        for x in self.state['resident_preconditioning'] + self.state['warmups'] + self.state['samples']:
            x['swap_activity'] = swap_observation(x['environment_before'], x['environment_after'], 'paging-aware')
        row = self.state['samples'][0]
        row['environment_after'] = dict(row['environment_after'], swapins=4)
        row['swap_activity'] = swap_observation(row['environment_before'], row['environment_after'], 'paging-aware')
        return row

    def test_paging_aware_profile_requires_complete_logs(self):
        row = self.paging_fixture()
        self.assertEqual(validate_baseline(self.state, self.contract), [])
        row.pop('swap_activity')
        self.assertIn('Declared paging activity missing or inconsistent', validate_baseline(self.state, self.contract))

    def test_paging_aware_never_allows_swapouts_or_growth(self):
        for key, value in (('swapouts', 1), ('swap_used_mib', 101)):
            self.setUp()
            row = self.paging_fixture()
            row['environment_after'][key] = value
            row['swap_activity'] = swap_observation(row['environment_before'], row['environment_after'], 'paging-aware')
            self.assertIn('Environment invalidation retained', validate_baseline(self.state, self.contract))

    def test_paging_aware_requires_explicit_human_approval(self):
        self.state['swap_policy'] = 'paging-aware'
        self.assertTrue(validate_policy_review(self.state, {}))
        self.assertTrue(validate_policy_review(self.state, {'swap_policy': 'paging-aware'}))
        self.assertEqual(validate_policy_review(self.state, {'swap_policy': 'paging-aware',
            'paging_aware_approved': True}), [])

    def test_failed_correctness_blocks_freeze(self):
        self.state['expert_correctness'][0]['passed']=False
        self.assertIn('Expert correctness evidence incomplete or failed',validate_baseline(self.state,self.contract))

    def test_missing_quiet_preflight_blocks_freeze(self):
        self.state['quiet_preflights'] = []
        self.assertIn('Untimed quiet preflight evidence incomplete', validate_baseline(self.state,self.contract))

    def test_paging_in_final_priming_rounds_blocks_freeze(self):
        self.state['resident_preconditioning'][-1]['environment_events'] = ['swapins increased during request']
        self.assertIn('Two clean untimed resident-priming rounds missing', validate_baseline(self.state,self.contract))

    def test_bad_pair_order_blocks_freeze(self):
        self.state['samples'][0]['lane']='B'
        errors=validate_baseline(self.state,self.contract)
        self.assertTrue(any('Pair ordering' in x for x in errors))

    def test_environment_event_blocks_freeze(self):
        self.state['samples'][0]=copy.deepcopy(self.state['samples'][0])
        self.state['samples'][0]['environment_after']=dict(self.state['samples'][0]['environment_after'])
        self.state['samples'][0]['environment_after']['swapouts']=1
        self.assertIn('Environment invalidation retained',validate_baseline(self.state,self.contract))

    def test_missing_control_or_hash_blocks_freeze(self):
        self.state['controller_receipts']=[]
        self.state['checkpoint_files']['0'].pop('observed_sha256')
        errors=validate_baseline(self.state,self.contract)
        self.assertIn('Actual checkpoint hash not verified',errors)
        self.assertTrue(any('controller checkpoints' in x for x in errors))


class SessionCompletionTests(unittest.TestCase):
    def test_session_and_supervisor_must_identify_the_same_finished_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resident = dict(session_id='session', worker_pid=123, retained_failed_attempts=[])
            state = dict(resident_session=resident)
            session = dict(status='completed_pending_review', model_loads=1,
                session_id='session', worker_pid=123, failed_attempts=[], current_attempt=str(root))
            (root / 'resident-session.json').write_text(json.dumps(session))
            receipt = dict(status='benchmark_finished', benchmark_pid=123,
                benchmark_returncode=0, controller_returncode=0, shutdown_complete=True)
            path = root / 'collection-session.json'
            path.write_text(json.dumps(receipt))
            self.assertEqual(validate_session_completion(state, root, path), (session, receipt))
            for change in (dict(benchmark_pid=999), dict(controller_returncode=1),
                           dict(status='session_error'), dict(shutdown_complete=False),
                           dict(shutdown_errors=['worker could not stop']),
                           dict(external_service_lease={'unchanged': True})):
                with self.subTest(change=change):
                    path.write_text(json.dumps(dict(receipt, **change)))
                    with self.assertRaises(ValueError):
                        validate_session_completion(state, root, path)
            path.write_text(json.dumps(receipt))
            (root / 'resident-session.json').write_text(json.dumps(dict(session, model_loads=2)))
            with self.assertRaises(ValueError):
                validate_session_completion(state, root, path)

    def test_protocol_reviews_require_booleans_and_reject_legacy_app_controls(self):
        state = dict(swap_policy='paging-aware', mode=dict(per_request_control=True),
                     resident_session=dict(max_environment_retries=2))
        review = dict(swap_policy='paging-aware', paging_aware_approved=True,
                      per_request_control_approved=True,
                      resident_retries_approved=True)
        self.assertEqual(validate_policy_review(state, review), [])
        for key in ('paging_aware', 'per_request_control', 'resident_retries'):
            self.assertTrue(validate_policy_review(state, dict(review, **{key + '_approved': 'yes'})))
        self.assertTrue(validate_policy_review(state, dict(review, ui_renderer_lease='legacy.json')))


class ResidentChainTests(unittest.TestCase):
    def test_single_verified_load_without_retry(self):
        state = dict(run_id='first', resident_session=dict(session_id='session', worker_pid=1,
            attempt_index=1, model_load_run_id='first', load_reused=False,
            max_environment_retries=2, retained_failed_attempts=[]))
        self.assertEqual(validate_resident_chain(state), [])
        state['resident_session']['load_reused'] = True
        self.assertTrue(validate_resident_chain(state))

    def test_retry_requires_unchanged_environment_only_failure_and_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            prior = dict(run_id='first', status='failed', failure_category='environment',
                errors=['EnvironmentInvalid: swapins'], driver_sha256='frozen',
                resident_session=dict(session_id='session', worker_pid=1, attempt_index=1,
                    model_load_run_id='first', retained_failed_attempts=[]))
            path = Path(temporary) / 'attempt.json'
            raw = json.dumps(prior).encode()
            path.write_bytes(raw)
            entry = dict(run_id='first', artifact=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                         failure_category='environment', attempt_index=1)
            state = dict(run_id='second', driver_sha256='frozen',
                resident_session=dict(session_id='session', worker_pid=1, attempt_index=2,
                    model_load_run_id='first', load_reused=True, max_environment_retries=2,
                    retained_failed_attempts=[entry]))
            self.assertEqual(validate_resident_chain(state), [])
            for key, value in (('driver_sha256', 'changed'), ('failure_category', 'gpu'),
                               ('errors', ['RuntimeError: correctness failure'])):
                changed = dict(prior, **{key: value})
                path.write_text(json.dumps(changed))
                entry['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertTrue(validate_resident_chain(state), key)
            path.write_bytes(raw + b' ')
            entry['sha256'] = hashlib.sha256(raw).hexdigest()
            self.assertTrue(validate_resident_chain(state), 'tampered artifact hash')


if __name__ == '__main__':
    unittest.main()
