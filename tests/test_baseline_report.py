import copy
import unittest
import json
import hashlib
import tempfile
from pathlib import Path

from evaluator.baseline_report import validate_baseline, validate_policy_review, validate_resident_chain, validate_ui_renderer_lease
from evaluator.native_baseline import stability_screen, swap_observation
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
        self.assertEqual(validate_policy_review(self.state, dict(per_request_control_approved=True,
            per_request_control_response='Approve the reviewed cadence')), [])

    def test_separate_screen_never_replaces_official_samples_and_is_revalidated(self):
        self.state['mode']['stability_screen'] = True
        self.state['driver_correctness'] = [dict(case='prefill', prompt_tokens=n, passed=True,
            actual_token_ids=[1]*128, expected_token_ids=[1]*128) for n in (512,2048,4096)] + [
            dict(case='decode',prompt_tokens=1,passed=True,actual_token_ids=[1]*128,expected_token_ids=[1]*128),
            dict(case='held_out_inputs',prompt_tokens=257,passed=True,actual_token_ids=[1]*128,expected_token_ids=[1]*128)]
        rows = copy.deepcopy([r for r in self.state['samples'] if r['trial'] <= 2])
        for row in rows:
            row['swap_activity'] = swap_observation(row['environment_before'], row['environment_after'], 'strict')
        screen = dict(excluded_from_baseline=True,status='passed',rows=rows,
            policy=dict(pairs_per_size=2,maximum_pair_ratio_deviation=.02,maximum_relative_mad=.01,maximum_relative_full_range=.04),
            controller_receipts=[{}]*6,quiet_preflights=self.state['quiet_preflights'][:6],
            summary=stability_screen(rows,(512,2048,4096)))
        self.state['stability_screen'] = screen
        self.assertEqual(validate_baseline(self.state,self.contract),[])
        screen['rows'][0]['metrics']['decode_tokens_per_second'] = 15
        self.assertIn('Separate predeclared stability-screen evidence incomplete or failed',validate_baseline(self.state,self.contract))
        screen['rows'][0]['metrics']['decode_tokens_per_second'] = 20
        self.state['samples'] = self.state['samples'][:-1]
        self.assertTrue(validate_baseline(self.state,self.contract))

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

    def test_paging_aware_requires_exact_human_approval(self):
        self.state['swap_policy'] = 'paging-aware'
        self.assertTrue(validate_policy_review(self.state, {}))
        self.assertTrue(validate_policy_review(self.state, {'swap_policy': 'paging-aware'}))
        self.assertEqual(validate_policy_review(self.state, {'swap_policy': 'paging-aware',
            'paging_aware_response': 'Approve paging-aware protocol'}), [])

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


class UiLeaseTests(unittest.TestCase):
    def test_worker_identity_full_coverage_and_restoration_required(self):
        executable = '/Applications/ChatGPT.app/Contents/Frameworks/Codex Framework.framework/Versions/test/Helpers/Codex (Renderer).app/Contents/MacOS/Codex (Renderer)'
        lease = dict(schema='kernelmax_ui_renderer_lease_v1', status='restored', errors=[],
            end_reason='worker_exited', worker=dict(pid=123), targets=[dict(pid=2,uid=501,executable=executable)],
            paused_at=5, paused=[dict(pid=2,at=5)], resumed=[dict(pid=2,at=25)])
        row = dict(environment_before=dict(observed_at=10),environment_after=dict(observed_at=20))
        state = dict(warmups=[row],samples=[row],resident_session=dict(worker_pid=123))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'lease.json'
            path.write_text(json.dumps(lease))
            review = dict(ui_renderer_lease=str(path))
            self.assertEqual(validate_ui_renderer_lease(state, review)['receipt'], lease)
            for change in (dict(worker=dict(pid=999)), dict(status='paused'),
                           dict(paused_at=11), dict(resumed=[dict(pid=2,at=19)]),
                           dict(targets=[dict(pid=2,uid=501,executable='/usr/bin/python')])):
                path.write_text(json.dumps(dict(lease, **change)))
                with self.assertRaises(ValueError):
                    validate_ui_renderer_lease(state, review)


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
