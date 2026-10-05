"""Independent arithmetic/readback audit; does not import evaluator helpers."""
import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-6), (actual, expected)


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    return (ordered[(n - 1) // 2] + ordered[n // 2]) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    raw = args.attempt.read_bytes()
    state = json.loads(raw)
    assert state['status'] == 'completed_pending_review' and not state['errors']
    policy = state['swap_policy']
    assert policy in ('strict', 'paging-aware')
    resident = state['resident_session']
    failures = resident['retained_failed_attempts']
    assert resident['attempt_index'] == len(failures) + 1 <= 3
    assert resident['load_reused'] is bool(failures)
    assert resident['model_load_run_id'] == (failures[0]['run_id'] if failures else state['run_id'])
    seen = {state['run_id']}
    for index, entry in enumerate(failures, 1):
        prior_raw = Path(entry['artifact']).read_bytes()
        prior = json.loads(prior_raw)
        assert hashlib.sha256(prior_raw).hexdigest() == entry['sha256']
        assert prior['status'] == 'failed' and prior['failure_category'] == 'environment'
        assert prior['run_id'] == entry['run_id'] and prior['run_id'] not in seen
        assert all(e.startswith('EnvironmentInvalid: ') for e in prior['errors'])
        assert prior['resident_session']['session_id'] == resident['session_id']
        assert prior['resident_session']['worker_pid'] == resident['worker_pid']
        assert prior['resident_session']['attempt_index'] == index
        assert prior['resident_session']['retained_failed_attempts'] == failures[:index-1]
        for key in ('driver_sha256', 'contract_sha256', 'inputs_sha256', 'checkpoint_files',
                    'driver_correctness', 'expert_correctness', 'swap_policy', 'calibration_policy', 'stability_screen'):
            assert prior.get(key) == state.get(key)
        seen.add(prior['run_id'])
    lengths = state['contract']['workload']['prompt_lengths']
    seed = state['contract']['workload']['seed']
    generator = random.Random(seed)
    vocab = 201088
    expected_inputs = {str(n): [generator.randrange(vocab) for _ in range(n)] for n in lengths}
    assert expected_inputs == state['inputs']
    assert hashlib.sha256(json.dumps(expected_inputs, sort_keys=True).encode()).hexdigest() == state['inputs_sha256']
    stock = {r['prompt_tokens']: r['expected_token_ids'] for r in state['driver_correctness'] if r['case'] == 'prefill'}
    assert set(stock) == set(lengths)
    assert len(state['warmups']) == 15 and len(state['samples']) == 120
    per_request = bool(state.get('mode', {}).get('per_request_control'))
    assert len(state['controller_receipts']) == len(state['quiet_preflights']) == (135 if per_request else 75)
    assert all(state[k] for k in ('runtime_source_sha256', 'runtime_binary_sha256', 'evaluator_source_sha256'))
    diagnostic = state.get('stability_screen') if state.get('mode', {}).get('stability_screen') else None
    diagnostic_rows = []
    if diagnostic:
        assert diagnostic['status'] == 'passed' and diagnostic['excluded_from_baseline'] is True
        diagnostic_rows = diagnostic['rows']
        assert len(diagnostic_rows) == 12
        assert len(diagnostic['controller_receipts']) == len(diagnostic['quiet_preflights']) == (12 if per_request else 6)
        assert diagnostic['policy'] == dict(pairs_per_size=2,maximum_pair_ratio_deviation=.02,
                                           maximum_relative_mad=.01,maximum_relative_full_range=.04)
        for n in lengths:
            chosen = [r for r in diagnostic_rows if r['prompt_tokens'] == n]
            assert [(r['trial'], r['lane']) for r in chosen] == [(1,'A'),(1,'B'),(2,'B'),(2,'A')]
            values = [r['metrics']['decode_tokens_per_second'] for r in chosen]
            midpoint = median(values)
            mad = median([abs(v-midpoint) for v in values]) / midpoint
            spread = max(values)/min(values)-1
            ratios = [values[1]/values[0], values[2]/values[3]]
            saved = diagnostic['summary'][str(n)]
            assert values == saved['rates'] and ratios == saved['paired_b_over_a_ratios']
            close(saved['relative_mad'], mad)
            close(saved['relative_full_range'], spread)
            assert saved['passed'] and mad <= .01 and spread <= .04 and all(abs(v-1) <= .02 for v in ratios)
    elif state.get('mode', {}).get('stability_screen'):
        raise AssertionError('Required diagnostic screen missing')
    rows = state['warmups'] + state['samples'] + diagnostic_rows
    for row in rows:
        assert row['output_token_ids'] == stock[row['prompt_tokens']]
        assert row['cache_dtypes'] == ['mlx.core.bfloat16'] and row['last_logits_finite']
        intervals = row['inter_token_intervals_ms']
        assert len(intervals) == 127 and all(math.isfinite(v) and v > 0 for v in intervals)
        m = row['metrics']
        close(m['decode_tokens_per_second'], 127000 / sum(intervals))
        close(m['inter_token_latency_ms'], median(intervals))
        close(m['full_model_wall_time_ms'], m['time_to_first_token_ms'] + sum(intervals))
        assert m['time_to_first_token_ms'] >= row['prompt_tokens'] / m['prefill_tokens_per_second'] * 1000
        assert not row['invalid_reasons'] and set(row['thermal_states_during_request']) == {0}
        before, after = row['environment_before'], row['environment_after']
        assert before['swapouts'] == after['swapouts']
        if policy == 'strict':
            assert before['swapins'] == after['swapins']
        else:
            assert before['swapins'] <= after['swapins']
        activity = row['swap_activity']
        assert activity == {'policy': policy, 'swapins_permitted': policy == 'paging-aware',
                            'swapin_pages': after['swapins'] - before['swapins'],
                            'swapout_pages': 0,
                            'swap_growth_mib': after['swap_used_mib'] - before['swap_used_mib']}
        assert after['swap_used_mib'] <= before['swap_used_mib']
        assert before['thermal_state'] == after['thermal_state'] == 0
        assert before['thermal'] == after['thermal'] and before['power_settings'] == after['power_settings']
        assert "'AC Power'" in before['power'] and "'AC Power'" in after['power']
    report = json.loads(args.report.read_bytes()) if args.report else None
    if report:
        assert report['official'] is True and report['candidate_acceptance_enabled'] is False
        assert report['raw_sha256'] == hashlib.sha256(raw).hexdigest()
        assert report['run_id'] == state['run_id']
        assert report['swap_policy'] == policy
        assert report.get('stability_screen') == diagnostic
        if diagnostic:
            assert report['human_review']['quiet_reference_response'] == 'do what it takes to get our official baseline'
        ui = report.get('ui_renderer_control')
        if ui:
            ui_bytes = (args.report.parent / 'ui-renderer-lease.json').read_bytes()
            assert hashlib.sha256(ui_bytes).hexdigest() == ui['sha256']
            lease = json.loads(ui_bytes)
            assert lease == ui['receipt'] and lease['status'] == 'restored' and not lease['errors']
            assert lease['end_reason'] == 'worker_exited' and lease['worker']['pid'] == state['resident_session']['worker_pid']
            targets = {entry['pid'] for entry in lease['targets']}
            assert targets and targets == {entry['pid'] for entry in lease['paused']} == {entry['pid'] for entry in lease['resumed']}
            for target in lease['targets']:
                assert target['uid'] == 501 and target['executable'].startswith('/Applications/ChatGPT.app/Contents/Frameworks/Codex Framework.framework/Versions/')
                assert target['executable'].endswith('/Helpers/Codex (Renderer).app/Contents/MacOS/Codex (Renderer)')
            first = min(row['environment_before']['observed_at'] for row in rows)
            last = max(row['environment_after']['observed_at'] for row in rows)
            assert lease['paused_at'] <= first and all(entry['at'] <= first for entry in lease['paused'])
            assert all(entry['at'] >= last for entry in lease['resumed'])
        external = report['environment_session'].get('external_service_lease')
        if external:
            snapshot = args.report.parent / 'background-service-lease-at-freeze.json'
            assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == external['sha256']
            assert external['unchanged'] and not any(external['loaded_before_session'].values()) and not any(external['loaded_after_session'].values())
        if policy == 'paging-aware':
            assert report['human_review']['paging_aware_response'] == 'Approve paging-aware protocol'
        for phase in ('resident_preconditioning', 'warmups', 'samples'):
            phase_rows = state[phase]
            ins = [r['environment_after']['swapins'] - r['environment_before']['swapins'] for r in phase_rows]
            outs = [r['environment_after']['swapouts'] - r['environment_before']['swapouts'] for r in phase_rows]
            growth = [r['environment_after']['swap_used_mib'] - r['environment_before']['swap_used_mib'] for r in phase_rows]
            assert report['swap_activity_summary'][phase] == dict(requests=len(phase_rows),
                requests_with_swapins=sum(n > 0 for n in ins), total_swapin_pages=sum(ins),
                maximum_swapin_pages_per_request=max(ins), total_swapout_pages=sum(outs),
                maximum_swap_growth_mib=max(growth))
    result = {}
    for length in lengths:
        selected = [r for r in state['samples'] if r['prompt_tokens'] == length]
        a = [r for r in selected if r['lane'] == 'A']
        b = [r for r in selected if r['lane'] == 'B']
        assert sorted(r['trial'] for r in a) == sorted(r['trial'] for r in b) == list(range(1, 21))
        ratios = []
        for trial in range(1, 21):
            pair = [r for r in selected if r['trial'] == trial]
            assert [r['lane'] for r in pair] == (['A', 'B'] if trial % 2 else ['B', 'A'])
            by_lane = {r['lane']: r['metrics']['decode_tokens_per_second'] for r in pair}
            ratios.append(by_lane['B'] / by_lane['A'])
        rng = random.Random(seed + length)
        boot = sorted(median([ratios[int(rng.random() * 20)] for _ in range(20)]) for _ in range(10000))
        ci = [boot[249], boot[9749]]
        saved_noise = state['aa_calibration'][str(length)]
        assert ratios == saved_noise['paired_b_over_a_ratios']
        assert ci == saved_noise['bootstrap_95pct_ci'] and median(ratios) == saved_noise['median_ratio']
        assert ci[0] <= 1 <= ci[1] and ci[1] - ci[0] <= .04 and saved_noise['passed']
        if report:
            saved = report['metrics'][str(length)]
            for metric in a[0]['metrics']:
                values = sorted(r['metrics'][metric] for r in a)
                central = median(values)
                close(saved[metric]['median'], central)
                close(saved[metric]['p90'], values[17])
                close(saved[metric]['mad'], median([abs(v-central) for v in values]))
                assert saved[metric]['trials'] == 20
            close(saved['decode_drift_first5_to_last5_percent'],
                  100 * (median([r['metrics']['decode_tokens_per_second'] for r in a[-5:]]) /
                         median([r['metrics']['decode_tokens_per_second'] for r in a[:5]]) - 1))
        result[str(length)] = {'primary_trials': len(a), 'calibration_trials': len(b),
                               'median_decode_tokens_per_second': median([r['metrics']['decode_tokens_per_second'] for r in a]),
                               'paired_ratio_ci': ci}
    print(json.dumps({'independent_audit': 'passed', 'run_id': state['run_id'],
                      'checked_requests': len(rows), 'output_ids_match_stock': True,
                      'official_warmups': len(state['warmups']), 'official_samples': len(state['samples']),
                      'excluded_diagnostic_requests': len(diagnostic_rows),
                      'metrics': result}, indent=2))


if __name__ == '__main__':
    main()
