"""Protected baseline-only report writer; no candidate verdicts or acceptance."""
import argparse
import hashlib
import json
import math
import statistics
import shutil
from pathlib import Path

from evaluator.native_baseline import digest, environment_issues, swap_observation, verify_resident_inputs
from evaluator.schemas import load_contract
from evaluator.baseline_validation import aa_noise


def validate_resident_chain(s):
    """Bind fresh complete attempts to one verified resident load and all failures."""
    resident = s.get('resident_session')
    if not resident:
        return ['Resident session evidence missing']
    failures = resident.get('retained_failed_attempts', [])
    index = resident.get('attempt_index', 0)
    errors = []
    if (not 1 <= index <= 3 or len(failures) != index - 1 or
            not 0 <= resident.get('max_environment_retries', -1) <= 2 or
            index > resident['max_environment_retries'] + 1 or
            resident.get('load_reused') is not (index > 1)):
        errors.append('Resident retry count or load-reuse identity invalid')
    expected_load = failures[0]['run_id'] if failures else s['run_id']
    if resident.get('model_load_run_id') != expected_load:
        errors.append('Resident original verified load identity mismatch')
    ids = {s['run_id']}
    protected = ('contract_sha256', 'driver_sha256', 'runtime', 'checkpoint_files',
                 'inputs_sha256', 'runtime_source_sha256', 'runtime_binary_sha256',
                 'evaluator_source_sha256', 'model_support_sha256',
                 'checkpoint_stat_at_verification', 'driver_correctness', 'expert_correctness',
                 'swap_policy', 'calibration_policy')
    for prior_index, entry in enumerate(failures, 1):
        try:
            raw = Path(entry['artifact']).read_bytes()
            prior = json.loads(raw)
            r = prior['resident_session']
            if (hashlib.sha256(raw).hexdigest() != entry['sha256'] or
                    prior['run_id'] != entry['run_id'] or entry['run_id'] in ids or
                    prior['status'] != 'failed' or prior.get('failure_category') != 'environment' or
                    entry.get('failure_category') != 'environment' or
                    entry.get('attempt_index') != prior_index or r['attempt_index'] != prior_index or
                    r['session_id'] != resident['session_id'] or r['worker_pid'] != resident['worker_pid'] or
                    r['model_load_run_id'] != expected_load or
                    r['retained_failed_attempts'] != failures[:prior_index - 1] or
                    not prior.get('errors') or
                    any(not e.startswith('EnvironmentInvalid: ') for e in prior['errors']) or
                    any(prior.get(k) != s.get(k) for k in protected)):
                errors.append('Retained resident failure identity/hash/protected-input mismatch')
            ids.add(entry['run_id'])
        except (OSError, KeyError, TypeError, ValueError):
            errors.append('Retained resident failure evidence missing or malformed')
    return sorted(set(errors))


def validate_session_completion(s, root, session_receipt):
    session = json.loads((root / 'resident-session.json').read_text())
    resident = s['resident_session']
    if (session['status'] != 'completed_pending_review' or session['model_loads'] != 1 or
            session['session_id'] != resident['session_id'] or session['worker_pid'] != resident['worker_pid'] or
            session['failed_attempts'] != resident['retained_failed_attempts'] or
            Path(session['current_attempt']) != session_receipt.parent):
        raise ValueError('Resident session did not complete with one verified model load')
    receipt = json.loads(session_receipt.read_text())
    if (receipt.get('status') != 'benchmark_finished' or
            receipt.get('benchmark_pid') != resident['worker_pid'] or
            receipt.get('benchmark_returncode') != 0 or receipt.get('controller_returncode') != 0 or
            receipt.get('shutdown_complete') is not True or receipt.get('shutdown_errors') or
            receipt.get('external_service_lease') or receipt.get('paused')):
        raise ValueError('Collection supervisor did not finish cleanly')
    return session, receipt


def validate_baseline(s, contract):
    errors = []
    policy = s.get('swap_policy', 'strict')
    if policy not in ('strict', 'paging-aware'):
        return ['Unknown declared swap policy']
    if s.get('status') != 'completed_pending_review' or s.get('errors'):
        errors.append('Collection did not complete cleanly')
    if s.get('contract_sha256') != contract.sha256:
        errors.append('Contract identity mismatch')
    if not all(s.get('mode', {}).get(k) for k in ('validate_reference', 'paired_aa', 'controlled', 'rehash_checkpoint')):
        errors.append('Required validation/control mode absent')
    if s.get('primary_baseline_lane') != 'A':
        errors.append('Primary baseline lane must be predeclared A')
    spec = contract.to_dict()
    for row in s.get('checkpoint_files', {}).values():
        if row.get('observed_sha256') != row.get('sha256'):
            errors.append('Actual checkpoint hash not verified')
    if len(s.get('checkpoint_files', {})) != 3:
        errors.append('Expected three native checkpoint shards')
    driver = s.get('driver_correctness', [])
    if len(driver) != 5 or not all(r.get('passed') and r.get('actual_token_ids') == r.get('expected_token_ids') for r in driver):
        errors.append('Stock-generation correctness evidence incomplete')
    expert = s.get('expert_correctness', [])
    required = set(spec['correctness']['required_cases'])
    if len(expert) != 21 or not all(r.get('passed') for r in expert):
        errors.append('Expert correctness evidence incomplete or failed')
    for layer in (0, 12, 23):
        if {r['case'] for r in expert if r['layer'] == layer} != required:
            errors.append(f'Correctness coverage missing for layer {layer}')
    per_request = bool(s.get('mode', {}).get('per_request_control'))
    checkpoints = 135 if per_request else 75
    if len(s.get('controller_receipts', [])) != checkpoints:
        errors.append(f'Expected {checkpoints} official controller checkpoints')
    if s.get('mode', {}).get('stability_screen') or s.get('stability_screen'):
        errors.append('Legacy stability-screen attempts require their original frozen writer')
    quiet = s.get('quiet_preflights', [])
    if len(quiet) != checkpoints or any(
            x.get('quiet_seconds_required') != 10 or x.get('waited_seconds', 0) < 10
            or x.get('swap_policy', 'strict') != policy
            or len(x.get('observations', [])) < 5
            or any(o.get('issues') for o in x.get('observations', [])[-5:])
            for x in quiet):
        errors.append('Untimed quiet preflight evidence incomplete')
    priming = s.get('resident_preconditioning', [])
    if not (6 <= len(priming) <= 15 and len(priming) % 3 == 0) or any(
            not x.get('excluded_from_baseline') or x.get('environment_events')
            or environment_issues(x['environment_before'], x['environment_after'], policy)
            for x in priming[-6:]):
        errors.append('Two clean untimed resident-priming rounds missing')
    if priming and [x['prompt_tokens'] for x in priming] != spec['workload']['prompt_lengths'] * (len(priming) // 3):
        errors.append('Resident-priming workload coverage mismatch')
    for length in spec['workload']['prompt_lengths']:
        warm = [x for x in s['warmups'] if x['prompt_tokens'] == length]
        if len(warm) != spec['benchmark']['warmups']:
            errors.append(f'Warmup count mismatch: {length}')
        rows = [x for x in s['samples'] if x['prompt_tokens'] == length]
        for lane in ('A', 'B'):
            trials = [x['trial'] for x in rows if x['lane'] == lane]
            if sorted(trials) != list(range(1, 21)):
                errors.append(f'Trial count mismatch: {length}/{lane}')
        for trial in range(1, 21):
            lanes = [x['lane'] for x in rows if x['trial'] == trial]
            if lanes != (['A', 'B'] if trial % 2 else ['B', 'A']):
                errors.append(f'Pair ordering mismatch: {length}/{trial}')
    for x in s.get('warmups', []) + s.get('samples', []):
        if x.get('invalid_reasons') or environment_issues(x['environment_before'], x['environment_after'], policy):
            errors.append('Environment invalidation retained')
        if any(t != 0 for t in x.get('thermal_states_during_request', [1])):
            errors.append('Native thermal state not continuously nominal')
        if len(x['output_token_ids']) != 128 or len(x['inter_token_intervals_ms']) != 127:
            errors.append('Output/interval count mismatch')
        if any(not math.isfinite(v) or v <= 0 for v in x['metrics'].values()):
            errors.append('Invalid numerical measurement')
        if set(x['metrics']) != set(spec['benchmark']['required_metrics']):
            errors.append('Metric coverage mismatch')
    for x in priming + s.get('warmups', []) + s.get('samples', []):
        if policy == 'paging-aware' or 'swap_activity' in x:
            if x.get('swap_activity') != swap_observation(x['environment_before'], x['environment_after'], policy):
                errors.append('Declared paging activity missing or inconsistent')
    try:
        noise = aa_noise(s['samples'], spec['workload']['prompt_lengths'], spec['workload']['seed'])
        if noise != s.get('aa_calibration') or not all(v['passed'] for v in noise.values()):
            errors.append('Predeclared A/A calibration failed')
    except ValueError as error:
        errors.append(str(error))
    for path, sha in {**s.get('runtime_source_sha256', {}), **s.get('runtime_binary_sha256', {}),
                      **s.get('evaluator_source_sha256', {})}.items():
        if digest(path) != sha:
            errors.append(f'Protected source/runtime changed: {path}')
    return sorted(set(errors))


def validate_policy_review(s, review):
    """Record reviewed protocol options without embedding historical chat replies."""
    errors = []
    policy = s.get('swap_policy', 'strict')
    if review.get('swap_policy', 'strict') != policy:
        errors.append('Human review swap-policy identity mismatch')
    required = {'paging_aware': policy == 'paging-aware',
                'resident_retries': s.get('resident_session', {}).get('max_environment_retries', 0) > 0,
                'per_request_control': s.get('mode', {}).get('per_request_control', False)}
    for option, enabled in required.items():
        if enabled and review.get(option + '_approved') is not True:
            errors.append(f'Explicit human {option} protocol review missing')
    if review.get('ui_renderer_lease'):
        errors.append('Legacy app-suspension evidence is unsupported by this writer')
    return errors


def paging_summary(s):
    result = {}
    for phase in ('resident_preconditioning', 'warmups', 'samples'):
        rows = s[phase]
        deltas = [swap_observation(x['environment_before'], x['environment_after'], s['swap_policy']) for x in rows]
        result[phase] = {'requests': len(rows),
            'requests_with_swapins': sum(x['swapin_pages'] > 0 for x in deltas),
            'total_swapin_pages': sum(x['swapin_pages'] for x in deltas),
            'maximum_swapin_pages_per_request': max((x['swapin_pages'] for x in deltas), default=0),
            'total_swapout_pages': sum(x['swapout_pages'] for x in deltas),
            'maximum_swap_growth_mib': max((x['swap_growth_mib'] for x in deltas), default=0)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--human-review', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    source = args.attempt.read_bytes()
    s = json.loads(source)
    repo = Path(__file__).resolve().parents[1]
    contract = load_contract(repo / 'kernelmaxxing.yaml')
    issues = validate_baseline(s, contract)
    issues.extend(validate_resident_chain(s))
    verify_resident_inputs(s, repo / 'kernelmaxxing.yaml')
    review = json.loads(args.human_review.read_text())
    issues.extend(validate_policy_review(s, review))
    if review.get('approved') is not True or review.get('run_id') != s['run_id'] or review.get('driver_sha256') != s['driver_sha256']:
        issues.append('Explicit human reference-protocol review missing or wrong identity')
    if issues:
        raise ValueError('Official baseline gate failed: ' + '; '.join(issues))
    root = Path(review['resident_session_root'])
    session, receipt = validate_session_completion(s, root, args.attempt.parent / 'collection-session.json')
    args.destination.mkdir(parents=True, exist_ok=False)
    (args.destination / 'raw.json').write_bytes(source)
    shutil.copytree(args.attempt.parent / 'protected_sources', args.destination / 'protected_sources')
    if (args.attempt.parent / 'dense_reference_diagnostic.json').exists():
        shutil.copy2(args.attempt.parent / 'dense_reference_diagnostic.json', args.destination / 'dense_reference_diagnostic.json')
    (args.destination / 'resident-session.json').write_bytes((root / 'resident-session.json').read_bytes())
    shutil.copy2(args.attempt.parent / 'collection-session.json', args.destination / 'collection-session.json')
    failed_dir = args.destination / 'failed_attempts'
    if session['failed_attempts']:
        failed_dir.mkdir()
        for entry in session['failed_attempts']:
            shutil.copy2(entry['artifact'], failed_dir / (entry['run_id'] + '.json'))
    report = {'schema': 'kernelmax_official_reference_baseline_v2', 'official': True,
              'scope': 'Reference-only, synchronous batch-one MLX/Metal inference on pinned native checkpoint. Not a stock CLI speed claim or candidate acceptance.',
              'run_id': s['run_id'], 'contract_sha256': s['contract_sha256'],
              'driver_sha256': s['driver_sha256'], 'raw_sha256': hashlib.sha256(source).hexdigest(),
              'human_review': review, 'candidate_acceptance_enabled': False,
              'swap_policy': s['swap_policy'], 'swap_activity_summary': paging_summary(s),
              'resident_session': session, 'environment_session': receipt,
              'hardware': s['hardware'], 'runtime': s['runtime'], 'checkpoint_files': s['checkpoint_files'],
              'inputs_sha256': s['inputs_sha256'], 'metrics': {}, 'noise': s['aa_calibration'],
              'dense_reference_diagnostic': s.get('dense_reference_diagnostic'),
              'limitations': [
                  'Native operator checks sample layers 0/12/23 and verify dispatch invariance, not independent matmul accuracy.',
                  'Thermal state is sampled every 500 ms; background CPU/GPU isolation is not certified.',
                  'Priming requests are excluded; failed attempts and every measured sample are retained.',
                  'Speed is specific to this machine/session. Candidate comparisons require fresh paired measurements with the same driver/profile.',
                  'Candidate acceptance and numerical tolerances remain provisional.']}
    if s['swap_policy'] == 'paging-aware':
        report['limitations'].append('System swap-ins are allowed and logged; swap-outs or swap growth invalidate the attempt. Global counters cannot attribute paging to a process.')
    if s.get('dense_reference_diagnostic'):
        report['limitations'].append('The attached dense-BF16 diagnostic is excluded from passing native-reference correctness claims; inspect its retained failed cases.')
    if s.get('mode', {}).get('per_request_control'):
        report['limitations'].append('Each lane has its own untimed quiet window; these metrics describe controlled requests, not back-to-back serving.')
    for length in s['contract']['workload']['prompt_lengths']:
        rows = [r for r in s['samples'] if r['prompt_tokens'] == length and r['lane'] == 'A']
        values = {}
        for metric in rows[0]['metrics']:
            xs = sorted(r['metrics'][metric] for r in rows)
            median = statistics.median(xs)
            values[metric] = {'median': median, 'p90': xs[math.ceil(.9 * len(xs)) - 1],
                              'mad': statistics.median(abs(x - median) for x in xs), 'trials': len(xs)}
        ordered = [r['metrics']['decode_tokens_per_second'] for r in rows]
        values['decode_drift_first5_to_last5_percent'] = 100 * (statistics.median(ordered[-5:]) / statistics.median(ordered[:5]) - 1)
        report['metrics'][str(length)] = values
    (args.destination / 'baseline.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    lines = ['# Official reference baseline', '', report['scope'], '',
             '| Prompt tokens | Decode tok/s median | TTFT ms median | Prefill tok/s median | Peak GiB median |',
             '|---:|---:|---:|---:|---:|']
    for n, values in report['metrics'].items():
        lines.append(f"| {n} | {values['decode_tokens_per_second']['median']:.2f} | {values['time_to_first_token_ms']['median']:.1f} | {values['prefill_tokens_per_second']['median']:.2f} | {values['peak_memory_bytes']['median']/2**30:.2f} |")
    lines += ['', f"Swap policy: `{report['swap_policy']}`. Raw evidence is in `raw.json`; statistics, noise, protocol review and environment receipts are in `baseline.json`.",
              '', *('- ' + limitation for limitation in report['limitations']), '']
    (args.destination / 'README.md').write_text('\n'.join(lines))
    print(args.destination / 'README.md')


if __name__ == '__main__':
    main()
