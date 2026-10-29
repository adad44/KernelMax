"""Protected baseline-only report writer; no candidate verdicts or acceptance."""
import argparse
import hashlib
import json
import math
import statistics
import shutil
from pathlib import Path

from evaluator.native_baseline import digest, environment_issues, stability_screen, swap_observation, verify_resident_inputs
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
                 'swap_policy', 'calibration_policy', 'stability_screen')
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


def validate_session_completion(s, root, service_receipt):
    session = json.loads((root / 'resident-session.json').read_text())
    resident = s['resident_session']
    if (session['status'] != 'completed_pending_review' or session['model_loads'] != 1 or
            session['session_id'] != resident['session_id'] or session['worker_pid'] != resident['worker_pid'] or
            session['failed_attempts'] != resident['retained_failed_attempts'] or
            Path(session['current_attempt']) != service_receipt.parent):
        raise ValueError('Resident session did not complete with one verified model load')
    receipt = json.loads(service_receipt.read_text())
    labels = {r['label'] for r in receipt['initially_loaded']}
    if (receipt.get('benchmark_returncode') != 0 or not receipt.get('restoration_complete') or
            receipt.get('restoration_errors') or
            {r['label'] for r in receipt['paused']} != labels or
            {r['label'] for r in receipt['restored']} != labels):
        raise ValueError('Temporary environment session restoration evidence incomplete')
    external = receipt.get('external_service_lease')
    if external:
        expected_labels = {'com.memoryos.scheduler', 'com.memoryos.daemon', 'com.memoryos.backend'}
        if (not external.get('unchanged') or
                set(external.get('loaded_before_session', {})) != expected_labels or
                set(external.get('loaded_after_session', {})) != expected_labels or
                any(external['loaded_before_session'].values()) or any(external['loaded_after_session'].values()) or
                digest(external['path']) != external['sha256']):
            raise ValueError('Persistent background-service pause evidence incomplete or changed')
    return session, receipt


def validate_ui_renderer_lease(s, review):
    path = review.get('ui_renderer_lease')
    if not path:
        return None
    lease = json.loads(Path(path).read_text())
    targets = lease.get('targets', [])
    pids = {entry['pid'] for entry in targets}
    rows = s['warmups'] + s['samples'] + s.get('stability_screen', {}).get('rows', [])
    first = min(row['environment_before']['observed_at'] for row in rows)
    last = max(row['environment_after']['observed_at'] for row in rows)
    prefix = '/Applications/ChatGPT.app/Contents/Frameworks/Codex Framework.framework/Versions/'
    suffix = '/Helpers/Codex (Renderer).app/Contents/MacOS/Codex (Renderer)'
    if (lease.get('schema') != 'kernelmax_ui_renderer_lease_v1' or
            lease.get('status') != 'restored' or lease.get('errors') or
            lease.get('end_reason') != 'worker_exited' or
            lease.get('worker', {}).get('pid') != s['resident_session']['worker_pid'] or
            not pids or len(pids) != len(targets) or
            any(entry.get('uid') != 501 or not entry['executable'].startswith(prefix)
                or not entry['executable'].endswith(suffix) for entry in targets) or
            {entry['pid'] for entry in lease.get('paused', [])} != pids or
            {entry['pid'] for entry in lease.get('resumed', [])} != pids or
            lease.get('paused_at', float('inf')) > first or
            any(entry['at'] > first for entry in lease['paused']) or
            any(entry['at'] < last for entry in lease['resumed'])):
        raise ValueError('UI-renderer lease identity, restoration or complete timing coverage failed')
    return {'source': str(path), 'sha256': digest(path), 'receipt': lease}


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
    if s.get('mode', {}).get('stability_screen'):
        screen = s.get('stability_screen', {})
        try:
            calculated = stability_screen(screen['rows'], spec['workload']['prompt_lengths'])
            if (screen.get('excluded_from_baseline') is not True or screen.get('status') != 'passed'
                    or calculated != screen.get('summary') or not all(r['passed'] for r in calculated.values())
                    or len(screen.get('controller_receipts', [])) != (12 if per_request else 6)
                    or len(screen.get('quiet_preflights', [])) != (12 if per_request else 6)
                    or screen.get('policy') != {'pairs_per_size': 2, 'maximum_pair_ratio_deviation': .02,
                                               'maximum_relative_mad': .01, 'maximum_relative_full_range': .04}):
                errors.append('Separate predeclared stability-screen evidence incomplete or failed')
            if any(q.get('quiet_seconds_required') != 10 or q.get('waited_seconds', 0) < 10
                    or q.get('swap_policy', 'strict') != policy or len(q.get('observations', [])) < 5
                    or any(o.get('issues') for o in q.get('observations', [])[-5:])
                    for q in screen.get('quiet_preflights', [])):
                errors.append('Stability-screen quiet preflight evidence failed')
            golden = {r['prompt_tokens']: r['expected_token_ids'] for r in driver if r['case'] == 'prefill'}
            for row in screen['rows']:
                if (row['output_token_ids'] != golden[row['prompt_tokens']]
                        or row.get('invalid_reasons') or environment_issues(row['environment_before'], row['environment_after'], policy)
                        or any(t != 0 for t in row.get('thermal_states_during_request', [1]))
                        or row.get('swap_activity') != swap_observation(row['environment_before'], row['environment_after'], policy)):
                    errors.append('Stability-screen correctness/environment evidence failed')
        except (KeyError, TypeError, ValueError):
            errors.append('Separate stability-screen evidence malformed')
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
    policy = s.get('swap_policy', 'strict')
    if review.get('swap_policy', 'strict') != policy:
        return ['Human review swap-policy identity mismatch']
    if policy == 'paging-aware' and review.get('paging_aware_response') != 'Approve paging-aware protocol':
        return ['Explicit human paging-aware protocol review missing']
    if s.get('mode', {}).get('stability_screen') and review.get('quiet_reference_response') != 'do what it takes to get our official baseline':
        return ['Updated quiet-reference procedure human review missing']
    if s.get('mode', {}).get('per_request_control') and (review.get('per_request_control_approved') is not True
            or not review.get('per_request_control_response')):
        return ['Per-request quiet-window protocol human review missing']
    return []


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
    if review.get('persistent_worker_response') != 'Approve persistent-worker protocol':
        issues.append('Explicit human persistent-worker review missing')
    if issues:
        raise ValueError('Official baseline gate failed: ' + '; '.join(issues))
    root = Path(review['resident_session_root'])
    session, service = validate_session_completion(s, root, args.attempt.parent / 'memoryos-session.json')
    ui_lease = validate_ui_renderer_lease(s, review)
    args.destination.mkdir(parents=True, exist_ok=False)
    (args.destination / 'raw.json').write_bytes(source)
    shutil.copytree(args.attempt.parent / 'protected_sources', args.destination / 'protected_sources')
    if (args.attempt.parent / 'dense_reference_diagnostic.json').exists():
        shutil.copy2(args.attempt.parent / 'dense_reference_diagnostic.json', args.destination / 'dense_reference_diagnostic.json')
    (args.destination / 'resident-session.json').write_bytes((root / 'resident-session.json').read_bytes())
    shutil.copy2(args.attempt.parent / 'memoryos-session.json', args.destination / 'memoryos-session.json')
    if service.get('external_service_lease'):
        shutil.copy2(service['external_service_lease']['path'], args.destination / 'background-service-lease-at-freeze.json')
    if ui_lease:
        shutil.copy2(ui_lease['source'], args.destination / 'ui-renderer-lease.json')
    failed_dir = args.destination / 'failed_attempts'
    if session['failed_attempts']:
        failed_dir.mkdir()
        for entry in session['failed_attempts']:
            shutil.copy2(entry['artifact'], failed_dir / (entry['run_id'] + '.json'))
    report = {'schema': 'kernelmax_official_reference_baseline_v1', 'official': True,
              'scope': 'Reference-only, synchronous batch-one MLX/Metal inference on pinned native checkpoint. Not a stock CLI speed claim or candidate acceptance.',
              'run_id': s['run_id'], 'contract_sha256': s['contract_sha256'],
              'driver_sha256': s['driver_sha256'], 'raw_sha256': hashlib.sha256(source).hexdigest(),
              'human_review': review, 'candidate_acceptance_enabled': False,
              'swap_policy': s['swap_policy'], 'swap_activity_summary': paging_summary(s),
              'resident_session': session, 'environment_session': service,
              'ui_renderer_control': ui_lease,
              'stability_screen': s.get('stability_screen'),
              'hardware': s['hardware'], 'runtime': s['runtime'], 'checkpoint_files': s['checkpoint_files'],
              'inputs_sha256': s['inputs_sha256'], 'metrics': {}, 'noise': s['aa_calibration'],
              'dense_reference_diagnostic': s.get('dense_reference_diagnostic'),
              'limitations': ['Correctness operator cases sample layers 0/12/23, not every possible tensor/input.',
                              'Native operator checks establish dispatch invariance, not independent matmul numerical accuracy.',
                              'Alternate dense BF16 reconstruction failed large-activation layer12 tolerance; preserved as unresolved diagnostic, not relabeled passing.',
                              'Thermal state sampled every 500 ms, not per-kernel temperature/frequency.',
                              'Each request group waits for ten untimed seconds without policy-disallowed swap activity or thermal/power changes. No samples are discarded or retried within an attempt; complete environment-invalid attempts may be retried with fresh IDs/caches/warmups/samples, and every failure is retained.',
                              'Additional untimed resident-state priming precedes the five contract warmups; all priming events are retained separately from the primary samples.',
                              'No privileged GPU attribution; OS compositor/background effects enter measured A/A noise.',
                              'Absolute speed is machine/session-specific; candidates require fresh alternating paired comparisons.',
                              'All original provisional tolerances and candidate acceptance remain unchanged.']}
    if s['swap_policy'] == 'paging-aware':
        report['scope'] += ' Paging-aware profile: system swap-ins are allowed and logged; swap-outs or swap-usage growth invalidate the complete attempt.'
        report['limitations'].append('This is not a zero-paging baseline. A/A noise gates quantify paired differential noise, not common-mode slowdowns; global swap counters cannot attribute paging to a process.')
    if s.get('stability_screen'):
        report['limitations'].append('A separate twelve-request stability screen preceded a fresh full collection on the same load. All screen/setup rows are excluded from official metrics; the short screen alone is not repeatability proof.')
    if ui_lease:
        report['limitations'].append('Exact same-user Codex UI-renderer processes were reversibly suspended across screen, warmups and measurements, then resumed automatically when the worker exited. Main app/service/CLI and WindowServer were not suspended. This reduces one observed contention source, not proof of complete CPU/GPU isolation; reproduce this environment control for comparisons.')
    if s.get('mode', {}).get('per_request_control'):
        report['limitations'].append('Every measured request, including the second lane of each A/A pair, received its own untimed controller/settling/ten-second quiet window. Future comparisons must use the same per-request cadence; this is not back-to-back serving throughput.')
        report['limitations'].append('IORegistry GPU-client counters did not advance even for the running worker in the retained diagnostic. They are unvalidated/stale on this setup and cannot prove GPU isolation or attribution.')
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
    lines += ['', 'Untimed resident-state priming, then five contract warmups; twenty primary A measurements and twenty identical-reference B calibration measurements per size; alternating AB/BA; seed 42; 128 outputs.',
              '', f"Declared swap policy: `{report['swap_policy']}`. Every request's paging counters remain in the raw evidence; aggregate paging counts are in `baseline.json`. Future comparisons must use the same declared profile.",
              '', 'Raw inputs, outputs, correctness, versions, source/binary/checkpoint hashes, telemetry and all timing intervals are in `raw.json`. Detailed median/p90/MAD, confidence intervals, drift, scope and limitations are in `baseline.json`.',
              '', 'The alternate dense-BF16 reconstruction failed the large-activation test at layer 12. That failed diagnostic is preserved in `dense_reference_diagnostic.json`; it is not a passing numerical accuracy claim. The contract reference is unchanged native MLX-LM. Operator checks verify native dispatch invariance, not independent matmul accuracy.',
              '', 'Candidate acceptance is still disabled. This reference is not an ACCEPTED optimization or proof of zero background GPU activity.', '']
    if service.get('external_service_lease'):
        lines += ['The three approved MemoryOS services remained paused across attempts and through report freeze under the preserved external lease. The supervisor did not restore those externally paused jobs. Restoration after freeze is recorded separately.', '']
    if ui_lease:
        lines += ['Codex UI-renderer helpers were reversibly paused for all measurements and automatically resumed after worker exit. Exact process identities, timing coverage and restoration are preserved in `ui-renderer-lease.json`. The app service/CLI and OS compositor were not paused; this is not complete isolation.', '']
    if s.get('mode', {}).get('per_request_control'):
        lines += ['Every one of the 120 measured requests had its own untimed settling/quiet window (135 official windows including warmups). This is controlled per-request latency/throughput, not back-to-back serving throughput. Keep this cadence in future comparisons.', '']
    (args.destination / 'README.md').write_text('\n'.join(lines))
    print(args.destination / 'README.md')


if __name__ == '__main__':
    main()
