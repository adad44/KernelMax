"""Protected baseline-only report writer; no candidate verdicts or acceptance."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from evaluator.native_baseline import digest, environment_issues
from evaluator.schemas import load_contract
from evaluator.baseline_validation import aa_noise


def validate_baseline(s, contract):
    errors = []
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
    if len(s.get('controller_receipts', [])) != 75:
        errors.append('Expected 15 warmup and 60 timed-pair controller checkpoints')
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
        if x.get('invalid_reasons') or environment_issues(x['environment_before'], x['environment_after']):
            errors.append('Environment invalidation retained')
        if any(t != 0 for t in x.get('thermal_states_during_request', [1])):
            errors.append('Native thermal state not continuously nominal')
        if len(x['output_token_ids']) != 128 or len(x['inter_token_intervals_ms']) != 127:
            errors.append('Output/interval count mismatch')
        if any(not math.isfinite(v) or v <= 0 for v in x['metrics'].values()):
            errors.append('Invalid numerical measurement')
        if set(x['metrics']) != set(spec['benchmark']['required_metrics']):
            errors.append('Metric coverage mismatch')
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
    review = json.loads(args.human_review.read_text())
    if review.get('approved') is not True or review.get('run_id') != s['run_id'] or review.get('driver_sha256') != s['driver_sha256']:
        issues.append('Explicit human reference-protocol review missing or wrong identity')
    if issues:
        raise ValueError('Official baseline gate failed: ' + '; '.join(issues))
    args.destination.mkdir(parents=True, exist_ok=False)
    (args.destination / 'raw.json').write_bytes(source)
    report = {'schema': 'kernelmax_official_reference_baseline_v1', 'official': True,
              'scope': 'Reference-only, synchronous batch-one MLX/Metal inference on pinned native checkpoint. Not a stock CLI speed claim or candidate acceptance.',
              'run_id': s['run_id'], 'contract_sha256': s['contract_sha256'],
              'driver_sha256': s['driver_sha256'], 'raw_sha256': hashlib.sha256(source).hexdigest(),
              'human_review': review, 'candidate_acceptance_enabled': False,
              'hardware': s['hardware'], 'runtime': s['runtime'], 'checkpoint_files': s['checkpoint_files'],
              'inputs_sha256': s['inputs_sha256'], 'metrics': {}, 'noise': s['aa_calibration'],
              'limitations': ['Correctness operator cases sample layers 0/12/23, not every possible tensor/input.',
                              'Thermal state sampled every 500 ms, not per-kernel temperature/frequency.',
                              'No privileged GPU attribution; OS compositor/background effects enter measured A/A noise.',
                              'Absolute speed is machine/session-specific; candidates require fresh alternating paired comparisons.',
                              'All original provisional tolerances and candidate acceptance remain unchanged.']}
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
    lines += ['', 'Five warmups; twenty primary A measurements and twenty identical-reference B calibration measurements per size; alternating AB/BA; seed 42; 128 outputs.',
              '', 'Raw inputs, outputs, correctness, versions, source/binary/checkpoint hashes, telemetry and all timing intervals are in `raw.json`. Detailed median/p90/MAD, confidence intervals, drift, scope and limitations are in `baseline.json`.',
              '', 'Candidate acceptance is still disabled. This reference is not an ACCEPTED optimization or proof of zero background GPU activity.', '']
    (args.destination / 'README.md').write_text('\n'.join(lines))
    print(args.destination / 'README.md')


if __name__ == '__main__':
    main()
