"""Diagnostic association only; incomplete attempts never certify a baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def cpu(snapshot):
    result = {'WindowServer': 0., 'Codex_Renderer': 0., 'Codex_Service': 0.}
    for line in snapshot.splitlines()[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        _, percentage, name = parts
        for key, pattern in [('WindowServer', '/WindowServer'),
                             ('Codex_Renderer', 'Codex (Renderer)'),
                             ('Codex_Service', 'Codex (Service)')]:
            if pattern in name:
                result[key] += float(percentage)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.attempt.read_bytes()
    state = json.loads(raw)
    rows = []
    reference = {n: statistics.median(r['metrics']['decode_tokens_per_second']
                    for r in state['warmups'] if r['prompt_tokens'] == n and r['trial'] <= 3)
                 for n in (512, 2048, 4096)}
    for phase in ('warmups', 'samples'):
        for row in state[phase]:
            after = row['environment_after']
            rows.append({'phase': phase, 'length': row['prompt_tokens'], 'trial': row['trial'],
                'lane': row['lane'], 'observed_at': after['observed_at'],
                'tps': row['metrics']['decode_tokens_per_second'],
                'relative_to_first_three_warmups': row['metrics']['decode_tokens_per_second'] / reference[row['prompt_tokens']],
                'cpu_snapshot_percentages': cpu(after['process_cpu_snapshot']),
                'paging': row['swap_activity'],
                'invalid_reasons': row['invalid_reasons']})
    groups = {}
    for name, busy in [('WindowServer_at_most_5pct', False), ('WindowServer_above_5pct', True)]:
        selected = [r for r in rows if (r['cpu_snapshot_percentages']['WindowServer'] > 5) == busy]
        groups[name] = {'requests': len(selected),
            'median_normalized_throughput': statistics.median(r['relative_to_first_three_warmups'] for r in selected),
            'per_length': {str(n): {'requests': len(xs := [r['tps'] for r in selected if r['length'] == n]),
                           'median_tps': statistics.median(xs) if xs else None} for n in (512, 2048, 4096)}}
    result = {'purpose': 'Diagnose interrupted timing; not an official or passing baseline',
        'attempt': str(args.attempt), 'attempt_sha256': hashlib.sha256(raw).hexdigest(),
        'run_id': state['run_id'], 'status': state['status'],
        'grouping_rule': 'Post-request ps snapshot WindowServer CPU above versus at most 5 percent; exploratory association, not a validity gate',
        'normalization': 'Within-size median of first three warmups; diagnostic denominator, not an official performance number',
        'groups': groups, 'rows': rows,
        'limits': ['Snapshots are not interval-integrated CPU/GPU attribution.',
                   'Association does not prove causality; a controlled display/controller intervention is required.',
                   'No row is removed, reused or relabeled as official.']}
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite diagnostic evidence')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'run_id': result['run_id'], 'groups': groups, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
