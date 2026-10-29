"""Describe all retained diagnostics without issuing an acceptance verdict."""
import collections
import json
from pathlib import Path
import random
import statistics as st

WORK = Path(__file__).resolve().parent


def interval(values):
    rng = random.Random(42)
    boot = sorted(st.median(rng.choices(values, k=len(values))) for _ in range(10000))
    return [boot[249], boot[9749]]


def main():
    state = json.loads((WORK/'state.json').read_text())
    result = {'status': state['status'], 'official': False, 'verdict': None,
              'acceptance_enabled': False, 'baseline': {}, 'oracle': {}, 'operators': {}}
    requests = state['requests']
    for length in (512, 2048, 4096):
        rows = [r for r in requests if r['label'].startswith(f'baseline-{length}-')]
        if not rows:
            continue
        result['baseline'][str(length)] = {
            'requests': len(rows),
            'decode_tokens_per_second': st.median(r['metrics']['decode_tokens_per_second'] for r in rows),
            'milliseconds_per_token': st.median(1000/r['metrics']['decode_tokens_per_second'] for r in rows),
            'mean_build_ms_per_token': st.median(sum(p['build_ms'] for p in r['step_times'])/127 for r in rows),
            'mean_eval_wait_ms_per_token': st.median(sum(p['eval_wait_ms'] for p in r['step_times'])/127 for r in rows),
            'mean_readback_ms_per_token': st.median(sum(p['readback_ms'] for p in r['step_times'])/127 for r in rows),
            'environment_valid': all(not r['environment_issues'] for r in rows)}
    for mode in ('oracle_pointwise', 'oracle_up_gate', 'oracle_down', 'oracle_experts'):
        pairs = []
        for pair in range(3):
            a = next((r for r in requests if r['label'] == f'{mode}-pair-{pair}-normal'), None)
            b = next((r for r in requests if r['label'] == f'{mode}-pair-{pair}-{mode}'), None)
            if a is None or b is None:
                continue
            a_ms = 1000/a['metrics']['decode_tokens_per_second']
            b_ms = 1000/b['metrics']['decode_tokens_per_second']
            pairs.append({'pair': pair, 'baseline_ms': a_ms, 'oracle_ms': b_ms,
                          'saved_ms': a_ms-b_ms, 'speedup': a_ms/b_ms,
                          'fraction_removed': 1-b_ms/a_ms,
                          'environment_valid': not a['environment_issues'] and not b['environment_issues'],
                          'exact_logits': a['max_abs_logit_error'] == b['max_abs_logit_error'] == 0})
        if pairs:
            f = st.median(p['fraction_removed'] for p in pairs)
            result['oracle'][mode] = {'pairs': pairs, 'median_saved_ms_per_token': st.median(p['saved_ms'] for p in pairs),
                                     'median_speedup': st.median(p['speedup'] for p in pairs),
                                     'median_removed_fraction': f,
                                     'conditional_amdahl_2x_stage_speedup': 1/(1-f/2),
                                     'interpretation': 'optimistic cached-output intervention on this trajectory; not a deployable candidate or a universal bound',
                                     'environment_valid': all(p['environment_valid'] for p in pairs)}
    operators = state['operators']
    by_kind = collections.defaultdict(dict)
    for row in operators:
        by_kind[row['kind']][row['pair']] = row
    for kind, rows in by_kind.items():
        result['operators'][kind] = {'blocks': len(rows),
                                    'median_ms_per_call': st.median(r['elapsed_ms']/r['calls'] for r in rows.values())}
        if kind != 'native':
            ids = sorted(set(rows)&set(by_kind['native']))
            ratios = [by_kind['native'][i]['elapsed_ms']/rows[i]['elapsed_ms'] for i in ids]
            result['operators'][kind].update({'paired_native_over_lane_median': st.median(ratios),
                                              'descriptive_bootstrap_95pct': interval(ratios)})
    serial_path = WORK/'serial-stages.json'
    if serial_path.exists():
        serial = json.loads(serial_path.read_text())
        stages = collections.defaultdict(float)
        pending = collections.defaultdict(float)
        for row in serial:
            stages[row['kind']] += (row['build_ms']+row['eval_ms'])/16
            pending[row['kind']] += row['pending_ms']/16
        request = next(r for r in requests if r['label'] == 'serial-stage-512')
        total = st.mean(p['total_ms'] for p in request['step_times'][:16])
        normal = result['baseline']['512']['milliseconds_per_token']
        result['serial'] = {'mean_first_16_instrumented_ms_per_token': total,
                            'uninstrumented_ms_per_token': normal, 'inflation_ratio': total/normal,
                            'stage_ms_per_token': dict(sorted(stages.items(), key=lambda p:-p[1])),
                            'pending_before_stage_ms_per_token': dict(pending),
                            'warning': 'synchronized serial attribution includes barrier effects; not the uninstrumented GPU critical path'}
    k = n = 2880
    layers = 24
    selected = 4
    packed = selected*k*n*(.5+1/32)
    expert_bytes = int(layers*3*packed)
    attention_bytes = layers*(2*k*4096 + 2*k*512)*2
    head_bytes = 201088*k*2
    result['logical_byte_accounting'] = {
        'expert_weights_and_scales_bytes_per_token': expert_bytes,
        'attention_projection_weights_bytes_per_token': attention_bytes,
        'lm_head_weights_bytes_per_token': head_bytes,
        'combined_bytes_per_token': expert_bytes+attention_bytes+head_bytes,
        'candidate1_projection_result_traffic_saved_bytes_per_token': layers*4*selected*n*2,
        'candidate2_projection_result_traffic_saved_bytes_per_token': layers*8*selected*n*2,
        'warning': 'analytical logical bytes, not measured DRAM traffic; intermediate estimate excludes bias-gather buffers, other tensors, caching and re-reads'}
    result['environment_issues'] = sorted({issue for r in requests for issue in r['environment_issues']} |
                                          {issue for r in state.get('operator_environment', []) for issue in r['issues']})
    result['source_and_checkpoint_unchanged'] = state.get('protected_unchanged') and state.get('checkpoint_identity_unchanged')
    (WORK/'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
