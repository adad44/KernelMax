"""Build an inspectable engineering readout from every retained profiling sample."""
from pathlib import Path
import collections
import hashlib
import json
import random
import statistics as st
import argparse

ROOT = Path(__file__).resolve().parent
WORK = ROOT/'controller-repair'


def bootstrap(values):
    rng = random.Random(42)
    medians = sorted(st.median(rng.choices(values, k=len(values))) for _ in range(10000))
    return [medians[249], medians[9749]]


def main():
    global WORK
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--output', type=Path, default=ROOT/'MEASUREMENTS.md')
    args = parser.parse_args()
    WORK = args.work.resolve()
    state = json.loads((WORK/'state.json').read_text())
    summary = json.loads((WORK/'summary.json').read_text())
    reqs = state['requests']
    blocks = state['operators']
    issues = summary['environment_issues']
    checks = state.get('real_operator_checks', [])
    compared_requests = [r for r in reqs if r['label']=='capture-512'
                         or r['label'].startswith('warmup-oracle_') or r['label'].startswith('oracle_')]
    audit = {
        'completed': state['status'] == 'completed_diagnostic',
        'requests_retained': len(reqs), 'timing_blocks_retained': len(blocks),
        'all_requests_have_128_tokens': all(len(r['tokens']) == 128 for r in reqs),
        'all_requests_have_127_decode_intervals': all(len(r['step_times']) == len(r['inter_token_intervals_ms']) == 127 for r in reqs),
        'all_operator_blocks_have_192_calls': all(r['calls'] == 192 for r in blocks),
        'real_operator_checks': len(checks), 'real_operator_checks_passed': sum(bool(r['passed']) for r in checks),
        'real_operator_max_abs_error': max((r['max_abs_error'] for r in checks), default=None),
        'sources_unchanged': state.get('protected_unchanged'),
        'checkpoint_identity_unchanged': state.get('checkpoint_identity_unchanged'),
        'distinct_candidate_modules': len({r['module'] for r in state.get('candidate_imports', [])}) == 2,
        'logit_comparison_requests': len(compared_requests),
        'all_logit_comparisons_exact': all(r['max_abs_logit_error'] == 0 for r in compared_requests) if compared_requests else None,
        'real_activation_hashes_match': all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() == r['sha256'] for r in state.get('real_cases', [])),
        'prepared_hashes_match': all(hashlib.sha256((WORK/n).read_bytes()).hexdigest() == h
                                     for n, h in json.loads((WORK/'prepared-sha256.json').read_text()).items()),
        'environment_issues': issues,
        'measurement_environment_policy': state.get('measurement_environment_policy', 'strict'),
        'aa_screen': state.get('aa_screen'),
    }
    measured = [r for r in reqs if not r['label'].startswith('warmup-')]
    audit['warmup_requests_retained'] = len(reqs)-len(measured)
    audit['measured_requests_with_environment_issues'] = sum(bool(r['environment_issues']) for r in measured)
    audit['operator_groups_with_environment_issues'] = sum(bool(r['issues']) for r in state.get('operator_environment', []))
    audit['measured_swap_activity'] = {
        'swapin_pages_total': sum(r['after'].get('swapins',0)-r['before'].get('swapins',0) for r in measured),
        'swapin_pages_max_request': max((r['after'].get('swapins',0)-r['before'].get('swapins',0) for r in measured), default=0),
        'swapout_pages_total': sum(r['after'].get('swapouts',0)-r['before'].get('swapouts',0) for r in measured),
        'swap_growth_mib_max_request': max((r['after']['swap_used_mib']-r['before']['swap_used_mib'] for r in measured), default=0),
    }
    postrun_path = WORK/'postrun-checkpoint-audit.json'
    if postrun_path.exists():
        postrun = json.loads(postrun_path.read_text())
        audit['independent_postrun_checkpoint_identity_matches'] = postrun['independent_postrun_identity_matches']
        audit['recording_bug_explanation'] = postrun['cause']
    paired = collections.defaultdict(dict)
    for row in blocks:
        paired[row['kind']][row['pair']] = row
    comparisons = {}
    for a, b in [('native','candidate1'), ('native','candidate2'),
                  ('up_gate_activation','candidate1_up_gate'),
                  ('up_gate_activation','candidate2_up_gate'),
                  ('up_gate_activation','raw_up_gate')]:
        ids = sorted(set(paired[a]) & set(paired[b]))
        if not ids:
            continue
        ratios = [paired[a][i]['elapsed_ms']/paired[b][i]['elapsed_ms'] for i in ids]
        differences = [(paired[a][i]['elapsed_ms']-paired[b][i]['elapsed_ms'])/192 for i in ids]
        comparisons[a+'_over_'+b] = {'pairs': len(ids), 'ratios': ratios,
                                     'median_ratio': st.median(ratios), 'descriptive_bootstrap_95pct': bootstrap(ratios),
                                     'median_saved_ms_per_call': st.median(differences)}
        strata = {}
        for parity in (0,1):
            selected = [paired[a][i]['elapsed_ms']/paired[b][i]['elapsed_ms'] for i in ids if i%2==parity]
            if selected:
                strata[str(parity)] = {'pairs': len(selected), 'median_ratio': st.median(selected),
                                       'descriptive_bootstrap_95pct': bootstrap(selected)}
        comparisons[a+'_over_'+b]['block_order_strata'] = strata
    audit['paired_comparisons'] = comparisons
    (WORK/'audit.json').write_text(json.dumps(audit, indent=2, sort_keys=True)+'\n')
    lines = ['# Decode bottleneck measurements', '',
             f"Status: **{state['status']}**. This is a diagnostic, not an acceptance benchmark or candidate verdict.", '',
             f"All {len(reqs)} requests and {len(blocks)} operator timing blocks are retained. No outliers were removed.", '',
             f"Environment policy: {audit['measurement_environment_policy']}. Flags, including warmups: **{', '.join(issues) if issues else 'none'}**. Of {len(measured)} non-warmup requests, {audit['measured_requests_with_environment_issues']} have boundary flags; {audit['operator_groups_with_environment_issues']} of {len(state.get('operator_environment', []))} operator groups have flags. Interpret each group's validity below. Warmups stay in the record and are excluded from the planned statistics.", '',
             '## Unchanged full-model decode', '',
             'One warmup and three measured requests per context; batch one, native weights, BF16 activations/cache, greedy 128-token generation and fresh caches. Model loading is excluded.', '',
             '| Context | Decode tok/s | ms/token | Host graph build ms/token | Eval/wait ms/token | Token readback ms/token | Valid |',
             '| ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for length, row in summary['baseline'].items():
        lines.append(f"| {length} | {row['decode_tokens_per_second']:.3f} | {row['milliseconds_per_token']:.3f} | {row['mean_build_ms_per_token']:.3f} | {row['mean_eval_wait_ms_per_token']:.3f} | {row['mean_readback_ms_per_token']:.4f} | {row['environment_valid']} |")
    if state.get('aa_screen'):
        lines += ['', 'A/A stability screen (excluded from baseline metrics): '+json.dumps(state['aa_screen'],sort_keys=True)]
    lines += ['', 'Measured-request swap activity: '+json.dumps(audit['measured_swap_activity'],sort_keys=True)+'. These are system-wide counters; they do not identify which process paged or establish that swap-ins had no timing cost.']
    lines += ['', 'Eval/wait includes scheduling, GPU execution and waiting. It is not a measurement of GPU idle gaps. The graph-build interval is measured wall time; these host boundaries are not hardware timestamps.', '',
              '## Optimistic stage-removal replays', '',
              'Context 512, three alternating paired requests per stage. Required upstream dependencies remain; cached outputs replace the removed computation. Every replay logit is checked against the capture. This is an optimistic intervention on one trajectory, not a deployable candidate or a universal upper bound. The conditional 2× estimate assumes additive stage cost.', '',
              '| Removed stage | Median saved ms/token | All paired savings ms/token | Median replay speedup | Conditional total speedup for 2× stage | Valid |',
              '| --- | ---: | --- | ---: | ---: | --- |']
    labels = {'oracle_pointwise':'Up/gate output biases + SwiGLU', 'oracle_up_gate':'Up/gate products + biases + SwiGLU',
              'oracle_down':'Down product + bias', 'oracle_experts':'Entire expert execution'}
    for mode, row in summary['oracle'].items():
        savings = ', '.join(f"{p['saved_ms']:.3f}" for p in row['pairs'])
        lines.append(f"| {labels[mode]} | {row['median_saved_ms_per_token']:.3f} | {savings} | {row['median_speedup']:.4f}× | {row['conditional_amdahl_2x_stage_speedup']:.4f}× | {row['environment_valid']} |")
    controls = [r for r in reqs if r['label'] == 'capture-control-512']
    if controls:
        control_ms = 1000/controls[0]['metrics']['decode_tokens_per_second']
        base_ms = summary['baseline']['512']['milliseconds_per_token']
        lines += ['', f"The normal-hook control took {control_ms:.3f} ms/token versus {base_ms:.3f} ms/token for the plain baseline. Their {control_ms-base_ms:+.3f} ms difference also includes run noise and logit retention; it is not a calibrated subtraction. Oracle savings include changed Python graph construction and dispatch work. The impossible cached-output dependency nodes can change scheduling and memory lifetime."]
    lines += ['', '## Candidate diagnosis on real activations', '',
              'Layers 0/12/23, eight evenly spread real decode steps per layer, 24 cases × eight cycles = 192 synchronized expert calls per block. Twenty alternating blocks. Candidate 1 includes its routing-ID conversion. These operator times include per-call host/synchronization overhead and cannot be multiplied into a full-model speedup. Isolated substages are non-additive: timing bias/activation alone pays synchronization costs that do not describe its incremental cost inside the larger graph.', '',
              '| Operator or region | Median ms/call |', '| --- | ---: |']
    for kind, row in summary['operators'].items():
        lines.append(f"| {kind} | {row['median_ms_per_call']:.4f} |")
    lines += ['', '| Paired comparison: reference / replacement | Median ratio | Descriptive bootstrap 95% interval | Median saved ms/call |',
              '| --- | ---: | --- | ---: |']
    for name, row in comparisons.items():
        lo, hi = row['descriptive_bootstrap_95pct']
        lines.append(f"| {name} | {row['median_ratio']:.4f}× | [{lo:.4f}, {hi:.4f}] | {row['median_saved_ms_per_call']:.4f} |")
    lines += ['', 'Block-order strata (even forward / odd reverse) are retained in audit.json. Inspect them before interpreting small pooled operator gains: a single native call first/last in a block can expose warmup/order effects even with counterbalancing. No order stratum is dropped or used to select a favourable result.']
    lines += ['', 'Candidate region functions are diagnostic extractions of each frozen implementation through its `activated` result, excluding only the unchanged final down call. Their source and hashes are retained. Regions and full outputs are checked before timing. Confidence intervals are descriptive across timing blocks, not an acceptance test or a task-quality result.']
    serial = summary.get('serial')
    if serial:
        lines += ['', '## Serialized stage diagnostic', '',
                  f"Synchronizing stage boundaries gave {serial['mean_first_16_instrumented_ms_per_token']:.3f} ms/token across the first 16 decode steps, versus {serial['uninstrumented_ms_per_token']:.3f} ms/token for the plain full-request baseline ({serial['inflation_ratio']:.3f}×). These durations expose serial costs and barrier effects; they are not the uninstrumented GPU critical path.", '',
                  '| Stage | Serial build + eval ms/token |', '| --- | ---: |']
        for name, ms in serial['stage_ms_per_token'].items():
            lines.append(f'| {name} | {ms:.3f} |')
        pending = sum(serial['pending_before_stage_ms_per_token'].values())
        lines += ['', f"An additional {pending:.3f} ms/token was spent evaluating pending graph work before leaf stages, including routing/reduction/residual work. The raw pending attribution and every event are in serial-stages.json."]
    lines += ['', '## Traffic and dispatch limits', '',
              'Logical weight references are about 3.70 GB per decode token: experts 1.269 GB, attention projections 1.274 GB, vocabulary projection 1.158 GB. Both candidates retain these weight values and logical reads. A simple projection-result traffic model estimates only 2.21 MB/token saved by candidate 1 and 4.42 MB by candidate 2, excluding bias-gather buffers. These are analytical byte counts, not measured DRAM traffic. Caches, re-reads and physical transactions remain unmeasured.', '',
              'No GPU timeline or bandwidth/occupancy/register counters were available: full Xcode/xctrace is absent. CPU eval/wait cannot establish dispatch bubbles. A megakernel remains a hypothesis; these measurements alone cannot prove that dispatch gaps justify it.', '',
              '## Integrity and evidence', '',
              f"Controller completed: {audit['completed']}; protected sources unchanged: {audit['sources_unchanged']}; checkpoint identity unchanged: {audit['checkpoint_identity_unchanged']}; exact-logit checks: {audit['all_logit_comparisons_exact']}; real operator checks: {audit['real_operator_checks_passed']}/{audit['real_operator_checks']}.", '',
              f'Raw evidence: [state.json]({WORK.name}/state.json), [summary.json]({WORK.name}/summary.json), [audit.json]({WORK.name}/audit.json), [serial stages]({WORK.name}/serial-stages.json), [measurement plan]({WORK.name}/PLAN.md). The interrupted preparation and first pass remain intact. No candidate was created, promoted or accepted; acceptance remains disabled.']
    if postrun_path.exists():
        lines += ['', f"The raw final checkpoint guard recorded false because the controller reused its checkpoint `expected` variable for a numerical expected-output array. An independent post-run dictionary/file-identity comparison returned **{postrun['independent_postrun_identity_matches']}**. The checkpoint did not change according to that comparison. The raw state is retained; see [independent audit]({WORK.name}/postrun-checkpoint-audit.json). This bookkeeping error is repaired in the quiet pass."]
    args.output.write_text('\n'.join(lines)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
