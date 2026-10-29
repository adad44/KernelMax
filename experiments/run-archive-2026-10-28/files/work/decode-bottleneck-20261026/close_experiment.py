"""Conservative final readout; retain all failed experiments and original data."""
import hashlib
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    names = ('controller-repair', 'quiet-pass', 'paging-aware-512', 'controlled-512')
    states = {name: read(ROOT/name/'state.json') for name in names}
    first = states['controller-repair']
    summary = read(ROOT/'controller-repair/summary.json')
    audit = read(ROOT/'controller-repair/audit.json')
    independent = read(ROOT/'controller-repair/postrun-checkpoint-audit.json')
    controlled = states['controlled-512']
    screen = [r for r in controlled['requests'] if r['label'].startswith('screen-')]
    host = {
        'scope': 'four environmental-boundary-valid unchanged-model screen requests; overall A/A screen failed',
        'median_ms_per_token': st.median(1000/r['metrics']['decode_tokens_per_second'] for r in screen),
        'median_mean_graph_build_ms_per_token': st.median(st.mean(t['build_ms'] for t in r['step_times']) for r in screen),
        'median_mean_eval_wait_ms_per_token': st.median(st.mean(t['eval_wait_ms'] for t in r['step_times']) for r in screen),
        'median_mean_readback_ms_per_token': st.median(st.mean(t['readback_ms'] for t in r['step_times']) for r in screen),
        'all_observed_thermal_states': sorted({t for r in screen for t in r['thermal_states_during_request']}),
        'swapin_pages_per_request': [r['after']['swapins']-r['before']['swapins'] for r in screen],
        'swapout_pages_per_request': [r['after']['swapouts']-r['before']['swapouts'] for r in screen],
    }
    result = {'status': 'stopped_inconclusive', 'official': False, 'verdict': None,
              'acceptance_enabled': False, 'new_candidate_created': False,
              'certified_baseline_modified': False, 'reliable_bottleneck_fraction_established': False,
              'dispatch_gaps_measured': False, 'dram_traffic_measured': False,
              'real_operator_checks_passed': audit['real_operator_checks_passed'],
              'real_operator_checks': audit['real_operator_checks'],
              'real_operator_max_abs_error': audit['real_operator_max_abs_error'],
              'exact_logit_replay_requests': audit['logit_comparison_requests'],
              'controlled_screen': controlled['aa_screen'], 'host_intervals': host,
              'inventory': {}, 'checkpoint_independent_first_pass_identity_matches': independent['independent_postrun_identity_matches']}
    for name, state in states.items():
        hashes = read(ROOT/name/'prepared-sha256.json')
        result['inventory'][name] = {
            'status': state['status'], 'requests_retained': len(state['requests']),
            'operator_blocks_retained': len(state['operators']),
            'aa_screen': state.get('aa_screen'), 'protected_sources_unchanged': state.get('protected_unchanged'),
            'raw_checkpoint_guard': state.get('checkpoint_identity_unchanged'),
            'prepared_hashes_match': all(hashlib.sha256((ROOT/name/file).read_bytes()).hexdigest()==expected for file, expected in hashes.items())}
    (ROOT/'experiment-status.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    lines = ['# Decode profiling: inconclusive timing, no new candidate', '',
             'The measured controls are not yet repeatable enough to support a bottleneck-based optimization choice. The existing certified baseline is unchanged. No new candidate, speedup, acceptance, PR or merge is produced by this diagnostic.', '',
             '## What stopped the clean comparison', '',
             'The first complete pass collected 43 native requests and 180 operator timing blocks. All timing groups were affected by strict paging/thermal flags, so its region durations and deletion budgets remain exploratory. Every sample is retained.', '',
             'A strict ten-second quiet gate then failed after 180 seconds, before any request: thermals were nominal and there were no recent swap-outs or swap growth, but small global swap-ins continued. The certified reference already uses a paging-aware policy; the initial diagnostic should have used that documented policy. Neither strict attempt was relabelled valid.', '',
             'A fresh paging-aware 512-context screen allowed logged global swap-ins, rejected swap-outs/growth and required nominal thermals. Its unchanged A/A pairs differed by 10.46% and 8.45%, failing the existing screen limits. A new intervention then matched the reference cadence: 12-second controller hold, transient display sleep, five-second settling and ten-second policy-quiet preflight before every request, plus 0.5-second thermal observations. No background service or renderer was paused.', '',
             'The controlled A/A screen still failed:', '',
             '| Pair | A decode tok/s | B decode tok/s | A/B ratio |',
             '| --- | ---: | ---: | ---: |']
    for pair in range(2):
        a = next(r for r in screen if r['label']==f'screen-{pair}-A')
        b = next(r for r in screen if r['label']==f'screen-{pair}-B')
        lines.append(f"| {pair+1} | {a['metrics']['decode_tokens_per_second']:.3f} | {b['metrics']['decode_tokens_per_second']:.3f} | {a['metrics']['decode_tokens_per_second']/b['metrics']['decode_tokens_per_second']:.5f} |")
    lines += ['', 'Both pair deviations exceeded the predeclared 2% limit (about +3.03% and -2.53%). Relative MAD was 1.285%, above the 1% limit; full range was 3.775%, within its 4% limit. All four requests passed their paging-aware environment boundaries and every observed thermal state was nominal. They recorded zero swap-outs, no swap growth, and global swap-in counts of 220/188/200/700 pages. These counters cannot identify which process paged.', '',
              'The screen stopped the run before deletion or candidate timings. It was not rerun again. This establishes remaining measurement instability; it does not establish its GPU, CPU, allocator or background-process cause.', '',
              '## Where decode time goes: what is actually measured', '',
              f"In the controlled screen, median total latency was {host['median_ms_per_token']:.2f} ms/token. Median per-request mean intervals were {host['median_mean_graph_build_ms_per_token']:.2f} ms for Python/model graph construction, {host['median_mean_eval_wait_ms_per_token']:.2f} ms for evaluation/wait, and {host['median_mean_readback_ms_per_token']:.4f} ms for token readback. Eval/wait includes C++ scheduling, GPU execution and waiting; it is not GPU kernel time or dispatch-gap time. These descriptive values do not override the failed A/A calibration.", '',
              'The first pass’s stage-by-stage synchronization expanded latency from about 42 to 101 ms/token (2.42×). Its attention, expert and vocabulary-projection durations cannot be treated as the original critical-path shares. No GPU timeline, bandwidth, occupancy or register counters were available; full Xcode/xctrace is absent.', '',
              '## How much could removal save?', '',
              '**No reliable headroom ceiling was established.** The following exact-output deletion replays all had invalid timing conditions. They are useful only for choosing what to confirm, not for selecting or accepting a candidate:', '',
              '| Optimistically removed work | Exploratory median saved ms/token | All three paired savings ms/token |',
              '| --- | ---: | --- |']
    labels = {'oracle_pointwise':'Up/gate biases + SwiGLU', 'oracle_up_gate':'Up/gate products + biases + SwiGLU',
              'oracle_down':'Down product + bias', 'oracle_experts':'Entire expert execution'}
    for key in ('oracle_pointwise','oracle_up_gate','oracle_down','oracle_experts'):
        row = summary['oracle'][key]
        values = ', '.join(f"{p['saved_ms']:.2f}" for p in row['pairs'])
        lines.append(f"| {labels[key]} | {row['median_saved_ms_per_token']:.2f} | {values} |")
    lines += ['', 'Each replay preserved upstream dependencies and every compared logit exactly. Cached-output deletion is an optimistic intervention on this fixed trajectory, not a deployable optimization or universal bound. Once a stable fraction f is measured, complete removal gives the additive Amdahl estimate 1/(1−f), while a stage speedup r gives 1/((1−f)+f/r). The current invalid fractions must not be used as validated ceilings.', '',
              '## Why the candidates have not established a gain', '',
              'Both frozen candidates passed all 96 supplementary full/region checks on real model activations with zero observed error. The complete first pass also checked exact logits in 29 capture/replay requests. Correctness on these inputs is not the missing evidence.', '',
              'Candidate 1 targets output biases plus an already compiled SwiGLU and retains all three matrix products. The exploratory real-input up/gate region ratio was 0.9915× (native/candidate), with its descriptive interval including one: there is no demonstrated region improvement. The small-target explanation is plausible, but the invalid timing prevents a quantitative causal conclusion.', '',
              'Candidate 2 fuses up/gate products and activation but retains both weight matrices and the native down product. The exploratory real-input region ratio was about 1.0402×, a small local effect that requires valid confirmation. It does not establish model-level improvement or explain the hardware cause of the prior near-ties. The earlier valid synthetic operator test still has intervals including one.', '',
              'Simple logical-byte accounting gives about 3.70 GB of weight references per token across experts, attention projections and vocabulary projection. The candidates save only an estimated 2.21/4.42 MB/token of projection-result traffic in that simplified model. These are analytical logical bytes, not measured DRAM traffic; they do not prove a bandwidth bottleneck. Dispatch savings remain unmeasured.', '',
              'A megakernel therefore has no evidence-backed justification yet. The next measurement work is explaining the remaining A/A variation and obtaining an analyzed GPU timeline; another speculative kernel would not resolve those gaps.', '',
              '## Evidence and integrity', '',
              '- [First complete exploratory pass](FIRST-PASS-MEASUREMENTS.md): every native request, operator block, real input, hash and failed condition.',
              '- [Controlled screen](CONTROLLED-SCREEN.md) and [raw controlled state](controlled-512/state.json): failed A/A screen with its fixed limits.',
              '- [Strict quiet-gate failure](quiet-pass/state.json) and [initial paging-aware screen failure](paging-aware-512/state.json).',
              '- [Machine-readable status](experiment-status.json) and [controlled protocol](controlled-512/PLAN.md).', '',
              'Protected source hashes and prepared controller hashes match. The first complete controller’s raw final checkpoint guard was false because its dictionary variable was overwritten by a numerical expected-output array. An independent file-identity comparison matched the original verified checkpoint identities; the raw result and [audit](controller-repair/postrun-checkpoint-audit.json) remain intact. Later controllers fixed the variable shadowing and their final checkpoint guards passed. The initial pre-execution candidate-import aliasing issue was also repaired and tested before any candidate comparison.', '',
              'All profiling workers and task-scoped sleep assertions have ended. The display was woken after the controlled screen stopped. Saved power settings, installed model/runtime, protected repository files and the existing certified baseline were not changed.']
    (ROOT/'CONCLUSION.md').write_text('\n'.join(lines)+'\n')
    old = ROOT/'RESULT.md'
    if not (ROOT/'PREPARATION.md').exists():
        (ROOT/'PREPARATION.md').write_text(old.read_text())
    old.write_text('# Latest profiling result\n\n**Stopped: timing inconclusive; no new candidate.**\n\nSee [final conclusion](CONCLUSION.md) and [machine-readable status](experiment-status.json). The earlier preparation status is preserved in [PREPARATION.md](PREPARATION.md).\n')
    (ROOT/'MEASUREMENTS.md').write_text('# Latest decode profiling readout\n\nSee [CONCLUSION.md](CONCLUSION.md). All earlier raw attempts and exploratory measurements remain retained.\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
