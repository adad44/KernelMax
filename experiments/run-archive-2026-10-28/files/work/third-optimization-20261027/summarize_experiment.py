"""Read all retained candidate-3 evidence; never issue a model verdict."""
import hashlib
import json
from pathlib import Path
import statistics as st

WORK=Path(__file__).resolve().parent


def main():
    approval=json.loads((WORK/'approval.json').read_text())
    pipeline=json.loads((WORK/'pipeline.json').read_text())
    tiny=json.loads((WORK/'tiny-native-checks.json').read_text())
    reports=list((WORK/'native-correctness/artifacts').glob('*/report.json'))
    report=json.loads(reports[0].read_text()) if len(reports)==1 else None
    native_checks=[]
    if report:
        for row in report['correctness']:
            try:
                detail=json.loads(row['detail'])
            except (ValueError,TypeError):
                continue
            if isinstance(detail,dict):
                native_checks.extend(detail.get('checks',[]))
    op_path=WORK/'operator-diagnostic.json'
    op=json.loads(op_path.read_text()) if op_path.exists() else None
    result={'candidate_id':approval['candidate_id'],'snapshot_sha256':approval['snapshot_sha256'],
            'pipeline_status':pipeline['status'],'pipeline_errors':pipeline['errors'],
            'acceptance_enabled':False,'verdict':None,'full_model_speedup_established':False,
            'committed':False,'pushed':False,'tiny_checks':len(tiny['checks']),
            'tiny_checks_passed':sum(r['passed'] for r in tiny['checks']),
            'native_status':report['status'] if report else None,
            'native_checks':len(native_checks),
            'native_checks_passed':sum((r.get('result') or {}).get('passed',False) for r in native_checks),
            'native_errors':report['errors'] if report else [],
            'native_report':str(reports[0]) if report else None,
            'native_report_sha256':hashlib.sha256(reports[0].read_bytes()).hexdigest() if report else None,
            'prepared_sources_unchanged':all(hashlib.sha256((WORK/n).read_bytes()).hexdigest()==sha
                                            for n,sha in json.loads((WORK/'prepared-sha256.json').read_text()).items())}
    if op:
        result.update({'operator_status':op['status'],'operator_errors':op['errors'],
                       'real_decode_checks':len(op['additional_correctness']),
                       'real_decode_checks_passed':sum(r['passed'] for r in op['additional_correctness']),
                       'paired_samples_retained':len(op['samples']),
                       'all_sample_call_counts_256':all(r['calls']==256 for r in op['samples']),
                       'environment_issues':op['environment_issues'],
                       'protected_sources_unchanged':op['protected_sources_unchanged'],
                       'checkpoint_identity_unchanged':op['checkpoint_identity_unchanged'],
                       'snapshot_unchanged':op['snapshot_unchanged'],
                       'operator_summaries':op['summaries']})
        frontend={}
        for layer in (0,12,23):
            rows={r['lane']:r for r in op['instrumented_frontend'] if r['layer']==layer}
            if len(rows)==2:
                frontend[str(layer)]={'baseline_mean_build_ms':rows['baseline']['mean_build_ms'],
                                      'candidate_mean_build_ms':rows['candidate']['mean_build_ms'],
                                      'baseline_mean_eval_wait_ms':rows['baseline']['mean_eval_wait_ms'],
                                      'candidate_mean_eval_wait_ms':rows['candidate']['mean_eval_wait_ms'],
                                      'environment_valid':all(not r['environment_issues'] for r in rows.values())}
        result['instrumented_frontend_summaries']=frontend
        if len(frontend)==3:
            result['hypothesis_assessment']=(
                'No consistent reduction in instrumented graph-construction cost; both lanes remained around '
                '9–10 microseconds per expert call. Compiling the complete graph did not deliver the intended '
                'frontend saving in this operator experiment. The exact internal compiler/hardware cause is unmeasured.'
                if not all(r['candidate_mean_build_ms']<r['baseline_mean_build_ms'] for r in frontend.values())
                else 'Instrumented construction cost decreased at all sampled layers; paired wall time remains the performance evidence.')
        for check in op['additional_correctness']:
            if hashlib.sha256(Path(check['arrays']).read_bytes()).hexdigest()!=check['arrays_sha256']:
                raise ValueError('Saved numerical evidence changed')
        valid=(op['status']=='completed_diagnostic' and not op['errors'] and not op['environment_issues']
               and op['protected_sources_unchanged'] and op['checkpoint_identity_unchanged'] and op['snapshot_unchanged'])
        intervals=[r['descriptive_bootstrap_95pct'] for r in op['summaries'].values()]
        if valid and len(intervals)==3 and all(v[0]>1 for v in intervals):
            disposition='Consistent expert-operator improvement on sampled real inputs; full-model integration and performance remain untested.'
        elif not valid:
            disposition='Inconclusive diagnostic; preserve errors/environment/integrity flags. No promotion.'
        else:
            disposition='No consistent expert-operator improvement established across sampled layers. Park candidate; no promotion.'
    else:
        disposition='Correctness or pipeline did not complete; operator timing not established.'
    result['research_disposition']=disposition
    (WORK/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    lines=['# Candidate 3: complete compiled native expert graph','',disposition,'',
           f"Candidate: `{approval['candidate_id']}`; frozen SHA256 `{approval['snapshot_sha256']}`.",'',
           'The hypothesis compiles all three original native MXFP4 products, their output biases and GPT-OSS SwiGLU as one graph. Weights/scales/biases are explicit function arguments; routing, precision and prefill stay unchanged. This targets graph-construction overhead, not a custom matrix kernel or a megakernel. The earlier full-model profiling screen remains failed and is not reused as valid timing evidence.','',
           f"Small fixture checks: {result['tiny_checks_passed']}/{result['tiny_checks']}. Protected checkpoint checks: {result['native_checks_passed']}/{result['native_checks']}.",'']
    if op:
        lines += [f"Real decode comparisons, including both ID types: {result['real_decode_checks_passed']}/{result['real_decode_checks']}. Paired timing samples retained: {len(op['samples'])}.",'',
                  'Twenty alternating AB/BA pairs per layer, eight actual decode cases repeated 32 times: 256 synchronized calls per lane. Five warmup cycles per lane. Model loading and native byte verification are outside timing. The pipeline ran as a temporary launchd Interactive job, equally for baseline and candidate. These are expert-operator wall times, not full-model tokens/s.','',
                  '| Layer | Baseline ms / 256 calls | Candidate ms / 256 calls | Paired median ratio | Descriptive bootstrap 95% | Valid |',
                  '| ---: | ---: | ---: | ---: | --- | --- |']
        for layer,row in op['summaries'].items():
            lo,hi=row['descriptive_bootstrap_95pct']
            lines.append(f"| {layer} | {row['median_baseline_ms']:.3f} | {row['median_candidate_ms']:.3f} | {row['median_ratio']:.4f}× | [{lo:.4f}, {hi:.4f}] | {row['environment_valid']} |")
        lines += ['', 'All samples and order strata are retained in [operator-diagnostic.json](operator-diagnostic.json). Ratios are baseline/candidate; above one means faster candidate. Intervals are descriptive and are not a candidate acceptance verdict. No outliers were removed and no failed statistical result was rerun.','',
                  '| Layer | Baseline graph build ms/call | Candidate graph build ms/call | Instrumented environment valid |',
                  '| ---: | ---: | ---: | --- |']
        for layer,row in result['instrumented_frontend_summaries'].items():
            lines.append(f"| {layer} | {row['baseline_mean_build_ms']:.5f} | {row['candidate_mean_build_ms']:.5f} | {row['environment_valid']} |")
        lines += ['', 'Graph-build measurements are from a separate instrumented pass, excluded from paired wall-time samples. Eval/wait includes dispatch and GPU work; no GPU hardware timeline or DRAM counter is claimed.','',
                  result.get('hypothesis_assessment','Frontend interpretation not available.'),'',
                  'Environment policy: AC, nominal thermals, no swap-outs or swap-usage growth, with global swap-ins permitted and retained under the certified reference paging-aware policy. Flags: '+(', '.join(op['environment_issues']) if op['environment_issues'] else 'none')+'.','',
                  f"Protected source hashes unchanged: {op['protected_sources_unchanged']}; checkpoint file identities unchanged: {op['checkpoint_identity_unchanged']}; frozen snapshot unchanged: {op['snapshot_unchanged']}.",'']
    lines += ['No full-model speedup, held-out quality result or acceptance is established. Acceptance remains disabled. Nothing was committed or pushed, per the user’s instruction.','',
              'Evidence: [plan](PLAN.md), [pipeline](pipeline.json), [small checks](tiny-native-checks.json), [native summary](native-correctness-summary.json), [machine-readable result](result.json), [controller call-count verification](call-count-verification.json). All earlier attempts remain intact.']
    (WORK/'RESULT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
