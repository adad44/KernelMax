"""Retain and audit experiment evidence; no evaluator verdict or promotion."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path('/Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop')
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from orchestrator.candidate_manager import FrozenCandidate,validate_candidate,verify_frozen_candidate


def main():
    approval=json.loads((WORK/'approval.json').read_text())
    candidate=validate_candidate(approval['snapshot'],Path(approval['snapshot']).parent)
    files={p.relative_to(candidate.directory).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for p in candidate.directory.rglob('*') if p.is_file()}
    frozen=FrozenCandidate(**vars(candidate),sha256=approval['snapshot_sha256'],file_sha256=files)
    verify_frozen_candidate(frozen)
    for name,digest in files.items():
        assert hashlib.sha256((ROOT/'candidates'/candidate.candidate_id/name).read_bytes()).hexdigest()==digest
    paths=list((WORK/'native-correctness/artifacts').glob('execution-*/report.json'))
    assert len(paths)==1, 'Retain all attempts; do not silently select a report'
    native_path=paths[0]
    native=json.loads(native_path.read_text())
    assert native['candidate']['sha256']==frozen.sha256
    assert not native['performance'] and native['verdict'] is None
    checks=[c for row in native['correctness'] for c in json.loads(row['detail'])['checks']]
    tiny=json.loads((WORK/'synthetic-fixture-repair/tiny-native-checks.json').read_text())
    for row in tiny['checks']:
        assert hashlib.sha256(Path(row['arrays']).read_bytes()).hexdigest()==row['arrays_sha256']
    original_setup=json.loads((WORK/'tiny-native-checks.json').read_text())
    result={'candidate_id':candidate.candidate_id,'snapshot_sha256':frozen.sha256,
            'native_run_id':native['run_id'],'native_status':native['status'],
            'protected_correctness_report':str(native_path),
            'protected_correctness_report_sha256':hashlib.sha256(native_path.read_bytes()).hexdigest(),
            'native_checks':len(checks),
            'native_checks_passed':sum(bool(c.get('result') and c['result']['passed']) for c in checks),
            'native_errors':native['errors'],
            'optimized_native_checks':sum(c['case_name']=='decode' for c in checks),
            'declared_reference_fallback_checks':sum(c['case_name']!='decode' for c in checks),
            'synthetic_checks_passed':sum(c['passed'] for c in tiny['checks']),
            'initial_fixture_setup_failure_retained':original_setup['errors'],
            'verdict':None,'acceptance_enabled':False,'full_model_speedup_established':False}
    diagnostic_path=WORK/'timing-controller-repair/operator-diagnostic.json'
    diagnostic=None
    if diagnostic_path.exists():
        diagnostic=json.loads(diagnostic_path.read_text())
        assert diagnostic['snapshot_sha256']==frozen.sha256
        for row in diagnostic['additional_correctness']:
            assert hashlib.sha256(Path(row['arrays']).read_bytes()).hexdigest()==row['arrays_sha256']
        for name,digest in diagnostic.get('protected_sources_after',{}).items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
        result.update({'operator_diagnostic_status':diagnostic['status'],
                       'timing_plan':diagnostic['plan'],
                       'additional_decode_checks_passed':sum(c['passed'] for c in diagnostic['additional_correctness']),
                       'timing_samples_retained':len(diagnostic['samples']),
                       'environment_issues':diagnostic.get('environment_issues',[]),
                       'protected_sources_unchanged':diagnostic.get('protected_sources_unchanged'),
                       'checkpoint_identity_unchanged':diagnostic.get('checkpoint_identity_unchanged'),
                       'operator_summaries':diagnostic['summaries'],'diagnostic_errors':diagnostic['errors']})
        if diagnostic['status']!='completed_diagnostic' or diagnostic['errors']:
            disposition='Diagnostic failed; retain failure, no performance conclusion.'
        elif diagnostic.get('environment_issues'):
            disposition='Timing is environmentally inconclusive; no promotion.'
        elif all(s['descriptive_bootstrap_95pct_interval'][0]>1 for s in diagnostic['summaries'].values()):
            disposition='Promising operator evidence at all sampled layers; full-model evaluation still required.'
        elif all(s['descriptive_bootstrap_95pct_interval'][1]<1 for s in diagnostic['summaries'].values()):
            disposition='Operator diagnostic indicates a slowdown at all sampled layers; park candidate.'
        else:
            disposition='No consistent operator improvement established across sampled layers; no promotion.'
    else:
        disposition='Native correctness did not complete successfully; timing was not run.'
    result['research_disposition']=disposition
    result['timing_protocol_mismatch_retained']=json.loads((WORK/'timing-protocol-audit.json').read_text())
    result['corrected_call_count_verification']=json.loads((WORK/'timing-controller-repair/call-count-verification.json').read_text())
    assert result['corrected_call_count_verification']['observed_calls_per_sample']=={'baseline':256,'candidate':256}
    for earlier in json.loads((WORK/'operator-diagnostic.json').read_text())['additional_correctness']:
        assert hashlib.sha256(Path(earlier['arrays']).read_bytes()).hexdigest()==earlier['arrays_sha256']
    initial=json.loads((WORK/'preflight.json').read_text())
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest for name,digest in initial['protected_sources'].items())
    result['initial_protected_sources_unchanged']=True
    (WORK/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    lines=['# Joint up/gate MXFP4 candidate result','',f'**{disposition}**','',
           f'Candidate: `{candidate.candidate_id}`.',f'Frozen snapshot: `{frozen.sha256}`.','',
           'One custom Metal dispatch computes both original up/gate projections, their biases and SwiGLU during one-token decode. Native packed weights, scales and routing IDs remain unchanged; the down projection and prefill fallback use the original implementation.','',
           f'- Synthetic fixture checks: {result["synthetic_checks_passed"]} passed. An earlier incompatible N=72 fixture failed during native quantization setup; it was repaired to N=96, and the initial failed setup remains retained. Candidate bytes were unchanged.',
           f'- Protected native suite: {result["native_checks_passed"]}/{result["native_checks"]} passed. Three decode comparisons exercise optimized code; eighteen other comparisons use declared reference fallbacks.',
           f'- Native execution errors: {native["errors"]}.']
    if diagnostic:
        lines += [f'- Additional decode comparisons passed: {result["additional_decode_checks_passed"]}.',
                  f'- Timing samples retained: {len(diagnostic["samples"])}.',
                  '- The sample length was predeclared as 256 calls per lane (eight cycles of the same 32 routing inputs), equally for baseline and candidate, before any candidate #2 timing was observed.',
                  f'- Environment issues: {diagnostic.get("environment_issues",[])}.',
                  f'- Protected-source and checkpoint-identity checks: {diagnostic.get("protected_sources_unchanged")}, {diagnostic.get("checkpoint_identity_unchanged")}.','',
                  '| Layer | Median paired baseline/candidate operator-time ratio | Descriptive 95% interval | All samples environment-valid |',
                  '| --- | ---: | --- | --- |']
        for layer,s in diagnostic['summaries'].items():
            lo,hi=s['descriptive_bootstrap_95pct_interval']
            lines.append(f'| {layer} | {s["median_ratio"]:.3f} | [{lo:.3f}, {hi:.3f}] | {s["environment_valid"]} |')
        lines += ['', 'Ratios above 1 favor the candidate for this operator diagnostic. All samples, including invalid observations, remain retained; no outliers were removed. Raw times cannot be directly compared to candidate #1 because the lane sample length differs.']
    lines += ['', 'These are operator-level observations, not full-model tokens/sec or task-quality results. No acceptance gate, tolerance, checkpoint, workload, protected source or historical baseline was changed. Formal evaluator verdict remains null; acceptance remains disabled.','',
              'Evidence:','',f'- [Native evaluator report]({native_path}).',f'- [Structured result]({WORK/"result.json"}).',
              f'- [Synthetic fixture result]({WORK/"synthetic-fixture-repair/tiny-native-checks.json"}).',f'- [Initial fixture setup failure]({WORK/"tiny-native-checks.json"}).',
              f'- [Candidate Python]({ROOT/"candidates"/candidate.candidate_id/"implementation.py"}).',f'- [Metal kernel]({ROOT/"candidates"/candidate.candidate_id/"fused_up_gate.metal"}).','']
    if diagnostic:
        lines += [f'- [Raw corrected operator diagnostic]({diagnostic_path}).',
                  f'- [Initial timing protocol mismatch and repair]({WORK/"timing-protocol-audit.json"}).',
                  f'- [Initial raw timing report, retained but ineligible under the declared 256-call plan]({WORK/"operator-diagnostic.json"}).','',
                  'The first timing attempt labeled samples as 256 calls while its loop executed 32. The mismatch was found before a performance conclusion was issued. Both the incorrect controller and its complete raw report remain unchanged. A fresh corrected run was executed after independently verifying 256 calls per lane with counting stubs. Only that corrected run supports the table above.','']
    (WORK/'RESULT.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in result.items() if k!='operator_summaries'},indent=2))

if __name__=='__main__':
    main()
