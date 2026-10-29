"""Real-input candidate-3 operator diagnostic; no full-model verdict."""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics as st
import sys
import time
import traceback

ROOT=Path('/Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop')
WORK=Path(__file__).resolve().parent
SOURCE=WORK.parent/'decode-bottleneck-20261026/controller-repair'
sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from evaluator.candidate_execution import _checkpoint_identity,_protected_files
from evaluator.candidate_worker import _candidate_entry
from evaluator.correctness import compare_outputs
from evaluator.native_baseline import environment,environment_issues,materialize_parameters
from evaluator.schemas import load_contract
from orchestrator.candidate_manager import FrozenCandidate,validate_candidate,verify_frozen_candidate
from scripts.run_reviewed_candidate import native_runtime
from target.adapter import bind_native_experts
from target.reference_moe import ReferenceMoE


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if (WORK/'operator-diagnostic.json').exists():
        raise ValueError('Existing attempt retained; use a new directory')
    approval=json.loads((WORK/'approval.json').read_text())
    report_path=Path(sys.argv[1]).resolve()
    report=json.loads(report_path.read_text())
    contract=load_contract(ROOT/'kernelmaxxing.yaml')
    if not (report['status']=='completed' and not report['errors']
            and report['candidate']['sha256']==approval['snapshot_sha256']
            and report['contract_sha256']==contract.sha256
            and len(report['correctness'])==7 and all(r['passed'] for r in report['correctness'])):
        raise ValueError('Protected native checks did not pass for frozen candidate')
    for row in report['correctness']:
        detail=json.loads(row['detail'])
        if detail.get('input_mode')!='mlx_native' or len(detail['checks'])!=3:
            raise ValueError('Incomplete native checkpoint evidence')
    c=validate_candidate(approval['snapshot'],Path(approval['snapshot']).parent)
    frozen=FrozenCandidate(**vars(c),sha256=approval['snapshot_sha256'],
                           file_sha256={p.relative_to(c.directory).as_posix():digest(p)
                                        for p in c.directory.rglob('*') if p.is_file()})
    verify_frozen_candidate(frozen)
    expected_identity=json.loads((report_path.parent/'checkpoint_identity.json').read_text())
    runtime=native_runtime(Path('/Volumes/BankOfSouls/models/gpt-oss-20b'),contract)
    identity=lambda:json.loads(json.dumps(_checkpoint_identity(runtime)))
    protected=_protected_files([__file__,WORK/'PLAN.md'])
    result={'schema':'candidate3_real_decode_operator_diagnostic_v1','status':'running',
            'candidate_id':c.candidate_id,'snapshot_sha256':frozen.sha256,
            'scope':'native expert operator on saved real decode activations; no full-model speedup or verdict',
            'acceptance_enabled':False,'verdict':None,'contract_sha256':contract.sha256,
            'protected_report':str(report_path),'protected_report_sha256':digest(report_path),
            'protected_sources_before':protected,'checkpoint_identity_before':expected_identity,
            'environment_before':environment(),'additional_correctness':[],
            'instrumented_frontend':[],'samples':[],'summaries':{},'errors':[],
            'plan':{'layers':[0,12,23],'real_cases_per_layer':8,'cycles_per_sample':32,
                    'calls_per_sample':256,'pairs_per_layer':20,'warmup_cycles_per_lane':5,
                    'order':'alternating_ab_ba','environment_policy':'paging-aware',
                    'per_call_timing_in_paired_samples':False,
                    'execution_class':'temporary launchd Interactive job'}}
    def save():
        p=WORK/'operator-diagnostic.json'; tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(p)
    save()
    try:
        if identity()!=expected_identity:
            raise ValueError('Checkpoint identity changed after native correctness')
        import mlx.core as mx
        import numpy as np
        from mlx_lm import load
        from mlx.utils import tree_flatten
        with mx.stream(mx.cpu):
            model,_=load(str(runtime.model_path),lazy=True)
            params=[(f'layer{layer}.{name}',v) for layer in (0,12,23)
                    for name,v in tree_flatten(model.model.layers[layer].mlp.experts.parameters())]
            materialize_parameters(params,mx,lambda done,total,name,size:print(f'native_load {done}/{total}',flush=True)
                                   if done%25==0 or done==total else None)
        if identity()!=expected_identity:
            raise ValueError('Checkpoint changed during sampled load')
        adapter=bind_native_experts(model,mx=mx)
        experts=dict(adapter.experts)
        reference=ReferenceMoE(model)
        entry=_candidate_entry(frozen)
        source=json.loads((SOURCE/'state.json').read_text())
        result['real_input_provenance']={'source_state':str(SOURCE/'state.json'),
                                        'source_state_sha256':digest(SOURCE/'state.json'),
                                        'scope':'original unchanged model, seed-42 prompt512 decode steps1..127'}
        inputs=WORK/'real-inputs';inputs.mkdir()
        arrays=WORK/'diagnostic-arrays';arrays.mkdir()
        for layer in (0,12,23):
            cases=[]
            rows=[r for r in source['real_cases'] if r['layer']==layer]
            assert len(rows)==8
            for row in rows:
                path=Path(row['path'])
                if digest(path)!=row['sha256']:
                    raise ValueError('Original real activation changed')
                dest=inputs/path.name;shutil.copy2(path,dest)
                with np.load(dest,allow_pickle=False) as data:
                    x=mx.array(data['x'],dtype=mx.bfloat16)
                    ids=mx.array(data['ids'],dtype=mx.uint32)
                    if not np.array_equal(np.array(x.astype(mx.float32)),data['x']):
                        raise ValueError('Stored activation does not roundtrip BF16 exactly')
                mx.eval(x,ids);cases.append((x,ids))
                baseline_output=reference(layer,x,ids)
                for dtype in (mx.int32,mx.uint32):
                    actual=entry(experts[layer],x,ids.astype(dtype))
                    comparison=compare_outputs(baseline_output,actual,case_name='decode',policy=contract.correctness)
                    out=arrays/f'layer-{layer}-step-{row["step"]}-{str(dtype).split(".")[-1]}.npz'
                    np.savez_compressed(out,x=np.array(x.astype(mx.float32)),ids=np.array(ids.astype(dtype)),
                                        reference=np.array(baseline_output.astype(mx.float32)),
                                        candidate=np.array(actual.astype(mx.float32)))
                    result['additional_correctness'].append({'layer':layer,'step':row['step'],
                                                              'ids_dtype':str(dtype),'passed':comparison.passed,
                                                              'detail':comparison.detail,'arrays':str(out),
                                                              'arrays_sha256':digest(out),'input_sha256':digest(dest)})
                    save()
                    if not comparison.passed:
                        raise ValueError(f'Real decode check failed: layer{layer},step{row["step"]},{dtype}')
            def value(lane,x,ids):
                return reference(layer,x,ids) if lane=='baseline' else entry(experts[layer],x,ids)
            def invoke(lane):
                count=0
                for _ in range(32):
                    for x,ids in cases:
                        mx.eval(value(lane,x,ids));count+=1
                assert count==256
                return count
            for _ in range(5):
                invoke('baseline');invoke('candidate')
            # Separate construction instrumentation, excluded from all paired wall samples.
            for lane in ('baseline','candidate'):
                parts=[];before=environment()
                for _ in range(32):
                    for x,ids in cases:
                        start=time.perf_counter_ns();v=value(lane,x,ids)
                        built=time.perf_counter_ns();mx.eval(v);end=time.perf_counter_ns()
                        parts.append({'build_ms':(built-start)/1e6,'eval_wait_ms':(end-built)/1e6})
                after=environment()
                result['instrumented_frontend'].append({'layer':layer,'lane':lane,'calls':len(parts),
                                                       'mean_build_ms':st.mean(p['build_ms'] for p in parts),
                                                       'mean_eval_wait_ms':st.mean(p['eval_wait_ms'] for p in parts),
                                                       'parts':parts,'before':before,'after':after,
                                                       'environment_issues':environment_issues(before,after,swap_policy='paging-aware')})
                save()
            for pair in range(1,21):
                lanes=('baseline','candidate') if pair%2 else ('candidate','baseline')
                for lane in lanes:
                    verify_frozen_candidate(frozen)
                    if identity()!=expected_identity:
                        raise ValueError('Checkpoint changed before sample')
                    before=environment()
                    if "'AC Power'" not in before['power']:
                        raise RuntimeError('AC unavailable; no timing started')
                    mx.reset_peak_memory();mx.synchronize()
                    start=time.perf_counter();count=invoke(lane);mx.synchronize()
                    elapsed_ms=(time.perf_counter()-start)*1000;peak=mx.get_peak_memory()
                    after=environment()
                    result['samples'].append({'layer':layer,'pair':pair,'lane':lane,'calls':count,
                                              'elapsed_ms':elapsed_ms,'peak_memory_bytes':peak,
                                              'before':before,'after':after,
                                              'environment_issues':environment_issues(before,after,swap_policy='paging-aware')})
                    save()
                print(f'layer {layer}: pair {pair}/20 retained',flush=True)
            selected=[r for r in result['samples'] if r['layer']==layer]
            a={r['pair']:r for r in selected if r['lane']=='baseline'}
            b={r['pair']:r for r in selected if r['lane']=='candidate'}
            ratios=[a[i]['elapsed_ms']/b[i]['elapsed_ms'] for i in range(1,21)]
            rng=random.Random(42+layer)
            boot=sorted(st.median(rng.choices(ratios,k=20)) for _ in range(10000))
            result['summaries'][str(layer)]={'median_baseline_ms':st.median(r['elapsed_ms'] for r in a.values()),
                                            'median_candidate_ms':st.median(r['elapsed_ms'] for r in b.values()),
                                            'paired_ratios':ratios,'median_ratio':st.median(ratios),
                                            'descriptive_bootstrap_95pct':[boot[249],boot[9749]],
                                            'ab_order_median_ratio':st.median(ratios[0::2]),
                                            'ba_order_median_ratio':st.median(ratios[1::2]),
                                            'environment_valid':all(not r['environment_issues'] for r in selected)}
            save()
        result['status']='completed_diagnostic'
    except BaseException:
        result['status']='failed_diagnostic';result['errors'].append(traceback.format_exc());raise
    finally:
        result['protected_sources_after']=_protected_files([__file__,WORK/'PLAN.md'])
        result['protected_sources_unchanged']=protected==result['protected_sources_after']
        result['checkpoint_identity_unchanged']=identity()==expected_identity
        result['environment_after']=environment()
        try:
            verify_frozen_candidate(frozen);result['snapshot_unchanged']=True
        except Exception as e:
            result['snapshot_unchanged']=False;result['errors'].append(str(e))
        result['environment_issues']=sorted({i for r in result['samples'] for i in r['environment_issues']})
        result['interpretation_status']='inconclusive_environment' if result['environment_issues'] else 'operator_only_no_full_model_claim'
        save()


if __name__=='__main__':
    main()
