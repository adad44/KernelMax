"""Native-only expert dependency-preserving deletion diagnostic.

Not a candidate, deployable implementation, full-model profile or verdict.
All outputs must equal the native operator before timing. No stage barriers
are inserted into timed graphs. All samples and failed screens are retained.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import statistics as st
import sys
import time
import traceback

ROOT=Path('/Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop')
WORK=Path(__file__).resolve().parent
CONTROL=WORK.parent
SOURCE=CONTROL.parent/'decode-bottleneck-20261026/controller-repair'
sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from evaluator.candidate_execution import _protected_files
from evaluator.candidate_protocol import file_digest
from evaluator.checkpoint_verification import checkpoint_identity, verify_checkpoint, validate_receipt
from evaluator.native_baseline import environment, environment_issues, materialize_parameters, quiet_preflight
from evaluator.schemas import load_contract
from scripts.loop_control import load_authorization
from scripts.run_reviewed_candidate import native_runtime


def main():
    output=WORK/'result.json'
    if output.exists():
        raise ValueError('Retain previous attempt; do not overwrite')
    search=json.loads((CONTROL/'search-20261027.json').read_text())
    deadline=search['deadline_unix']
    contract=load_contract(ROOT/'kernelmaxxing.yaml')
    session=load_authorization(CONTROL/'session.json',contract)
    protected=_protected_files([__file__,WORK/'supervise.py'])
    state={'schema':1,'scope':'isolated native expert operator only; not full-model bottleneck shares',
           'status':'verifying_checkpoint','pid':os.getpid(),'started_unix':time.time(),
           'contract_sha256':contract.sha256,'sources_before':protected,
           'plan':{'layers':[0,12,23],'cases_per_layer':8,'cycles':32,'calls_per_sample':256,
                   'warmups_per_lane':5,'aa_pairs':2,'region_pairs':20,
                   'noise_limits':{'maximum_pair_deviation':0.02,'relative_mad':0.01,'full_range':0.04},
                   'order':'alternating_ab_ba','paging_policy':'paging-aware',
                   'interpretation':'Deletion savings apply only to this sampled expert operator. Model contribution and hardware cause remain unverified.'},
           'correctness':[],'samples':[],'screens':{},'summaries':{},'errors':[]}
    def save():
        tmp=output.with_suffix('.tmp')
        tmp.write_text(json.dumps(state,indent=2,sort_keys=True,allow_nan=False)+'\n')
        tmp.replace(output)
    def check_time():
        if time.time()>=deadline:
            raise TimeoutError('Search deadline reached')
    def bootstrap(values,seed):
        rng=random.Random(seed)
        samples=sorted(st.median(rng.choices(values,k=len(values))) for _ in range(10000))
        return [samples[249],samples[9749]]
    save()
    receipt=None
    try:
        check_time()
        runtime=native_runtime(Path(session['model_path']),contract)
        receipt,reused=verify_checkpoint(runtime.model_path,runtime.model_revision,
                                         runtime.model_files,Path(session['checkpoint_cache']))
        state['checkpoint_verification']={'receipt':receipt,'reused':reused};save()
        state['status']='loading_sampled_experts';save()
        import mlx.core as mx
        import numpy as np
        from mlx_lm import load
        from mlx.utils import tree_flatten
        with mx.stream(mx.cpu):
            model,_=load(str(runtime.model_path),lazy=True)
            parameters=[(f'layer{layer}.{name}',value) for layer in (0,12,23)
                        for name,value in tree_flatten(model.model.layers[layer].mlp.experts.parameters())]
            materialize_parameters(parameters,mx,lambda done,total,name,size:
                                   print(f'native_load {done}/{total}',flush=True) if done%25==0 or done==total else None)
        validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files)
        source=json.loads((SOURCE/'state.json').read_text())
        state['input_source_sha256']=file_digest(SOURCE/'state.json')
        for layer in (0,12,23):
            check_time()
            experts=model.model.layers[layer].mlp.experts
            def activation(x,ids):
                expanded=mx.expand_dims(x,(-2,-3))
                return experts.activation(experts.up_proj(expanded,ids),experts.gate_proj(expanded,ids))
            cases=[]
            for row in [r for r in source['real_cases'] if r['layer']==layer]:
                path=Path(row['path'])
                if file_digest(path)!=row['sha256']:
                    raise ValueError('Real input hash changed')
                with np.load(path,allow_pickle=False) as data:
                    x=mx.array(data['x'],dtype=mx.bfloat16)
                    ids=mx.array(data['ids'],dtype=mx.uint32)
                    if not np.array_equal(np.array(x.astype(mx.float32)),data['x']):
                        raise ValueError('Input no longer roundtrips BF16')
                a=activation(x,ids);y=experts(x,ids);mx.eval(x,ids,a,y)
                cases.append((x,ids,a,y))
            assert len(cases)==8
            def value(mode,case):
                x,ids,cached_activation,cached_output=case
                if mode in ('native','native_repeat'):
                    return experts(x,ids)
                if mode=='without_up_gate':
                    a=mx.depends(cached_activation,[x,ids])
                    return experts.down_proj(a,ids).squeeze(-2)
                if mode=='without_down':
                    a=activation(x,ids)
                    return mx.depends(cached_output,[a,ids])
                if mode=='without_experts':
                    return mx.depends(cached_output,[x,ids])
                raise ValueError(mode)
            for mode in ('native_repeat','without_up_gate','without_down','without_experts'):
                for i,case in enumerate(cases):
                    actual=value(mode,case);mx.eval(actual)
                    exact=bool(mx.array_equal(actual,case[3]).item())
                    state['correctness'].append({'layer':layer,'case':i,'mode':mode,'exact':exact})
                    if not exact:
                        raise ValueError('Deletion output differs from native')
            save()
            def invoke(mode):
                calls=0
                for _ in range(32):
                    for case in cases:
                        mx.eval(value(mode,case));calls+=1
                assert calls==256
                return calls
            def sample(mode,pair,group):
                check_time()
                validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files)
                before=environment()
                if "'AC Power'" not in before['power']:
                    raise RuntimeError('AC unavailable before timing')
                mx.synchronize();start=time.perf_counter_ns()
                count=invoke(mode);mx.synchronize()
                elapsed=(time.perf_counter_ns()-start)/1e6
                after=environment()
                row={'layer':layer,'group':group,'pair':pair,'mode':mode,'calls':count,
                     'elapsed_ms':elapsed,'before':before,'after':after,
                     'environment_issues':environment_issues(before,after,swap_policy='paging-aware')}
                state['samples'].append(row);save()
                return row
            state['status']=f'layer_{layer}_aa_screen';save()
            state.setdefault('quiet_preflights',[]).append({'layer':layer,**quiet_preflight(
                quiet_seconds=10,timeout_seconds=60,swap_policy='paging-aware')});save()
            for _ in range(5):
                invoke('native');invoke('native_repeat')
            aa=[]
            for pair in (1,2):
                for mode in (('native','native_repeat') if pair%2 else ('native_repeat','native')):
                    aa.append(sample(mode,pair,'aa'))
            ratios=[next(r['elapsed_ms'] for r in aa if r['pair']==p and r['mode']=='native')/
                    next(r['elapsed_ms'] for r in aa if r['pair']==p and r['mode']=='native_repeat') for p in (1,2)]
            times=[r['elapsed_ms'] for r in aa];median=st.median(times)
            mad=st.median(abs(v-median) for v in times)/median
            span=(max(times)-min(times))/median
            valid=not any(r['environment_issues'] for r in aa)
            passed=valid and all(abs(r-1)<=.02 for r in ratios) and mad<=.01 and span<=.04
            state['screens'][str(layer)]={'ratios':ratios,'relative_mad':mad,'full_range':span,
                                           'environment_valid':valid,'passed':passed};save()
            if not passed:
                state['status']='stopped_aa_screen';save()
                print(f'AA screen failed layer {layer}; no deletion timing or unchanged retry',flush=True)
                return
            for mode in ('without_up_gate','without_down','without_experts'):
                state['status']=f'layer_{layer}_{mode}';save()
                for _ in range(5):
                    invoke('native');invoke(mode)
                retained=[]
                for pair in range(1,21):
                    for lane in (('native',mode) if pair%2 else (mode,'native')):
                        retained.append(sample(lane,pair,mode))
                baseline={r['pair']:r for r in retained if r['mode']=='native'}
                deletion={r['pair']:r for r in retained if r['mode']==mode}
                savings=[(baseline[p]['elapsed_ms']-deletion[p]['elapsed_ms'])/256 for p in range(1,21)]
                fractions=[1-deletion[p]['elapsed_ms']/baseline[p]['elapsed_ms'] for p in range(1,21)]
                state['summaries'][f'{layer}:{mode}']={
                    'native_ms_per_call':st.median(r['elapsed_ms']/256 for r in baseline.values()),
                    'deletion_ms_per_call':st.median(r['elapsed_ms']/256 for r in deletion.values()),
                    'paired_savings_ms_per_call':savings,'median_saved_ms_per_call':st.median(savings),
                    'descriptive_95pct_saved_ms':bootstrap(savings,42+layer),
                    'median_isolated_operator_fraction':st.median(fractions),
                    'environment_valid':not any(r['environment_issues'] for r in retained),
                    'full_model_share_established':False}
                save();print(f'layer {layer} {mode} retained 20 pairs',flush=True)
            print(f'layer {layer} complete',flush=True)
        state['status']='completed_operator_diagnostic'
    except BaseException:
        state['status']='failed';state['errors'].append(traceback.format_exc());raise
    finally:
        state['sources_unchanged']=protected==_protected_files([__file__,WORK/'supervise.py'])
        if receipt is not None:
            try:
                validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files)
                state['checkpoint_unchanged']=True
            except Exception as exc:
                state['checkpoint_unchanged']=False;state['errors'].append(str(exc))
        state['ended_unix']=time.time();save()


if __name__=='__main__':
    main()
