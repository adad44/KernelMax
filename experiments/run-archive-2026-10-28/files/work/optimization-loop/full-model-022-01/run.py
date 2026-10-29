"""Fresh matched full-model diagnostic for the frozen operator shortlist.

No acceptance verdict or task-quality claim. Full-model teacher-forced numerical,
route and greedy checks precede A/A and candidate timing. All data are retained.
"""
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import statistics as st
import sys
import threading
import time
import traceback

WORK=Path(__file__).resolve().parent
CONTROL=WORK.parent
ROOT=CONTROL.parent.parent/'kernelmax-first-loop'
sys.path.insert(0,str(ROOT))
from evaluator.candidate_execution import _protected_files
from evaluator.candidate_protocol import file_digest
from evaluator.candidate_worker import _candidate_entry
from evaluator.checkpoint_verification import validate_receipt
from evaluator.correctness import compare_outputs
from evaluator.native_baseline import (environment,environment_issues,fixed_inputs,
    materialize_parameters,quiet_preflight,request,swap_observation)
from evaluator.baseline_validation import aa_noise,thermal_state
from evaluator.schemas import load_contract
from orchestrator.candidate_manager import FrozenCandidate,validate_candidate,verify_frozen_candidate
from scripts.loop_control import load_authorization
from scripts.run_reviewed_candidate import native_runtime


def paired_interval(values,seed):
    rng=random.Random(seed)
    boot=sorted(st.median(rng.choices(values,k=len(values))) for _ in range(10000))
    return [boot[249],boot[9749]]


class ExpertDispatch:
    """Runtime-only expert binding; original method always handles fallback."""
    def __init__(self,model,entry):
        self.cls=type(model.model.layers[0].mlp.experts)
        if any(type(l.mlp.experts) is not self.cls for l in model.model.layers):
            raise ValueError('Mixed expert classes are unsupported')
        self.original=self.cls.__call__
        self.layers={id(l.mlp.experts):i for i,l in enumerate(model.model.layers)}
        self.entry=entry
        self.capture=None

    @contextlib.contextmanager
    def lane(self,candidate=False,capture=False):
        original=self.original;entry=self.entry;owner=self
        def wrapped(module,x,ids,*args,**kwargs):
            if owner.capture is not None:
                owner.capture.append((owner.layers[id(module)],ids))
            supported=(x.ndim==3 and x.shape[:2]==(1,1) and ids.shape==(1,1,4)
                       and module.up_proj.output_dims%4==0)
            if candidate and supported:
                return entry(module,x,ids)
            return original(module,x,ids,*args,**kwargs)
        self.cls.__call__=wrapped if candidate or capture else original
        try:yield
        finally:
            self.capture=None
            self.cls.__call__=original


def main():
    output=WORK/'state.json'
    if output.exists():raise ValueError('Retain existing full-model attempt')
    contract=load_contract(ROOT/'kernelmaxxing.yaml')
    session=load_authorization(CONTROL/'session.json',contract)
    selected=json.loads((WORK/'selected.json').read_text())
    source=Path(selected['directory'])
    approval=json.loads((source/'approval.json').read_text())
    report_path=Path(approval['native_report'])
    native=json.loads(report_path.read_text())
    operator=json.loads((source/'operator-diagnostic.json').read_text())
    if not (selected['shortlisted'] and native['status']=='completed' and not native['errors']
            and all(r['passed'] for r in native['correctness'])
            and operator['status']=='completed_diagnostic'
            and all(operator.get(k) is True for k in ('snapshot_unchanged','protected_sources_unchanged','checkpoint_identity_unchanged'))
            and all(v['passed'] for v in operator['aa_screens'].values())
            and all(v['descriptive_bootstrap_95pct'][0]>1 for v in operator['summaries'].values())):
        raise ValueError('A complete valid operator shortlist is required')
    info=json.loads((report_path.parent/'candidate/request.json').read_text())['snapshot']
    c=validate_candidate(info['directory'],Path(info['directory']).parent)
    frozen=FrozenCandidate(**vars(c),sha256=info['sha256'],file_sha256=info['file_sha256'])
    if frozen.candidate_id!=selected['candidate_id'] or frozen.sha256!=approval['snapshot_sha256']:
        raise ValueError('Selected snapshot identity mismatch')
    verify_frozen_candidate(frozen)
    runtime=native_runtime(Path(session['model_path']),contract)
    receipt=json.loads(Path(session['checkpoint_cache']).read_text())
    validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files)
    deadline=json.loads((CONTROL/'search-20261027.json').read_text())['deadline_unix']
    import mlx.core as mx
    import numpy as np
    import mlx_lm.models.gpt_oss as gpt_oss
    import mlx_lm.models.switch_layers as switch_layers
    from mlx_lm import load
    from mlx_lm.generate import wired_limit
    from mlx_lm.models.cache import make_prompt_cache
    from mlx.utils import tree_flatten
    extra=[__file__,WORK/'plan.json',WORK/'selected.json',WORK/'supervise.py',
           gpt_oss.__file__,switch_layers.__file__,mx.__file__]
    protected=_protected_files(extra)
    state={'schema':1,'status':'loading','scope':'full-model matched diagnostic; no acceptance or task-quality verdict',
           'official':False,'verdict':None,'acceptance_enabled':False,
           'full_model_speedup_established':False,
           'candidate_id':frozen.candidate_id,'snapshot_sha256':frozen.sha256,
           'contract_sha256':contract.sha256,'checkpoint_verification':receipt,
           'protected_sources_before':protected,'started_unix':time.time(),
           'packages':{n:importlib.metadata.version(n) for n in ('mlx','mlx-lm','numpy')},
           'operator_selection':str(source/'result.json'),'full_model_correctness':[],
           'requests':[],'aa_screens':{},'summaries':{},'errors':[],
           'plan':{'prompts':[512,2048,4096],'outputs':128,'fresh_caches':True,'seed':42,
                   'warmups_per_lane':5,'aa_pairs':20,'candidate_pairs':20,
                   'per_request_hold_seconds':12,'settle_seconds':5,'quiet_seconds':10,
                   'paging_policy':'paging-aware','order':'alternating_ab_ba',
                   'aa_gate':'reference20pair_bootstrap_ci_includes1_width<=0.04',
                   'all_failed_samples_retained':True,'no_statistical_rerolls':True}}
    def save():
        tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(output)
    def check():
        if time.time()>=deadline or (WORK/'STOP').exists():raise TimeoutError('Deadline or explicit stop reached')
        verify_frozen_candidate(frozen)
        validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files)
    save();dispatch=None
    try:
        check()
        with mx.stream(mx.cpu):
            model,_=load(str(runtime.model_path),lazy=True)
            def progress(done,total,name,size):
                if done%25==0 or done==total:
                    state['load_progress']={'done':done,'total':total,'name':name};save()
                    print(f'native_load {done}/{total}',flush=True)
            materialize_parameters(tree_flatten(model.parameters()),mx,progress)
        for layer in model.model.layers:
            for name in ('up_proj','gate_proj','down_proj'):
                p=getattr(layer.mlp.experts,name)
                if (p.mode,p.bits,p.group_size)!=('mxfp4',4,32):raise ValueError('Native format changed')
        check()
        dispatch=ExpertDispatch(model,_candidate_entry(frozen))
        inputs=fixed_inputs([512,2048,4096],model.args.vocab_size,42)
        state['input_token_ids']=inputs;save()
        arrays=WORK/'correctness-arrays';arrays.mkdir()
        def trace(ids,lane,forced=None):
            check();cache=make_prompt_cache(model)
            prompt=mx.array([ids],dtype=mx.int32);mx.eval(prompt)
            with dispatch.lane(candidate=lane=='candidate',capture=True):
                remaining=prompt
                while remaining.shape[1]>1:
                    n=min(4096,remaining.shape[1]-1)
                    model(remaining[:,:n],cache=cache);mx.eval([c.state for c in cache])
                    remaining=remaining[:,n:];mx.clear_cache()
                all_logits=[];routes=[];tokens=[];dtypes=[]
                for step in range(128):
                    check();dispatch.capture=[]
                    logits=model(remaining,cache=cache)[:,-1,:];mx.eval(logits)
                    ids_by_layer=sorted(dispatch.capture,key=lambda row:row[0])
                    if [r[0] for r in ids_by_layer]!=list(range(24)):
                        raise ValueError('Incomplete route capture')
                    routes.append(np.stack([np.array(r[1]).reshape(4) for r in ids_by_layer]))
                    dtypes.append(str(logits.dtype));all_logits.append(np.array(logits.astype(mx.float32)))
                    token=mx.argmax(logits,axis=-1);mx.eval(token);tokens.append(token.item())
                    value=tokens[-1] if forced is None else forced[step]
                    remaining=mx.array([[value]],dtype=mx.int32)
                cache_dtypes=sorted({str(t.dtype) for item in cache for t in item.state})
            return np.concatenate(all_logits,axis=0),np.stack(routes),tokens,sorted(set(dtypes)),cache_dtypes
        def collect(length,lane,phase,pair):
            check();time.sleep(12);time.sleep(5)
            preflight=quiet_preflight(quiet_seconds=10,timeout_seconds=min(300,max(1,deadline-time.time())),swap_policy='paging-aware')
            check();before=environment()
            issues=environment_issues(before,before,swap_policy='paging-aware')
            if issues:
                state.setdefault('blocked_preflights',[]).append({'length':length,'lane':lane,'phase':phase,'pair':pair,'environment':before,'issues':issues});save()
                raise RuntimeError('Environment invalid before request: '+str(issues))
            observed=[before['thermal_state']];stop=threading.Event()
            def observe():
                while not stop.wait(.5):observed.append(thermal_state())
            observer=threading.Thread(target=observe,daemon=True)
            observer.start()
            try:
                with dispatch.lane(candidate=lane=='candidate'):
                    row=request(model,inputs[str(length)],4096,mx,make_prompt_cache)
            finally:
                stop.set();observer.join()
            after=environment()
            issues=environment_issues(before,after,swap_policy='paging-aware')
            if any(t!=0 for t in observed):issues.append('nonnominal thermal state during request')
            row.update(length=length,prompt_tokens=length,lane=lane,phase=phase,pair=pair,trial=pair,
                       environment_before=before,environment_after=after,quiet_preflight=preflight,
                       thermal_states=observed,swap_activity=swap_observation(before,after,'paging-aware'),invalid_reasons=sorted(set(issues)))
            state['requests'].append(row);save()
            print(f'{length} {phase} pair{pair} {lane}: {row["metrics"]["decode_tokens_per_second"]:.4f} tok/s; flags={issues}',flush=True)
            if issues:raise RuntimeError('Environment invalid; complete request retained: '+str(issues))
            return row
        with wired_limit(model):
            state['status']='full_model_correctness';save()
            for length in (512,2048,4096):
                reference,ref_routes,ref_tokens,ref_dtypes,ref_cache=trace(inputs[str(length)],'baseline')
                candidate,cand_routes,cand_tokens,cand_dtypes,cand_cache=trace(inputs[str(length)],'candidate',ref_tokens)
                comparison=compare_outputs(reference,candidate,case_name='decode',policy=contract.correctness)
                path=arrays/f'context-{length}.npz'
                np.savez_compressed(path,reference_logits=reference,candidate_logits=candidate,
                                    reference_routes=ref_routes,candidate_routes=cand_routes,
                                    reference_tokens=np.array(ref_tokens),candidate_predicted_tokens=np.array(cand_tokens))
                row={'length':length,'numerical_passed':comparison.passed,'detail':comparison.detail,
                     'all_routes_equal':bool(np.array_equal(ref_routes,cand_routes)),
                     'all_greedy_tokens_equal':ref_tokens==cand_tokens,
                     'original_logit_dtypes':{'reference':ref_dtypes,'candidate':cand_dtypes},
                     'cache_dtypes':{'reference':ref_cache,'candidate':cand_cache},
                     'arrays':str(path),'arrays_sha256':file_digest(path)}
                row['passed']=(row['numerical_passed'] and row['all_routes_equal'] and row['all_greedy_tokens_equal']
                               and ref_dtypes==cand_dtypes==['mlx.core.bfloat16']
                               and ref_cache==cand_cache==['mlx.core.bfloat16'])
                state['full_model_correctness'].append(row);save()
                del reference,candidate,ref_routes,cand_routes;gc.collect();mx.clear_cache()
                print(f'full-model correctness context{length}: {row["passed"]}',flush=True)
                if not row['passed']:raise ValueError('Full-model correctness failed; timing blocked')
            for length in (512,2048,4096):
                state['status']=f'context_{length}_aa';save()
                for warm in range(1,6):
                    collect(length,'A','aa_warmup',warm);collect(length,'B','aa_warmup',warm)
                aa=[]
                for pair in range(1,21):
                    for lane in (('A','B') if pair%2 else ('B','A')):aa.append(collect(length,lane,'aa',pair))
                calibration=aa_noise(aa,[length])[str(length)]
                state['aa_screens'][str(length)]=calibration;save()
                if not calibration['passed']:
                    state['status']='stopped_aa_screen';save();return
                for warm in range(1,6):
                    collect(length,'baseline','candidate_warmup',warm);collect(length,'candidate','candidate_warmup',warm)
                a={};b={}
                state['status']=f'context_{length}_candidate_timing';save()
                for pair in range(1,21):
                    for lane in (('baseline','candidate') if pair%2 else ('candidate','baseline')):
                        row=collect(length,lane,'candidate',pair)
                        (a if lane=='baseline' else b)[pair]=row
                ratios=[b[p]['metrics']['decode_tokens_per_second']/a[p]['metrics']['decode_tokens_per_second'] for p in range(1,21)]
                interval=paired_interval(ratios,42+length)
                aa_interval=calibration['bootstrap_95pct_ci']
                noise=max(abs(v-1) for v in aa_interval)
                summary={'decode_paired_ratios':ratios,'median_ratio':st.median(ratios),
                         'descriptive_bootstrap_95pct':interval,'aa_interval':aa_interval,
                         'improvement_above_measured_noise':interval[0]>1+noise,'metrics':{}}
                for metric in a[1]['metrics']:
                    summary['metrics'][metric]={}
                    for lane,rows in [('baseline',a),('candidate',b)]:
                        values=[r['metrics'][metric] for r in rows.values()];center=st.median(values)
                        summary['metrics'][metric][lane]={'median':center,'p90':float(np.percentile(values,90)),
                                                          'mad':st.median(abs(v-center) for v in values)}
                state['summaries'][str(length)]=summary;save()
            state['status']='completed_full_model_diagnostic'
            state['full_model_speedup_established']=all(s['improvement_above_measured_noise'] for s in state['summaries'].values())
    except BaseException:
        state['status']='failed';state['errors'].append(traceback.format_exc());raise
    finally:
        if dispatch is not None:dispatch.cls.__call__=dispatch.original
        state['expert_class_method_restored']=dispatch is None or dispatch.cls.__call__ is dispatch.original
        state['protected_sources_unchanged']=protected==_protected_files(extra)
        try:
            verify_frozen_candidate(frozen)
            validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files)
            state['checkpoint_and_snapshot_unchanged']=True
        except Exception as error:
            state['checkpoint_and_snapshot_unchanged']=False;state['errors'].append(str(error))
        if not all(state[k] for k in ('expert_class_method_restored','protected_sources_unchanged','checkpoint_and_snapshot_unchanged')):
            state['full_model_speedup_established']=False
            state['status']='failed_integrity'
            state['errors'].append('Full-model integrity/restoration guard failed')
        state['ended_unix']=time.time();save()


if __name__=='__main__':
    main()
