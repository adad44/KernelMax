"""Controller-owned full-model correctness and timing for a decode scheduler."""
import contextlib
import hashlib
import importlib.util
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
ROOT=WORK.parent.parent/'kernelmax-first-loop'
sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from evaluator.schemas import load_contract
from evaluator.candidate_execution import _protected_files
from evaluator.checkpoint_verification import validate_receipt
from evaluator.native_baseline import (environment,environment_issues,fixed_inputs,
    materialize_parameters,metrics,quiet_preflight,request,swap_observation)
from evaluator.baseline_validation import aa_noise,thermal_state
from orchestrator.candidate_manager import FrozenCandidate,validate_candidate,verify_frozen_candidate
from scripts.loop_control import load_authorization
from scripts.run_reviewed_candidate import native_runtime


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)


def interval(values,seed):
    rng=random.Random(seed)
    boot=sorted(st.median(rng.choices(values,k=len(values))) for _ in range(10000))
    return [boot[249],boot[9749]]


class CountedModel:
    def __init__(self,model,trace=False):
        self.model=model;self.trace=trace;self.calls=[];self.logits=[];self.routes=[];self.cache=None
    def __getattr__(self,name):return getattr(self.model,name)
    def __call__(self,tokens,cache=None):
        self.calls.append(list(tokens.shape));self.cache=cache
        if self.trace:self.current=[]
        result=self.model(tokens,cache=cache)
        if self.trace and tokens.shape[1]==1:
            if [r[0] for r in self.current]!=list(range(24)):raise ValueError('Incomplete route capture')
            self.logits.append(result[:,-1,:]);self.routes.append(self.current)
        return result


@contextlib.contextmanager
def capture_routes(proxy):
    cls=type(proxy.model.model.layers[0].mlp.experts);original=cls.__call__
    indices={id(l.mlp.experts):i for i,l in enumerate(proxy.model.model.layers)}
    def wrapped(experts,x,ids,*args,**kwargs):
        proxy.current.append((indices[id(experts)],ids))
        return original(experts,x,ids,*args,**kwargs)
    cls.__call__=wrapped
    try:yield
    finally:cls.__call__=original


def main(preloaded_model=None):
    output=WORK/'candidate-state.json'
    if output.exists():raise ValueError('Fresh candidate attempts only')
    contract=load_contract(ROOT/'kernelmaxxing.yaml')
    session=load_authorization(WORK/'session.json',contract)
    frozen_data=json.loads((WORK/'candidate-frozen.json').read_text())
    c=validate_candidate(frozen_data['directory'],Path(frozen_data['directory']).parent)
    frozen=FrozenCandidate(**vars(c),sha256=frozen_data['sha256'],file_sha256=frozen_data['file_sha256'])
    verify_frozen_candidate(frozen)
    runtime=native_runtime(Path(session['model_path']),contract)
    receipt=json.loads(Path(session['checkpoint_cache']).read_text())
    validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files)
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from mlx_lm.generate import wired_limit
    from mlx_lm.models.cache import make_prompt_cache
    from mlx.utils import tree_flatten
    import mlx_lm.models.gpt_oss as gpt_oss
    import mlx_lm.models.switch_layers as switch_layers
    extra=[__file__,WORK/'candidate-plan.json',WORK/'session.json',WORK/'candidate-frozen.json',gpt_oss.__file__,switch_layers.__file__,mx.__file__]
    protected=_protected_files(extra)
    # Exact frozen implementation imported only after validation/hash checks.
    spec=importlib.util.spec_from_file_location('frozen_decode_schedule',frozen.directory/'implementation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    candidate_request=module.decode_request
    state={'schema':1,'status':'loading','official':False,'verdict':None,'acceptance_enabled':False,
        'candidate_id':frozen.candidate_id,'snapshot_sha256':frozen.sha256,'contract_sha256':contract.sha256,
        'protected_before':protected,'requests':[],'correctness':[],'aa':{},'summaries':{},'errors':[],
        'pid':os.getpid(),'started_unix':time.time(),'full_model_speedup_established':False}
    save=lambda:write(output,state)
    def check():
        verify_frozen_candidate(frozen)
        validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files)
        if time.time()-state['started_unix']>18000:raise TimeoutError('Five-hour diagnostic cap')
    save()
    try:
        if preloaded_model is None:
            with mx.stream(mx.cpu):
                model,_=load(str(runtime.model_path),lazy=True)
                def progress(done,total,name,size):
                    if done%25==0 or done==total:
                        state['load_progress']={'done':done,'total':total};save();print(f'load {done}/{total}',flush=True)
                materialize_parameters(tree_flatten(model.parameters()),mx,progress)
        else:
            check();model=preloaded_model
            state['load_reused_from_timeline_probe']=True;save()
        inputs=fixed_inputs([512,2048,4096],model.args.vocab_size,42)
        arrays=WORK/'candidate-correctness-arrays';arrays.mkdir()
        reference_tokens={}
        def invoke(length,lane,trace=False):
            proxy=CountedModel(model,trace=trace)
            fn=candidate_request if lane=='candidate' else request
            entry_start=time.perf_counter()
            with capture_routes(proxy) if trace else contextlib.nullcontext():
                result=fn(proxy,inputs[str(length)],4096,mx,make_prompt_cache)
            entry_returned=time.perf_counter()
            mx.synchronize()
            result['controller_observed_entry_wall_ms']=(entry_returned-entry_start)*1000
            if lane=='candidate':
                observed=result['timing_observations'];times=observed['tokens_at']
                if not (len(times)==128 and entry_start<=observed['start']<observed['prefill_end']<times[0]
                    and times[-2]<observed['last_token_observed_at']<=observed['gpu_drained_at']==times[-1]<=entry_returned
                    and observed['async_submissions']==127):
                    raise ValueError('Candidate timing/submission observations are inconsistent')
                computed,gaps=metrics(length,observed['start'],observed['prefill_end'],times,observed['peak_memory_bytes'])
                if computed!=result['metrics'] or gaps!=result['inter_token_intervals_ms']:
                    raise ValueError('Trusted metric recomputation disagrees with candidate')
            expected=[[1,length-1]]+[[1,1]]*128
            if proxy.calls!=expected:raise ValueError('Incorrect forward count/shapes; extra or missing work')
            result['forward_calls']=len(proxy.calls);result['single_token_forward_calls']=sum(v==[1,1] for v in proxy.calls)
            actual_cache_dtypes=sorted({str(t.dtype) for c in proxy.cache for t in c.state})
            if actual_cache_dtypes!=['mlx.core.bfloat16'] or result['cache_dtypes']!=actual_cache_dtypes:
                raise ValueError('Cache precision changed or candidate metadata disagrees')
            if not trace:return result,None
            if len(proxy.logits)!=128 or len(proxy.routes)!=128:
                raise ValueError('Trace must contain every generated step')
            if any(v.shape!=(1,model.args.vocab_size) or v.dtype!=mx.bfloat16 for v in proxy.logits):
                raise ValueError('Logit shape/dtype changed')
            logits=np.concatenate([np.array(v.astype(mx.float32)) for v in proxy.logits],axis=0)
            if not np.all(np.isfinite(logits)):raise ValueError('Nonfinite logits')
            routes=np.stack([np.stack([np.array(ids).reshape(4) for _,ids in step]) for step in proxy.routes])
            cache_arrays=[np.array(t.astype(mx.float32)) for c in proxy.cache for t in c.state]
            cache_meta=[str(c.meta_state) for c in proxy.cache]
            return result,(logits,routes,cache_arrays,cache_meta)
        def collect(length,lane,phase,pair):
            check();time.sleep(12);time.sleep(5)
            quiet=quiet_preflight(quiet_seconds=10,timeout_seconds=180,swap_policy='paging-aware')
            before=environment();issues=environment_issues(before,before,'paging-aware')
            if issues:raise RuntimeError('Invalid preflight: '+str(issues))
            seen=[before['thermal_state']];stop=threading.Event()
            def observe():
                while not stop.wait(.5):seen.append(thermal_state())
            thread=threading.Thread(target=observe,daemon=True);thread.start()
            try:result,_=invoke(length,lane)
            finally:stop.set();thread.join()
            after=environment();issues=environment_issues(before,after,'paging-aware')
            if any(t!=0 for t in seen):issues.append('Nonnominal thermal during request')
            result.update(prompt_tokens=length,lane=lane,phase=phase,trial=pair,
                environment_before=before,environment_after=after,thermal_states=seen,
                quiet_preflight=quiet,swap_activity=swap_observation(before,after,'paging-aware'),invalid_reasons=issues)
            state['requests'].append(result);save()
            print(f'{length} {phase} {pair} {lane}: {1000/result["metrics"]["decode_tokens_per_second"]:.3f} ms/token; flags={issues}',flush=True)
            if issues:raise RuntimeError('Environment invalid; complete sample retained: '+str(issues))
            if result['output_token_ids']!=reference_tokens[length]:raise ValueError('Greedy tokens changed during timing')
            return result
        with wired_limit(model):
            state['status']='full_model_correctness';save()
            for length in (512,2048,4096):
                baseline,a=invoke(length,'baseline',True)
                candidate,b=invoke(length,'candidate',True)
                reference_tokens[length]=baseline['output_token_ids']
                equal_logits=bool(np.array_equal(a[0],b[0]));equal_routes=bool(np.array_equal(a[1],b[1]))
                equal_cache=len(a[2])==len(b[2]) and all(np.array_equal(x,y) for x,y in zip(a[2],b[2]))
                equal_meta=a[3]==b[3]
                path=arrays/f'context-{length}.npz'
                payload={'reference_logits':a[0],'candidate_logits':b[0],'reference_routes':a[1],'candidate_routes':b[1]}
                for i,(x,y) in enumerate(zip(a[2],b[2])):payload[f'reference_cache_{i}']=x;payload[f'candidate_cache_{i}']=y
                np.savez_compressed(path,**payload)
                row={'length':length,'all_logits_bit_exact':equal_logits,'all_routes_equal':equal_routes,
                    'all_final_cache_arrays_bit_exact':equal_cache,'cache_metadata_equal':equal_meta,
                    'all_greedy_tokens_equal':baseline['output_token_ids']==candidate['output_token_ids'],
                    'reference_calls':baseline['forward_calls'],'candidate_calls':candidate['forward_calls'],
                    'arrays':str(path),'arrays_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'cache_metadata':{'reference':a[3],'candidate':b[3]}}
                row['passed']=all(row[k] for k in ('all_logits_bit_exact','all_routes_equal','all_final_cache_arrays_bit_exact','cache_metadata_equal','all_greedy_tokens_equal'))
                state['correctness'].append(row);save();print(f'correctness {length}: {row["passed"]}',flush=True)
                del a,b,payload;mx.clear_cache()
                if not row['passed']:raise ValueError('Full-model correctness failed; timing blocked')
            for length in (512,2048,4096):
                state['status']=f'context_{length}_warmup';save()
                # Retained hydration is untimed; strict environment gates apply to measured requests.
                for lane in ('baseline','candidate'):
                    result,_=invoke(length,lane);state.setdefault('hydration',[]).append({'length':length,'lane':lane,**result});save()
                state.setdefault('quiet_preflights',[]).append(quiet_preflight(quiet_seconds=10,timeout_seconds=180,swap_policy='paging-aware'));save()
                for n in range(1,6):
                    collect(length,'A','aa_warmup',n);collect(length,'B','aa_warmup',n)
                aa=[];state['status']=f'context_{length}_aa';save()
                for pair in range(1,21):
                    for lane in (('A','B') if pair%2 else ('B','A')):aa.append(collect(length,lane,'aa',pair))
                state['aa'][str(length)]=aa_noise(aa,[length])[str(length)];save()
                if not state['aa'][str(length)]['passed']:state['status']='stopped_aa';save();return
                for n in range(1,6):
                    collect(length,'baseline','candidate_warmup',n);collect(length,'candidate','candidate_warmup',n)
                a={};b={};state['status']=f'context_{length}_candidate';save()
                for pair in range(1,21):
                    for lane in (('baseline','candidate') if pair%2 else ('candidate','baseline')):
                        r=collect(length,lane,'candidate',pair);(a if lane=='baseline' else b)[pair]=r
                savings=[1000/a[p]['metrics']['decode_tokens_per_second']-1000/b[p]['metrics']['decode_tokens_per_second'] for p in range(1,21)]
                ratios=[b[p]['metrics']['decode_tokens_per_second']/a[p]['metrics']['decode_tokens_per_second'] for p in range(1,21)]
                ci=interval(ratios,length);noise=max(abs(v-1) for v in state['aa'][str(length)]['bootstrap_95pct_ci'])
                summary={'saved_ms_per_token':savings,'median_saved_ms_per_token':st.median(savings),
                    'bootstrap_95pct_saved_ms':interval(savings,length+1),'median_ratio':st.median(ratios),
                    'bootstrap_95pct_ratio':ci,'improvement_above_measured_noise':ci[0]>1+noise,'metrics':{}}
                for key in a[1]['metrics']:
                    summary['metrics'][key]={}
                    for lane,rows in [('baseline',a),('candidate',b)]:
                        vals=[r['metrics'][key] for r in rows.values()];center=st.median(vals)
                        summary['metrics'][key][lane]={'median':center,'p90':float(np.percentile(vals,90)),'mad':st.median(abs(v-center) for v in vals)}
                state['summaries'][str(length)]=summary;save()
                if not summary['improvement_above_measured_noise']:state['status']='stopped_futility';save();return
            state['status']='completed_candidate_diagnostic'
            state['full_model_speedup_established']=True
    except BaseException:
        state['status']='failed';state['errors'].append(traceback.format_exc());raise
    finally:
        state['protected_unchanged']=protected==_protected_files(extra)
        try:check();state['checkpoint_and_snapshot_unchanged']=True
        except Exception:state['checkpoint_and_snapshot_unchanged']=False;state['errors'].append(traceback.format_exc())
        state['finished_unix']=time.time();save()


if __name__=='__main__':main()
