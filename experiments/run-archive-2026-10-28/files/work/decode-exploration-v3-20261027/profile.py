"""Bounded whole-decode exploration: retain swap flags, never claim a speedup."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics as st
import sys
import threading
import time
import traceback

WORK=Path(__file__).resolve().parent
ROOT=WORK.parent.parent/'kernelmax-first-loop'
HISTORY=WORK.parent/'decode-search-20261027'
sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from evaluator.schemas import load_contract
from evaluator.candidate_execution import _protected_files
from evaluator.checkpoint_verification import validate_receipt
from evaluator.native_baseline import (environment,fixed_inputs,materialize_parameters,swap_observation)
from evaluator.baseline_validation import thermal_state
from evaluator.measurement import (assess_environment,BoundedExploration,ExplorationComplete,
                                   PhaseMemoryObserver,ObservedModel)
from scripts.loop_control import load_authorization
from scripts.run_reviewed_candidate import native_runtime


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temp.replace(path)


def main():
    output=WORK/'state.json'
    if output.exists():raise ValueError('Fresh exploration attempt required')
    contract=load_contract(ROOT/'kernelmaxxing.yaml')
    session=load_authorization(WORK/'session.json',contract)
    runtime=native_runtime(Path(session['model_path']),contract)
    receipt=json.loads(Path(session['checkpoint_cache']).read_text())
    validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files)
    plan=json.loads((WORK/'plan.json').read_text())
    budget=BoundedExploration(contract.measurement.exploration,max_requests=plan['max_requests'],max_seconds=plan['max_wall_time_seconds'])
    driver_path=HISTORY/'cooled-profile/profile.py'
    spec=importlib.util.spec_from_file_location('preserved_decode_functions',driver_path)
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    import mlx.core as mx
    import mlx_lm.models.gpt_oss as gpt_oss
    import mlx_lm.models.switch_layers as switch_layers
    from mlx_lm import load
    from mlx_lm.generate import wired_limit
    from mlx_lm.models.cache import make_prompt_cache
    from mlx.utils import tree_flatten
    extra=[__file__,WORK/'session.json',WORK/'plan.json',driver_path,gpt_oss.__file__,switch_layers.__file__,mx.__file__]
    protected=_protected_files(extra)
    prior_path=HISTORY/'cooled-profile/profile-state.json'
    prior_raw=prior_path.read_bytes();prior=json.loads(prior_raw)
    state={'schema':1,'status':'loading','mode':'exploration','official':False,'verdict':None,
        'acceptance_enabled':False,'speedup_claim_eligible':False,'contract_sha256':contract.sha256,
        'pid':os.getpid(),'started_unix':time.time(),'protected_before':protected,'requests':[],
        'summaries':{},'errors':[],'plan':plan,
        'preserved_work':{'source':str(prior_path),'source_sha256':hashlib.sha256(prior_raw).hexdigest(),
            'source_contract_sha256':prior['contract_sha256'],'completed_aa_calibration':prior['aa'],
            'source_attempt_status':prior['status'],'old_samples_regraded':False,
            'use':'historical calibration/correctness context for hypothesis discovery only; confirmation must be fresh'}}
    save=lambda:write(output,state)
    save();hooks=None
    def check():
        validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files)
        if protected!=_protected_files(extra):raise RuntimeError('Protected sources changed')
    try:
        with mx.stream(mx.cpu):
            model,_=load(str(runtime.model_path),lazy=True)
            def progress(done,total,name,size):
                if done%25==0 or done==total:
                    state['load_progress']={'done':done,'total':total};save();print(f'load {done}/{total}',flush=True)
            materialize_parameters(tree_flatten(model.parameters()),mx,progress)
        check();hooks=driver.Hooks(model,mx)
        ids=fixed_inputs([512,2048,4096],model.args.vocab_size,42)['512']
        expected_tokens=None
        def collect(label,mode='normal',reference=None,pair=None):
            check();time.sleep(12);time.sleep(5);time.sleep(10)
            budget.begin_request();hooks.mode=mode
            before=environment();pre=assess_environment(before,before,mode='exploration')
            if not pre['continue_pass']:raise RuntimeError('Power/thermal/observation preflight failed: '+str(pre))
            observer=PhaseMemoryObserver(len(ids),mx)
            proxy=ObservedModel(model,observer)
            seen=[before['thermal_state']];stop=threading.Event()
            def observe():
                while not stop.wait(.5):seen.append(thermal_state())
            thread=threading.Thread(target=observe,daemon=True)
            observer.start();thread.start()
            result=None;logs=None;error=None
            try:result,logs=driver.request(proxy,ids,mx,make_prompt_cache,hooks,reference)
            except BaseException:error=traceback.format_exc()
            finally:
                stop.set();thread.join();memory=observer.finish()
            after=environment();assessment=assess_environment(before,after,mode='exploration',thermal_states=seen)
            row={'label':label,'mode':mode,'pair':pair,'environment_before':before,'environment_after':after,
                 'environment_assessment':assessment,'thermal_states':seen,'phase_memory':memory,
                 'swap_activity':swap_observation(before,after,'paging-aware'),'error':error,
                 'confirmation_eligible':False}
            if result is not None:row.update(result)
            state['requests'].append(row);save()
            print(f'{label}: flags={assessment["flags"]}; swap_affected={assessment["swap_affected"]}; continue={assessment["continue_pass"]}',flush=True)
            if error:raise RuntimeError('Numerical/GPU/request failure; retained row: '+error)
            if not assessment['continue_pass']:raise RuntimeError('Hard environment failure; retained row: '+str(assessment))
            if expected_tokens is not None and result['output_token_ids']!=expected_tokens:
                raise ValueError('Greedy trajectory changed')
            if reference is not None and result['max_abs_logit_error']!=0:
                raise ValueError('Cached deletion did not preserve exact logits')
            return row,logs
        with wired_limit(model):
            state['status']='resident_warmup';save()
            for n in range(2):
                row,_=collect(f'warmup-{n}')
                if expected_tokens is None:expected_tokens=row['output_token_ids']
            state['status']='capture';save()
            captured,reference=collect('capture',mode='capture')
            state['capture_entries']=len(hooks.saved);save()
            for region in ('experts','attention','lm_head','small_ops'):
                mode='delete_'+region;state['status']=mode;save()
                for n in range(5):collect(f'{mode}-warmup-{n}',mode,reference)
                controls={};deleted={}
                for pair in range(1,6):
                    for lane in (('normal',mode) if pair%2 else (mode,'normal')):
                        row,_=collect(f'{mode}-{pair}-{lane}',lane,reference,pair)
                        (controls if lane=='normal' else deleted)[pair]=row
                # Every pair is retained, including swap-affected pairs. These
                # are nonadditive deletion budgets, not implementation speedups.
                savings=[1000/controls[p]['metrics']['decode_tokens_per_second']-1000/deleted[p]['metrics']['decode_tokens_per_second'] for p in range(1,6)]
                state['summaries'][region]={'all_pair_saved_ms_per_token':savings,
                    'median_saved_ms_per_token':st.median(savings),'bootstrap_descriptive_95pct_ms':driver.interval(savings),
                    'swap_affected_pairs':[p for p in range(1,6) if any(r['environment_assessment']['swap_affected'] for r in (controls[p],deleted[p]))],
                    'all_pairs_included':True,'confirmation_eligible':False,
                    'scope':'optimistic nonadditive cached-output deletion budget; hypothesis discovery only'}
                save()
            state['status']='completed_bounded_exploration'
    except ExplorationComplete as error:
        state['status']='completed_at_exploration_bound';state['bound_reason']=str(error)
    except BaseException:
        state['status']='failed';state['errors'].append(traceback.format_exc());raise
    finally:
        if hooks:hooks.restore()
        state['requests_started']=budget.requests
        state['protected_unchanged']=protected==_protected_files(extra)
        try:validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files);state['checkpoint_unchanged']=True
        except Exception:state['checkpoint_unchanged']=False;state['errors'].append(traceback.format_exc())
        state['finished_unix']=time.time();save()


if __name__=='__main__':main()
