"""Lightweight command-buffer probe after paging-invalid cached deletion."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
ROOT=WORK.parent.parent/'kernelmax-first-loop'
sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from evaluator.schemas import load_contract
from evaluator.candidate_execution import _protected_files
from evaluator.checkpoint_verification import validate_receipt
from evaluator.native_baseline import (environment,environment_issues,fixed_inputs,
    materialize_parameters,quiet_preflight,request,swap_observation)
from evaluator.baseline_validation import thermal_state
from scripts.loop_control import load_authorization
from scripts.run_reviewed_candidate import native_runtime
from probe import Timeline,trace_decode


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)


def main():
    output=HERE/'state.json'
    if output.exists():raise ValueError('Fresh probe attempts only')
    # Authoritative process check prevents overlapping the running profiler.
    import subprocess
    ps=subprocess.run(['ps','-p','71655','-o','lstart=,command='],text=True,capture_output=True)
    if ps.returncode==0 and 'cooled-profile/profile.py' in ps.stdout:
        raise RuntimeError('Prior profiler is still live; no second model worker started')
    profile=json.loads((WORK/'cooled-profile/profile-state.json').read_text())
    if not (profile.get('aa',{}).get('passed') and profile.get('protected_unchanged') and profile.get('checkpoint_unchanged')):
        raise RuntimeError('Recorded valid A/A and integrity guards required; failed deletion timings are never reused')
    validation=json.loads((HERE/'validation.json').read_text())
    if (validation.get('passed') is not True or validation.get('original_method_restored') is not True
        or any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in validation['source_sha256'].items())):
        raise RuntimeError('A current native observer fixture pass is required')
    contract=load_contract(ROOT/'kernelmaxxing.yaml')
    session=load_authorization(WORK/'session.json',contract)
    runtime=native_runtime(Path(session['model_path']),contract)
    receipt=json.loads(Path(session['checkpoint_cache']).read_text())
    validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files)
    import mlx.core as mx
    import mlx_lm.models.gpt_oss as gpt_oss
    import mlx_lm.models.switch_layers as switch_layers
    from mlx_lm import load
    from mlx_lm.generate import wired_limit
    from mlx_lm.models.cache import make_prompt_cache
    from mlx.utils import tree_flatten
    extra=[__file__,HERE/'probe.py',HERE/'command_timeline.m',HERE/'command_timeline.dylib',
        HERE/'plan.json',HERE/'validation.json',WORK/'session.json',WORK/'evaluate_schedule.py',gpt_oss.__file__,switch_layers.__file__,mx.__file__]
    protected=_protected_files(extra)
    state={'schema':1,'status':'loading','official':False,'verdict':None,'acceptance_enabled':False,
        'contract_sha256':contract.sha256,'protected_before':protected,'requests':[],
        'prior_profile_status':profile['status'],'prior_aa_calibration':profile['aa'],
        'prior_profile_path':str(WORK/'cooled-profile/profile-state.json'),
        'pid':os.getpid(),'started_unix':time.time(),'errors':[]}
    save=lambda:write(output,state)
    def check():
        validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files)
        if time.time()-state['started_unix']>7200:raise TimeoutError('Timeline preparation cap')
        if protected!=_protected_files(extra):raise RuntimeError('Prepared sources changed')
    save();timeline=None
    try:
        with mx.stream(mx.cpu):
            model,_=load(str(runtime.model_path),lazy=True)
            def progress(done,total,name,size):
                if done%25==0 or done==total:
                    state['load_progress']={'done':done,'total':total};save();print(f'load {done}/{total}',flush=True)
            materialize_parameters(tree_flatten(model.parameters()),mx,progress)
        ids=fixed_inputs([512,2048,4096],model.args.vocab_size,42)['512']
        with wired_limit(model):
            for n in range(2):
                row=request(model,ids,4096,mx,make_prompt_cache)
                state.setdefault('hydration',[]).append(row);save()
            expected=None
            for pair in range(1,4):
                for mode in (('control','timeline') if pair%2 else ('timeline','control')):
                    check();state['status']=f'{mode}_cooldown';save();time.sleep(12);time.sleep(5)
                    quiet=quiet_preflight(quiet_seconds=10,timeout_seconds=180,swap_policy='paging-aware')
                    before=environment();pre=environment_issues(before,before,'paging-aware')
                    if pre:raise RuntimeError('Preflight invalid: '+str(pre))
                    seen=[before['thermal_state']];stop=threading.Event()
                    def observe():
                        while not stop.wait(.5):seen.append(thermal_state())
                    thread=threading.Thread(target=observe,daemon=True)
                    state['status']=f'{mode}_request';save();thread.start()
                    try:
                        if mode=='timeline':
                            timeline=Timeline(HERE/'command_timeline.dylib')
                            class_name=timeline.cls.decode()
                            row=trace_decode(model,ids,mx,make_prompt_cache,timeline)
                            timeline.restore();timeline=None;row['hook_restored']=True;row['command_buffer_class']=class_name
                            tokens=row['tokens']
                        else:
                            row=request(model,ids,4096,mx,make_prompt_cache);tokens=row['output_token_ids']
                    finally:stop.set();thread.join()
                    after=environment();issues=environment_issues(before,after,'paging-aware')
                    if any(v!=0 for v in seen):issues.append('Nonnominal thermal during probe')
                    row.update(pair=pair,mode=mode,quiet_preflight=quiet,environment_before=before,
                        environment_after=after,thermal_states=seen,invalid_reasons=issues,
                        swap_activity=swap_observation(before,after,'paging-aware'))
                    state['requests'].append(row);save();print(f'{mode} pair{pair}: flags={issues}',flush=True)
                    if issues:raise RuntimeError('Environment invalid; all rows retained: '+str(issues))
                    if expected is None:expected=tokens
                    if tokens!=expected:raise ValueError('Timestamp observer changed greedy outputs')
            state['status']='waiting_candidate_selection';state['selection_wait_started_unix']=time.time();save()
            print('Timeline complete; original command-buffer method restored; waiting for frozen candidate selection',flush=True)
            selection=WORK/'run-selected-candidate.json'
            deadline=time.monotonic()+1200
            while not selection.exists():
                check()
                if time.monotonic()>deadline:
                    state['status']='completed_probe_no_candidate_selected';save();return
                time.sleep(2)
            chosen=json.loads(selection.read_text())
            if chosen.get('action')!='run_schedule':raise ValueError('Unsupported controller handoff')
            check();state['status']='candidate_handoff';state['selected_candidate']=chosen;save()
            spec=importlib.util.spec_from_file_location('decode_schedule_evaluator',WORK/'evaluate_schedule.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            module.main(preloaded_model=model)
            state['status']='completed_candidate_handoff'
    except BaseException:
        state['status']='failed';state['errors'].append(traceback.format_exc());raise
    finally:
        if timeline is not None:
            try:timeline.finish();timeline.restore()
            except Exception:state['errors'].append(traceback.format_exc())
        state['protected_unchanged']=protected==_protected_files(extra)
        try:validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files);state['checkpoint_unchanged']=True
        except Exception:state['checkpoint_unchanged']=False;state['errors'].append(traceback.format_exc())
        state['finished_unix']=time.time();save()


if __name__=='__main__':main()
