"""Untimed full-model native repeatability and local boundary divergence probe."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
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
from evaluator.native_baseline import materialize_parameters
from evaluator.schemas import load_contract
from orchestrator.candidate_manager import FrozenCandidate,validate_candidate,verify_frozen_candidate
from scripts.run_reviewed_candidate import native_runtime


def main():
    output=WORK/'state.json'
    if output.exists():raise ValueError('Retain previous diagnostic')
    selected=json.loads((WORK/'selected.json').read_text())
    source=Path(selected['directory'])
    approval=json.loads((source/'approval.json').read_text())
    request=json.loads((Path(approval['native_report']).parent/'candidate/request.json').read_text())
    info=request['snapshot'];candidate=validate_candidate(info['directory'],Path(info['directory']).parent)
    frozen=FrozenCandidate(**vars(candidate),sha256=info['sha256'],file_sha256=info['file_sha256'])
    verify_frozen_candidate(frozen)
    contract=load_contract(ROOT/'kernelmaxxing.yaml')
    runtime=native_runtime(Path('/Volumes/BankOfSouls/models/gpt-oss-20b'),contract)
    receipt=json.loads((CONTROL/'checkpoint-verification.json').read_text())
    validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files)
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from mlx_lm.generate import wired_limit
    from mlx_lm.models.cache import make_prompt_cache
    from mlx.utils import tree_flatten
    full_source=CONTROL/'full-model-012/run.py'
    spec=importlib.util.spec_from_file_location('_trusted_fullmodel_trace',full_source)
    full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full)
    protected=_protected_files([__file__,WORK/'selected.json',WORK/'supervise.py',full_source])
    state={'status':'loading','scope':'untimed numerical diagnosis; no performance measurements',
           'candidate_id':frozen.candidate_id,'snapshot_sha256':frozen.sha256,
           'started_unix':time.time(),'sources_before':protected,'boundary_checks':[],'errors':[]}
    def save():
        tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n');tmp.replace(output)
    deadline=json.loads((WORK/'process.json').read_text())['deadline_unix']
    def check():
        if time.time()>=deadline:raise TimeoutError('Diagnostic deadline reached')
        verify_frozen_candidate(frozen)
        validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files)
    save();original=None;cls=None
    try:
        with mx.stream(mx.cpu):
            model,_=load(str(runtime.model_path),lazy=True)
            def progress(done,total,name,size):
                if done%25==0 or done==total:
                    state['load_progress']={'done':done,'total':total};save();print(f'load {done}/{total}',flush=True)
            materialize_parameters(tree_flatten(model.parameters()),mx,progress)
        entry=_candidate_entry(frozen)
        dispatch=full.ExpertDispatch(model,entry)
        # Reuse the exact trusted trace function from the failed full-model test.
        tree=ast.parse(full_source.read_text())
        node=copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='trace'))
        scope={'check':check,'make_prompt_cache':make_prompt_cache,'model':model,'mx':mx,'np':np,'dispatch':dispatch}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(full_source)+'#trace','exec'),scope)
        trace=scope['trace']
        prior=json.loads((CONTROL/'full-model-012/state.json').read_text())
        prompt=prior['input_token_ids']['512']
        with np.load(CONTROL/'full-model-012/correctness-arrays/context-512.npz',allow_pickle=False) as arrays:
            previous_reference=arrays['reference_logits'];previous_routes=arrays['reference_routes'];forced=arrays['reference_tokens'].tolist()
        state['status']='native_repeatability';save()
        with wired_limit(model):
            logits,routes,tokens,dtypes,cache_dtypes=trace(prompt,'baseline',forced)
            difference=np.abs(logits-previous_reference)
            state['native_repeatability']={'all_logits_exact':bool(np.array_equal(logits,previous_reference)),
                'max_abs_logit_error':float(difference.max()),'all_routes_exact':bool(np.array_equal(routes,previous_routes)),
                'all_tokens_equal':tokens==forced,'logit_dtypes':dtypes,'cache_dtypes':cache_dtypes}
            np.savez_compressed(WORK/'native-repeat.npz',logits=logits,routes=routes,tokens=np.array(tokens))
            del logits,routes,previous_reference,previous_routes,difference
            state['status']='local_boundary_checks';save()
            # Derive only the activation return from the same frozen candidate.
            implementation=frozen.directory/'implementation.py'
            tree=ast.parse(implementation.read_text())
            function=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run'))
            if not isinstance(function.body[-1],ast.Return):raise ValueError('Unexpected candidate return')
            function.name='diagnostic_activation'
            function.body[-1].value=ast.Name(id='activated',ctx=ast.Load())
            region_tree=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
            region_scope=dict(entry.__globals__)
            exec(compile(region_tree,'<frozen-candidate-activation-diagnostic>','exec'),region_scope)
            region=region_scope['diagnostic_activation']
            (WORK/'derived-activation.py').write_text(ast.unparse(region_tree)+'\n')
            cls=type(model.model.layers[0].mlp.experts);original=cls.__call__
            layers={id(l.mlp.experts):i for i,l in enumerate(model.model.layers)}
            capture=[]
            def recorded(module,x,ids,*args,**kwargs):
                y=original(module,x,ids,*args,**kwargs)
                capture.append((layers[id(module)],module,x,ids,y))
                return y
            cache=make_prompt_cache(model);remaining=mx.array([prompt],dtype=mx.int32)
            while remaining.shape[1]>1:
                n=min(4096,remaining.shape[1]-1)
                model(remaining[:,:n],cache=cache);mx.eval([c.state for c in cache]);remaining=remaining[:,n:];mx.clear_cache()
            saved=0
            for step in range(8):
                check();capture=[];cls.__call__=recorded
                logits=model(remaining,cache=cache)[:,-1,:];mx.eval(logits)
                cls.__call__=original
                if len(capture)!=24:raise ValueError('Incomplete expert capture')
                for layer,experts,x,ids,expected in capture:
                    actual=entry(experts,x,ids);mx.eval(expected,actual)
                    ref=np.array(expected.astype(mx.float32));cand=np.array(actual.astype(mx.float32))
                    comparison=compare_outputs(expected,actual,case_name='decode',policy=contract.correctness)
                    exact=bool(np.array_equal(ref,cand))
                    row={'step':step,'layer':layer,'bit_exact':exact,'passed_existing_operator_tolerance':comparison.passed,
                         'detail':comparison.detail,'max_abs':float(np.abs(ref-cand).max()),'different_elements':int(np.count_nonzero(ref!=cand))}
                    if not exact and saved<12:
                        expanded=mx.expand_dims(x,(-2,-3))
                        up=experts.up_proj(expanded,ids);gate=experts.gate_proj(expanded,ids)
                        reference_activation=experts.activation(up,gate)
                        candidate_activation=region(experts,x,ids)
                        mx.eval(up,gate,reference_activation,candidate_activation)
                        native_a=np.array(reference_activation.astype(mx.float32));candidate_a=np.array(candidate_activation.astype(mx.float32))
                        row['activation_exact']=bool(np.array_equal(native_a,candidate_a))
                        row['activation_max_abs']=float(np.abs(native_a-candidate_a).max())
                        path=WORK/f'boundary-step{step}-layer{layer}.npz'
                        np.savez_compressed(path,x=np.array(x.astype(mx.float32)),ids=np.array(ids),
                            reference=ref,candidate=cand,up=np.array(up.astype(mx.float32)),gate=np.array(gate.astype(mx.float32)),
                            native_activation=native_a,candidate_activation=candidate_a)
                        row['arrays']=str(path);row['arrays_sha256']=file_digest(path);saved+=1
                    state['boundary_checks'].append(row)
                save();print(f'boundary step{step}: {sum(not r["bit_exact"] for r in state["boundary_checks"] if r["step"]==step)} nonexact layers',flush=True)
                remaining=mx.array([[forced[step]]],dtype=mx.int32)
            cls.__call__=original
        state['status']='completed_numerical_diagnostic'
    except BaseException:
        state['status']='failed';state['errors'].append(traceback.format_exc());raise
    finally:
        if cls is not None and original is not None:cls.__call__=original
        state['sources_unchanged']=protected==_protected_files([__file__,WORK/'selected.json',WORK/'supervise.py',full_source])
        try:
            verify_frozen_candidate(frozen)
            validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files)
            state['checkpoint_and_snapshot_unchanged']=True
        except Exception as error:
            state['checkpoint_and_snapshot_unchanged']=False;state['errors'].append(str(error))
        state['ended_unix']=time.time();save()


if __name__=='__main__':main()
