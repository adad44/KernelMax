"""Prepared, bounded same-input diagnostic at the retained first-failure steps.

Never run concurrently with a resident model worker. One native 128-output
request, 120 selected layer probes, no timing evidence and no new candidate ID.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
import traceback

WORK=Path(__file__).resolve().parent
ROOT=WORK.parent.parent/'kernelmax-first-loop'
sys.path.insert(0,str(ROOT))


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir()
    state={'status':'preflight','performance_evidence':False,'full_model_requests':0,
           'max_full_model_requests':1,'max_stage_probes':120,'cases':[],'errors':[]}
    save=lambda:write(args.output/'state.json',state)
    save()
    from evaluator.schemas import load_contract
    from evaluator.candidate_execution import _protected_files
    from evaluator.checkpoint_verification import validate_receipt
    from evaluator.native_baseline import fixed_inputs,materialize_parameters,request,environment
    from evaluator.measurement import assess_environment
    from evaluator.baseline_validation import thermal_state
    from scripts.loop_control import load_authorization
    from scripts.run_reviewed_candidate import native_runtime
    from orchestrator.candidate_manager import FrozenCandidate,validate_candidate,verify_frozen_candidate
    contract=load_contract(ROOT/'kernelmaxxing.yaml')
    session=load_authorization(WORK/'session.json',contract)
    runtime=native_runtime(Path(session['model_path']),contract)
    receipt=json.loads(Path(session['checkpoint_cache']).read_text())
    validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files)
    info=json.loads((WORK.parent/'decode-sprint-20261028/moe-frozen.json').read_text())
    candidate=validate_candidate(info['directory'],Path(info['directory']).parent)
    frozen=FrozenCandidate(**vars(candidate),sha256=info['sha256'],file_sha256=info['file_sha256'])
    verify_frozen_candidate(frozen)
    old_arrays=WORK.parent/'decode-sprint-20261028/command-004-confirm-cand_3f4c02fcfc71/correctness-2048-128.npz'
    if hashlib.sha256(old_arrays.read_bytes()).hexdigest()!='218581d49fbed41014cc594127d1b65d8e3ac2c49716cc8d2d9b661595673921':
        raise ValueError('Retained failure artifact changed')
    import mlx.core as mx
    import mlx_lm.models.gpt_oss as gpt_oss
    import mlx_lm.models.switch_layers as switch_layers
    import numpy as np
    from mlx_lm import load
    from mlx_lm.generate import wired_limit
    from mlx_lm.models.cache import make_prompt_cache
    from mlx.utils import tree_flatten
    protected_paths=[__file__,WORK/'session.json',WORK/'plan.json',mx.__file__,gpt_oss.__file__,switch_layers.__file__]
    protected=_protected_files(protected_paths)
    started=time.monotonic()
    state.update(status='loading',candidate_source_sha256=frozen.sha256,contract_sha256=contract.sha256)
    save()
    originals={}
    stop=threading.Event()
    observer=None
    try:
        with mx.stream(mx.cpu):
            model,_=load(str(runtime.model_path),lazy=True)
            def progress(done,total,name,size):
                if done%25==0 or done==total:
                    state['load_progress']={'done':done,'total':total};save()
            materialize_parameters(tree_flatten(model.parameters()),mx,progress)
        spec=importlib.util.spec_from_file_location('old_fused_moe_for_diagnosis',frozen.directory/'implementation.py')
        old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
        route_source=(frozen.directory/'route.metal').read_text()+'\nif(tid<32) all_scores[tid]=scores[tid];\n'
        route_probe=mx.fast.metal_kernel(name='r2_actual_input_scores',
            input_names=['residual','attention','norm_weight','router_weight','router_bias'],
            output_names=['norm_out','residual_out','ids','weights','all_scores'],
            source=route_source,header=old._HEADER,ensure_row_contiguous=True,
            compile_options={'math_mode':'safe'})
        layers=model.model.layers
        block_cls=type(layers[0]);attention_cls=type(layers[0].self_attn)
        norm_cls=type(layers[0].post_attention_layernorm);linear_cls=type(layers[0].mlp.router)
        originals={cls:cls.__call__ for cls in (block_cls,attention_cls,norm_cls,linear_cls)}
        layer_ids={id(layer):i for i,layer in enumerate(layers)}
        context={'step':None,'active':None,'calls':[]}
        def attention_call(layer,*a,**kw):
            result=originals[attention_cls](layer,*a,**kw)
            active=context['active']
            if active is not None and layer is active['layer'].self_attn:
                active['attention']=result
            return result
        def norm_call(layer,x):
            result=originals[norm_cls](layer,x)
            active=context['active']
            if active is not None and layer is active['layer'].post_attention_layernorm:
                active['merged'],active['norm']=x,result
            return result
        def linear_call(layer,x):
            result=originals[linear_cls](layer,x)
            active=context['active']
            if active is not None and layer is active['layer'].mlp.router:
                active['scores']=result
            return result
        def difference(a,b):
            av=np.array(a.astype(mx.float32)).astype(np.float64)
            bv=np.array(b.astype(mx.float32)).astype(np.float64)
            return {'bit_exact':bool(np.array_equal(np.array(a.view(mx.uint16)),np.array(b.view(mx.uint16)))),
                    'different_elements':int(np.count_nonzero(av!=bv)),
                    'tolerance_violations':int(np.count_nonzero(np.abs(av-bv)>.02+.02*np.abs(av))),
                    'max_abs_error':float(np.max(np.abs(av-bv)))}
        def block_call(layer,x,mask,cache=None):
            selected=context['step'] in (104,105,106,107,108) and x.shape[:2]==(1,1)
            if not selected: return originals[block_cls](layer,x,mask,cache)
            if len(state['cases'])>=120 or time.monotonic()-started>1800:
                raise TimeoutError('Diagnostic probe/time bound')
            active={'layer':layer,'input':x}
            context['active']=active
            try:
                native=originals[block_cls](layer,x,mask,cache)
            finally:
                context['active']=None
            for name in ('attention','merged','norm','scores'):
                if name not in active: raise ValueError('Missing native stage '+name)
            h=x.shape[-1]
            fn,fr,fi,fw,fs=route_probe(inputs=[x,active['attention'],layer.post_attention_layernorm.weight,
                    layer.mlp.router.weight,layer.mlp.router.bias],template=[('T',mx.bfloat16),('H',h)],
                grid=(736,1,1),threadgroup=(736,1,1),
                output_shapes=[(1,1,h),(1,1,h),(1,1,4),(1,1,4),(1,1,32)],
                output_dtypes=[mx.bfloat16,mx.bfloat16,mx.uint32,mx.bfloat16,mx.bfloat16])
            fused=old.expert_reduce(layer.mlp.experts,fn,fi,fw,fr)
            ni=mx.argpartition(active['scores'],kth=-4,axis=-1)[...,-4:]
            nw=mx.softmax(mx.take_along_axis(active['scores'],ni,axis=-1),axis=-1,precise=True)
            mx.eval(native,fused,active['merged'],active['norm'],active['scores'],fr,fn,fs,fi,fw,ni,nw)
            row={'step_zero_based':context['step'],'layer':layer_ids[id(layer)],
                 'residual':difference(active['merged'],fr),'norm':difference(active['norm'],fn),
                 'scores':difference(active['scores'],fs),'weights':difference(nw,fw),
                 'block_output':difference(native,fused),'native_ids':np.array(ni).reshape(-1).tolist(),
                 'fused_ids':np.array(fi).reshape(-1).tolist()}
            if any(not row[k]['bit_exact'] for k in ('residual','norm','scores','weights','block_output')):
                path=args.output/f"step-{context['step']:03d}-layer-{layer_ids[id(layer)]:02d}.npz"
                np.savez_compressed(path,**{name:np.array(value.view(mx.uint16)) for name,value in {
                    'input':x,'attention':active['attention'],'native_norm':active['norm'],
                    'fused_norm':fn,'native_scores':active['scores'],'fused_scores':fs,
                    'native_block':native,'fused_block':fused}.items()})
                row['arrays']=str(path)
            state['cases'].append(row)
            return native
        attention_cls.__call__=attention_call;norm_cls.__call__=norm_call
        linear_cls.__call__=linear_call;block_cls.__call__=block_call
        class ObservedNative:
            def __getattr__(self,name): return getattr(model,name)
            def __call__(self,tokens,cache=None):
                context['calls'].append(list(tokens.shape))
                context['step']=None if tokens.shape[1]>1 else len(context['calls'])-2
                return model(tokens,cache=cache)
        ids=fixed_inputs([512,2048,4096],model.args.vocab_size,42)['2048']
        before=environment();seen=[before['thermal_state']]
        def observe():
            while not stop.wait(.5): seen.append(thermal_state())
        observer=threading.Thread(target=observe,daemon=True);observer.start()
        state['status']='diagnostic_native_request';state['full_model_requests']=1;save()
        with wired_limit(model): result=request(ObservedNative(),ids,4096,mx,make_prompt_cache)
        stop.set();observer.join();observer=None
        after=environment()
        flags=assess_environment(before,after,mode='exploration',thermal_states=seen)
        with np.load(old_arrays,allow_pickle=False) as arrays:
            matched=result['output_token_ids']==arrays['reference_tokens'].tolist()
        state.update(status='completed_diagnostic',result=result,environment_before=before,
            environment_after=after,environment_assessment=flags,native_reference_tokens_match=matched,
            native_forward_shapes=context['calls'])
        if not matched or context['calls']!=[[1,2047]]+[[1,1]]*128 or len(state['cases'])!=120:
            raise ValueError('Diagnostic workload/coverage differs from native reference')
        if not flags['continue_pass']: raise ValueError('Hard environment failure in diagnostic')
    except BaseException:
        state['status']='failed_diagnostic';state['errors'].append(traceback.format_exc());raise
    finally:
        stop.set()
        if observer is not None: observer.join()
        for cls,original in originals.items(): cls.__call__=original
        state['methods_restored']=all(cls.__call__ is original for cls,original in originals.items())
        state['protected_unchanged']=protected==_protected_files(protected_paths)
        try:
            verify_frozen_candidate(frozen)
            validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files)
            state['checkpoint_and_snapshot_unchanged']=True
        except Exception:
            state['checkpoint_and_snapshot_unchanged']=False;state['errors'].append(traceback.format_exc())
        state['wall_seconds']=time.monotonic()-started
        save()


if __name__=='__main__': main()
