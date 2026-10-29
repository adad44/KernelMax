"""Whole-decode causal deletion diagnostic; never an acceptance evaluator."""
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

WORK = Path(__file__).resolve().parent
ROOT = Path('/Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop')
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from evaluator.schemas import load_contract
from evaluator.candidate_execution import _protected_files
from evaluator.checkpoint_verification import validate_receipt
from evaluator.native_baseline import (environment, environment_issues, fixed_inputs,
    materialize_parameters, metrics, quiet_preflight, swap_observation)
from evaluator.baseline_validation import aa_noise, thermal_state
from scripts.loop_control import load_authorization
from scripts.run_reviewed_candidate import native_runtime


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    tmp.replace(path)


def interval(values, seed=42):
    rng = random.Random(seed)
    b = sorted(st.median(rng.choices(values, k=len(values))) for _ in range(10000))
    return [b[249], b[9749]]


class Hooks:
    def __init__(self, model, mx):
        self.mx = mx
        self.step = None
        self.mode = 'normal'
        self.saved = {}
        self.originals = {}
        self.modules = {}
        for i, layer in enumerate(model.model.layers):
            for kind, module in [('attention', layer.self_attn),
                                 ('experts', layer.mlp.experts),
                                 ('small_ops', layer.input_layernorm),
                                 ('small_ops', layer.post_attention_layernorm),
                                 ('small_ops', layer.mlp.router)]:
                self.modules[id(module)] = (i, kind, id(module))
        for kind, module in [('small_ops', model.model.embed_tokens),
                             ('small_ops', model.model.norm), ('lm_head', model.lm_head)]:
            self.modules[id(module)] = (-1, kind, id(module))
        objects = [model.model.embed_tokens, model.model.norm, model.lm_head]
        for l in model.model.layers:
            objects.extend([l.self_attn, l.mlp.experts, l.input_layernorm,
                            l.post_attention_layernorm, l.mlp.router])
        for cls in {type(o) for o in objects}:
            original = cls.__call__
            self.originals[cls] = original
            def wrapped(module, *args, _original=original, **kwargs):
                key = self.modules.get(id(module))
                if key is None or self.step is None:
                    return _original(module, *args, **kwargs)
                slot = (self.step, key[2])
                if self.mode == 'delete_'+key[1]:
                    # Build attention to advance Python cache offsets/layout.
                    # Its unused lazy GPU/cache graph is intentionally deleted.
                    # Thus this measures the entire attention+cache upper budget.
                    if key[1] == 'attention':
                        _original(module, *args, **kwargs)
                    return mx.depends(self.saved[slot], list(args[:1]))
                out = _original(module, *args, **kwargs)
                if self.mode == 'capture':
                    self.saved[slot] = out
                return out
            cls.__call__ = wrapped

    def restore(self):
        for cls, original in self.originals.items():
            cls.__call__ = original


def request(model, ids, mx, make_prompt_cache, hooks, reference=None):
    cache = make_prompt_cache(model)
    prompt = mx.array([ids], dtype=mx.int32)
    mx.eval(prompt); mx.clear_cache(); mx.reset_peak_memory(); mx.synchronize()
    hooks.step = None
    start = time.perf_counter()
    model(prompt[:, :-1], cache=cache); mx.eval([c.state for c in cache]); mx.clear_cache()
    logits = model(prompt[:, -1:], cache=cache)[:, -1, :]
    mx.eval(logits); mx.synchronize()
    prefill_end = time.perf_counter()
    token = mx.argmax(logits, axis=-1); mx.eval(token)
    tokens = [token.item()]; stamps = [time.perf_counter()]
    parts = []; logs = []
    for step in range(1,128):
        hooks.step = step
        c0 = time.process_time(); t0 = time.perf_counter()
        logits = model(token[:, None], cache=cache)[:, -1, :]
        token = mx.argmax(logits, axis=-1)
        t1 = time.perf_counter(); c1 = time.process_time()
        mx.eval(token)
        t2 = time.perf_counter(); c2 = time.process_time()
        tokens.append(token.item())
        t3 = time.perf_counter(); stamps.append(t3)
        parts.append({'build_cpu_ms': (c1-c0)*1000, 'eval_cpu_ms': (c2-c1)*1000, 'build_ms': (t1-t0)*1000, 'eval_wait_ms': (t2-t1)*1000,
                      'readback_ms': (t3-t2)*1000, 'total_ms': (t3-t0)*1000})
        if hooks.mode != 'normal' or reference is not None:
            logs.append(logits)
    mx.synchronize(); hooks.step = None
    values, gaps = metrics(len(ids), start, prefill_end, stamps, mx.get_peak_memory())
    max_error = None
    if reference is not None:
        max_error = 0.
        for actual, expected in zip(logs, reference, strict=True):
            max_error = max(max_error, mx.max(mx.abs(actual.astype(mx.float32)-expected.astype(mx.float32))).item())
    finite = bool(mx.all(mx.isfinite(logits)).item())
    dtypes = sorted({str(t.dtype) for c in cache for t in c.state})
    if not finite or dtypes != ['mlx.core.bfloat16']:
        raise ValueError('Native precision/finite validation failed')
    return {'metrics':values, 'inter_token_intervals_ms':gaps, 'step_times':parts,
            'output_token_ids':tokens, 'cache_dtypes':dtypes,
            'max_abs_logit_error':max_error}, logs


def main():
    output = WORK/'profile-state.json'
    if output.exists(): raise ValueError('Fresh attempts only')
    contract = load_contract(ROOT/'kernelmaxxing.yaml')
    session = load_authorization(WORK/'session.json', contract)
    runtime = native_runtime(Path(session['model_path']), contract)
    receipt = json.loads(Path(session['checkpoint_cache']).read_text())
    validate_receipt(receipt, runtime.model_path, runtime.model_revision, runtime.model_files)
    import mlx.core as mx
    import mlx_lm.models.gpt_oss as gpt_oss
    import mlx_lm.models.switch_layers as switch_layers
    from mlx_lm import load
    from mlx_lm.generate import wired_limit
    from mlx_lm.models.cache import make_prompt_cache
    from mlx.utils import tree_flatten
    extra = [__file__,WORK/'plan.json',WORK/'session.json',gpt_oss.__file__,switch_layers.__file__,mx.__file__]
    protected = _protected_files(extra)
    state = {'schema':1, 'status':'loading', 'official':False, 'verdict':None,
        'acceptance_enabled':False, 'contract_sha256':contract.sha256,
        'protected_before':protected, 'checkpoint_receipt':receipt,
        'packages':{n:importlib.metadata.version(n) for n in ('mlx','mlx-lm','numpy')},
        'requests':[], 'summaries':{}, 'errors':[], 'pid':os.getpid(), 'started_unix':time.time()}
    save = lambda:write(output,state)
    save(); hooks = None
    def check():
        validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files)
        if time.time()-state['started_unix']>10800: raise TimeoutError('Three-hour profiling cap')
    try:
        with mx.stream(mx.cpu):
            model,_ = load(str(runtime.model_path),lazy=True)
            def progress(done,total,name,size):
                if done%25==0 or done==total:
                    state['load_progress']={'done':done,'total':total,'name':name};save()
                    print(f'load {done}/{total}',flush=True)
            materialize_parameters(tree_flatten(model.parameters()),mx,progress)
        check()
        state['model_bytes']=sum(t.nbytes for _,t in tree_flatten(model.parameters()))
        state['lm_head_shape']=list(model.lm_head.weight.shape)
        hooks=Hooks(model,mx)
        ids=fixed_inputs([512,2048,4096],model.args.vocab_size,42)['512']
        state['input_token_ids']=ids
        ref_tokens=None
        def collect(label,mode='normal',reference=None,lane=None,pair=None):
            check(); time.sleep(12); time.sleep(5)
            quiet=quiet_preflight(quiet_seconds=10,timeout_seconds=180,swap_policy='paging-aware')
            hooks.mode=mode
            before=environment(); pre=environment_issues(before,before,'paging-aware')
            if pre:raise RuntimeError('Pre-request invalid: '+str(pre))
            seen=[before['thermal_state']]; stop=threading.Event()
            def observe():
                while not stop.wait(.5):seen.append(thermal_state())
            thread=threading.Thread(target=observe,daemon=True);thread.start()
            try: result,logs=request(model,ids,mx,make_prompt_cache,hooks,reference)
            finally:stop.set();thread.join()
            after=environment(); issues=environment_issues(before,after,'paging-aware')
            if any(t!=0 for t in seen):issues.append('Nonnominal thermal during request')
            row={'label':label,'mode':mode,'lane':lane,'trial':pair,'prompt_tokens':512,
                 'environment_before':before,'environment_after':after,'thermal_states':seen,
                 'invalid_reasons':issues,'quiet_preflight':quiet,'swap_activity':swap_observation(before,after,'paging-aware'),**result}
            state['requests'].append(row);save()
            print(f'{label}: {1000/result["metrics"]["decode_tokens_per_second"]:.3f} ms/token; flags={issues}',flush=True)
            if issues:raise RuntimeError('Environment invalid; retained sample: '+str(issues))
            if ref_tokens is not None and row['output_token_ids']!=ref_tokens:
                raise ValueError('Token trajectory changed')
            if reference is not None and row['max_abs_logit_error']!=0:
                raise ValueError('Deletion failed exact logit replay')
            return row,logs
        with wired_limit(model):
            state['status']='resident_warmup';save()
            # Untimed hydration is retained but may page; no sample selection.
            for n in range(2):
                hooks.mode='normal'; row,_=request(model,ids,mx,make_prompt_cache,hooks)
                state.setdefault('hydration',[]).append(row);save()
            state['quiet_preflight']=quiet_preflight(quiet_seconds=10,timeout_seconds=180,swap_policy='paging-aware');save()
            for n in range(5):
                row,_=collect(f'warmup-{n}')
                if ref_tokens is None:ref_tokens=row['output_token_ids']
            state['status']='aa_calibration';save(); aa=[]
            for pair in range(1,21):
                for lane in (('A','B') if pair%2 else ('B','A')):
                    row,_=collect(f'aa-{pair}-{lane}',lane=lane,pair=pair);aa.append(row)
            state['aa']=aa_noise(aa,[512])['512'];save()
            if not state['aa']['passed']:
                state['status']='stopped_aa';save();return
            state['status']='capture';save()
            captured,reference=collect('capture',mode='capture')
            state['capture_entries']=len(hooks.saved);save()
            # Full-step removal is only an optimistic budget, never an implementation.
            for region in ('experts','attention','lm_head','small_ops'):
                mode='delete_'+region;state['status']=mode;save()
                for n in range(5):collect(f'{mode}-warmup-{n}',mode,reference)
                controls={};deleted={}
                for pair in range(1,6):
                    for lane in (('normal',mode) if pair%2 else (mode,'normal')):
                        row,_=collect(f'{mode}-{pair}-{lane}',lane,reference,pair=pair)
                        (controls if lane=='normal' else deleted)[pair]=row
                savings=[1000/controls[p]['metrics']['decode_tokens_per_second']-1000/deleted[p]['metrics']['decode_tokens_per_second'] for p in range(1,6)]
                state['summaries'][region]={'saved_ms_per_token':savings,'median_saved_ms_per_token':st.median(savings),
                    'bootstrap_95pct_saved_ms':interval(savings),'scope':'exact-output cached deletion; optimistic nonadditive budget, not a deployable optimization'}
                save()
            state['status']='completed_profile'
    except BaseException:
        state['status']='failed';state['errors'].append(traceback.format_exc());raise
    finally:
        if hooks:hooks.restore()
        state['protected_unchanged']=protected==_protected_files(extra)
        try:check();state['checkpoint_unchanged']=True
        except Exception:state['checkpoint_unchanged']=False;state['errors'].append(traceback.format_exc())
        state['finished_unix']=time.time();save()


if __name__=='__main__': main()
