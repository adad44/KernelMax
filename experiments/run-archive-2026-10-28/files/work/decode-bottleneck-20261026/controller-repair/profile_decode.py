"""External diagnostic controller. No verdicts; no protected source mutations."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import ast
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import types

BASE = Path('/Users/alandiaz/Downloads/KernelMaxxxing')
ROOT = BASE / 'kernelmax-first-loop'
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from evaluator.candidate_execution import _checkpoint_identity, _protected_files
from evaluator.native_baseline import environment, environment_issues, fixed_inputs, materialize_parameters, metrics
from evaluator.schemas import load_contract
from orchestrator.candidate_manager import FrozenCandidate, validate_candidate, verify_frozen_candidate
from scripts.run_reviewed_candidate import native_runtime


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def candidate_entry(frozen):
    """Keep frozen submissions in distinct packages in this multi-candidate process."""
    name = '_bottleneck_' + frozen.candidate_id
    package = types.ModuleType(name)
    package.__path__ = [str(frozen.directory)]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name+'.wrapper', frozen.directory/'wrapper.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    entry = module.run
    if Path(entry.__code__.co_filename).resolve() != (frozen.directory/'implementation.py').resolve():
        raise ValueError('candidate imported from a different snapshot')
    return entry


def up_gate_region(entry):
    """Extract the unchanged candidate function prefix ending at activated.

    This diagnostic region includes the candidate's own input validation, casts,
    projections and activation, and excludes only its final down-projection call.
    Retain source and verify against the native activation before any timings.
    """
    tree = ast.parse(inspect.getsource(entry))
    function = tree.body[0]
    last = function.body[-1]
    if not isinstance(last, ast.Return) or not isinstance(last.value, ast.Call):
        raise ValueError('candidate final return changed')
    if ast.unparse(last.value) not in (
        'down(activated, selected_expert_ids, sorted_indices=False).squeeze(-2)',
        'experts.down_proj(activated, selected_expert_ids, sorted_indices=False).squeeze(-2)',
    ):
        raise ValueError('candidate region no longer ends at the unchanged down projection')
    function.name = 'diagnostic_up_gate_region'
    last.value = ast.Name(id='activated', ctx=ast.Load())
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree)
    scope = dict(entry.__globals__)
    exec(compile(tree, '<diagnostic_up_gate_region>', 'exec'), scope)
    return scope['diagnostic_up_gate_region'], source


class Hooks:
    def __init__(self, model, mx):
        self.mx = mx
        self.mode = 'normal'
        self.step = None
        self.saved = {}
        self.serial = []
        self.originals = {}
        self.modules = {}
        for i, layer in enumerate(model.model.layers):
            for kind, module in [('input_norm', layer.input_layernorm),
                                 ('attention', layer.self_attn),
                                 ('post_norm', layer.post_attention_layernorm),
                                 ('router', layer.mlp.router), ('experts', layer.mlp.experts),
                                 ('up', layer.mlp.experts.up_proj), ('gate', layer.mlp.experts.gate_proj),
                                 ('activation', layer.mlp.experts.activation),
                                 ('down', layer.mlp.experts.down_proj)]:
                self.modules[id(module)] = (i, kind)
        for kind, module in [('embedding', model.model.embed_tokens),
                             ('final_norm', model.model.norm), ('lm_head', model.lm_head)]:
            self.modules[id(module)] = (-1, kind)
        objects = [model.model.embed_tokens, model.model.norm, model.lm_head]
        for layer in model.model.layers:
            objects.extend([layer.input_layernorm, layer.self_attn, layer.post_attention_layernorm,
                            layer.mlp.router, layer.mlp.experts, layer.mlp.experts.up_proj,
                            layer.mlp.experts.gate_proj, layer.mlp.experts.activation,
                            layer.mlp.experts.down_proj])
        for cls in {type(obj) for obj in objects}:
            original = cls.__call__
            self.originals[cls] = original
            def wrapped(module, *args, _original=original, **kwargs):
                key = self.modules.get(id(module))
                if self.step is None or key is None:
                    return _original(module, *args, **kwargs)
                layer, kind = key
                slot = (self.step, layer, kind)
                if self.mode == 'oracle_experts' and kind == 'experts':
                    return mx.depends(self.saved[slot], [args[0], args[1]])
                if self.mode == 'oracle_up_gate' and kind == 'experts':
                    activated = mx.depends(self.saved[(self.step, layer, 'activation')],
                                           [args[0], args[1]])
                    return module.down_proj(activated, args[1], sorted_indices=False).squeeze(-2)
                if self.mode == 'oracle_down' and kind == 'down':
                    return mx.depends(self.saved[slot], [args[0], args[1]])
                if self.mode == 'oracle_pointwise':
                    if kind in ('up', 'gate'):
                        raw = raw_projection(module, args[0], args[1], mx)
                        return mx.depends(self.saved[slot], raw)
                    if kind == 'activation':
                        return mx.depends(self.saved[slot], [args[0], args[1]])
                if self.mode == 'serial' and kind != 'experts' and self.step <= 16:
                    start = time.perf_counter()
                    mx.eval(args, kwargs)
                    ready = time.perf_counter()
                    out = _original(module, *args, **kwargs)
                    built = time.perf_counter()
                    mx.eval(out)
                    end = time.perf_counter()
                    self.serial.append({'step': self.step, 'layer': layer, 'kind': kind,
                                        'pending_ms': (ready-start)*1000,
                                        'build_ms': (built-ready)*1000,
                                        'eval_ms': (end-built)*1000})
                    return out
                out = _original(module, *args, **kwargs)
                if self.mode == 'capture' and kind in ('up', 'gate', 'activation', 'down', 'experts'):
                    self.saved[slot] = out
                    if kind == 'experts':
                        self.saved[(self.step, layer, 'x')] = args[0]
                        self.saved[(self.step, layer, 'ids')] = args[1]
                return out
            cls.__call__ = wrapped

    def restore(self):
        for cls, original in self.originals.items():
            cls.__call__ = original


def raw_projection(projection, x, ids, mx):
    return mx.gather_qmm(x, projection.weight, projection.scales, projection.get('biases'),
                         rhs_indices=ids, transpose=True, group_size=projection.group_size,
                         bits=projection.bits, mode=projection.mode, sorted_indices=False)


def request(model, ids, mx, make_prompt_cache, hooks=None, reference_logits=None, retain_logits=False):
    cache = make_prompt_cache(model)
    prompt = mx.array([ids], dtype=mx.int32)
    mx.eval(prompt)
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.synchronize()
    if hooks:
        hooks.step = None
    start = time.perf_counter()
    model(prompt[:, :-1], cache=cache)
    mx.eval([c.state for c in cache])
    mx.clear_cache()
    logits = model(prompt[:, -1:], cache=cache)[:, -1, :]
    mx.eval(logits)
    mx.synchronize()
    prefill_end = time.perf_counter()
    token = mx.argmax(logits, axis=-1)
    mx.eval(token)
    tokens = [token.item()]
    timestamps = [time.perf_counter()]
    parts = []
    captured_logits = []
    max_error = 0.0
    for step in range(1, 128):
        if hooks:
            hooks.step = step
        t0 = time.perf_counter()
        logits = model(token[:, None], cache=cache)[:, -1, :]
        token = mx.argmax(logits, axis=-1)
        t1 = time.perf_counter()
        mx.eval(token)
        t2 = time.perf_counter()
        tokens.append(token.item())
        t3 = time.perf_counter()
        timestamps.append(t3)
        parts.append({'step': step, 'build_ms': (t1-t0)*1000, 'eval_wait_ms': (t2-t1)*1000,
                      'readback_ms': (t3-t2)*1000, 'total_ms': (t3-t0)*1000})
        # Retain evaluated logits. All comparisons run after the request interval;
        # no validation GPU work or readbacks are inserted between decode steps.
        if retain_logits or (hooks and hooks.mode == 'capture') or reference_logits is not None:
            captured_logits.append(logits)
    mx.synchronize()
    if hooks:
        hooks.step = None
    values, intervals = metrics(len(ids), start, prefill_end, timestamps, mx.get_peak_memory())
    if reference_logits is not None:
        for actual, expected in zip(captured_logits, reference_logits, strict=True):
            error = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32)))
            max_error = max(max_error, error.item())
    finite = mx.all(mx.isfinite(logits)).item()
    cache_dtypes = sorted({str(t.dtype) for c in cache for t in c.state})
    if not finite or cache_dtypes != ['mlx.core.bfloat16']:
        raise ValueError('native output/cache mismatch')
    return {'metrics': values, 'step_times': parts, 'inter_token_intervals_ms': intervals,
            'tokens': tokens, 'max_abs_logit_error': max_error,
            'cache_dtypes': cache_dtypes}, captured_logits


def main():
    if (WORK / 'state.json').exists():
        raise ValueError('Existing attempt retained; use a fresh directory')
    contract = load_contract(ROOT / 'kernelmaxxing.yaml')
    runtime = native_runtime(Path('/Volumes/BankOfSouls/models/gpt-oss-20b'), contract)
    expected = json.loads((BASE / 'work/second-optimization-20261026/native-correctness/artifacts/execution-ka4ns_ag/checkpoint_identity.json').read_text())
    identity = lambda: json.loads(json.dumps(_checkpoint_identity(runtime)))
    if identity() != expected:
        raise ValueError('checkpoint identity changed since byte verification')
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import wired_limit
    from mlx_lm.models.cache import make_prompt_cache
    from mlx.utils import tree_flatten
    import mlx_lm.models.gpt_oss as gpt_oss
    import mlx_lm.models.switch_layers as switch_layers
    extra = [__file__, WORK / 'PLAN.md', gpt_oss.__file__, switch_layers.__file__]
    protected = _protected_files(extra)
    state = {'status': 'loading', 'official': False, 'verdict': None, 'acceptance_enabled': False,
             'contract_sha256': contract.sha256, 'protected_before': protected,
             'checkpoint_identity': expected, 'runtime': runtime.payload(),
             'packages': {n: importlib.metadata.version(n) for n in ('mlx', 'mlx-lm', 'numpy')},
             'environment_at_start': environment(), 'requests': [], 'operators': [], 'errors': []}
    save = lambda: write(WORK / 'state.json', state)
    save()
    hooks = None
    try:
        if "'AC Power'" not in environment()['power']:
            raise RuntimeError('AC unavailable; no checkpoint load or measured request started')
        candidates = []
        for dirname in ('first-optimization-20261026', 'second-optimization-20261026'):
            approval = json.loads((BASE/'work'/dirname/'approval.json').read_text())
            c = validate_candidate(Path(approval['snapshot']), Path(approval['snapshot']).parent)
            hashes = {p.relative_to(c.directory).as_posix(): digest(p)
                      for p in c.directory.rglob('*') if p.is_file()}
            frozen = FrozenCandidate(**vars(c), sha256=approval['snapshot_sha256'], file_sha256=hashes)
            verify_frozen_candidate(frozen)
            entry = candidate_entry(frozen)
            region, source = up_gate_region(entry)
            path = WORK / (frozen.candidate_id+'-diagnostic-region.py')
            path.write_text(source+'\n')
            candidates.append((frozen, entry, region))
        assert candidates[0][1] is not candidates[1][1]
        assert candidates[0][1].__module__ != candidates[1][1].__module__
        state['candidate_imports'] = [{'candidate_id': f.candidate_id, 'snapshot_sha256': f.sha256,
                                       'module': entry.__module__, 'source_file': entry.__code__.co_filename,
                                       'region_source_sha256': digest(WORK/(f.candidate_id+'-diagnostic-region.py'))}
                                      for f, entry, region in candidates]
        save()
        with mx.stream(mx.cpu):
            model, _ = load(str(runtime.model_path), lazy=True)
            def progress(done, total, name, size):
                if done % 25 == 0 or done == total:
                    print(f'load {done}/{total}: {name}', flush=True)
                    state['load_progress'] = {'done': done, 'total': total, 'name': name}
                    save()
            materialize_parameters(tree_flatten(model.parameters()), mx, progress)
        for layer in model.model.layers:
            for name in ('up_proj', 'gate_proj', 'down_proj'):
                p = getattr(layer.mlp.experts, name)
                assert (p.mode, p.bits, p.group_size) == ('mxfp4', 4, 32)
        inputs = fixed_inputs([512, 2048, 4096], model.args.vocab_size, 42)
        state['input_token_ids'] = inputs
        state['status'] = 'waiting_for_ac'
        save()
        deadline = time.monotonic() + 600
        while "'AC Power'" not in environment()['power']:
            if time.monotonic() > deadline:
                raise RuntimeError('AC unavailable; no measured request started')
            print('Untimed load finished; waiting for AC', flush=True)
            time.sleep(10)
        def run(label, length, mode='normal', ref=None, retain_logits=False):
            if identity() != expected:
                raise ValueError('checkpoint changed')
            before = environment()
            if "'AC Power'" not in before['power']:
                raise RuntimeError('AC removed; no additional timing started')
            if hooks:
                hooks.mode = mode
            print(f'request {label}: context={length}, mode={mode}', flush=True)
            result, logs = request(model, inputs[str(length)], mx, make_prompt_cache, hooks, ref, retain_logits)
            after = environment()
            row = {'label': label, 'length': length, 'mode': mode,
                   'before': before, 'after': after,
                   'environment_issues': environment_issues(before, after), **result}
            state['requests'].append(row)
            save()
            print(f"done {label}: {result['metrics']['decode_tokens_per_second']:.3f} tok/s; issues={row['environment_issues']}", flush=True)
            return row, logs
        with wired_limit(model):
            state['status'] = 'baseline'
            save()
            reference_tokens = {}
            for length in (512, 2048, 4096):
                warm, _ = run(f'warmup-{length}', length)
                reference_tokens[length] = warm['tokens']
                for repeat in range(3):
                    row, _ = run(f'baseline-{length}-{repeat}', length)
                    if row['tokens'] != reference_tokens[length]:
                        raise ValueError('unchanged model token trajectory changed')
            hooks = Hooks(model, mx)
            state['status'] = 'capture'
            _, control_logits = run('capture-control-512', 512, retain_logits=True)
            captured, logs = run('capture-512', 512, 'capture', control_logits)
            if captured['tokens'] != reference_tokens[512] or captured['max_abs_logit_error'] != 0:
                raise ValueError('capture changed tokens or logits')
            del control_logits
            assert len(logs) == 127 and len(hooks.saved) == 127*24*7
            cases = []
            import numpy as np
            arrays = WORK / 'real-activations'
            arrays.mkdir()
            steps = (1, 19, 37, 55, 73, 91, 109, 127)
            for layer in (0, 12, 23):
                for step in steps:
                    x = hooks.saved[(step, layer, 'x')]
                    ids = hooks.saved[(step, layer, 'ids')]
                    cases.append((step, layer, x, ids))
                    path = arrays / f'layer-{layer}-step-{step}.npz'
                    np.savez_compressed(path, x=np.array(x.astype(mx.float32)), ids=np.array(ids))
            state['real_cases'] = [{'step': s, 'layer': l, 'ids': np.array(i).tolist(),
                                    'path': str(arrays/f'layer-{l}-step-{s}.npz'),
                                    'sha256': digest(arrays/f'layer-{l}-step-{s}.npz')}
                                   for s, l, x, i in cases]
            state['captured_entries'] = len(hooks.saved)
            save()
            # Untimed first execution primes replay dependencies; all validation retained.
            state['status'] = 'oracle_replays'
            save()
            for mode in ('oracle_pointwise', 'oracle_up_gate', 'oracle_down', 'oracle_experts'):
                warm, _ = run(f'warmup-{mode}', 512, mode, logs)
                if warm['max_abs_logit_error'] != 0 or warm['tokens'] != captured['tokens']:
                    raise ValueError('oracle replay did not preserve exact logits')
                for pair in range(3):
                    lanes = ('normal', mode) if pair % 2 == 0 else (mode, 'normal')
                    for lane in lanes:
                        row, _ = run(f'{mode}-pair-{pair}-{lane}', 512, lane, logs)
                        if row['max_abs_logit_error'] != 0 or row['tokens'] != captured['tokens']:
                            raise ValueError('oracle/control did not preserve exact logits')
            state['status'] = 'serial_stages'
            save()
            serial, _ = run('serial-stage-512', 512, 'serial')
            if serial['tokens'] != captured['tokens']:
                raise ValueError('serial instrumentation changed tokens')
            write(WORK / 'serial-stages.json', hooks.serial)
            # Restore classes before operator comparisons, including candidate type checks.
            hooks.restore()
            saved = hooks.saved
            hooks = None
            # All operands for substage timings are already materialized real model values.
            prepared = []
            for step, layer, x, ids in cases:
                e = model.model.layers[layer].mlp.experts
                raw_up = raw_projection(e.up_proj, mx.expand_dims(x, (-2, -3)), ids, mx)
                raw_gate = raw_projection(e.gate_proj, mx.expand_dims(x, (-2, -3)), ids, mx)
                mx.eval(raw_up, raw_gate)
                prepared.append((e, x, ids, raw_up, raw_gate, saved[(step, layer, 'activation')]))
            def op(kind, case):
                e, x, ids, raw_up, raw_gate, act = case
                xx = mx.expand_dims(x, (-2, -3))
                if kind == 'native':
                    return e(x, ids)
                if kind == 'candidate1':
                    return candidates[0][1](e, x, ids.astype(mx.int32))
                if kind == 'candidate2':
                    return candidates[1][1](e, x, ids)
                if kind == 'candidate1_up_gate':
                    return candidates[0][2](e, x, ids.astype(mx.int32))
                if kind == 'candidate2_up_gate':
                    return candidates[1][2](e, x, ids)
                if kind == 'raw_up_gate':
                    return (raw_projection(e.up_proj, xx, ids, mx), raw_projection(e.gate_proj, xx, ids, mx))
                if kind == 'up_gate_activation':
                    return e.activation(e.up_proj(xx, ids), e.gate_proj(xx, ids))
                if kind == 'pointwise':
                    return e.activation(raw_up + mx.expand_dims(e.up_proj.bias[ids], -2),
                                        raw_gate + mx.expand_dims(e.gate_proj.bias[ids], -2))
                if kind == 'down':
                    return e.down_proj(act, ids)
                raise ValueError(kind)
            checks = []
            for case in prepared:
                ref = op('native', case)
                for kind in ('candidate1', 'candidate2', 'candidate1_up_gate', 'candidate2_up_gate'):
                    expected = op('up_gate_activation', case) if kind.endswith('_up_gate') else ref
                    value = op(kind, case)
                    delta_array = mx.abs(value.astype(mx.float32)-expected.astype(mx.float32))
                    delta = mx.max(delta_array).item()
                    passed = mx.all(delta_array <= .02 + .02*mx.abs(expected.astype(mx.float32))).item()
                    checks.append({'lane': kind, 'max_abs_error': delta, 'passed': passed})
                    if not passed:
                        raise ValueError('candidate failed on real activation')
            state['real_operator_checks'] = checks
            state['status'] = 'operator_timings'
            save()
            kinds = ('native', 'candidate1', 'candidate2', 'raw_up_gate', 'up_gate_activation',
                     'candidate1_up_gate', 'candidate2_up_gate', 'pointwise', 'down')
            def invoke(kind):
                count = 0
                for _ in range(8):
                    for case in prepared:
                        mx.eval(op(kind, case))
                        count += 1
                assert count == 192
                return count
            for _ in range(5):
                for kind in kinds:
                    invoke(kind)
            for pair in range(20):
                order = kinds if pair % 2 == 0 else tuple(reversed(kinds))
                before = environment()
                if "'AC Power'" not in before['power']:
                    raise RuntimeError('AC removed')
                for kind in order:
                    mx.synchronize()
                    start = time.perf_counter()
                    count = invoke(kind)
                    mx.synchronize()
                    elapsed = (time.perf_counter()-start)*1000
                    state['operators'].append({'pair': pair, 'kind': kind, 'calls': count,
                                               'elapsed_ms': elapsed})
                after = environment()
                state.setdefault('operator_environment', []).append({'pair': pair, 'before': before, 'after': after,
                                                                     'issues': environment_issues(before, after)})
                save()
                print(f'operator block {pair+1}/20 retained', flush=True)
            for frozen, entry, region in candidates:
                verify_frozen_candidate(frozen)
        state['status'] = 'completed_diagnostic'
    except BaseException:
        state['status'] = 'failed_diagnostic'
        state['errors'].append(traceback.format_exc())
        raise
    finally:
        if hooks:
            hooks.restore()
        state['protected_after'] = _protected_files(extra)
        state['protected_unchanged'] = state['protected_after'] == protected
        state['checkpoint_identity_unchanged'] = identity() == expected
        state['environment_at_end'] = environment()
        save()


if __name__ == '__main__':
    main()
