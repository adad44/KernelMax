"""Controller-owned decode diagnostic; never issues a candidate verdict.

Requires the exact approved snapshot to have passed the protected native suite.
Keeps every paired sample, environment observation, source hash, and failure.
This is not the contract's full-model 128-token benchmark or task-quality gate.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path('/Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop')
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluator.candidate_execution import _checkpoint_identity, _protected_files
from evaluator.candidate_worker import _candidate_entry
from evaluator.correctness import compare_outputs
from evaluator.native_baseline import environment, environment_issues, materialize_parameters
from evaluator.schemas import load_contract
from orchestrator.candidate_manager import FrozenCandidate, validate_candidate, verify_frozen_candidate
from scripts.run_reviewed_candidate import native_runtime
from target.adapter import bind_native_experts
from target.reference_moe import ReferenceMoE


def identity(runtime):
    return json.loads(json.dumps(_checkpoint_identity(runtime)))


def load_verified_experts(runtime, expected_identity):
    """Reuse completed byte verification only while all file identities match.

    Both correctness workers have independently hashed every checkpoint file.
    This separate diagnostic does not replace or change either worker check.
    """
    if identity(runtime) != expected_identity:
        raise ValueError('checkpoint file identity differs from the completed native run')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import mlx.core as mx
    from mlx_lm import load
    from mlx.utils import tree_flatten
    with mx.stream(mx.cpu):
        model, _ = load(str(runtime.model_path), lazy=True)
        parameters = [(f'layer{layer}.{name}', tensor)
                      for layer in (0, 12, 23)
                      for name, tensor in tree_flatten(model.model.layers[layer].mlp.experts.parameters())]
        materialize_parameters(parameters, mx,
                               lambda done, total, name, size: print(f'native_load {done}/{total}', flush=True)
                               if done % 25 == 0 or done == total else None)
    for layer in model.model.layers:
        for name in ('up_proj', 'gate_proj', 'down_proj'):
            module = getattr(layer.mlp.experts, name)
            if (module.mode, module.bits, module.group_size) != ('mxfp4', 4, 32):
                raise ValueError('native expert weight format changed')
    if identity(runtime) != expected_identity:
        raise ValueError('checkpoint changed during diagnostic model load')
    adapter = bind_native_experts(model, mx=mx)
    return mx, ReferenceMoE(model), dict(adapter.experts)


def save(record):
    path = WORK / 'operator-diagnostic.json'
    staging = path.with_suffix('.tmp')
    staging.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    staging.replace(path)


def main():
    approval = json.loads((WORK / 'approval.json').read_text())
    protected_report = Path(sys.argv[1]).resolve()
    evidence = json.loads(protected_report.read_text())
    contract = load_contract(ROOT / 'kernelmaxxing.yaml')
    if not (
        evidence['candidate']['sha256'] == approval['snapshot_sha256']
        and evidence['contract_sha256'] == contract.sha256
        and evidence['status'] == 'completed'
        and not evidence['errors']
        and len(evidence['correctness']) == 7
        and all(row['passed'] for row in evidence['correctness'])
    ):
        raise ValueError('protected native correctness has not passed for this exact snapshot')
    for row in evidence['correctness']:
        detail = json.loads(row['detail'])
        if detail.get('input_mode') != 'mlx_native' or len(detail['checks']) != 3:
            raise ValueError('protected report is not complete native evidence')
    candidate = validate_candidate(Path(approval['snapshot']), Path(approval['snapshot']).parent)
    digests = {p.relative_to(candidate.directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in candidate.directory.rglob('*') if p.is_file()}
    frozen = FrozenCandidate(**vars(candidate), sha256=approval['snapshot_sha256'], file_sha256=digests)
    verify_frozen_candidate(frozen)
    protected = _protected_files([__file__])
    result = {
        'schema': 'kernelmax_operator_diagnostic_v1', 'status': 'running',
        'interpretation_status': 'not_evaluated',
        'scope': 'native one-token expert operator; synthetic activations and routing; not full-model speedup',
        'candidate_id': candidate.candidate_id, 'snapshot_sha256': frozen.sha256,
        'protected_correctness_report': str(protected_report),
        'protected_correctness_report_sha256': hashlib.sha256(protected_report.read_bytes()).hexdigest(),
        'protected_sources_before': protected, 'contract_sha256': contract.sha256,
        'acceptance_enabled': False, 'verdict': None,
        'plan': {'layers': [0, 12, 23], 'warmup_cycles_per_lane': 5,
                 'pairs_per_layer': 20, 'decode_calls_per_sample': 256, 'routing_inputs': 32, 'routing_cycles_per_sample': 8,
                 'order': 'alternating_ab_ba', 'seed': 42,
                 'sample_metric': 'wall_time_ms_per_256_synchronized_decode_operator_calls',
                 'environment_policy': 'strict; invalid samples retained; any issue makes summary inconclusive'},
        'additional_correctness': [], 'samples': [], 'summaries': {}, 'errors': [],
        'environment_before': environment(),
    }
    save(result)
    expected_identity = None
    try:
        runtime = native_runtime(Path('/Volumes/BankOfSouls/models/gpt-oss-20b'), contract)
        result['runtime'] = runtime.payload()
        expected_identity = json.loads((protected_report.parent / 'checkpoint_identity.json').read_text())
        result['checkpoint_identity_from_completed_native_run'] = expected_identity
        print('Rechecking verified checkpoint identity and loading sampled native expert layers', flush=True)
        mx, reference, experts = load_verified_experts(runtime, expected_identity)
        entry = _candidate_entry(frozen)
        result['packages'] = {name: importlib.metadata.version(name) for name in ('mlx', 'mlx-lm', 'numpy')}
        import numpy as np
        array_root = WORK / 'diagnostic-arrays'
        array_root.mkdir(exist_ok=True)
        for layer in (0, 12, 23):
            rng = np.random.default_rng(42 + layer)
            cases = []
            for index in range(32):
                x = mx.array(rng.normal(size=(1, 1, runtime.hidden_size)).astype(np.float32), dtype=mx.bfloat16)
                ids = mx.array([[[(index + k * 7) % runtime.num_experts for k in range(4)]]], dtype=mx.int32)
                mx.eval(x, ids)
                cases.append((x, ids))
                baseline_output = reference(layer, x, ids)
                candidate_output = entry(experts[layer], x, ids)
                comparison = compare_outputs(baseline_output, candidate_output,
                                             case_name='decode', policy=contract.correctness)
                arrays = array_root / f'layer-{layer}-input-{index}.npz'
                np.savez_compressed(arrays, x=np.array(x.astype(mx.float32)), ids=np.array(ids),
                                    reference=np.array(baseline_output.astype(mx.float32)),
                                    candidate=np.array(candidate_output.astype(mx.float32)))
                result['additional_correctness'].append({'layer': layer, 'input_index': index,
                                                          'passed': comparison.passed, 'detail': comparison.detail,
                                                          'expert_ids': np.array(ids).tolist(),
                                                          'original_dtype': str(candidate_output.dtype),
                                                          'arrays': str(arrays),
                                                          'arrays_sha256': hashlib.sha256(arrays.read_bytes()).hexdigest()})
                if not comparison.passed:
                    raise ValueError(f'additional diagnostic numerical failure: layer {layer}, input {index}')
            save(result)
            def invoke(lane):
                for x, ids in cases:
                    value = reference(layer, x, ids) if lane == 'baseline' else entry(experts[layer], x, ids)
                    mx.eval(value)
            for _ in range(5):
                invoke('baseline')
                invoke('candidate')
            for pair in range(1, 21):
                lanes = ('baseline', 'candidate') if pair % 2 else ('candidate', 'baseline')
                for lane in lanes:
                    verify_frozen_candidate(frozen)
                    if identity(runtime) != expected_identity:
                        raise ValueError('checkpoint changed before diagnostic sample')
                    before = environment()
                    mx.reset_peak_memory()
                    mx.synchronize()
                    start = time.perf_counter()
                    invoke(lane)
                    mx.synchronize()
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    peak = mx.get_peak_memory()
                    after = environment()
                    result['samples'].append({'layer': layer, 'pair': pair, 'lane': lane,
                                              'elapsed_ms': elapsed_ms, 'peak_memory_bytes': peak,
                                              'before': before, 'after': after,
                                              'environment_issues': environment_issues(before, after)})
                    save(result)
                print(f'layer {layer}: pair {pair}/20 retained', flush=True)
            rows = [row for row in result['samples'] if row['layer'] == layer]
            a = {row['pair']: row for row in rows if row['lane'] == 'baseline'}
            b = {row['pair']: row for row in rows if row['lane'] == 'candidate'}
            ratios = [a[i]['elapsed_ms'] / b[i]['elapsed_ms'] for i in range(1, 21)]
            bootstrap_rng = random.Random(42 + layer)
            boot = sorted(statistics.median(bootstrap_rng.choices(ratios, k=20)) for _ in range(10000))
            result['summaries'][str(layer)] = {
                'median_baseline_ms': statistics.median(row['elapsed_ms'] for row in a.values()),
                'median_candidate_ms': statistics.median(row['elapsed_ms'] for row in b.values()),
                'paired_baseline_over_candidate_ratios': ratios,
                'median_ratio': statistics.median(ratios),
                'descriptive_bootstrap_95pct_interval': [boot[249], boot[9749]],
                'environment_valid': all(not row['environment_issues'] for row in rows),
            }
            save(result)
        result['status'] = 'completed_diagnostic'
        result['environment_issues'] = sorted({issue for row in result['samples']
                                               for issue in row['environment_issues']})
        result['interpretation_status'] = (
            'inconclusive_environment' if result['environment_issues']
            else 'operator_diagnostic_only_not_full_model_speedup'
        )
    except BaseException as exc:
        result['status'] = 'failed_diagnostic'
        result['errors'].append(f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['environment_after'] = environment()
        result['protected_sources_after'] = _protected_files([__file__])
        result['protected_sources_unchanged'] = protected == result['protected_sources_after']
        if expected_identity is not None:
            result['checkpoint_identity_unchanged'] = identity(runtime) == expected_identity
            if not result['checkpoint_identity_unchanged']:
                result['status'] = 'invalid_diagnostic'
                result['errors'].append('checkpoint file identity changed')
        try:
            verify_frozen_candidate(frozen)
        except Exception as exc:
            result['status'] = 'invalid_diagnostic'
            result['errors'].append(f'snapshot integrity failure: {type(exc).__name__}: {exc}')
        if not result['protected_sources_unchanged']:
            result['status'] = 'invalid_diagnostic'
            result['errors'].append('protected sources changed')
        save(result)


if __name__ == '__main__':
    main()
