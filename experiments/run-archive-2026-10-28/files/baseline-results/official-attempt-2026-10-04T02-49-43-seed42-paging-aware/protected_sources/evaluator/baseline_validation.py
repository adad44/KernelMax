"""Reference correctness and predeclared A/A calibration, not candidate grading."""
from __future__ import annotations

import ctypes
import math
import random
import statistics


def thermal_state():
    ctypes.CDLL('/System/Library/Frameworks/Foundation.framework/Foundation')
    objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    pointer = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', objc))
    integer = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', objc))
    process = pointer(objc.objc_getClass(b'NSProcessInfo'), objc.sel_registerName(b'processInfo'))
    return integer(process, objc.sel_registerName(b'thermalState'))


def aa_noise(samples, lengths, seed=42):
    """Bootstrap paired median B/A throughput ratio; preserve all twenty pairs.

    Calibration gates are declared before this run: 95% CI includes one and
    spans <=4 percentage points. These do not enable candidate acceptance.
    """
    result = {}
    for length in lengths:
        rows = [s for s in samples if s['prompt_tokens'] == length]
        a = {s['trial']: s for s in rows if s['lane'] == 'A'}
        b = {s['trial']: s for s in rows if s['lane'] == 'B'}
        if set(a) != set(range(1, 21)) or set(b) != set(a):
            raise ValueError('A/A requires exactly twenty complete pairs per size')
        ratios = [b[i]['metrics']['decode_tokens_per_second'] / a[i]['metrics']['decode_tokens_per_second']
                  for i in range(1, 21)]
        rng = random.Random(seed + length)
        boot = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10000))
        lo, hi = boot[249], boot[9749]
        result[str(length)] = {'paired_b_over_a_ratios': ratios,
                               'median_ratio': statistics.median(ratios),
                               'bootstrap_95pct_ci': [lo, hi],
                               'passed': lo <= 1 <= hi and hi - lo <= .04}
    return result


def driver_checks(model, inputs, held_out, step_size, request, mx, make_prompt_cache):
    """Compare exact output IDs with the unchanged stock generate_step."""
    from mlx_lm.generate import generate_step, generation_stream
    records = []
    cases = [('prefill', ids) for ids in inputs.values()] + [('decode', [1]), ('held_out_inputs', held_out)]
    for name, ids in cases:
        actual = request(model, ids, step_size, mx, make_prompt_cache)['output_token_ids']
        expected = [token for token, _ in generate_step(mx.array(ids, dtype=mx.int32), model,
                    max_tokens=128, prefill_step_size=step_size)]
        mx.synchronize(generation_stream)
        record = {'case': name, 'prompt_tokens': len(ids), 'output_tokens': 128,
                  'expected_token_ids': expected, 'actual_token_ids': actual,
                  'passed': len(expected) == 128 and actual == expected,
                  'reference': 'unmodified mlx_lm.generate.generate_step; greedy; fresh cache'}
        records.append(record)
        print(f"Stock generation comparison {name}/{len(ids)}: {record['passed']}", flush=True)
        if not record['passed']:
            return records
    return records


def dense_expert_checks(model, mx, atol, rtol):
    """Check gather/sort/unsort against individually dequantized dense experts.

    Full-model weights stay native MXFP4. Dense BF16 expert matrices are only
    temporary correctness references; never used in timed inference.
    """
    import numpy as np
    records = []
    cases = [('decode', 1, 'uneven', 1), ('prefill', 16, 'varied', 1),
             ('uneven_expert_load', 17, 'uneven', 1),
             ('unused_experts', 17, 'unused', 1),
             ('non_divisible_token_counts', 19, 'varied', 1),
             ('large_finite_activations', 17, 'varied', 64),
             ('held_out_inputs', 23, 'varied', 1)]
    for layer_index in (0, 12, 23):
        experts = model.model.layers[layer_index].mlp.experts
        for case_index, (name, count, distribution, magnitude) in enumerate(cases):
            rng = np.random.default_rng(4200 + layer_index * 31 + case_index)
            data = rng.normal(size=(1, count, model.args.hidden_size)).astype(np.float32) * magnitude
            ids = np.empty((1, count, 4), dtype=np.int32)
            for i in range(count):
                ids[0, i] = ([0, 1, 2, 3] if distribution == 'unused' else
                             [0, 1, 2, (3 if i % 5 else 4)] if distribution == 'uneven' else
                             [(i + j * 7) % model.args.num_local_experts for j in range(4)])
            x, indices = mx.array(data, dtype=mx.bfloat16), mx.array(ids)
            actual = experts(x, indices)
            mx.eval(actual)
            expected = mx.zeros(actual.shape, dtype=mx.bfloat16)
            # Group positions by expert; dense GEMM is an independent projection path.
            for expert_id in sorted(set(ids.reshape(-1).tolist())):
                positions = np.argwhere(ids[0] == expert_id)
                rows, slots = positions[:, 0], positions[:, 1]
                values = x[0, mx.array(rows)]
                projections = []
                for projection in (experts.up_proj, experts.gate_proj):
                    weight = mx.dequantize(projection.weight[expert_id], projection.scales[expert_id],
                                           group_size=32, bits=4, mode='mxfp4', dtype=mx.bfloat16)
                    projections.append(values @ weight.T + projection.bias[expert_id])
                hidden = experts.activation(*projections)
                down = experts.down_proj
                weight = mx.dequantize(down.weight[expert_id], down.scales[expert_id],
                                       group_size=32, bits=4, mode='mxfp4', dtype=mx.bfloat16)
                projected = hidden @ weight.T + down.bias[expert_id]
                expected[0, mx.array(rows), mx.array(slots), :] = projected
                mx.eval(expected)
            mx.eval(expected)
            difference = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
            allowed = atol + rtol * mx.abs(expected.astype(mx.float32))
            record = {'case': name, 'layer': layer_index, 'tokens': count,
                      'shape': list(actual.shape), 'dtype': str(actual.dtype),
                      'reference_dtype': str(expected.dtype),
                      'max_absolute_error': float(mx.max(difference).item()),
                      'max_tolerance_ratio': float(mx.max(difference / allowed).item()),
                      'finite': bool(mx.all(mx.isfinite(actual)).item()) and bool(mx.all(mx.isfinite(expected)).item()),
                      'reference': 'native MXFP4 dequantization to BF16 plus dense per-expert matmul',
                      'atol': atol, 'rtol': rtol}
            record['passed'] = (record['finite'] and actual.shape == expected.shape
                                and actual.dtype == expected.dtype and bool(mx.all(difference <= allowed).item()))
            records.append(record)
            print(f"Expert check {name}/layer{layer_index}: {record['passed']}; error={record['max_absolute_error']}", flush=True)
            del actual, expected, difference, allowed, x, indices, weight, projected, hidden, projections, values
            mx.clear_cache()
            if not record['passed']:
                return records
    return records


def expert_checks(model, mx, atol, rtol):
    """Native-reference dispatch invariance; not an independent matmul oracle.

    The contract's golden operator is unchanged MLX-LM, not dense reconstruction.
    Reorder tokens and expert slots, call the native operator independently,
    and restore both permutations. Dispatch must commute with this operation.
    The separately retained dense failure is not hidden or relabeled a pass.
    """
    import hashlib
    import numpy as np
    records = []
    cases = [('decode', 1, 'uneven', 1), ('prefill', 16, 'varied', 1),
             ('uneven_expert_load', 17, 'uneven', 1), ('unused_experts', 17, 'unused', 1),
             ('non_divisible_token_counts', 19, 'varied', 1),
             ('large_finite_activations', 17, 'varied', 64), ('held_out_inputs', 23, 'varied', 1)]
    for layer_index in (0, 12, 23):
        experts = model.model.layers[layer_index].mlp.experts
        for case_index, (name, count, distribution, magnitude) in enumerate(cases):
            seed = 4200 + layer_index * 31 + case_index
            rng = np.random.default_rng(seed)
            data = rng.normal(size=(1, count, model.args.hidden_size)).astype(np.float32) * magnitude
            ids = np.empty((1, count, 4), dtype=np.int32)
            for i in range(count):
                ids[0, i] = ([0, 1, 2, 3] if distribution == 'unused' else
                             [0, 1, 2, (3 if i % 5 else 4)] if distribution == 'uneven' else
                             [(i + j * 7) % model.args.num_local_experts for j in range(4)])
            x, indices = mx.array(data, dtype=mx.bfloat16), mx.array(ids)
            token_permutation = mx.array(list(range(count - 1, -1, -1)))
            slot_permutation = mx.array([3, 2, 1, 0])
            actual = experts(x, indices)
            permuted = experts(x[:, token_permutation, :],
                               indices[:, token_permutation, :][:, :, slot_permutation])
            expected = permuted[:, token_permutation, :, :][:, :, slot_permutation, :]
            mx.eval(actual, expected)
            difference = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
            allowed = atol + rtol * mx.abs(expected.astype(mx.float32))
            record = {'case': name, 'layer': layer_index, 'tokens': count,
                      'input_seed': seed, 'input_magnitude': magnitude,
                      'input_bf16_as_float32_sha256': hashlib.sha256(np.array(x.astype(mx.float32)).tobytes()).hexdigest(),
                      'selected_expert_ids': ids.tolist(), 'shape': list(actual.shape),
                      'dtype': str(actual.dtype), 'reference_dtype': str(expected.dtype),
                      'max_absolute_error': float(mx.max(difference).item()),
                      'max_tolerance_ratio': float(mx.max(difference / allowed).item()),
                      'finite': bool(mx.all(mx.isfinite(actual)).item()) and bool(mx.all(mx.isfinite(expected)).item()),
                      'reference': 'unchanged native MLX-LM operator; independently permuted token/expert-slot dispatch',
                      'atol': atol, 'rtol': rtol,
                      'scope': 'dispatch invariance and native reference consistency; not independent kernel numerical accuracy'}
            record['passed'] = (record['finite'] and actual.shape == expected.shape
                                and actual.dtype == expected.dtype and bool(mx.all(difference <= allowed).item()))
            records.append(record)
            print(f"Native expert check {name}/layer{layer_index}: {record['passed']}; error={record['max_absolute_error']}", flush=True)
            del actual, expected, difference, allowed, x, indices, permuted
            mx.clear_cache()
            if not record['passed']:
                return records
    return records
