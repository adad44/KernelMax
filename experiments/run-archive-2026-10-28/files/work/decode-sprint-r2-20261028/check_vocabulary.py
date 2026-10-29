"""Bounded new decoder/lifetime fixtures; no checkpoint or full model load."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    name = 'r2_vocabulary_fixture'
    spec = importlib.util.spec_from_file_location(name, args.candidate/'implementation.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    records = []
    # Full BF16 word coverage, both tightly grouped and random escape layouts.
    all_words = np.arange(65536, dtype=np.uint16)
    rng = np.random.default_rng(20261028)
    for label, words in [('all_words', np.tile(all_words,2)),
                         ('escape_permutation', np.tile(rng.permutation(all_words),2))]:
        if time.monotonic()-start > 120:
            raise TimeoutError('Two-minute fixture bound')
        weight = mx.array(words.reshape(1024,128), dtype=mx.uint16).view(mx.bfloat16)
        projection = module._reader.prepare_projection(weight)
        actual = projection.reconstruct().view(mx.uint16)
        mx.eval(actual)
        exact = bool(np.array_equal(np.array(actual), words.reshape(1024,128)))
        records.append({'fixture': label, 'words': len(words), 'exact': exact,
                        'escaped_blocks': projection.statistics['escaped_blocks']})
        if not exact:
            raise ValueError('Lossless decoder failed '+label)
        del actual, weight, projection
        mx.clear_cache()
    # Native linear fallback must retain exact arithmetic with a temporary
    # reconstructed weight, and restore the placeholder even before evaluation.
    layer = nn.Linear(128,32,bias=False)
    layer.weight = mx.array(rng.normal(size=(32,128)), dtype=mx.bfloat16)
    model = type('FixtureModel', (), {'lm_head': layer})()
    adapter = module.prepare(model)
    original = type(layer).__call__
    reference_weight = layer.weight
    stored = dict(adapter.export_layout())
    adapter.unload_layout()
    if adapter.packed_bytes() != 0:
        raise ValueError('Packed arrays retained after unload')
    adapter.load_layout(stored)
    del stored
    for tokens in (1,3):
        x = mx.array(rng.normal(size=(1,tokens,128)), dtype=mx.bfloat16)
        expected = original(layer,x)
        mx.eval(expected)
        placeholder = mx.zeros((0,),dtype=mx.bfloat16)
        layer.weight = placeholder
        actual = adapter.call_projection(original,layer,x)
        restored = layer.weight is placeholder
        mx.eval(actual)
        exact = bool(np.array_equal(np.array(actual.view(mx.uint16)),np.array(expected.view(mx.uint16))))
        maximum = float(mx.max(mx.abs(actual.astype(mx.float32)-expected.astype(mx.float32))).item())
        a, b = np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32))
        within = bool(np.all(np.isfinite(a)) and np.all(np.abs(a-b) <= .02+.02*np.abs(b)))
        records.append({'fixture':'projection','tokens':tokens,'exact':exact,
                        'max_abs_error':maximum,'within_existing_tolerance':within,
                        'placeholder_restored':restored})
        if not within or not restored:
            raise ValueError('Projection/fallback mismatch')
        layer.weight = reference_weight
    adapter.unload_layout()
    if adapter.packed_bytes():
        raise ValueError('Final unload retained arrays')
    result = {'passed':True,'records':records,'wall_seconds':time.monotonic()-start,
              'no_checkpoint_loaded':True,'full_model_correctness':False,
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
