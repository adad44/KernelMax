"""Bounded real-MLX serialization/exchange fixture; no checkpoint load."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weight_residency import WeightResidency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    start=time.monotonic()
    spec=importlib.util.spec_from_file_location('r2_storage_fixture',args.candidate/'implementation.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    rng=np.random.default_rng(20261028)
    layer=nn.Linear(128,32,bias=False)
    layer.weight=mx.array(rng.normal(size=(32,128)),dtype=mx.bfloat16)
    model=type('FixtureModel',(),{'lm_head':layer})()
    original=type(layer).__call__
    vectors={tokens:mx.array(rng.normal(size=(1,tokens,128)),dtype=mx.bfloat16) for tokens in (1,3)}
    reference={}
    for tokens,x in vectors.items():
        value=original(layer,x)
        mx.eval(value)
        reference[tokens]=np.array(value.astype(mx.float32))
    del value
    adapter=module.prepare(model)
    storage=WeightResidency(model,adapter,mx,args.output.with_suffix('.backing'))
    rows=[]
    for enabled in (True,False,True,False):
        if time.monotonic()-start>120:
            raise TimeoutError('Storage fixture time bound')
        storage.switch(enabled)
        storage.assert_lane(enabled)
        for tokens,x in vectors.items():
            result=adapter.call_projection(original,layer,x) if enabled else original(layer,x)
            mx.eval(result)
            actual=np.array(result.astype(mx.float32))
            expected=reference[tokens]
            within=bool(np.all(np.isfinite(actual)) and np.all(np.abs(actual-expected)<=.02+.02*np.abs(expected)))
            rows.append({'candidate':enabled,'tokens':tokens,'within_existing_tolerance':within,
                         'exact':bool(np.array_equal(actual,expected)),**storage.snapshot()})
            if not within: raise ValueError('Storage exchange changed projection output')
            storage.assert_lane(enabled)
    if not storage.verify_final(): raise ValueError('Backing store validation failed')
    result={'passed':True,'no_checkpoint_loaded':True,'full_model_correctness':False,
            'actual_matrix_unpack_bit_exact':storage.actual_matrix_unpack_bit_exact,
            'native_roundtrip_bit_exact':storage.native_roundtrip_bit_exact,
            'rows':rows,'storage':storage.description(),'wall_seconds':time.monotonic()-start}
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))


if __name__=='__main__': main()
