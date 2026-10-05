"""Untimed diagnostic of the preserved large-activation reference mismatch."""
import json
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx.utils import tree_flatten
from evaluator.native_baseline import materialize_parameters


def main():
    with mx.stream(mx.cpu):
        model, _ = load('/Volumes/BankOfSouls/models/gpt-oss-20b', lazy=True)
        experts = model.model.layers[12].mlp.experts
        materialize_parameters(tree_flatten(experts.parameters()), mx,
                               lambda i,n,key,size: print(f'expert tensors {i}/{n}: {key}',flush=True))
    rng = np.random.default_rng(4200 + 12 * 31 + 5)
    x = mx.array(rng.normal(size=(1,17,2880)).astype(np.float32)*64, dtype=mx.bfloat16)
    ids = mx.array([[[(i+j*7)%32 for j in range(4)] for i in range(17)]],dtype=mx.int32)
    native = experts(x,ids)
    permutation=mx.array(list(range(16,-1,-1)))
    slots=mx.array([3,2,1,0])
    permuted=experts(x[:,permutation,:],ids[:,permutation,:][:,:,slots])[:,permutation,:,:][:,:,slots,:]
    repeated=experts(x,ids)
    mx.eval(native,permuted,repeated)
    print(json.dumps({'native_permutation_max_absolute_error':float(mx.max(mx.abs(native-permuted)).item()),
                      'native_repeat_max_absolute_error':float(mx.max(mx.abs(native-repeated)).item()),
                      'dtype':str(native.dtype),'finite':bool(mx.all(mx.isfinite(native)).item())}),flush=True)


if __name__=='__main__':
    main()
