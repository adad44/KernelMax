"""One bounded same-input routing diagnostic; no model load or timing claim."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    start=time.monotonic()
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    spec=importlib.util.spec_from_file_location('r2_old_moe_stage_probe',args.source/'implementation.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    # Expose the old shader's own scores through one appended diagnostic store;
    # arithmetic, barriers, workgroup and selection body remain byte-identical.
    original=(args.source/'route.metal').read_text()
    code=original+'\nif(tid<32) all_scores[tid]=scores[tid];\n'
    probe=mx.fast.metal_kernel(name='r2_same_input_route_probe',
        input_names=['residual','attention','norm_weight','router_weight','router_bias'],
        output_names=['norm_out','residual_out','ids','weights','all_scores'],
        source=code,header=module._HEADER,ensure_row_contiguous=True,
        compile_options={'math_mode':'safe'})
    rows=[]
    h=2880
    for seed in range(17):
        if time.monotonic()-start>90: raise TimeoutError('Routing diagnostic bound')
        rng=np.random.default_rng(20261028+seed)
        residual=mx.array(rng.normal(size=(1,1,h))*.2,dtype=mx.bfloat16)
        attention=mx.array(rng.normal(size=(1,1,h))*.2,dtype=mx.bfloat16)
        norm=nn.RMSNorm(h,eps=1e-5)
        norm.weight=mx.array(1+rng.normal(size=h)*.1,dtype=mx.bfloat16)
        router=nn.Linear(h,32,bias=True)
        router.weight=mx.array(rng.normal(size=(32,h))*.01,dtype=mx.bfloat16)
        router.bias=mx.array(rng.normal(size=32)*.05,dtype=mx.bfloat16)
        if seed==16:
            router.weight=mx.zeros((32,h),dtype=mx.bfloat16)
            router.bias=mx.zeros((32,),dtype=mx.bfloat16)
        merged=residual+attention
        normalized=norm(merged)
        scores=router(normalized)
        ids=mx.argpartition(scores,kth=-4,axis=-1)[...,-4:]
        selected=mx.take_along_axis(scores,ids,axis=-1)
        weights=mx.softmax(selected,axis=-1,precise=True)
        outputs=probe(inputs=[residual,attention,norm.weight,router.weight,router.bias],
            template=[('T',mx.bfloat16),('H',h)],grid=(736,1,1),threadgroup=(736,1,1),
            output_shapes=[(1,1,h),(1,1,h),(1,1,4),(1,1,4),(1,1,32)],
            output_dtypes=[mx.bfloat16,mx.bfloat16,mx.uint32,mx.bfloat16,mx.bfloat16])
        mx.eval(merged,normalized,scores,ids,weights,*outputs)
        fused_norm,fused_merged,fused_ids,fused_weights,fused_scores=outputs
        def details(a,b):
            av,bv=np.array(a.astype(mx.float32)),np.array(b.astype(mx.float32))
            return {'bit_exact':bool(np.array_equal(np.array(a.view(mx.uint16)),np.array(b.view(mx.uint16)))),
                    'different_elements':int(np.count_nonzero(av!=bv)),
                    'max_abs_error':float(np.max(np.abs(av-bv)))}
        rows.append({'seed':seed,'zero_router_ties':seed==16,
            'merged':details(merged,fused_merged),'norm':details(normalized,fused_norm),
            'scores':details(scores,fused_scores),'weights':details(weights,fused_weights),
            'native_ids':np.array(ids).reshape(-1).tolist(),
            'fused_ids':np.array(fused_ids).reshape(-1).tolist(),
            'native_scores':np.array(scores.astype(mx.float32)).reshape(-1).tolist(),
            'fused_scores':np.array(fused_scores.astype(mx.float32)).reshape(-1).tolist()})
    result={'no_checkpoint_loaded':True,'performance_evidence':False,'fixtures':17,
        'original_route_sha256':hashlib.sha256(original.encode()).hexdigest(),
        'instrumented_route_sha256':hashlib.sha256(code.encode()).hexdigest(),
        'wall_seconds':time.monotonic()-start,'rows':rows,
        'scope':'Same-input synthetic stage compatibility only; actual failed model trajectory root cause still requires matching evidence.'}
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'fixtures':17,'wall_seconds':result['wall_seconds'],
        'norm_nonexact_cases':sum(not r['norm']['bit_exact'] for r in rows),
        'score_nonexact_cases':sum(not r['scores']['bit_exact'] for r in rows),
        'route_different_cases':sum(r['native_ids']!=r['fused_ids'] for r in rows),
        'zero_router_native_ids':rows[-1]['native_ids'],'zero_router_fused_ids':rows[-1]['fused_ids']}))


if __name__=='__main__': main()
