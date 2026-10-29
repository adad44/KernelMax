"""Small native-format fixtures for the frozen eight-lane up/gate candidate."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop')
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from evaluator.candidate_worker import _candidate_entry
from evaluator.correctness import compare_outputs
from evaluator.schemas import load_contract
from orchestrator.candidate_manager import FrozenCandidate, validate_candidate, verify_frozen_candidate


def main():
    approval = json.loads((WORK/'approval.json').read_text())
    c = validate_candidate(approval['snapshot'], Path(approval['snapshot']).parent)
    hashes = {p.relative_to(c.directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in c.directory.rglob('*') if p.is_file()}
    frozen = FrozenCandidate(**vars(c), sha256=approval['snapshot_sha256'], file_sha256=hashes)
    verify_frozen_candidate(frozen)
    record = {'status':'running', 'scope':'small random fixtures; not pinned checkpoint evidence',
              'snapshot_sha256':frozen.sha256, 'checks':[], 'errors':[]}
    save = lambda: (WORK/'tiny-native-checks.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    save()
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from mlx_lm.models.gpt_oss import SwiGLU
        from mlx_lm.models.switch_layers import SwitchGLU
        entry = _candidate_entry(frozen)
        policy = load_contract(ROOT/'kernelmaxxing.yaml').correctness
        mx.random.seed(42)
        arrays = WORK/'tiny-native-arrays'
        arrays.mkdir()
        def check(experts, x, ids, label, details):
            expected_output = experts(x,ids)
            actual = entry(experts,x,ids)
            compared = compare_outputs(expected_output,actual,case_name='decode' if x.shape[1]==1 else 'prefill',policy=policy)
            path = arrays/(label+'.npz')
            np.savez_compressed(path,x=np.array(x.astype(mx.float32)),ids=np.array(ids),
                                reference=np.array(expected_output.astype(mx.float32)),
                                candidate=np.array(actual.astype(mx.float32)))
            row = {**details,'label':label,'passed':compared.passed,'detail':compared.detail,
                   'arrays':str(path),'arrays_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            record['checks'].append(row)
            save()
            print(json.dumps(row),flush=True)
            if not compared.passed:
                raise ValueError('small fixture failed; native run blocked')
            return actual
        for k,n in ((64,64),(96,96),(2880,64)):
            experts = SwitchGLU(k,n,8,activation=SwiGLU(),bias=True)
            for p in (experts.up_proj,experts.gate_proj,experts.down_proj):
                p.bias = mx.random.normal((8,p.output_dims)).astype(mx.bfloat16)
            nn.quantize(experts,group_size=32,bits=4,mode='mxfp4')
            experts.eval()
            for tokens,magnitude in ((1,1),(1,64),(16,1),(17,1)):
                x = (mx.random.normal((1,tokens,k))*magnitude).astype(mx.bfloat16)
                for dtype in (mx.int32,mx.uint32):
                    ids = mx.array([[[(i+j*2)%8 for j in range(4)] for i in range(tokens)]],dtype=dtype)
                    check(experts,x,ids,f'k{k}-n{n}-t{tokens}-m{magnitude}-{str(dtype).split(".")[-1]}',
                          {'k':k,'n':n,'tokens':tokens,'magnitude':magnitude,'ids_dtype':str(dtype),
                           'path':'octet_decode' if tokens==1 else 'reference_fallback'})
            x = mx.random.normal((1,1,k)).astype(mx.bfloat16)
            ids = mx.array([[[0,2,4,6]]],dtype=mx.uint32)
            original = entry(experts,x,ids)
            mx.eval(original)
            experts.down_proj.bias = experts.down_proj.bias + mx.array(.125,dtype=mx.bfloat16)
            for dtype in (mx.int32,mx.uint32):
                actual = check(experts,x,ids.astype(dtype),f'k{k}-n{n}-changed-bias-{str(dtype).split(".")[-1]}',
                               {'path':'octet_decode_updated_explicit_parameter','ids_dtype':str(dtype)})
                if not mx.any(original != actual).item():
                    raise ValueError('parameter update did not change candidate output')
        record['status'] = 'passed_synthetic_only'
    except BaseException as e:
        record['status'] = 'failed_synthetic'
        record['errors'].append(f'{type(e).__name__}: {e}')
        raise
    finally:
        verify_frozen_candidate(frozen)
        save()


if __name__=='__main__':
    main()
