"""Controller-owned synthetic fixture checks of an explicitly approved snapshot."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path('/Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop')
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

from evaluator.candidate_worker import _candidate_entry
from evaluator.correctness import compare_outputs
from evaluator.schemas import load_contract
from orchestrator.candidate_manager import FrozenCandidate,validate_candidate,verify_frozen_candidate


def main():
    approval=json.loads((WORK/'approval.json').read_text())
    candidate=validate_candidate(approval['snapshot'],Path(approval['snapshot']).parent)
    files={p.relative_to(candidate.directory).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for p in candidate.directory.rglob('*') if p.is_file()}
    frozen=FrozenCandidate(**vars(candidate),sha256=approval['snapshot_sha256'],file_sha256=files)
    verify_frozen_candidate(frozen)
    result={'scope':'synthetic random native-MXFP4 expert fixtures; not pinned-checkpoint evidence',
            'snapshot_sha256':frozen.sha256,'checks':[],'errors':[],'status':'running'}
    path=WORK/'tiny-native-checks.json'
    def save():
        path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    save()
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from mlx_lm.models.gpt_oss import SwiGLU
        from mlx_lm.models.switch_layers import SwitchGLU
        run=_candidate_entry(frozen)
        contract=load_contract(ROOT/'kernelmaxxing.yaml')
        mx.random.seed(42)
        arrays=WORK/'tiny-native-arrays'
        arrays.mkdir()
        for k,n in ((64,64),(96,96),(2880,64)):
            experts=SwitchGLU(k,n,8,activation=SwiGLU(),bias=True)
            for projection in (experts.up_proj,experts.gate_proj,experts.down_proj):
                projection.bias=mx.random.normal((8,projection.output_dims)).astype(mx.bfloat16)
            nn.quantize(experts,group_size=32,bits=4,mode='mxfp4')
            experts.eval()
            for tokens,magnitude in ((1,1),(1,64),(16,1),(17,1)):
                x=(mx.random.normal((1,tokens,k))*magnitude).astype(mx.bfloat16)
                ids=mx.array([[[(i+j*2)%8 for j in range(4)] for i in range(tokens)]],dtype=mx.int32)
                reference=experts(x,ids)
                actual=run(experts,x,ids)
                check=compare_outputs(reference,actual,case_name='decode' if tokens==1 else 'prefill',policy=contract.correctness)
                array_path=arrays/f'k{k}-n{n}-t{tokens}-m{magnitude}.npz'
                np.savez_compressed(array_path,x=np.array(x.astype(mx.float32)),ids=np.array(ids),reference=np.array(reference.astype(mx.float32)),candidate=np.array(actual.astype(mx.float32)))
                row={'k':k,'n':n,'tokens':tokens,'magnitude':magnitude,'passed':check.passed,
                     'detail':check.detail,'path':'fused_metal_decode' if tokens==1 else 'reference_fallback',
                     'arrays':str(array_path),'arrays_sha256':hashlib.sha256(array_path.read_bytes()).hexdigest()}
                result['checks'].append(row)
                save()
                print(json.dumps(row),flush=True)
                if not check.passed:
                    raise ValueError('candidate failed synthetic numerical comparison; native run blocked')
        result['status']='passed_synthetic_only'
    except BaseException as exc:
        result['status']='failed_synthetic'
        result['errors'].append(f'{type(exc).__name__}: {exc}')
        raise
    finally:
        verify_frozen_candidate(frozen)
        save()

if __name__=='__main__':
    main()
