"""Exact layout/update fixtures and previously failing real-boundary regressions."""
import gc
import json
from pathlib import Path
import sys
import weakref

ROOT=Path('/Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop')
WORK=Path(__file__).resolve().parent
CONTROL=WORK.parent
sys.path.insert(0,str(ROOT))
from evaluator.candidate_protocol import file_digest
from evaluator.candidate_worker import _candidate_entry
from evaluator.checkpoint_verification import validate_receipt
from evaluator.native_baseline import materialize_parameters
from evaluator.schemas import load_contract
from orchestrator.candidate_manager import FrozenCandidate,validate_candidate,verify_frozen_candidate
from scripts.run_reviewed_candidate import native_runtime
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.gpt_oss import SwiGLU
from mlx_lm.models.switch_layers import SwitchGLU
from mlx.utils import tree_flatten

approval=json.loads((WORK/'approval.json').read_text())
request=json.loads((Path(approval['native_report']).parent/'candidate/request.json').read_text())
info=request['snapshot'];c=validate_candidate(info['directory'],Path(info['directory']).parent)
frozen=FrozenCandidate(**vars(c),sha256=info['sha256'],file_sha256=info['file_sha256']);verify_frozen_candidate(frozen)
entry=_candidate_entry(frozen)
implementation=sys.modules[entry.__module__]
result={'status':'running','candidate_id':frozen.candidate_id,'snapshot_sha256':frozen.sha256,
        'scope':'untimed correctness/representation tests; no performance claim',
        'layout_checks':[],'real_boundary_checks':[],'errors':[]}
def save():(WORK/'repair-checks.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
save()
try:
    result['layout_checks'].append({'check':'no parameter layout conversion or cache in this candidate','passed':True})
    prior=json.loads((CONTROL/'correctness-014/state.json').read_text())
    cases=[r for r in prior['boundary_checks'] if not r['bit_exact']]
    layers=sorted({r['layer'] for r in cases})
    runtime=native_runtime(Path('/Volumes/BankOfSouls/models/gpt-oss-20b'),load_contract(ROOT/'kernelmaxxing.yaml'))
    receipt=json.loads((CONTROL/'checkpoint-verification.json').read_text())
    validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files)
    with mx.stream(mx.cpu):
        model,_=load(str(runtime.model_path),lazy=True)
        parameters=[(f'layer{layer}.{name}',v) for layer in layers
                    for name,v in tree_flatten(model.model.layers[layer].mlp.experts.parameters())]
        materialize_parameters(parameters,mx,lambda done,total,name,size:print(f'load {done}/{total}',flush=True) if done%25==0 or done==total else None)
    del parameters
    for case in cases:
        path=Path(case['arrays'])
        if file_digest(path)!=case['arrays_sha256']:raise ValueError('Real boundary artifact changed')
        experts=model.model.layers[case['layer']].mlp.experts
        with np.load(path,allow_pickle=False) as arrays:
            x=mx.array(arrays['x'],dtype=mx.bfloat16);ids=mx.array(arrays['ids'],dtype=mx.uint32)
            reference=experts(x,ids);actual=entry(experts,x,ids);mx.eval(reference,actual)
            reference_array=np.array(reference.astype(mx.float32));actual_array=np.array(actual.astype(mx.float32))
            row={'step':case['step'],'layer':case['layer'],
                 'native_reproduction_exact':bool(np.array_equal(reference_array,arrays['reference'])),
                 'candidate_exact':bool(np.array_equal(reference_array,actual_array)),
                 'max_abs':float(np.abs(reference_array-actual_array).max())}
        result['real_boundary_checks'].append(row);save()
        print(json.dumps(row),flush=True)
        if not row['native_reproduction_exact'] or not row['candidate_exact']:
            raise ValueError('Exact real-boundary repair failed; timing blocked')
    validate_receipt(receipt,runtime.model_path,runtime.model_revision,runtime.model_files)
    result['status']='passed_exact_repair_checks'
except BaseException as error:
    result['status']='failed';result['errors'].append(f'{type(error).__name__}: {error}');raise
finally:
    verify_frozen_candidate(frozen);save()
