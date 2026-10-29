"""Untimed pointwise isolation on saved actual failing full-model boundaries."""
import ast
import json
from pathlib import Path
import sys

ROOT=Path('/Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop')
WORK=Path(__file__).resolve().parent
CONTROL=WORK.parent
sys.path.insert(0,str(ROOT))
from evaluator.candidate_protocol import file_digest
import mlx.core as mx
import numpy as np
from mlx_lm.models.gpt_oss import swiglu

selected=json.loads((CONTROL/'correctness-014/selected.json').read_text())
approval=json.loads((Path(selected['directory'])/'approval.json').read_text())
snapshot=Path(approval['snapshot'])
tree=ast.parse((snapshot/'implementation.py').read_text())
header=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_HEADER' for t in n.targets))
source=(snapshot/'projection.metal').read_text()
body=source[source.index('    const T clipped_up'):source.index('    activated[slot * N + row]')]
body+='    activated[i] = T(gated * shifted_up);\n'
body='const uint i=thread_position_in_grid.x;\nif(i>=S)return;\nconst T biased_up=up[i];\nconst T biased_gate=gate[i];\n'+body
kernel=mx.fast.metal_kernel(name='kernelmax_diagnostic_pointwise_isolation',input_names=['up','gate'],
    output_names=['activated'],source=body,header=header,ensure_row_contiguous=True,compile_options={'math_mode':'safe'})
state=json.loads((CONTROL/'correctness-014/state.json').read_text())
results=[]
for row in state['boundary_checks']:
    if row['bit_exact']:continue
    path=Path(row['arrays'])
    if file_digest(path)!=row['arrays_sha256']:raise ValueError('Boundary artifact changed')
    with np.load(path,allow_pickle=False) as arrays:
        up=mx.array(arrays['up'],dtype=mx.bfloat16);gate=mx.array(arrays['gate'],dtype=mx.bfloat16)
        actual=kernel(inputs=[up,gate],template=[('T',mx.bfloat16),('S',up.size)],grid=(up.size,1,1),
                      threadgroup=(256,1,1),output_shapes=[up.shape],output_dtypes=[mx.bfloat16])[0]
        native=swiglu(up,gate);mx.eval(actual,native)
        a=np.array(actual.astype(mx.float32));n=np.array(native.astype(mx.float32))
        results.append({'step':row['step'],'layer':row['layer'],
                        'native_reproduction_exact':bool(np.array_equal(n,arrays['native_activation'])),
                        'custom_epilogue_on_identical_projection_inputs_exact':bool(np.array_equal(a,n)),
                        'custom_epilogue_differences':int(np.count_nonzero(a!=n)),
                        'original_candidate_activation_differences':int(np.count_nonzero(arrays['candidate_activation']!=n)),
                        'max_pointwise_abs_error':float(np.abs(a-n).max())})
result={'scope':'untimed correctness isolation; no performance claim','checks':results,
        'all_native_reproductions_exact':all(r['native_reproduction_exact'] for r in results),
        'all_custom_pointwise_exact_on_native_inputs':all(r['custom_epilogue_on_identical_projection_inputs_exact'] for r in results)}
(WORK/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2))
