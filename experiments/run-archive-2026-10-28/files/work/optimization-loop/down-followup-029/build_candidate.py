"""Generate one down-projection attempt only after valid measured down budget."""
import ast,json,secrets
from pathlib import Path
HERE=Path(__file__).resolve().parent
CONTROL=HERE.parent
ROOT=CONTROL.parent.parent/'kernelmax-first-loop'

SHADER=r'''// Native32-lane/four-row accumulation with a BF16 projection then bias.
const uint lane=thread_index_in_simdgroup;
const uint row0=threadgroup_position_in_grid.x*8+simdgroup_index_in_threadgroup*4;
const uint slot=threadgroup_position_in_grid.y;
const uint expert=uint(ids[slot]);
if(expert>=E){
    if(lane==0){for(uint row=0;row<4;++row)output[slot*N+row0+row]=T(as_type<float>(0x7fc00000u));}
    return;
}
float sum[4]={0.0f,0.0f,0.0f,0.0f};
for(uint base=0;base<K;base+=256){
    const uint column=base+lane*8;
    if(column<K){
        float values[8];
        #pragma unroll
        for(uint j=0;j<8;++j)values[j]=float(x[slot*K+column+j]);
        #pragma unroll
        for(uint row=0;row<4;++row){
            const uint output_row=expert*N+row0+row;
            const uint packed_offset=output_row*(K/8)+column/8;
            const uint scale_offset=output_row*(K/32)+column/32;
            sum[row]+=km_scale(scales[scale_offset])*km_dot8(weight[packed_offset],values);
        }
    }
}
#pragma unroll
for(uint row=0;row<4;++row){
    const float raw=simd_sum(sum[row]);
    if(lane==0){
        const uint output_row=expert*N+row0+row;
        const T rounded=T(raw);
        output[slot*N+row0+row]=T(float(rounded)+float(bias[output_row]));
    }
}
'''

BODY=r'''
_DOWN=mx.fast.metal_kernel(
    name="KERNEL_NAME",input_names=["x","ids","weight","scales","bias"],
    output_names=["output"],source=Path(__file__).with_name("down_bias.metal").read_text(),
    header=_HEADER,ensure_row_contiguous=True,compile_options={"math_mode":"safe"})

def run(experts,x,selected_expert_ids):
    if x.shape[:2]!=(1,1) or selected_expert_ids.shape!=(1,1,4):
        return experts(x,selected_expert_ids)
    if x.ndim!=3 or x.dtype!=mx.bfloat16:
        raise ValueError("Down decode requires BF16[1,1,H]")
    if selected_expert_ids.dtype not in (mx.int32,mx.uint32):
        raise ValueError("Down decode requires int32/uint32 IDs")
    if type(experts.activation) is not SwiGLU:
        raise ValueError("Unchanged native SwiGLU required")
    down=experts.down_proj;k=down.input_dims;n=down.output_dims;e=down.num_experts
    if k<=0 or k%32 or n<=0 or n%8 or n!=x.shape[-1]:
        raise ValueError("Unsupported down dimensions")
    for p in (experts.up_proj,experts.gate_proj,down):
        if (p.mode,p.bits,p.group_size)!=("mxfp4",4,32):
            raise ValueError("Native MXFP4 required")
    if (down.weight.dtype!=mx.uint32 or down.scales.dtype!=mx.uint8
        or down.weight.shape!=(e,n,k//8) or down.scales.shape!=(e,n,k//32)
        or down.get("biases") is not None or "bias" not in down
        or down.bias.dtype!=mx.bfloat16 or down.bias.shape!=(e,n)):
        raise ValueError("Unsupported packed down parameters")
    expanded=mx.expand_dims(x,(-2,-3))
    activated=experts.activation(experts.up_proj(expanded,selected_expert_ids),
                                 experts.gate_proj(expanded,selected_expert_ids))
    return _DOWN(inputs=[activated,selected_expert_ids,down.weight,down.scales,down.bias],
                 template=[("T",mx.bfloat16),("K",k),("N",n),("E",e)],
                 grid=((n//8)*64,4,1),threadgroup=(64,1,1),
                 output_shapes=[(1,1,4,n)],output_dtypes=[mx.bfloat16])[0]
'''

def generate(measurement):
    if measurement['status']!='completed_operator_diagnostic' or not measurement['sources_unchanged'] or not measurement['checkpoint_unchanged']:
        raise ValueError('Complete guarded down budget required')
    if set(measurement['screens'])!={'0','12','23'} or not all(r['passed']for r in measurement['screens'].values()):
        raise ValueError('All3fresh A/A screens required')
    rows=[measurement['summaries'][f'{layer}:without_down']for layer in (0,12,23)]
    if not all(r['environment_valid'] and r['descriptive_95pct_saved_ms'][0]>.05 for r in rows):
        raise ValueError('Down budget unsupported; do not generate')
    cid='cand_'+secrets.token_hex(6);candidate=ROOT/'candidates'/cid;candidate.mkdir()
    base=ROOT/'candidates/cand_90f0c66e0d03'
    source=(base/'implementation.py').read_text();header=source[source.index('_HEADER ='):source.index('_FUSED =')]
    implementation='"""Native-order down projection and BF16 bias epilogue, decode only."""\nfrom pathlib import Path\nimport mlx.core as mx\nfrom mlx_lm.models.gpt_oss import SwiGLU\n'+header+BODY.replace('KERNEL_NAME','kernelmax_down_'+cid)
    ast.parse(implementation)
    (candidate/'implementation.py').write_text(implementation);(candidate/'down_bias.metal').write_text(SHADER)
    (candidate/'wrapper.py').write_bytes((base/'wrapper.py').read_bytes())
    (candidate/'manifest.json').write_text(json.dumps({'candidate_id':cid,'name':'Native-order down projection and bias','implementation_type':'mlx','description':'Targets freshly measured native down projection cost with one custom native32-lane/four-row MXFP4 kernel and BF16 projection-rounding then BF16 bias. Native up/gate/SwiGLU, selected IDs, all weights/scales/precision, prefill fallback preserved. No parameter conversion, caching or routing weighting. Arithmetic structure follows native QMV; exact real-boundary checks required. No speedup claim.'},indent=2)+'\n')
    work=CONTROL/f'down-candidate-025-{cid}';work.mkdir()
    plan={'candidate_id':cid,'measured_target':str(CONTROL/'down-budget-023/result.json'),'measured_down_deletion':rows,'mechanism':'Custom down QMV fuses BF16 bias and tests LUT conversion while preserving native per-lane accumulation order. Dispatch and hardware cause unmeasured. Prior up/gate changes provided small operator gains; down is a distinct target.','model_contribution':'Unverified; hypothetical20percent down reduction saves24*0.2*region_ms per token only under additive representative critical-path assumptions. Full-model validation required.','planning_20percent_ms_per_token_interval':[[24*.2*r['descriptive_95pct_saved_ms'][0],24*.2*r['descriptive_95pct_saved_ms'][1]]for r in rows],'exact_repair_required':True,'no_rerolls':True}
    (work/'plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    for name in ['tiny_native_checks.py','operator_diagnostic.py','paired_noise.py','repair_checks.py','pipeline.py']:
        code=(CONTROL/'candidate-019'/name).read_text().replace('cand_d5987f75f173',cid)
        ast.parse(code);(work/name).write_text(code)
    return cid,work
