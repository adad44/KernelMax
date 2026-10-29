"""Native-order down projection and BF16 bias epilogue, decode only."""
from pathlib import Path
import mlx.core as mx
from mlx_lm.models.gpt_oss import SwiGLU
_HEADER = r"""
#include <metal_simdgroup>

constant float km_fp4_values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

inline float km_fp4(uint code) {
    float value = km_fp4_values[code & 7u];
    return (code & 8u) ? -value : value;
}

inline float km_scale(uchar code) {
    uint bits = code == 0 ? 0x00400000u : (uint(code) << 23);
    return as_type<float>(bits);
}

inline float km_dot8(uint word, thread const float* x) {
    float first = ((x[0] * km_fp4(word) + x[1] * km_fp4(word >> 4))
                   + x[2] * km_fp4(word >> 8)) + x[3] * km_fp4(word >> 12);
    float second = ((x[4] * km_fp4(word >> 16) + x[5] * km_fp4(word >> 20))
                    + x[6] * km_fp4(word >> 24)) + x[7] * km_fp4(word >> 28);
    return first + second;
}

template <typename T> inline T km_sigmoid(T x) {
    auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    return x < 0 ? T(y) : T(1 - y);
}
"""


_DOWN=mx.fast.metal_kernel(
    name="kernelmax_down_cand_38d341fb17de",input_names=["x","ids","weight","scales","bias"],
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
