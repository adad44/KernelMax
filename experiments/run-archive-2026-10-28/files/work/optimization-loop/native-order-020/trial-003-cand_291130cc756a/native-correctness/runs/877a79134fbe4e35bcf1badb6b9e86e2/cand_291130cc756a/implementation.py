"""One-token native-MXFP4 up/gate projections and SwiGLU in one Metal dispatch."""

from pathlib import Path

import mlx.core as mx
from mlx_lm.models.gpt_oss import SwiGLU


_HEADER = r"""
#include <metal_simdgroup>

// Native-style exact FP4 conversion; preserve the original32-lane dot/reduction.
inline float km_fp4(uint code) {
    half converted = as_type<half>(ushort((code & 7u) << 9));
    converted *= 16384.0;
    return float((code & 8u) ? -converted : converted);
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

_FUSED = mx.fast.metal_kernel(
    name="kernelmax_native_group_cand_291130cc756a",
    input_names=["x", "ids", "up_w", "up_s", "up_bias", "gate_w", "gate_s", "gate_bias"],
    output_names=["activated"],
    source=Path(__file__).with_name("fused_up_gate.metal").read_text(),
    header=_HEADER,
    ensure_row_contiguous=True,
    compile_options={"math_mode": "safe"},
)


def run(experts, x, selected_expert_ids):
    if x.shape[:2] != (1, 1) or selected_expert_ids.shape != (1, 1, 4):
        return experts(x, selected_expert_ids)
    if x.ndim != 3 or x.dtype != mx.bfloat16:
        raise ValueError("fused decode requires BF16 [1,1,H] activations")
    if selected_expert_ids.dtype not in (mx.int32, mx.uint32):
        raise ValueError("fused decode requires int32 or uint32 routing IDs")
    if type(experts.activation) is not SwiGLU:
        raise ValueError("fused decode requires unchanged GPT-OSS SwiGLU")
    up, gate, down = experts.up_proj, experts.gate_proj, experts.down_proj
    k, n, e = x.shape[-1], up.output_dims, up.num_experts
    if k <= 0 or k % 32 or n <= 0 or n % 8 or e <= 0:
        raise ValueError("fused decode requires K divisible by 32 and N divisible by 8")
    for projection in (up, gate, down):
        if (projection.mode, projection.bits, projection.group_size) != ("mxfp4", 4, 32):
            raise ValueError("fused decode requires native MXFP4 projections")
    for projection in (up, gate):
        if (
            projection.weight.dtype != mx.uint32
            or projection.scales.dtype != mx.uint8
            or projection.weight.shape != (e, n, k // 8)
            or projection.scales.shape != (e, n, k // 32)
            or projection.get("biases") is not None
            or "bias" not in projection
            or projection.bias.dtype != mx.bfloat16
            or projection.bias.shape != (e, n)
        ):
            raise ValueError("unsupported native packed weight, scale or bias layout")
    if down.input_dims != n or down.output_dims != k or down.num_experts != e:
        raise ValueError("down projection dimensions disagree with fused projections")
    activated = _FUSED(
        inputs=[x, selected_expert_ids, up.weight, up.scales, up.bias,
                gate.weight, gate.scales, gate.bias],
        template=[("T", mx.bfloat16), ("K", k), ("N", n), ("E", e)],
        grid=((n // 32) * 256, 4, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, 1, 4, 1, n)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return down(activated, selected_expert_ids, sorted_indices=False).squeeze(-2)
