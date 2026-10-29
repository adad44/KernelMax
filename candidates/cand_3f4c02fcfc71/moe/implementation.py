"""Fused three-stage post-attention decode pipeline; native prefill fallback."""

from pathlib import Path
import mlx.core as mx

_HEADER = r"""
#include <metal_simdgroup>
constant float km_fp4_values[8]={0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
inline float km_fp4(uint word) {float v=km_fp4_values[word&7u];return word&8u ? -v : v;}
inline float km_scale(uchar code) {return as_type<float>(code==0 ? 0x00400000u : uint(code)<<23);}
inline float km_dot8(uint word,thread const float* x) {
    float a=((x[0]*km_fp4(word)+x[1]*km_fp4(word>>4))+x[2]*km_fp4(word>>8))+x[3]*km_fp4(word>>12);
    float b=((x[4]*km_fp4(word>>16)+x[5]*km_fp4(word>>20))+x[6]*km_fp4(word>>24))+x[7]*km_fp4(word>>28);
    return a+b;
}
template<typename T> inline T km_sigmoid(T x) {
    auto y=1/(1+metal::precise::exp(metal::abs(x)));
    return x<0 ? T(y) : T(1-y);
}
"""

_ROUTE = mx.fast.metal_kernel(
    name="sprint_route_4a02c3189b7d",
    input_names=[
        "residual",
        "attention",
        "norm_weight",
        "router_weight",
        "router_bias",
    ],
    output_names=["norm_out", "residual_out", "ids", "weights"],
    source=Path(__file__).with_name("route.metal").read_text(),
    header=_HEADER,
    ensure_row_contiguous=True,
    compile_options={"math_mode": "safe"},
)
_UP = mx.fast.metal_kernel(
    name="sprint_up_gate_4a02c3189b7d",
    input_names=[
        "x",
        "ids",
        "up_w",
        "up_s",
        "up_bias",
        "gate_w",
        "gate_s",
        "gate_bias",
    ],
    output_names=["activated"],
    source=Path(__file__).with_name("up_gate.metal").read_text(),
    header=_HEADER,
    ensure_row_contiguous=True,
    compile_options={"math_mode": "safe"},
)
_DOWN = mx.fast.metal_kernel(
    name="sprint_down_reduce_4a02c3189b7d",
    input_names=["activated", "ids", "weight", "scales", "bias", "weights", "residual"],
    output_names=["output"],
    source=Path(__file__).with_name("down_reduce.metal").read_text(),
    header=_HEADER,
    ensure_row_contiguous=True,
    compile_options={"math_mode": "safe"},
)


def routed_norm(residual, attention, norm, router):
    h = residual.shape[-1]
    if (
        residual.dtype != mx.bfloat16
        or attention.dtype != mx.bfloat16
        or residual.shape != (1, 1, h)
        or attention.shape != residual.shape
        or h % 32
        or not 128 <= h <= 4096
        or norm.eps != 1e-5
        or router.weight.shape != (32, h)
        or router.bias.shape != (32,)
    ):
        raise ValueError("Unsupported fused normalization/routing shape or precision")
    threads = max(256, ((h + 127) // 128) * 32)
    return _ROUTE(
        inputs=[residual, attention, norm.weight, router.weight, router.bias],
        template=[("T", mx.bfloat16), ("H", h)],
        grid=(threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(1, 1, h), (1, 1, h), (1, 1, 4), (1, 1, 4)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.uint32, mx.bfloat16],
    )


def expert_reduce(experts, x, ids, weights, residual):
    up, gate, down = experts.up_proj, experts.gate_proj, experts.down_proj
    h = x.shape[-1]
    n = up.output_dims
    e = up.num_experts
    if x.shape != (1, 1, h) or x.dtype != mx.bfloat16 or h % 32 or n % 32 or e != 32:
        raise ValueError("Unsupported fused MoE decode dimensions")
    for p in (up, gate, down):
        if (
            (p.mode, p.bits, p.group_size) != ("mxfp4", 4, 32)
            or p.weight.dtype != mx.uint32
            or p.scales.dtype != mx.uint8
            or p.bias.dtype != mx.bfloat16
        ):
            raise ValueError("Native MXFP4 weights and BF16 biases required")
    if down.input_dims != n or down.output_dims != h:
        raise ValueError("Mismatched down projection")
    activated = _UP(
        inputs=[
            x,
            ids,
            up.weight,
            up.scales,
            up.bias,
            gate.weight,
            gate.scales,
            gate.bias,
        ],
        template=[("T", mx.bfloat16), ("K", h), ("N", n), ("E", e)],
        grid=((n // 4) * 32, 4, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, 1, 4, 1, n)],
        output_dtypes=[mx.bfloat16],
    )[0]
    output = _DOWN(
        inputs=[activated, ids, down.weight, down.scales, down.bias, weights, residual],
        template=[("T", mx.bfloat16), ("K", n), ("N", h), ("E", e)],
        grid=((h // 4) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, 1, h)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return output


class Adapter:
    kind = "moe"
    statistics = {
        "pipeline_stages_after_attention": 3,
        "original_weight_values_preserved": True,
        "precision": "native MXFP4/BF16",
        "prefill": "unchanged native TransformerBlock",
    }

    def __init__(self, model):
        if len(model.model.layers) != 24 or model.args.hidden_size != 2880:
            raise ValueError("Expected the pinned24layer GPT-OSS20B model")
        for layer in model.model.layers:
            if (
                layer.mlp.sharding_group is not None
                or layer.mlp.num_experts_per_tok != 4
            ):
                raise ValueError("Unsupported sharding/routing configuration")

    def call_block(self, original, layer, x, mask, cache, trace):
        if x.shape[:2] != (1, 1):
            return original(layer, x, mask, cache)
        attention = layer.self_attn(layer.input_layernorm(x), mask, cache)
        normalized, residual, ids, weights = routed_norm(
            x, attention, layer.post_attention_layernorm, layer.mlp.router
        )
        trace(ids)
        return expert_reduce(layer.mlp.experts, normalized, ids, weights, residual)


def prepare(model):
    return Adapter(model)
