"""Compile the native one-token expert graph with explicit weight inputs."""

import mlx.core as mx
from mlx_lm.models.gpt_oss import SwiGLU, swiglu


def _project(x, indices, weight, scales, bias):
    value = mx.gather_qmm(
        x,
        weight,
        scales,
        None,
        rhs_indices=indices,
        transpose=True,
        group_size=32,
        bits=4,
        mode="mxfp4",
        sorted_indices=False,
    )
    return value + mx.expand_dims(bias[indices], -2)


@mx.compile
def _expert_decode(
    x, indices, up_weight, up_scales, up_bias, gate_weight, gate_scales,
    gate_bias, down_weight, down_scales, down_bias,
):
    expanded = mx.expand_dims(x, (-2, -3))
    up = _project(expanded, indices, up_weight, up_scales, up_bias)
    gate = _project(expanded, indices, gate_weight, gate_scales, gate_bias)
    activated = swiglu(up, gate)
    return _project(activated, indices, down_weight, down_scales, down_bias).squeeze(-2)


def run(experts, x, selected_expert_ids):
    if x.shape[:2] != (1, 1) or selected_expert_ids.shape != (1, 1, 4):
        return experts(x, selected_expert_ids)
    if x.ndim != 3 or x.dtype != mx.bfloat16:
        raise ValueError("compiled decode requires BF16 [1,1,H] activations")
    if selected_expert_ids.dtype not in (mx.int32, mx.uint32):
        raise ValueError("compiled decode requires int32 or uint32 expert IDs")
    if type(experts.activation) is not SwiGLU:
        raise ValueError("compiled decode requires unchanged GPT-OSS SwiGLU")
    up, gate, down = experts.up_proj, experts.gate_proj, experts.down_proj
    for projection in (up, gate, down):
        if (projection.mode, projection.bits, projection.group_size) != ("mxfp4", 4, 32):
            raise ValueError("compiled decode requires native MXFP4 projections")
        if projection.get("biases") is not None or "bias" not in projection:
            raise ValueError("compiled decode requires native output biases and no affine offsets")
        if (projection.weight.dtype != mx.uint32 or projection.scales.dtype != mx.uint8
                or projection.bias.dtype != mx.bfloat16):
            raise ValueError("compiled decode requires unchanged native packed dtypes")
    return _expert_decode(
        x, selected_expert_ids,
        up.weight, up.scales, up.bias,
        gate.weight, gate.scales, gate.bias,
        down.weight, down.scales, down.bias,
    )
