"""Decode hypothesis: fuse projection bias additions with GPT-OSS SwiGLU.

Native MXFP4 projections, weights, routing and down-projection stay unchanged.
Prefill and all shapes except one-token, batch-one, four-expert decode use the
unchanged reference. No arrays or compiled closures capturing weights are cached.
"""

from functools import partial

import mlx.core as mx
from mlx_lm.models.gpt_oss import SwiGLU


@partial(mx.compile, shapeless=True)
def _bias_swiglu(up, gate, up_bias, gate_bias, indices):
    up = up + mx.expand_dims(up_bias[indices], -2)
    gate = gate + mx.expand_dims(gate_bias[indices], -2)
    gate = mx.clip(gate, a_min=None, a_max=7.0)
    up = mx.clip(up, a_min=-7.0, a_max=7.0)
    return (gate * mx.sigmoid(1.702 * gate)) * (up + 1)


def _project_without_output_bias(projection, x, indices):
    return mx.gather_qmm(
        x,
        projection.weight,
        projection.scales,
        projection.get("biases"),
        rhs_indices=indices,
        transpose=True,
        group_size=projection.group_size,
        bits=projection.bits,
        mode=projection.mode,
        sorted_indices=False,
    )


def run(experts, x, selected_expert_ids):
    if x.shape[:2] != (1, 1) or selected_expert_ids.shape != (1, 1, 4):
        return experts(x, selected_expert_ids)
    if x.ndim != 3 or x.dtype != mx.bfloat16 or selected_expert_ids.dtype != mx.int32:
        raise ValueError("decode specialization requires BF16 [1,1,H] and int32 [1,1,4]")
    if type(experts.activation) is not SwiGLU:
        raise ValueError("decode specialization requires unchanged GPT-OSS SwiGLU")
    for projection in (experts.up_proj, experts.gate_proj, experts.down_proj):
        if (projection.mode, projection.bits, projection.group_size) != ("mxfp4", 4, 32):
            raise ValueError("decode specialization requires native MXFP4 projections")
        if "bias" not in projection:
            raise ValueError("decode specialization requires native output biases")
    expanded = mx.expand_dims(x, (-2, -3))
    up = _project_without_output_bias(experts.up_proj, expanded, selected_expert_ids)
    gate = _project_without_output_bias(experts.gate_proj, expanded, selected_expert_ids)
    activated = _bias_swiglu(
        up, gate, experts.up_proj.bias, experts.gate_proj.bias, selected_expert_ids
    )
    return experts.down_proj(activated, selected_expert_ids, sorted_indices=False).squeeze(-2)
