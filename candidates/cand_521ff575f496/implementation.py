"""Test a cheaper inverse permutation around unchanged native expert maths.

The sorted-call sequence follows MLX-LM 0.31.3 SwitchGLU. Only inverse-index
construction differs: unique permutation destinations need one scatter instead
of a second argsort. NumPy is used only by candidate-local development tests.
"""


def _inverse_permutation(order, mx):
    """Map each original token/expert slot to its position in sorted order."""
    inverse = mx.zeros_like(order)
    # Even when expert IDs repeat, order contains every slot exactly once.
    inverse[order] = mx.arange(order.size, dtype=order.dtype)
    return inverse


def _run_sorted(experts, x, selected_expert_ids, mx):
    """Use native projections/activation on the same expert-sorted inputs."""
    flat_ids = selected_expert_ids.reshape(-1)
    order = mx.argsort(flat_ids)
    inverse = _inverse_permutation(order, mx)
    sorted_ids = flat_ids[order]
    # Four expert slots share each token's original activation vector.
    sorted_x = x.reshape(-1, 1, x.shape[-1])[order // 4]
    up = experts.up_proj(sorted_x, sorted_ids, sorted_indices=True)
    gate = experts.gate_proj(sorted_x, sorted_ids, sorted_indices=True)
    activated = experts.activation(up, gate)
    output = experts.down_proj(activated, sorted_ids, sorted_indices=True)
    return output[inverse].reshape(*selected_expert_ids.shape, x.shape[-1])


def _supports_native(experts, x, ids, mx):
    """Check metadata only; unsupported modules retain their own call behavior."""
    from mlx_lm.models.gpt_oss import SwiGLU
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU

    if (
        type(experts) is not SwitchGLU
        or experts.training
        or type(experts.activation) is not SwiGLU
        or not isinstance(x, mx.array)
        or not isinstance(ids, mx.array)
        or x.dtype != mx.bfloat16
        or ids.dtype not in (mx.int32, mx.uint32)
        or x.ndim != 3
        or x.shape[0] != 1
        or x.shape[1] < 16
        or x.shape[2] < 1
        or ids.shape != (*x.shape[:2], 4)
    ):
        return False
    up, gate, down = experts.up_proj, experts.gate_proj, experts.down_proj
    for projection in (up, gate, down):
        if (
            type(projection) is not QuantizedSwitchLinear
            or (projection.mode, projection.bits, projection.group_size)
            != ("mxfp4", 4, 32)
        ):
            return False
    return (
        up.input_dims == gate.input_dims == down.output_dims == x.shape[-1]
        and up.output_dims == gate.output_dims == down.input_dims
        and up.num_experts == gate.num_experts == down.num_experts
    )


def run(experts, x, selected_expert_ids):
    """Optimize supported native prefill; delegate all other calls unchanged."""
    # Keep plain fake fixtures usable without importing MLX or loading a model.
    if (
        type(x).__module__ != "mlx.core"
        or type(selected_expert_ids).__module__ != "mlx.core"
    ):
        return experts(x, selected_expert_ids)
    import mlx.core as mx

    if not _supports_native(experts, x, selected_expert_ids, mx):
        return experts(x, selected_expert_ids)
    return _run_sorted(experts, x, selected_expert_ids, mx)
