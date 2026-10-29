def diagnostic_activation(experts, x, selected_expert_ids):
    if x.shape[:2] != (1, 1) or selected_expert_ids.shape != (1, 1, 4):
        return experts(x, selected_expert_ids)
    if x.ndim != 3 or x.dtype != mx.bfloat16:
        raise ValueError('fused decode requires BF16 [1,1,H] activations')
    if selected_expert_ids.dtype not in (mx.int32, mx.uint32):
        raise ValueError('fused decode requires int32 or uint32 routing IDs')
    if type(experts.activation) is not SwiGLU:
        raise ValueError('fused decode requires unchanged GPT-OSS SwiGLU')
    up, gate, down = (experts.up_proj, experts.gate_proj, experts.down_proj)
    k, n, e = (x.shape[-1], up.output_dims, up.num_experts)
    if n % 4:
        return experts(x, selected_expert_ids)
    if k <= 0 or k % 32 or n <= 0 or n % 8 or (e <= 0):
        raise ValueError('fused decode requires K divisible by 32 and N divisible by 8')
    for projection in (up, gate, down):
        if (projection.mode, projection.bits, projection.group_size) != ('mxfp4', 4, 32):
            raise ValueError('fused decode requires native MXFP4 projections')
    for projection in (up, gate):
        if projection.weight.dtype != mx.uint32 or projection.scales.dtype != mx.uint8 or projection.weight.shape != (e, n, k // 8) or (projection.scales.shape != (e, n, k // 32)) or (projection.get('biases') is not None) or ('bias' not in projection) or (projection.bias.dtype != mx.bfloat16) or (projection.bias.shape != (e, n)):
            raise ValueError('unsupported native packed weight, scale or bias layout')
    if down.input_dims != n or down.output_dims != k or down.num_experts != e:
        raise ValueError('down projection dimensions disagree with fused projections')
    activated = _FUSED(inputs=[x, selected_expert_ids, up.weight, up.scales, up.bias, gate.weight, gate.scales, gate.bias], template=[('T', mx.bfloat16), ('K', k), ('N', n), ('E', e)], grid=(n // 4 * 64, 4, 1), threadgroup=(64, 1, 1), output_shapes=[(1, 1, 4, 1, n)], output_dtypes=[mx.bfloat16])[0]
    return activated
