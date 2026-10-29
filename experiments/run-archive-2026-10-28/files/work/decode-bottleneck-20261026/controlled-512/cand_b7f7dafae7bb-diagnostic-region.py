def diagnostic_up_gate_region(experts, x, selected_expert_ids):
    if x.shape[:2] != (1, 1) or selected_expert_ids.shape != (1, 1, 4):
        return experts(x, selected_expert_ids)
    if x.ndim != 3 or x.dtype != mx.bfloat16 or selected_expert_ids.dtype != mx.int32:
        raise ValueError('decode specialization requires BF16 [1,1,H] and int32 [1,1,4]')
    if type(experts.activation) is not SwiGLU:
        raise ValueError('decode specialization requires unchanged GPT-OSS SwiGLU')
    for projection in (experts.up_proj, experts.gate_proj, experts.down_proj):
        if (projection.mode, projection.bits, projection.group_size) != ('mxfp4', 4, 32):
            raise ValueError('decode specialization requires native MXFP4 projections')
        if 'bias' not in projection:
            raise ValueError('decode specialization requires native output biases')
    expanded = mx.expand_dims(x, (-2, -3))
    up = _project_without_output_bias(experts.up_proj, expanded, selected_expert_ids)
    gate = _project_without_output_bias(experts.gate_proj, expanded, selected_expert_ids)
    activated = _bias_swiglu(up, gate, experts.up_proj.bias, experts.gate_proj.bias, selected_expert_ids)
    return activated
