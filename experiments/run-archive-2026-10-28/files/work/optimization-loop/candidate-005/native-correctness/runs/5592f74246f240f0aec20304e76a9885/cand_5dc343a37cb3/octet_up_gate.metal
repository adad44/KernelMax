// Four independent eight-lane reductions per SIMD group; eight rows per group.
// Weights and scales are read directly in the unchanged native MXFP4 layout.
const uint simd_lane = thread_index_in_simdgroup;
const uint k_lane = simd_lane & 7u;
const uint row = threadgroup_position_in_grid.x * 8u
    + simdgroup_index_in_threadgroup * 4u + simd_lane / 8u;
const uint slot = threadgroup_position_in_grid.y;
const uint expert = uint(ids[slot]);
if (expert >= E) {
    if (k_lane == 0u) {
        activated[slot * N + row] = T(as_type<float>(0x7fc00000u));
    }
    return;
}

const uint output_row = expert * N + row;
float sum_up = 0.0f;
float sum_gate = 0.0f;
for (uint base = 0u; base < K; base += 256u) {
    const uint group_column = base + k_lane * 32u;
    if (group_column < K) {
        const uint scale_offset = output_row * (K / 32u) + group_column / 32u;
        const float up_scale = km_scale(up_s[scale_offset]);
        const float gate_scale = km_scale(gate_s[scale_offset]);
        #pragma unroll
        for (uint pack = 0u; pack < 4u; ++pack) {
            const uint column = group_column + pack * 8u;
            float values[8];
            #pragma unroll
            for (uint j = 0u; j < 8u; ++j) {
                values[j] = float(x[column + j]);
            }
            const uint packed_offset = output_row * (K / 8u) + column / 8u;
            sum_up += up_scale * km_dot8(up_w[packed_offset], values);
            sum_gate += gate_scale * km_dot8(gate_w[packed_offset], values);
        }
    }
}

sum_up += simd_shuffle_down(sum_up, 4);
sum_gate += simd_shuffle_down(sum_gate, 4);
sum_up += simd_shuffle_down(sum_up, 2);
sum_gate += simd_shuffle_down(sum_gate, 2);
sum_up += simd_shuffle_down(sum_up, 1);
sum_gate += simd_shuffle_down(sum_gate, 1);
if (k_lane == 0u) {
    // Match BF16 projection rounding before biases and each activation step.
    const T rounded_up = T(sum_up);
    const T rounded_gate = T(sum_gate);
    const T biased_up = T(float(rounded_up) + float(up_bias[output_row]));
    const T biased_gate = T(float(rounded_gate) + float(gate_bias[output_row]));
    const T clipped_up = T(metal::clamp(float(biased_up), -7.0f, 7.0f));
    const T clipped_gate = T(metal::min(float(biased_gate), 7.0f));
    const T scaled_gate = T(T(1.702f) * clipped_gate);
    const T sigmoid = km_sigmoid(scaled_gate);
    const T gated = T(clipped_gate * sigmoid);
    const T shifted_up = T(clipped_up + T(1.0f));
    activated[slot * N + row] = T(gated * shifted_up);
}
