// Two SIMD groups compute eight output rows for one selected expert.
// Each lane reads eight activations and reuses them across up/gate projections.
const uint lane = thread_index_in_simdgroup;
const uint row0 = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup * 4;
const uint slot = threadgroup_position_in_grid.y;
const uint expert = uint(ids[slot]);
if (expert >= E) {
    if (lane == 0) {
        for (uint row = 0; row < 4; ++row) {
            activated[slot * N + row0 + row] = T(as_type<float>(0x7fc00000u));
        }
    }
    return;
}

float sum_up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
float sum_gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
for (uint base = 0; base < K; base += 256) {
    const uint column = base + lane * 8;
    if (column < K) {
        float values[8];
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            values[j] = float(x[column + j]);
        }
        #pragma unroll
        for (uint row = 0; row < 4; ++row) {
            const uint output_row = expert * N + row0 + row;
            const uint packed_offset = output_row * (K / 8) + column / 8;
            const uint scale_offset = output_row * (K / 32) + column / 32;
            sum_up[row] += km_scale(up_s[scale_offset]) * km_dot8(up_w[packed_offset], values);
            sum_gate[row] += km_scale(gate_s[scale_offset]) * km_dot8(gate_w[packed_offset], values);
        }
    }
}

#pragma unroll
for (uint row = 0; row < 4; ++row) {
    const float raw_up = simd_sum(sum_up[row]);
    const float raw_gate = simd_sum(sum_gate[row]);
    if (lane == 0) {
        const uint output_row = expert * N + row0 + row;
        // Preserve the native BF16 projection outputs and subsequent BF16 steps.
        const T rounded_up = T(raw_up);
        const T rounded_gate = T(raw_gate);
        const T biased_up = T(float(rounded_up) + float(up_bias[output_row]));
        const T biased_gate = T(float(rounded_gate) + float(gate_bias[output_row]));
        const T clipped_up = T(metal::clamp(float(biased_up), -7.0f, 7.0f));
        const T clipped_gate = T(metal::min(float(biased_gate), 7.0f));
        const T scaled_gate = T(T(1.702f) * clipped_gate);
        const T sigmoid = km_sigmoid(scaled_gate);
        const T gated = T(clipped_gate * sigmoid);
        const T shifted_up = T(clipped_up + T(1.0f));
        activated[slot * N + row0 + row] = T(gated * shifted_up);
    }
}
