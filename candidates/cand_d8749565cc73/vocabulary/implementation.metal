// Ordered BF16 GEMV, with exact compressed-word loads. No weight approximation.
const uint lane = thread_index_in_simdgroup;
const uint simd = simdgroup_index_in_threadgroup;
const uint row = threadgroup_position_in_grid.x * (BM * 4) + simd * 4;
if (row >= N) return;
float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
for (uint base = 0; base < K; base += 128) {
    const uint column = base + lane * 4;
    float v[4];
    #pragma unroll
    for (uint q = 0; q < 4; ++q) {
        v[q] = column + q < K ? float(x[column + q]) : 0.0f;
    }
    #pragma unroll
    for (uint r = 0; r < 4; ++r) {
        float4 weights = float4(0.0f);
        if (column < K) {
            weights = km_load_bf16_4(payload, headers, escapes,
                                      (row + r) * (K / 64) + column / 64,
                                      column & 63u);
        }
        #pragma unroll
        for (uint q = 0; q < 4; ++q) {
            sums[r] += weights[q] * v[q];
        }
    }
}
#pragma unroll
for (uint r = 0; r < 4; ++r) {
    #pragma unroll
    for (ushort shift = 16; shift >= 1; shift >>= 1) {
        sums[r] += simd_shuffle_down(sums[r], shift);
    }
}
if (lane == 0) {
    #pragma unroll
    for (uint r = 0; r < 4; ++r) {
        if (HAS_BIAS) {
            // Same BF16 accumulator rounding and unit-scale expression as
            // the native biased GEMV, before the final output conversion.
            output[row + r] = T(1.0f) * T(sums[r]) + T(1.0f) * bias[row + r];
        } else {
            output[row + r] = T(sums[r]);
        }
    }
}
