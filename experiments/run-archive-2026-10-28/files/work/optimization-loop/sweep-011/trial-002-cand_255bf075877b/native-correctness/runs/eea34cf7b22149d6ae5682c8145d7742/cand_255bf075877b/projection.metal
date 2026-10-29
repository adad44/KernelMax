// Generated numeric variant: lanes=16, simdgroups=2, values/lane=8.
const uint simd_lane=thread_index_in_simdgroup;
const uint k_lane=simd_lane & 15u;
const uint row=threadgroup_position_in_grid.x*4u
    +simdgroup_index_in_threadgroup*2u+simd_lane/16u;
const uint slot=threadgroup_position_in_grid.y;
const uint expert=uint(ids[slot]);
if(expert>=E){if(k_lane==0u)activated[slot*N+row]=T(as_type<float>(0x7fc00000u));return;}
const uint output_row=expert*N+row;
float sum_up=0.0f,sum_gate=0.0f;
for(uint base=0u;base<K;base+=128u){
    const uint column_base=base+k_lane*8u;
    #pragma unroll
    for(uint pack=0u;pack<1u;++pack){
        const uint column=column_base+pack*8u;
        if(column<K){
            float x_values[8];
            #pragma unroll
            for(uint j=0u;j<8u;++j)x_values[j]=float(x[column+j]);
            const uint w_offset=output_row*(K/8u)+column/8u;
            const uint s_offset=output_row*(K/32u)+column/32u;
            sum_up+=km_scale(up_s[s_offset])*km_dot8(up_w[w_offset],x_values);
            sum_gate+=km_scale(gate_s[s_offset])*km_dot8(gate_w[w_offset],x_values);
        }
    }
}
sum_up+=simd_shuffle_down(sum_up,8);
sum_gate+=simd_shuffle_down(sum_gate,8);
sum_up+=simd_shuffle_down(sum_up,4);
sum_gate+=simd_shuffle_down(sum_gate,4);
sum_up+=simd_shuffle_down(sum_up,2);
sum_gate+=simd_shuffle_down(sum_gate,2);
sum_up+=simd_shuffle_down(sum_up,1);
sum_gate+=simd_shuffle_down(sum_gate,1);
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
