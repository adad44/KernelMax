// Native32-lane/four-row accumulation with a BF16 projection then bias.
const uint lane=thread_index_in_simdgroup;
const uint row0=threadgroup_position_in_grid.x*8+simdgroup_index_in_threadgroup*4;
const uint slot=threadgroup_position_in_grid.y;
const uint expert=uint(ids[slot]);
if(expert>=E){
    if(lane==0){for(uint row=0;row<4;++row)output[slot*N+row0+row]=T(as_type<float>(0x7fc00000u));}
    return;
}
float sum[4]={0.0f,0.0f,0.0f,0.0f};
for(uint base=0;base<K;base+=256){
    const uint column=base+lane*8;
    if(column<K){
        float values[8];
        #pragma unroll
        for(uint j=0;j<8;++j)values[j]=float(x[slot*K+column+j]);
        #pragma unroll
        for(uint row=0;row<4;++row){
            const uint output_row=expert*N+row0+row;
            const uint packed_offset=output_row*(K/8)+column/8;
            const uint scale_offset=output_row*(K/32)+column/32;
            sum[row]+=km_scale(scales[scale_offset])*km_dot8(weight[packed_offset],values);
        }
    }
}
#pragma unroll
for(uint row=0;row<4;++row){
    const float raw=simd_sum(sum[row]);
    if(lane==0){
        const uint output_row=expert*N+row0+row;
        const T rounded=T(raw);
        output[slot*N+row0+row]=T(float(rounded)+float(bias[output_row]));
    }
}
