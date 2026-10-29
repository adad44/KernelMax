// Four SIMD groups, one selected expert per group, four shared output rows.
const uint lane=thread_index_in_simdgroup;
const uint slot=simdgroup_index_in_threadgroup;
const uint row0=threadgroup_position_in_grid.x*4;
const uint expert=uint(ids[slot]);
threadgroup T products[16];
float sums[4]={0.0f,0.0f,0.0f,0.0f};
for(uint base=0;base<K;base+=256) {
    uint col=base+lane*8;
    if(col<K) {
        float values[8];
        #pragma unroll
        for(uint q=0;q<8;++q) values[q]=float(activated[slot*K+col+q]);
        #pragma unroll
        for(uint r=0;r<4;++r) {
            uint row=expert*N+row0+r;
            sums[r]+=km_scale(scales[row*(K/32)+col/32])*km_dot8(weight[row*(K/8)+col/8],values);
        }
    }
}
#pragma unroll
for(uint r=0;r<4;++r) {
    float raw=simd_sum(sums[r]);
    if(lane==0) {
        T rounded=T(raw);
        T biased=T(float(rounded)+float(bias[expert*N+row0+r]));
        products[slot*4+r]=T(float(biased)*float(weights[slot]));
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(slot==0 && lane<4) {
    // Native small-column reduction rounds the BF16 partial sum at each row.
    T total=T(0.0f);
    #pragma unroll
    for(uint q=0;q<4;++q) total=T(float(products[q*4+lane])+float(total));
    output[row0+lane]=T(float(residual[row0+lane])+float(total));
}
