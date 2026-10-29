// Single-token attention residual, RMS normalization, BF16 router and top4.
// RMS lane partition mirrors the native4-values/thread reduction for H<=4096.
const uint tid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
threadgroup float partial[32];
threadgroup float inverse[1];
threadgroup T normalized[H];
threadgroup T scores[32];
threadgroup T selected[4];
float values[4];
float squares = 0.0f;
#pragma unroll
for (uint q=0;q<4;++q) {
    uint col=tid*4+q;
    T merged=col<H ? T(float(residual[col])+float(attention[col])) : T(0.0f);
    values[q]=float(merged);
    squares+=values[q]*values[q];
    if(col<H) residual_out[col]=merged;
}
float subtotal=simd_sum(squares);
if(sg==0) partial[lane]=0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(lane==0) partial[sg]=subtotal;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0) {
    float total=simd_sum(partial[lane]);
    if(lane==0) inverse[0]=metal::precise::rsqrt(total/float(H)+1.0e-5f);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
for(uint q=0;q<4;++q) {
    uint col=tid*4+q;
    if(col<H) {
        T value=T(norm_weight[col]*T(values[q]*inverse[0]));
        normalized[col]=value;
        norm_out[col]=value;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg<8) {
    float sums[4]={0.0f,0.0f,0.0f,0.0f};
    for(uint base=0;base<H;base+=128) {
        uint col=base+lane*4;
        #pragma unroll
        for(uint r=0;r<4;++r) {
            #pragma unroll
            for(uint q=0;q<4;++q) {
                float activation=col+q<H ? float(normalized[col+q]) : 0.0f;
                float weight=col+q<H ? float(router_weight[(sg*4+r)*H+col+q]) : 0.0f;
                sums[r]+=weight*activation;
            }
        }
    }
    #pragma unroll
    for(uint r=0;r<4;++r) {
        #pragma unroll
        for(ushort shift=16;shift>=1;shift>>=1) sums[r]+=simd_shuffle_down(sums[r],shift);
        if(lane==0) scores[sg*4+r]=T(T(1.0f)*T(sums[r])+T(1.0f)*router_bias[sg*4+r]);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
// Native MLX0.32.3 ArgPartition uses stable ascending full sort. Rank ties by
// original index; emit the final four slots in the same ascending order.
if(tid<32) {
    float value=float(scores[tid]);
    uint rank=0;
    for(uint other=0;other<32;++other) {
        float v=float(scores[other]);
        rank+=uint(v<value || (v==value && other<tid));
    }
    if(rank>=28) { ids[rank-28]=tid; selected[rank-28]=scores[tid]; }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(tid==0) {
    float maximum=float(selected[3]);
    float exponents[4];float denominator=0.0f;
    #pragma unroll
    for(uint q=0;q<4;++q) { exponents[q]=metal::fast::exp(float(selected[q])-maximum);denominator+=exponents[q]; }
    float reciprocal=1.0f/denominator;
    #pragma unroll
    for(uint q=0;q<4;++q) weights[q]=T(exponents[q]*reciprocal);
}
