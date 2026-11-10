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
threadgroup float router_partials[64];
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
// Pinned MLX GEMV chooses BN=8 for K=2880,N=32. Each four-row output
// group therefore has eight independent K partials, then a sequential
// cross-partial reduction. Four iterations fit the eight output groups into
// this normalization workgroup without exceeding the hardware thread limit.
for(uint batch=0;batch<4;++batch) {
    const uint group_m=sg/8;
    const uint group_k=sg%8;
    const uint row0=(batch*2+group_m)*4;
    if(sg<16) {
        float sums[4]={0.0f,0.0f,0.0f,0.0f};
        for(uint base=0;base<H;base+=1024) {
            const uint col=base+group_k*128+lane*4;
            #pragma unroll
            for(uint r=0;r<4;++r) {
                #pragma unroll
                for(uint q=0;q<4;++q) {
                    const float activation=col+q<H ? float(normalized[col+q]) : 0.0f;
                    const float weight=col+q<H ? float(router_weight[(row0+r)*H+col+q]) : 0.0f;
                    sums[r]+=weight*activation;
                }
            }
        }
        #pragma unroll
        for(uint r=0;r<4;++r) {
            #pragma unroll
            for(ushort shift=16;shift>=1;shift>>=1) sums[r]+=simd_shuffle_down(sums[r],shift);
            if(lane==0) router_partials[group_m*32+group_k*4+r]=sums[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg<16 && group_k==0 && lane==0) {
        #pragma unroll
        for(uint r=0;r<4;++r) {
            float total=router_partials[group_m*32+r];
            #pragma unroll
            for(uint part=1;part<8;++part) total+=router_partials[group_m*32+part*4+r];
            scores[row0+r]=T(T(1.0f)*T(total)+T(1.0f)*router_bias[row0+r]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
