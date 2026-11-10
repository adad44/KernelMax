# Vocabulary residency and fused MoE with native router partition

This second candidate keeps the first candidate's exact compressed vocabulary
reader and weight-lifetime interface. Its MoE fusion covers attention residual,
normalization, routing/top-k/weights, paired up/gate/SwiGLU and down/weighted
expert reduction/residual in three post-attention stages. Attention and prefill
use the native implementation.

For the pinned router shape K=2880,N=32, MLX 0.32.3 selects BN=8. The old fusion
used a different floating-point partition. This candidate uses eight K partials
and native-order summation, multiplexing output groups within the existing
normalization workgroup. This is a source-aligned arithmetic change; the cause
of the old complete-model divergence remains unproven until diagnostic results
and fresh full-length correctness establish it.

Native dispatch and arithmetic references:

- https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/matmul.cpp
- https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/gemv.h
- https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/rms_norm.metal

Vocabulary sources and expert up/down stages are inherited unchanged. Exact
native/packed backing exchange remains controller-owned. No speedup, numerical
pass, task-quality certification or acceptance verdict is asserted here.
