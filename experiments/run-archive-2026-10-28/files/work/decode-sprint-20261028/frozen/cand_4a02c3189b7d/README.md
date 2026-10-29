# Fused post-attention MoE block

One-token24layer decode replaces the residual/normalization/routing/expert tail
with three Metal stages. Attention and its input normalization remain native;
prefill uses the original whole TransformerBlock. Checkpoint native MXFP4
products, BF16 boundaries and four selected experts are retained.

Stage1 combines the attention residual with RMSNorm and BF16 dense routing,
stable ascending top4 and precise-accumulation softmax. RMSNorm mirrors the
installed four-values/thread reduction including BF16 rounding before the scale
multiply; the32expert GEMV preserves its native32lane/four-value order. Native
MLX0.32.3 ArgPartition dispatches a complete ascending sort. Tie ordering and
softmax are independently checked in the native fixture.

Stage2 shares activation loads across joint up/gate MXFP4 products, BF16 biases,
clipping and SwiGLU. Its native-ordered arithmetic is vendored from the earlier
reviewed local `cand_35fff69ba1e1` up/gate implementation. Stage3 executes all
four selected-expert down products per output tile, then BF16 bias, weighting,
ordered BF16 expert reduction and final residual without materializing the
separate expert-output/weighted-output tensors. No global-barrier spinning or
additional GPU/CPU waits are introduced.

This is a three-dispatch fused pipeline, not a single whole-model megakernel.
The hypothesis is to remove launch/intermediate/scheduling costs across every
layer. It retains all weight reads; the measured14.7ms expert deletion budget is
an optimistic target, not a promised saving. Full-model decode determines the
result. Existing provisional BF16 tolerances apply; record routing/token/cache
differences and require fresh strict confirmation before a speedup statement.

Implementation references: Apple MLXv0.32.3 normalization.cpp/rms_norm.metal,
sort.cpp, and installed gemv.h/reduction/softmax headers (MIT, Apple2023–2024).
The custom stages are separately authored for this pinned model shape.
