# Sixteen-lane joint up/gate reduction

Eight-lane variants did not produce a reliable gain; direct integer decoding
regressed. This candidate keeps native-style exact half FP4 decoding and changes
only reduction width to sixteen lanes per row, two rows per SIMD group and four
rows per64threadgroup. Each lane now handles16 rather than32 K elements per
256element iteration. This tests a different parallelism/register/coalescing
balance; no occupancy or bandwidth cause is established.

The validated target remains0.37947ms/call of one isolated expert (46.7%). A20%
region gain would save0.0759ms/expert or hypothetical1.82ms/token across24 equal
additive layers. Neither layer representativeness nor the full-model critical
path/cache conditions are established. This is no model speedup claim.

Original packed native MXFP4 weights/scales/routes, float accumulations and BF16
projection/bias/SwiGLU rounding are preserved. Native down and prefill fallback
remain. No weight copying, requantization, tensor cache or layout conversion.
