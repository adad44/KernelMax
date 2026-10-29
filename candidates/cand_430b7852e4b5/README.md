# Native-style half FP4 decoder

Target: the audited up/gate region saved 0.37947 ms/call in a dependency-preserving
deletion experiment, about 46.7% of one sampled isolated expert call. This is
not a full-model share. A 20% region improvement would save 0.0759 ms/expert;
24 times that is a hypothetical 1.82 ms/token only under unverified additive,
representative-layer and cache assumptions.

The eight-lane LUT candidate showed only a 0.8% effect below calibration bias.
An integer-bit decoder then regressed 3.6% on valid layer-0 measurements. This
variant uses the installed MLX fp4.h half-conversion algorithm instead: bitcast
three magnitude bits into a half and multiply by 16384, then apply the sign.
All sixteen FP4 values are exactly representable. Actual dequantized values,
float accumulation and BF16 projection/bias/SwiGLU rounding stay unchanged.
The tiny local half temporary is not a model precision or weight-format change.

No hardware or compiler explanation is established by this source inspection.
Native correctness and fresh matched operator calibration remain required.
No persistent cache, copied weights or layout transformation. Native down and
prefill fallback remain. Acceptance is disabled.
