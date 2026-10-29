# Exact integer-bit FP4 decoding

The preceding eight-lane kernel produced only a 0.8% descriptive layer-0 effect,
smaller than its A/A calibration bias; layer-12 calibration failed. It is parked.
This variant isolates a different cost within the same measured up/gate region:
each dynamic FP4 lookup becomes integer construction of the exact float bits.
The installed native fp4.h also avoids a constant lookup, using half conversion;
this source uses direct float encodings and does not depend on half subnormals.
No assertion is made about emitted instructions, occupancy or hardware cause.

All 16 E2M1 encodings, including signed zero, are checked bit-for-bit on CPU
before execution. The underlying eight-lane projection, accumulation order,
scale handling and BF16 projection/bias/SwiGLU steps are unchanged. No tensor
layout transformation, copied weights, cache or requantization. The same
reference fallback covers prefill. Native correctness remains required.

The measured target is about 0.37947 ms per isolated expert call (46.7% of the
layer-0 operator), from the audited probe-004 deletion experiment. This is no
full-model share. The planning scenario of a 20% region improvement would save
about 0.0759 ms/expert; multiplying by 24 gives 1.82 ms/token only if all layers
and critical-path/cache conditions matched, which is unverified. The direct
bit-decode change might save nothing if weight movement or other work dominates.

Acceptance stays disabled. Only fresh valid paired results can establish an
operator improvement; full-model confirmation requires separate calibration.
