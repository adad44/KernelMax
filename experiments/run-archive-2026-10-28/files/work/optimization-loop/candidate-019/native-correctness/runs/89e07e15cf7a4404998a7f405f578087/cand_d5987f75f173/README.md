# Native-order reduction with exact half FP4 decoding

The sixteen-lane shortlist changed rare projection rounding results, which passed
operator tolerance but amplified into full-model routing/logit differences.
Pointwise isolation matched native exactly on the same inputs. This candidate
returns to the original four-output-row,32-lane reduction mapping used by the
first fused kernel and native generic QMV. It changes only that fused kernel's
FP4 decoder to the exact native-style half conversion.

The purpose is to preserve native projection arithmetic while improving the
implementation cost of the fused up/gate kernel. No bit-exact or speedup claim
is made before testing the saved real-boundary regressions. Native MXFP4 values,
scales, routes, BF16 intermediate rounding, native down and prefill fallback
remain. There is no packing, copied weight storage or persistent tensor cache.

The measured up/gate target is about0.379ms per sampled expert call. A20% region
gain would save0.0759ms/expert or a hypothetical1.82ms/token over24 equal additive
layers. Actual full-model contribution and hardware cause remain unverified.
