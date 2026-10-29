# Lossless BF16 dense decode reads

The candidate encodes64original BF16 words into64low bytes,32upper-byte
delta/sign bytes and a4byte header. A block whose upper exponent bits do not fit
the exact3bit range uses its original64raw words in an escape table. This is
lossless word encoding, with no rounding or requantization. Eligible blocks use
100bytes instead of128bytes, before escapes. The Metal reader reconstructs each
original BF16 word as FP32 exactly for the unchanged ordered matrix product.

`prepare_projection(weight)` builds CPU-prepared sidecars for a supported native
BF16 matrix. Its callable handles one BF16 vector and optional native bias.
The controller must use unchanged native fallbacks for prefill and unsupported
layouts. Intended sites are attention q/k/v/o and the vocabulary projection;
native MXFP4 experts and the rest of the model are unchanged. This is not an
expert-only `run(experts,x,ids)` submission.

The prototype keeps the reference's original weights for matched comparisons,
so sidecars increase allocation. Preparation cost, encoded bytes, escapes,
actual sidecar allocation and peak memory must all be reported. Reduced weight
reads do not establish reduced total residency or a performance gain.

Before timing require exhaustive CPU word roundtrips, native GPU decoding and
projection checks with bias and tail shapes, then whole-model logits, routes,
greedy tokens, BF16 caches and fallback/call-count checks. Preserve all samples
under the exploration/confirmation policies; only fresh strict confirmation
can support a speedup claim. This directory contains no verdict.
