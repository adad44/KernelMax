# Next candidate: exact dense-weight read layout

Candidate slot `cand_390712b41518` contains an unfrozen implementation; it has not
executed on native weights/GPU or established a performance result.

The three completed timestamp probes measured about43.64/44.79/42.28ms per
sampled step inside GPU command-buffer intervals and1.18/1.25/1.15ms outside.
Command-buffer intervals include stalls, so this is not kernel-active time or
a proof that every internal wait is necessary. It does not justify saving3ms
with the prepared plain async candidate `cand_75ca936df290`; that snapshot stays
unexecuted. The first observer/control pair also differed by about5ms; retain
that drift rather than calling the observer a speedup. Source artifact:
`../decode-search-20261027/bookkeeping/timeline-light-summary.json`.

The new candidate targets weight-read traffic in all24attention q/k/v/o products
and the vocabulary projection. Native checkpoint dimensions are H2880,
Q4096,KV512,vocab201088. These dense matrices total2,432,286,720BF16bytes.
An eligible64word block stores100bytes instead of128; without escapes this saves
532,062,720logical bytes/token. At a hypothetical80–120GB/s critical-path read
rate that is4.43–6.65ms of gross budget, before decoding/dispatch/cache costs.
This is conditional planning, not measured DRAM traffic, bandwidth, or speedup.
The remaining bounded region-removal pass will refine the actual target budget.

Representation: preserve all16BF16bits. Store64low bytes plus32nibble bytes,
encoding sign and a3bit delta of upper exponent bits from a block base, plus a
4byte header. Any block outside the exact range uses original raw words via an
escape table. No rounding/requantization; native MXFP4 expert weights remain
untouched. CPU preparation uses row chunks to bound temporary workspace.

The shader follows the installed MLX0.32.3 native GEMV ordering for supported
single-vector dense products:32lanes,4successive K values per lane,128K stride,
four row accumulators and explicit down-shuffles16/8/4/2/1;4simdgroups forN<4096
and8otherwise. Reject K<=64, K>=16N, non64-aligned K or non4-aligned N because
native dispatch/reduction differs. Real target dimensions satisfy these guards.
Safe math mode matches the installed CompileOptions default. Bias preserves the
native BF16 accumulator rounding before the unit-scale biased expression.
Primary implementation references: installed headers
`mlx/include/mlx/backend/metal/kernels/gemv.h` and upstream MLXv0.32.3
`mlx/backend/metal/matmul.cpp` (single-vector wide path disabled).

Before freezing/running: exhaustive integer CPU roundtrips including all65536
words ordered and shuffled, mixed escape blocks and chunk-boundary indexing.
Then freeze exact source, run native GPU decoder/projection tests (bias, tails,
both native dispatch sizes), and native-weight operator screening. Only after a
reliable local gain run full-model reference/candidate logits, routes, tokens,
BF16 cache arrays/metadata and fallback/call-count checks before model timing.

The first prototype holds compressed sidecars alongside reference weights to
support matched trials. It reduces candidate read traffic, not total prototype
allocation. Report encoded/escape/sidecar bytes, preparation time and absolute
peak overhead, including reference sidecars. Do not use a sidecar-inflated
baseline to claim memory savings or acceptance. Future production replacement
would need representation lifetime/prefill handling that actually frees originals.

Exploration retains and flags system-wide swap activity and finishes within
bounds. It cannot establish a speedup. Any speedup claim requires a fresh strict
matched confirmation and its validity/noise checks. Task-quality/acceptance gates
remain unchanged and disabled/planned; no publication or merge is authorized.
