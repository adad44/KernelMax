# Eight-lane up/gate reduction

Hypothesis: changing the reduction width/output-row mapping can improve the
measured up/gate region, whereas the previous fused kernel kept a four-row,
32-lane mapping comparable to the generic native QMV. Each eight-lane group now
computes one output row, streams quantization groups of 32 elements, and jointly
accumulates the original up/gate products. Register use, occupancy and hardware
bandwidth are hypotheses, not measured causes. Native down projection remains.

Controller evidence: work/optimization-loop/probe-004/audited-result.json outside
this checkout. Layer-0 up/gate deletion saved 0.37947 ms/call, descriptive 95%
interval [0.37236, 0.38048], about 46.7% of the isolated 0.81144 ms expert call.
Its A/A gate, all 20 pairs, environment and integrity checks passed. Other regions
had thermal flags and remain ineligible. The original probe summary had a
normalization bug; this hypothesis uses independently audited raw call counts.

A hypothetical 20% reduction in that region saves about 0.0759 ms per sampled
expert call, roughly 10% expert speedup. If all 24 layers had that same additive
saving, the arithmetic would be about 1.82 ms/token. That extrapolation is a
planning scenario, not measured full-model headroom: layer representativeness,
weight/cache behavior and the full-model critical path are unverified. A valid
full-model comparison is still required.

Preserve native MXFP4 values and BF16 intermediate rounding, routing IDs and
output shape. No conversion, weight copy, requantization or persistent tensor
cache. The wrapper validates native layouts and falls back to reference for
prefill or other token counts. This is an untested hypothesis until evaluator
results are produced. Acceptance remains disabled.
