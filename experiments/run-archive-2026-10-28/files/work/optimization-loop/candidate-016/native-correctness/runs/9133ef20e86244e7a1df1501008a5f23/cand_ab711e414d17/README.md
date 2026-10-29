# Native up/gate packing

The custom reduction shortlist changed rare BF16 projection results. Native
repeatability was exact, but those local differences amplified into full-model
logit and routing changes. The pointwise epilogue matched native on the same
biased projection inputs. This candidate keeps the native matrix and activation
implementations rather than reproducing their reduction arithmetic in Metal.

Up/gate native MXFP4 weights, scales and biases are concatenated along output
rows. One native gather_qmm produces both projections, which are split before
the unchanged native SwiGLU and down projection. This removes one native product
dispatch; its potential saving is a hypothesis, not measured dispatch-gap time.
The audited up/gate region cost about 0.379 ms per sampled expert call. Its
full-model contribution remains unverified.

Packing is a declared lossless layout transformation, with no requantization or
precision change. Original per-projection fields become views of the same packed
buffers. Only that representation is cached, with weak module ownership and
parameter-object signatures; no request data or outputs are cached. Changed
parameters trigger repacking. Native contiguous fields can be restored with
`prepare_native`, preserving the current values. Preparation/restore time and
temporary memory are setup costs and must be reported separately. Before any
paired timing, the controller must restore native layout for baseline lanes and
prepare packed layout for candidate lanes outside the timed interval.

Correctness must include byte-exact layout round trips, changed parameters,
prefill fallback and the saved real boundary cases that exposed the previous
failure. A full-model comparison and fresh calibration remain required before
claiming a model speedup. Acceptance stays disabled.
