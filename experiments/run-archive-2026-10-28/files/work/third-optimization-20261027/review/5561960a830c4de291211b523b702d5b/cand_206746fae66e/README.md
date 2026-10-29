# Complete native expert decode graph compilation

The profiling pass observed time in Python/model graph construction, with a
condition-dependent size of roughly 0.75–2.6 ms/token. Its failed A/A screens do
not establish a stable fraction or a speedup ceiling. This is a bounded new
hypothesis targeting that observed interval, not a claim that dispatch gaps or
intermediate traffic justify a megakernel.

Candidate 1 compiled only the up/gate output-bias and activation region. This
candidate compiles the full expert decode graph, including the native up, gate
and down products and all output biases. GPT-OSS's original compiled SwiGLU is
called inside that larger pure function. The native MXFP4 math and routing IDs
are retained; there is no custom Metal matrix multiply.

All nine native parameter arrays are explicit inputs on every invocation.
Ordinary MLX shape/dtype compilation caches may be reused across layers, but no
closure captures a layer's parameter values. There is no user-managed tensor
cache, packing buffer, ID cast, or precision conversion. Compilation uses fixed
shape caching; native production layers share the same parameter dimensions.

The optimized path is batch-one, one-token, four selected experts, BF16
activations, int32 or uint32 IDs and native biased MXFP4 projections. Every other
shape uses the original expert module, including prefill. Both input-ID dtypes,
changed explicit parameter values, real decode activations and native protected
cases must be checked before timing.

This submission has no verdict or performance claim. Operator diagnostics are
not full-model performance or the planned held-out task-quality gate.
