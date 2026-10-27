# Joint native up/gate decode candidate

The hypothesis is that one custom Metal dispatch can replace the separate up
and gate MXFP4 projection dispatches plus bias/SwiGLU operations during decode.
It shares each lane's eight activation loads across both projections, handles
eight output rows per 64-thread group, and invokes the unchanged native down
projection afterwards. Both distinct weight matrices are still read in full
for the selected experts; no reduction in their compulsory weight bytes is
claimed. Increased register pressure could make this slower than native MLX.

The kernel reads the existing packed weights and group-32 E8M0 scales directly.
There is no requantization, persistent cache, concatenated weight allocation,
model mutation or routing change. Accumulation uses FP32, with explicit BF16
rounding at projection outputs, bias additions and the SwiGLU arithmetic steps.
Input contiguity is ensured by MLX; any required copy is part of execution.

Only `[1,1,H]` BF16 activations and four selected int32/uint32 expert IDs take the
optimized path. Projection layouts and dimensions are checked. Other token
shapes call the unchanged reference. Thus only the three protected `decode`
checks exercise the optimization; the other eighteen use declared fallbacks.
Small compatible native-MXFP4 fixtures can exercise the same Metal kernel, but
cannot substitute for native-checkpoint correctness and full-model evaluation.

Format, accumulation grouping and sigmoid semantics were checked against the
installed MLX 0.32.3 headers `fp_quantized.h`, `fp4.h`, `fp8.h`, and `unary_ops.h`.
Native layout and GPT-OSS activation behavior were checked against installed
MLX-LM 0.31.3 `switch_layers.py` and `gpt_oss.py`.

All Python and Metal sources in this snapshot require human review before
execution under the current evaluator policy. Candidate metadata is not
approval. No numerical result, speedup, full-model quality or acceptance is
asserted. Tolerances, contract and acceptance gates remain unchanged.
