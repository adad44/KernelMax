# Official native M4 reference baseline

Run `c052ae53-8a6b-47d7-a9a7-d5c5414a43cd` is the official reference under the
approved paging-aware, per-request-settling profile. It is a reference measurement,
not a candidate acceptance result; **candidate acceptance remains disabled**.

| Prompt tokens | Median prefill (tok/s) | Median TTFT (ms) | Median decode (tok/s) |
| --- | ---: | ---: | ---: |
| 512 | 338.50 | 1,512.82 | 24.18 |
| 2048 | 378.42 | 5,412.21 | 23.75 |
| 4096 | 358.56 | 11,423.83 | 23.26 |

## What was measured

- Apple M4, 24 GiB unified memory; native OpenAI `gpt-oss-20b` weights.
- MLX 0.32.3 / MLX-LM 0.31.3; native MXFP4 experts, BF16 activations/cache.
- Batch 1, greedy decoding, seed 42, 128 generated tokens and fresh request caches.
- Five warmups and 20 alternating A/A pairs per prompt size.
- Twelve separate stability-screen requests excluded from official measurements.
- Model loaded once; per-request settling; sampled thermal state nominal.
- Paging-aware policy allowed and logged four swap-in pages; no measured swap-outs
  or swap-usage growth. This is not a zero-paging baseline.
- All five stock-generation comparisons and 21 native dispatch checks passed.
  Dispatch invariance is not an independent matrix-multiplication accuracy proof.
- The dense-BF16 layer-12 diagnostic remains unresolved; tolerances were unchanged.
- All three predeclared A/A noise gates passed; all collected samples were retained.
