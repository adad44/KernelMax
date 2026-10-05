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

## Compact publication and retained evidence

[summary.json](gpt-oss-20b-m4-week1/c052ae53-8a6b-47d7-a9a7-d5c5414a43cd/summary.json)
contains derived metrics, dispersion, runtime, protocol and provenance hashes.
[independent-audit.json](gpt-oss-20b-m4-week1/c052ae53-8a6b-47d7-a9a7-d5c5414a43cd/independent-audit.json)
is the original audit result: 147 requests checked, including 15 warmups and
120 official samples. These small files are an index, not standalone raw evidence.

The complete raw samples, original report, source snapshots, receipts and 16
earlier attempts remain in the
[historical evidence tree](https://github.com/adad44/KernelMax/tree/b3223bddb68848af054375263ffbeaf66543dfa9/reports/baselines).
That tree records the measured implementation; subsequent working-tree evaluator
changes are not part of this measurement.

A complete local archive is also preserved outside Git at
`../baseline-results/baseline-publication-full-e86ed7d.tar.gz`.
Its SHA-256 is
`ac67d8faff74216ce8fd41152ca2674169f825f2bd492d684d6d3cf3a1b3e3d3`.
The raw/report/audit hashes are in `summary.json`; use the full historical bundle
for reproduction or sample-level auditing. Failed attempts have not been deleted
or reclassified, and no benchmark was rerun for this packaging-only publication.
