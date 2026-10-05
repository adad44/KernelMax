# Official reference baseline

The current historical reference is
[gpt-oss-20b on the 24 GiB M4, seed 42](gpt-oss-20b-m4-week1/c052ae53-8a6b-47d7-a9a7-d5c5414a43cd/README.md),
run `c052ae53-8a6b-47d7-a9a7-d5c5414a43cd`, collected October 4, 2026.

| Prompt tokens | Median decode tokens/s | Median TTFT ms |
|---:|---:|---:|
| 512 | 24.1849 | 1512.8156 |
| 2048 | 23.7469 | 5412.2089 |
| 4096 | 23.2578 | 11423.8263 |

Native pinned OpenAI MXFP4 expert weights, BF16 activations/cache, unchanged
MLX-LM, synchronous batch one, fixed token IDs, and 128 outputs. All fifteen
warmups and 120 measurements are retained: twenty A and twenty identical-reference
B requests per size in alternating AB/BA order. Official statistics use only
the predeclared twenty A requests; twelve prior screen requests stay excluded.
The full independent audit checked 147 requests. All three median-ratio noise
confidence intervals passed the predeclared gates.

This is the approved **paging-aware, per-request settled** profile, not zero
paging, stock CLI speed, back-to-back serving throughput, quality evaluation,
or a candidate speedup. Four swap-in pages were logged in one measured request;
no measured swap-outs, swap growth or non-nominal sampled thermal state occurred.
Keep the same profile/cadence for fresh comparisons, not simply this absolute
tokens/s number as an acceptance threshold.

Outliers and roughly 1–1.3% first-five/last-five drift remain in the evidence;
passing the robust median-bootstrap gates does not mean every request was stable.
The failed alternate dense-BF16 layer-12 check is preserved and unresolved.
Native operator checks establish dispatch invariance, not independent matmul
accuracy. **Candidate acceptance remains disabled.**

See the report's [publication provenance](gpt-oss-20b-m4-week1/c052ae53-8a6b-47d7-a9a7-d5c5414a43cd/PUBLICATION.md)
and [`publication.json`](gpt-oss-20b-m4-week1/c052ae53-8a6b-47d7-a9a7-d5c5414a43cd/publication.json)
for the archived failed/interrupted attempts, exact hashes and source distinctions.
The [run/audit instructions](../../scripts/BASELINE.md) describe the portable
infrastructure; packaging it did not generate another benchmark.

Audit from any Python 3 environment, with no model or MLX installation:

```sh
python3 -B scripts/audit_baseline.py \
  --attempt reports/baselines/gpt-oss-20b-m4-week1/c052ae53-8a6b-47d7-a9a7-d5c5414a43cd/raw.json \
  --report reports/baselines/gpt-oss-20b-m4-week1/c052ae53-8a6b-47d7-a9a7-d5c5414a43cd/baseline.json
```
