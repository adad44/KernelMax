# Official reference baseline

Reference-only, synchronous batch-one MLX/Metal inference on pinned native checkpoint. Not a stock CLI speed claim or candidate acceptance. Paging-aware profile: system swap-ins are allowed and logged; swap-outs or swap-usage growth invalidate the complete attempt.

| Prompt tokens | Decode tok/s median | TTFT ms median | Prefill tok/s median | Peak GiB median |
|---:|---:|---:|---:|---:|
| 512 | 24.18 | 1512.8 | 338.50 | 13.23 |
| 2048 | 23.75 | 5412.2 | 378.42 | 13.50 |
| 4096 | 23.26 | 11423.8 | 358.56 | 13.84 |

Untimed resident-state priming, then five contract warmups; twenty primary A measurements and twenty identical-reference B calibration measurements per size; alternating AB/BA; seed 42; 128 outputs.

Declared swap policy: `paging-aware`. Every request's paging counters remain in the raw evidence; aggregate paging counts are in `baseline.json`. Future comparisons must use the same declared profile.

Raw inputs, outputs, correctness, versions, source/binary/checkpoint hashes, telemetry and all timing intervals are in `raw.json`. Detailed median/p90/MAD, confidence intervals, drift, scope and limitations are in `baseline.json`.

The alternate dense-BF16 reconstruction failed the large-activation test at layer 12. That failed diagnostic is preserved in `dense_reference_diagnostic.json`; it is not a passing numerical accuracy claim. The contract reference is unchanged native MLX-LM. Operator checks verify native dispatch invariance, not independent matmul accuracy.

Candidate acceptance is still disabled. This reference is not an ACCEPTED optimization or proof of zero background GPU activity.

The three approved MemoryOS services remained paused across attempts and through report freeze under the preserved external lease. The supervisor did not restore those externally paused jobs. Restoration after freeze is recorded separately.

Every one of the 120 measured requests had its own untimed settling/quiet window (135 official windows including warmups). This is controlled per-request latency/throughput, not back-to-back serving throughput. Keep this cadence in future comparisons.
