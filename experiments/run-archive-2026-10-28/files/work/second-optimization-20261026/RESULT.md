# Joint up/gate MXFP4 candidate result

**No consistent operator improvement established across sampled layers; no promotion.**

Candidate: `cand_90f0c66e0d03`.
Frozen snapshot: `93f523c916125738e860b796e6440170d192dcdeb207e327edaa26d826f14272`.

One custom Metal dispatch computes both original up/gate projections, their biases and SwiGLU during one-token decode. Native packed weights, scales and routing IDs remain unchanged; the down projection and prefill fallback use the original implementation.

- Synthetic fixture checks: 12 passed. An earlier incompatible N=72 fixture failed during native quantization setup; it was repaired to N=96, and the initial failed setup remains retained. Candidate bytes were unchanged.
- Protected native suite: 21/21 passed. Three decode comparisons exercise optimized code; eighteen other comparisons use declared reference fallbacks.
- Native execution errors: [].
- Additional decode comparisons passed: 96.
- Timing samples retained: 120.
- The sample length was predeclared as 256 calls per lane (eight cycles of the same 32 routing inputs), equally for baseline and candidate, before any candidate #2 timing was observed.
- Environment issues: [].
- Protected-source and checkpoint-identity checks: True, True.

| Layer | Median paired baseline/candidate operator-time ratio | Descriptive 95% interval | All samples environment-valid |
| --- | ---: | --- | --- |
| 0 | 1.002 | [0.986, 1.033] | True |
| 12 | 1.001 | [0.965, 1.020] | True |
| 23 | 1.024 | [0.997, 1.046] | True |

Ratios above 1 favor the candidate for this operator diagnostic. All samples, including invalid observations, remain retained; no outliers were removed. Raw times cannot be directly compared to candidate #1 because the lane sample length differs.

These are operator-level observations, not full-model tokens/sec or task-quality results. No acceptance gate, tolerance, checkpoint, workload, protected source or historical baseline was changed. Formal evaluator verdict remains null; acceptance remains disabled.

Evidence:

- [Native evaluator report](/Users/alandiaz/Downloads/KernelMaxxxing/work/second-optimization-20261026/native-correctness/artifacts/execution-ka4ns_ag/report.json).
- [Structured result](/Users/alandiaz/Downloads/KernelMaxxxing/work/second-optimization-20261026/result.json).
- [Synthetic fixture result](/Users/alandiaz/Downloads/KernelMaxxxing/work/second-optimization-20261026/synthetic-fixture-repair/tiny-native-checks.json).
- [Initial fixture setup failure](/Users/alandiaz/Downloads/KernelMaxxxing/work/second-optimization-20261026/tiny-native-checks.json).
- [Candidate Python](/Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop/candidates/cand_90f0c66e0d03/implementation.py).
- [Metal kernel](/Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop/candidates/cand_90f0c66e0d03/fused_up_gate.metal).

- [Raw corrected operator diagnostic](/Users/alandiaz/Downloads/KernelMaxxxing/work/second-optimization-20261026/timing-controller-repair/operator-diagnostic.json).
- [Initial timing protocol mismatch and repair](/Users/alandiaz/Downloads/KernelMaxxxing/work/second-optimization-20261026/timing-protocol-audit.json).
- [Initial raw timing report, retained but ineligible under the declared 256-call plan](/Users/alandiaz/Downloads/KernelMaxxxing/work/second-optimization-20261026/operator-diagnostic.json).

The first timing attempt labeled samples as 256 calls while its loop executed 32. The mismatch was found before a performance conclusion was issued. Both the incorrect controller and its complete raw report remain unchanged. A fresh corrected run was executed after independently verifying 256 calls per lane with counting stubs. Only that corrected run supports the table above.

Local source commit: `fc1dc28a4f18a12340668af2cee5cea9a0d6cbe7` on `candidate/first-optimization-loop`.
