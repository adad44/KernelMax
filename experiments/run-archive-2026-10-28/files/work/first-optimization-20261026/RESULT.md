# First optimization experiment result

**Candidate correctness passed; performance is inconclusive.**

Candidate `cand_b7f7dafae7bb` compiles native up/gate output-bias additions together with GPT-OSS SwiGLU during one-token decode. Native MXFP4 weights, matmuls, routing decisions, BF16 activations and the down-projection remain unchanged. Prefill uses the unchanged reference fallback.

- Local branch: `candidate/first-optimization-loop`, commit `9c110a2`.
- Human-approved snapshot: `04442d2a56c48c328cf812a86bbe0edc45f8736fb3765f223274ec8fd6212949`.
- Native run: `70a82e9f-5b06-4f9b-a3d0-a255d06b6ede`.
- Protected suite: **21/21 passed**. Three decode checks exercised the optimization; eighteen checked declared reference fallbacks. Decode absolute error was zero at layers 0, 12 and 23.
- Additional native decode comparisons: **96/96 passed**, using varied expert IDs and retained inputs/outputs.
- Exploratory operator timing: **120 samples retained**, twenty alternating baseline/candidate pairs at each of three layers.
- Protected sources and checkpoint file identities remained unchanged. Every retained diagnostic array hash was audited.

The Mac was on battery throughout timing. Every sample records `AC power unavailable`; all three descriptive ratio intervals include 1. Timing therefore does not establish a speedup or a regression. No candidate was promoted, no full-model candidate speedup or task-quality result was established, and acceptance remains disabled.

| Layer | Median paired baseline/candidate operator-time ratio | Descriptive 95% interval |
| --- | ---: | --- |
| 0 | 0.872 | [0.728, 1.106] |
| 12 | 1.027 | [0.819, 1.158] |
| 23 | 1.014 | [0.977, 1.152] |

Ratios above 1 favor the candidate for this operator diagnostic only. These are noisy battery-powered observations, not validated full-model speedups.

Next: connect AC and repeat the paired operator diagnostic into a fresh artifact directory, preserving this run. If a repeat shows a reliable benefit, connect the same reviewed candidate to a full-model comparison. The native model uses uint32 top-k IDs while this evaluator interface uses int32; a future integration must account for a lossless conversion and include its cost.

Evidence:

- [Native evaluator report](/Users/alandiaz/Downloads/KernelMaxxxing/work/first-optimization-20261026/native-correctness/artifacts/execution-eoik4s4_/report.json) — SHA256 `003663710bc9e4b9f6b387ddc53b8c3787e8d4a8bc81bbd3d903aaa8b6e2584a`.
- [Operator diagnostic](/Users/alandiaz/Downloads/KernelMaxxxing/work/first-optimization-20261026/operator-diagnostic.json).
- [Experiment plan and scope](/Users/alandiaz/Downloads/KernelMaxxxing/work/first-optimization-20261026/EXPERIMENT.md).
- [Candidate source](/Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop/candidates/cand_b7f7dafae7bb/implementation.py).
- [Structured result](/Users/alandiaz/Downloads/KernelMaxxxing/work/first-optimization-20261026/result.json).
