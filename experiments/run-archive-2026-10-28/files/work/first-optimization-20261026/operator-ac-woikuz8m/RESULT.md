# First candidate — AC timing repeat

**Correctness passed. No reliable performance improvement was established.**

The user connected AC power, and the same previously approved snapshot and unchanged diagnostic plan were repeated once in a fresh artifact directory. The earlier battery-powered run remains untouched.

- Candidate: `cand_b7f7dafae7bb`.
- Additional native decode correctness: **96/96 passed**. The earlier protected native result remains 21/21 (three optimized decode checks and eighteen declared reference fallbacks).
- Timing: **120 samples retained**, twenty alternating baseline/candidate pairs at each of layers 0, 12, and 23.
- All measurements had AC power. 1 sample(s) recorded swap-in activity; the strict protocol therefore marks the overall repeat inconclusive.
- Protected sources, frozen candidate and checkpoint identities passed the final audit.

| Layer | Median paired baseline/candidate operator-time ratio | Descriptive 95% interval | Environment valid for all samples |
| --- | ---: | --- | --- |
| 0 | 0.981 | [0.892, 1.124] | True |
| 12 | 1.015 | [0.920, 1.048] | False |
| 23 | 1.005 | [0.927, 1.113] | True |

Ratios above 1 favor the candidate. Every descriptive interval includes 1, including the two layers without environment failures. These operator observations do not establish a speedup or a regression. No samples were excluded, and no full-model candidate benchmark or task-quality evaluation was run.

Research disposition: **park this candidate without promoting it**. Profiling to choose a stronger hypothesis is more useful than repeatedly rerunning this candidate until favorable samples appear. Formal evaluator verdict remains null, and acceptance remains disabled.

Evidence:

- [Raw operator diagnostic](/Users/alandiaz/Downloads/KernelMaxxxing/work/first-optimization-20261026/operator-ac-woikuz8m/operator-diagnostic.json).
- [Structured repeat result](/Users/alandiaz/Downloads/KernelMaxxxing/work/first-optimization-20261026/operator-ac-woikuz8m/result.json).
- [Repeat provenance and AC preflight](/Users/alandiaz/Downloads/KernelMaxxxing/work/first-optimization-20261026/operator-ac-woikuz8m/repeat-provenance.json).
- [Prior protected correctness report](/Users/alandiaz/Downloads/KernelMaxxxing/work/first-optimization-20261026/native-correctness/artifacts/execution-eoik4s4_/report.json).
