# Decode profiling: inconclusive timing, no new candidate

The measured controls are not yet repeatable enough to support a bottleneck-based optimization choice. The existing certified baseline is unchanged. No new candidate, speedup, acceptance, PR or merge is produced by this diagnostic.

## What stopped the clean comparison

The first complete pass collected 43 native requests and 180 operator timing blocks. All timing groups were affected by strict paging/thermal flags, so its region durations and deletion budgets remain exploratory. Every sample is retained.

A strict ten-second quiet gate then failed after 180 seconds, before any request: thermals were nominal and there were no recent swap-outs or swap growth, but small global swap-ins continued. The certified reference already uses a paging-aware policy; the initial diagnostic should have used that documented policy. Neither strict attempt was relabelled valid.

A fresh paging-aware 512-context screen allowed logged global swap-ins, rejected swap-outs/growth and required nominal thermals. Its unchanged A/A pairs differed by 10.46% and 8.45%, failing the existing screen limits. A new intervention then matched the reference cadence: 12-second controller hold, transient display sleep, five-second settling and ten-second policy-quiet preflight before every request, plus 0.5-second thermal observations. No background service or renderer was paused.

The controlled A/A screen still failed:

| Pair | A decode tok/s | B decode tok/s | A/B ratio |
| --- | ---: | ---: | ---: |
| 1 | 22.307 | 21.651 | 1.03029 |
| 2 | 21.479 | 22.036 | 0.97472 |

Both pair deviations exceeded the predeclared 2% limit (about +3.03% and -2.53%). Relative MAD was 1.285%, above the 1% limit; full range was 3.775%, within its 4% limit. All four requests passed their paging-aware environment boundaries and every observed thermal state was nominal. They recorded zero swap-outs, no swap growth, and global swap-in counts of 220/188/200/700 pages. These counters cannot identify which process paged.

The screen stopped the run before deletion or candidate timings. It was not rerun again. This establishes remaining measurement instability; it does not establish its GPU, CPU, allocator or background-process cause.

## Where decode time goes: what is actually measured

In the controlled screen, median total latency was 45.78 ms/token. Median per-request mean intervals were 2.60 ms for Python/model graph construction, 43.09 ms for evaluation/wait, and 0.0056 ms for token readback. Eval/wait includes C++ scheduling, GPU execution and waiting; it is not GPU kernel time or dispatch-gap time. These descriptive values do not override the failed A/A calibration.

The first pass’s stage-by-stage synchronization expanded latency from about 42 to 101 ms/token (2.42×). Its attention, expert and vocabulary-projection durations cannot be treated as the original critical-path shares. No GPU timeline, bandwidth, occupancy or register counters were available; full Xcode/xctrace is absent.

## How much could removal save?

**No reliable headroom ceiling was established.** The following exact-output deletion replays all had invalid timing conditions. They are useful only for choosing what to confirm, not for selecting or accepting a candidate:

| Optimistically removed work | Exploratory median saved ms/token | All three paired savings ms/token |
| --- | ---: | --- |
| Up/gate biases + SwiGLU | 1.66 | 1.86, 0.13, 1.66 |
| Up/gate products + biases + SwiGLU | 8.89 | 9.69, 8.89, 6.92 |
| Down product + bias | 8.12 | 8.12, 4.96, 16.64 |
| Entire expert execution | 17.83 | 20.58, 10.68, 17.83 |

Each replay preserved upstream dependencies and every compared logit exactly. Cached-output deletion is an optimistic intervention on this fixed trajectory, not a deployable optimization or universal bound. Once a stable fraction f is measured, complete removal gives the additive Amdahl estimate 1/(1−f), while a stage speedup r gives 1/((1−f)+f/r). The current invalid fractions must not be used as validated ceilings.

## Why the candidates have not established a gain

Both frozen candidates passed all 96 supplementary full/region checks on real model activations with zero observed error. The complete first pass also checked exact logits in 29 capture/replay requests. Correctness on these inputs is not the missing evidence.

Candidate 1 targets output biases plus an already compiled SwiGLU and retains all three matrix products. The exploratory real-input up/gate region ratio was 0.9915× (native/candidate), with its descriptive interval including one: there is no demonstrated region improvement. The small-target explanation is plausible, but the invalid timing prevents a quantitative causal conclusion.

Candidate 2 fuses up/gate products and activation but retains both weight matrices and the native down product. The exploratory real-input region ratio was about 1.0402×, a small local effect that requires valid confirmation. It does not establish model-level improvement or explain the hardware cause of the prior near-ties. The earlier valid synthetic operator test still has intervals including one.

Simple logical-byte accounting gives about 3.70 GB of weight references per token across experts, attention projections and vocabulary projection. The candidates save only an estimated 2.21/4.42 MB/token of projection-result traffic in that simplified model. These are analytical logical bytes, not measured DRAM traffic; they do not prove a bandwidth bottleneck. Dispatch savings remain unmeasured.

A megakernel therefore has no evidence-backed justification yet. The next measurement work is explaining the remaining A/A variation and obtaining an analyzed GPU timeline; another speculative kernel would not resolve those gaps.

## Evidence and integrity

- [First complete exploratory pass](FIRST-PASS-MEASUREMENTS.md): every native request, operator block, real input, hash and failed condition.
- [Controlled screen](CONTROLLED-SCREEN.md) and [raw controlled state](controlled-512/state.json): failed A/A screen with its fixed limits.
- [Strict quiet-gate failure](quiet-pass/state.json) and [initial paging-aware screen failure](paging-aware-512/state.json).
- [Machine-readable status](experiment-status.json) and [controlled protocol](controlled-512/PLAN.md).

Protected source hashes and prepared controller hashes match. The first complete controller’s raw final checkpoint guard was false because its dictionary variable was overwritten by a numerical expected-output array. An independent file-identity comparison matched the original verified checkpoint identities; the raw result and [audit](controller-repair/postrun-checkpoint-audit.json) remain intact. Later controllers fixed the variable shadowing and their final checkpoint guards passed. The initial pre-execution candidate-import aliasing issue was also repaired and tested before any candidate comparison.

All profiling workers and task-scoped sleep assertions have ended. The display was woken after the controlled screen stopped. Saved power settings, installed model/runtime, protected repository files and the existing certified baseline were not changed.
