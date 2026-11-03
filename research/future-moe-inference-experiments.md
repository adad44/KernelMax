# Research plan: future MoE inference optimization experiments

## Why the first attempts fell short

The archived results point mainly to no verified performance gain, not correctness failures:

- **Fused up/gate bias + SwiGLU:** Correctness passed, but timing was done on battery and all three layer intervals included no change. The run established neither a speedup nor a regression.
- **Fused up/gate + SwiGLU in one Metal dispatch:** A call-count mismatch was caught and repaired before the corrected comparison. Correctness passed, but the corrected timing ratios were close to 1 and all intervals included no change, so there was no consistent gain.
- **Compile the full expert graph:** Correctness passed, but paired timing showed no consistent improvement. Graph construction remained around 9–10 μs per call, so the overhead this experiment targeted was too small to yield a measurable operator win.
- **Combined lossless BF16 layout for attention + vocabulary:** The vocabulary projection improved locally, but all 12 sampled attention projections regressed and 11 of 13 A/A calibrations were noisy. The combined placement was parked; its negative overall estimate was an extrapolation, not a full-model measurement.

The four results do not prove that the optimizations were incorrect. They show that local correctness, operator timing, and end-to-end inference gain are separate gates. None of the four established a repeatable end-to-end speedup. Keep their original records and avoid comparing raw times across different protocols.

## Proposed next experiments

Predictions are hypotheses for prioritizing experiments using a single change per candidate. Preserve the normal correctness tolerance, frozen snapshot, evaluator provenance, and final promotion gates.

| ID | Experiment | Measurable prediction | Decision rule |
|---|---|---|---|
| **R0 — A/A calibration** | Run the same baseline in both lanes with the exact planned timing protocol. | Establish a noise floor small enough to distinguish a 3% paired improvement. | If A/A spread is comparable to 3%, fix environment/protocol or increase pairs before testing candidates. |
| **R1 — Profile the decode path** | Capture one representative decode and attribute time across routing/dispatch, expert products, attention, and vocabulary projection. | Identify one operation responsible for the largest actionable share of critical-path time; do not assume a universal bottleneck. | Use traces/counters to choose one target. Profiling data is diagnostic, not the final benchmark. |
| **R2 — Vocabulary-only weight layout** | Test the existing lossless layout on the vocabulary projection only, if R1 confirms it is material. | Check whether the observed ~2.5 ms/operator saving repeats without the measured attention regressions. | Require bit-exact or normal-contract correctness, no paging regression, and a full-decode improvement before promotion. Do not use the prior −9.925 ms/token extrapolation as a result. |
| **R3 — Stride/copy path** | For one hot operator, compare MLX’s row-contiguous input path with a stride-aware custom kernel. | If a hidden copy is material, target ≥3% lower operator latency and lower temporary allocation; otherwise retain the simpler path. | Check contiguous and strided inputs; profile copy cost and confirm full decode does not regress. |
| **R4 — Routing/dispatch fusion** | Fuse one measured small routing/dispatch-adjacent operation with its consumer, preserving top-k IDs and semantics. | If launch/intermediate overhead is material, target ≥3% operator improvement and ≥1% end-to-end decode improvement. | Keep only if paired operator and end-to-end results both repeat and correctness passes. |
| **R5 — Threadgroup sweep** | Hold the kernel algorithm fixed and compare 64, 128, 256, and 512 threads per group where supported. | One configuration may improve the target kernel by ≥3%; no universal best size is assumed. | Use paired measurements plus occupancy and limiter counters; occupancy alone is not a success metric. |

R0 should precede every timing study. Run R1 before choosing between R2–R5. Keep cold-start compilation/load time separate from steady-state decode. If system-wide paging or thermal/power limits violate the declared protocol, preserve the run as an environment-inconclusive outcome rather than calling it a candidate regression.

## Research modes and guardrail boundaries

These modes improve the search process; they do not make kernels faster by themselves.

1. **Profiler-only mode — diagnostic value.** Capture traces/counters and stop before the full evaluator. Label the outcome `diagnostic_only`; never include it in accepted candidates. This can save evaluation time and steer work toward bottlenecks. Keep the final benchmark uninstrumented because profiling changes execution conditions.
2. **Operator-scope screening — strongest immediate value.** Measure a single operation or shape before a full-model run. A local win is a hypothesis only; promotion requires end-to-end decode confirmation because copies, dispatch, cache effects, or other layers can erase the gain.
3. **Exploratory numerical diagnostics — useful for approximate kernels.** Record max absolute error, relative error, RMS error, NaN/Inf counts, and edge-case failures to understand a lower-precision or relaxed-math idea. A wider diagnostic envelope must not satisfy correctness or promotion; keep the existing numerical tolerance for approval.
4. **Environment-failure classification — protects evidence quality.** Record swap, thermal, power, and process interruptions as inconclusive environment stops. Preserve all rows and provenance. Do not call them candidate regressions or retry them automatically outside the shared retry budget.

Keep candidate snapshots, report/run identity checks, and correctness promotion criteria strict.

## Measurement protocol for noisy results

1. **Time completed work.** MLX is lazy: evaluate the output with `mx.eval(...)` and synchronize the measured stream before stopping the timer; synchronize before starting to drain earlier work. Otherwise a benchmark may time graph construction or leave GPU work queued. [MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html) · [MLX stream synchronization](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html)
2. **Separate cold and warm behavior.** Warm baseline and candidate equally. Build/JIT custom kernels outside steady-state timing, while recording compile/load latency separately. MLX notes custom Metal kernel creation may create a library and incur JIT compilation. [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
3. **Use the same computer and track its state.** The same Mac controls for hardware differences, but temperature, power state, background activity, and memory pressure can still vary between runs. Keep the setup consistent, let the machine stabilize, and record those conditions. Apple provides GPU performance-state controls for profiling; use them to diagnose bottlenecks, not as a substitute for real application timing. [Apple GPU performance profiling](https://developer.apple.com/documentation/xcode/optimizing-gpu-performance) · [Apple consistent performance state](https://developer.apple.com/videos/play/tech-talks/10001/)
4. **Pair and balance order.** Use identical prompts, token counts, expert IDs, cache state, and memory condition. Alternate or randomize baseline/candidate order (AB/BA) to avoid one lane consistently benefiting from thermal drift or cache warmth.
5. **Calibrate noise with A/A.** Run baseline against itself with the same protocol. If the A/A spread is near the minimum useful gain, the benchmark cannot resolve that gain; fix the setup or gather more pairs before evaluating a candidate.
6. **Audit the controller’s work.** Verify call counts, trial IDs, workload hashes, and that timing boundaries include the same work. The second attempt’s 256-vs-32 mismatch shows why protocol assertions matter.
7. **Track memory pressure.** Record memory/swap deltas per run. If a no-swap condition is required and paging occurs, mark the run inconclusive and preserve it; separately test a constrained-memory scenario if that matches deployment.
8. **Report paired distributions.** Keep every sample, report paired median change and interval, and predeclare the minimum useful effect and stopping rule. Do not delete outliers or rerun only until a favorable result appears.
9. **Profile separately.** Use Xcode/Instruments timelines and counters to decide what to optimize. Use the regular uninstrumented harness for the performance claim. [Apple GPU counter analysis](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-counter-statistics)

Start with the existing 20-pair protocol plus an A/A calibration. Increase the pair count only when the interval is too wide to resolve the predeclared target. For a research target of ≥3%, aim for an A/A interval substantially narrower than 3%; set the exact sample count from observed variance rather than assuming 20 pairs always suffice.
