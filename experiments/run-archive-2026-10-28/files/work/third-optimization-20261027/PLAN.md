# Candidate 3: complete compiled expert graph

The user asked for another candidate run and explicitly deferred committing run
history. Do not commit or push this experiment. Standing authorization permits
implementing, freezing, correctness-testing and timing candidates without another
approval prompt. Acceptance stays disabled; no merge or publication is allowed.

Candidate `cand_206746fae66e` compiles the entire native one-token expert graph.
The previous profiling pass observed Python graph-build time, but failed its A/A
screen. This is a low-complexity hypothesis about that observed work, not proof
of a measured stable bottleneck and not a megakernel investment. All three
native MXFP4 products, weights, routing and precisions remain unchanged.

Freeze and hash the candidate before importing it. First run small native-MXFP4
fixtures for int32/uint32 IDs, two decode magnitudes and prefill shapes 16/17,
plus an explicit parameter-change test to detect stale compiled weight captures.
If those pass, use the protected native correctness runner (all seven cases on
layers 0/12/23; 21 comparisons). Preserve every error and failed attempt. Do not
modify the contract, evaluator, tolerance or protected model/runtime.

Only after native correctness passes, run a candidate-only expert-operator
diagnostic using the real activations/routes retained by the preceding full-model
capture. Verify their original hashes and cast the exact stored BF16-representable
values back to BF16. Use eight cases per layer (0/12/23). Check both ID dtypes
against the unchanged reference, retaining outputs and all comparisons.

Timing: five warmup cycles per lane, twenty alternating AB/BA pairs per layer,
eight real cases repeated 32 times = 256 synchronized expert calls per lane.
Return and assert the actual call count. Synchronize sample boundaries. Require
AC, nominal thermals, no swap growth/outs; permit and retain global swap-ins under
the certified reference's paging-aware policy. Retain every sample, including
environment-invalid samples; do not select outliers or rerun statistical failures.
Use all pairs for descriptive median and bootstrap intervals and show AB/BA order
strata. This operator protocol is not the full-model 128-token benchmark.

Record separate untimed instrumented graph-build/eval intervals after warmup to
test whether compilation reduces the hypothesized frontend work. Do not add
per-call instrumentation to the paired wall-time samples or claim GPU dispatch
timing from CPU intervals. Record all source/snapshot/checkpoint identity guards.

Run the native correctness and timing pipeline as one temporary, bounded user
launchd job with Interactive process type, matching the successful reference's
execution class instead of the prior PTY background execution. No other service
is paused or renderer suspended. Remove the temporary job after completion. This
changes execution context for a fresh run; no earlier failed timing is relabelled.

No full-model speedup or acceptance follows from these operator timings. A
correct candidate with no stable operator improvement is parked, with all data
retained. Any full-model comparison requires a separate valid calibration.
