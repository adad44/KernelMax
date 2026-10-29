# New candidate executed after complete-decode profiling

`cand_390712b41518` was executed against native GPU fixtures and the pinned
checkpoint's actual dense weights. The combined attention/vocabulary layout
hypothesis is parked: it did not establish a useful overall operator gain.
No full-model speedup or acceptance is claimed.

The complete bounded profiling pass finished all63requests and preserved all
samples. Cached-output deletion budgets were14.720ms/token for attention,
14.674ms for experts,12.205ms for the vocabulary projection and0.296ms for the
small operations group. These optimistic interventions are nonadditive; they
are not a decomposition into independently recoverable kernel time.

The new candidate is a lossless BF16 weight-read layout, a larger memory-execution
change. It preserves all16original weight bits while reducing most blocks from
128bytes to100bytes, with raw escapes for blocks that cannot be encoded exactly.
Applied across dense attention and vocabulary matrices, its conditional maximal
logical-byte saving was532MB/token, a plausible several-millisecond target under
the explicitly hypothetical bandwidth assumptions in `LAYOUT-HYPOTHESIS.md`.
This estimate was not a measured DRAM counter or speedup.

Validation and execution evidence:

- Frozen bundle SHA256:
  `bbe50c8a03c67d5ba3811eb657a8f15c0292238e66fe8a1ddf3a93a8a5813595`.
- CPU encoding fixtures checked136,256BF16words including every16bit pattern and
  chunk/escape handling. Native synthetic GPU fixtures passed26/26checks.
- Actual checkpoint attentionq/k/v/o at layers0/12/23 and `lm_head` passed52/52
  output comparisons bit for bit on varied synthetic BF16 vectors. These are
  operator checks, not captured model activations or full-model correctness.
- The screen retained100lane/warmup samples, each containing13operator timings
  with16actual calls per operator. Each operator had20A/A control pairs and20
  alternating baseline/candidate pairs after the declared warmups.
- Sampled dense native weights occupied1.318GB; encoded sidecars occupied1.036GB.
  The prototype holds both for comparison, so it increases resident allocation.

The vocabulary projection's exploratory paired median saving was2.504ms/call,
and its A/A calibration passed. All12sampled attention projections had negative
median savings;11of13overall A/A calibrations were noisy. An additive24layer
planning extrapolation was-9.925ms/token. That extrapolation is not a model
measurement, but the evidence does not justify testing the current combined
placement in the full model. A separate vocabulary-only placement could be a
future hypothesis; it has not been run or confirmed here.

All samples were retained under schema3 exploration policy; no recorded row in
this pass or screen was swap-affected. Source/checkpoint/frozen-snapshot guards
passed, and worker/sleep-assertion processes exited. GPT-6 Luna on high handled
the prepared script execution, process monitoring and artifact collection.
Historical failed runs and their original identities remain intact. Any future
speedup claim requires fresh strict matched confirmation. Acceptance remains
disabled and held-out task-quality evaluation remains planned.

Evidence: `state.json`, `native-gpu-fixture-001/state.json`,
`checkpoint-operator-001/state.json`, and
`../decode-search-20261027/bookkeeping/exploration-v3-20261027/completion-audit.json`.
