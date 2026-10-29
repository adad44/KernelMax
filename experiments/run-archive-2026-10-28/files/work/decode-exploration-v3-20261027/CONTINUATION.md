# Decode exploration v3 continuation

Updated: 2026-10-27 (profile, candidate screening, and primary disposition complete).

## Goal and guardrails

- Run status: the requested complete-decode profile and new candidate execution sequence completed. The primary parked C1 combined attention/vocabulary placement after its negative/inconclusive operator screen. No full-model run, speedup claim, candidate acceptance, or task-quality claim was made.
- Bookkeeping uses the user-authorized GPT-6 Luna high workflow. Keep original histories and failed data intact.
- Schema v3 contract SHA-256: `cb3fc638080fb65badca7854f753642ad93dec8d33fcf09d83841568a59c89bc`. Exploration policy: max 64 requests / 3600 seconds; planned 63 requests; retain and flag swap-affected samples; acceptance disabled.
- Procedure: smoke passed (passive telemetry, prefill/decode, 4 samples); `prepare.py` froze the session and job before launch. Full offline suite: 358 passed, 8 skipped. Ruff 0.17.3 formatted the four requested policy files; post-format targeted tests: 46 passed. Results: [policy-v3-tests summary](/Users/alandiaz/Downloads/KernelMaxxxing/work/decode-search-20261027/bookkeeping/policy-v3-tests/summary.json).

## Live exploration

- Launchd job `gui/501/com.kernelmax.exploration.v3.20261027`: PID 81514, born Tue Oct 27 19:36:30 2026; terminal, last exit 0. The completed state and final guards were independently checked.
- One-off sleep assertion `gui/501/com.kernelmax.exploration.caffeinate.v3.20261027`, PID 81552, `/usr/bin/caffeinate -i -s -w 81514`; it exited when the profile worker exited. Saved power settings were unchanged.
- Bounds and scope: 2 resident warmups + 1 capture + 4 regions × (5 warmups + 5 alternating matched pairs), 63 planned requests; no AA rerun. Exploration summaries are cached-output deletion budgets only, nonadditive, not confirmation or speedup evidence.
- Current state: `completed_bounded_exploration`, 63/63 request rows, 0 errors. `protected_unchanged=true`, `checkpoint_unchanged=true`; launchd exited 0. Regions summarized: attention, experts, lm_head, small_ops.
- Recorded request environmental flags: none; swap-affected rows: 0.

### Region budgets (hypothesis discovery only)

| Region | Saved ms/token, all pairs | Median | Descriptive interval | Swap-affected pairs |
|---|---:|---:|---:|---|
| attention | 14.832, 18.865, 14.675, 14.169, 14.720 | 14.720 | 14.169, 18.865 | [] |
| experts | 14.448, 14.744, 14.250, 15.073, 14.674 | 14.674 | 14.250, 15.073 | [] |
| lm_head | 13.245, 12.205, 12.037, 12.172, 12.406 | 12.205 | 12.037, 13.245 | [] |
| small_ops | 0.303, 0.164, 0.280, 0.385, 0.296 | 0.296 | 0.164, 0.385 | [] |

## Candidate disposition

- C0 `cand_75ca936df290` remains frozen and unexecuted/parked: light timeline showed roughly 1.15–1.25 ms/token outside sampled command-buffer intervals, which does not support a several-ms pure scheduling hypothesis. Command-buffer intervals include stalls, so this is not an absolute hardware ceiling. Disposition: [async-candidate-disposition.json](/Users/alandiaz/Downloads/KernelMaxxxing/work/decode-exploration-v3-20261027/async-candidate-disposition.json).
- C1 `cand_390712b41518` (combined attention/vocabulary placement) is parked by the primary after the completed screen: exact-layout dense-projection candidate. CPU fixture passed 3 cases / 136,256 BF16 words, including all-word coverage and chunked equivalence; synthetic native fixture also passed as described below. Frozen bundle SHA-256 `bbe50c8a03c67d5ba3811eb657a8f15c0292238e66fe8a1ddf3a93a8a5813595`. Freeze metadata: [layout-frozen.json](/Users/alandiaz/Downloads/KernelMaxxxing/work/decode-exploration-v3-20261027/layout-frozen.json). It passed the bounded real-checkpoint operator screen, but no full-model candidate run was performed.
- Synthetic native GPU fixture: [layout_gpu_fixture.py](/Users/alandiaz/Downloads/KernelMaxxxing/work/decode-exploration-v3-20261027/layout_gpu_fixture.py), formatted with isolated Ruff, AST passes, SHA-256 `6b46050612a9f54a52483c444a90fde2ef42c0859d65e57d7f5371701354dc7b`. It ran after profile PID 81514 exited: 26/26 synthetic decoder/projection cases passed; 0 flags, 0 swap-affected records; `protected_unchanged=true`, `snapshot_unchanged=true`; source hash unchanged; launchd exited 0. This is not real-checkpoint or full-model correctness and contains no timing. Summary: [native-gpu-fixture-001.json](/Users/alandiaz/Downloads/KernelMaxxxing/work/decode-search-20261027/bookkeeping/exploration-v3-20261027/native-gpu-fixture-001.json).
- Real-checkpoint operator screen completed from the prepared script: 13 actual tensor families, 52/52 synthetic-vector checks bit-exact, 100 timing rows retained per family (20 A/A pairs and 20 candidate pairs, plus warmups), no environment flags or swap-affected rows. A/A calibration passed only 2/13 families. Candidate median result improved lm_head by 2.504 ms/call but regressed all 12 sampled attention projections; conditional 24-layer planning estimate is -9.925 ms/token under the explicit representativeness/additivity assumptions. Sidecar residency was 1,036,300,954 B versus sampled native dense weights 1,317,567,360 B. Current screen evidence did not justify a full-model test; the primary parked the combined placement. No model speedup claim. Audit: [completion-audit.json](/Users/alandiaz/Downloads/KernelMaxxxing/work/decode-search-20261027/bookkeeping/exploration-v3-20261027/completion-audit.json). Raw state/logs and retained arrays are in `checkpoint-operator-001/` and adjacent logs.

## Artifacts

- Completed exploration raw state/logs: `/Users/alandiaz/Downloads/KernelMaxxxing/work/decode-exploration-v3-20261027/state.json`, `stdout.log`, `stderr.log`.
- Run setup: `plan.json`, `session.json`, `job.plist`; telemetry smoke: `telemetry-smoke.json`.
- Prior timeline summary: `/Users/alandiaz/Downloads/KernelMaxxxing/work/decode-search-20261027/bookkeeping/timeline-light-summary.json`.
- Procedural C1 logs: `/Users/alandiaz/Downloads/KernelMaxxxing/work/decode-search-20261027/bookkeeping/exploration-v3-20261027/c1-cpu-fixture/`.
- Completion audit: `/Users/alandiaz/Downloads/KernelMaxxxing/work/decode-search-20261027/bookkeeping/exploration-v3-20261027/completion-audit.json`.
- Final verification of report/disposition, raw counts and hashes, policy-doc alignment, guards, and terminal handles: `/Users/alandiaz/Downloads/KernelMaxxxing/work/decode-search-20261027/bookkeeping/exploration-v3-20261027/final-audit.json`.
- Primary report/disposition: `/Users/alandiaz/Downloads/KernelMaxxxing/work/decode-exploration-v3-20261027/RESULT.md`, `disposition.json`. No required run remains queued; a future vocabulary-only placement is a separate unexecuted hypothesis.
