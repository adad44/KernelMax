# Fresh paging-aware diagnostic at context 512

This is a controller-owned diagnostic, not an official baseline, candidate verdict or acceptance run. No optimization candidate is built. Preserve all prior attempts, including the exploratory complete pass and the strict quiet-preflight timeout, without reclassifying them.

The certified reference protocol (`baseline-tools/official-reference-protocol-review.json` and `baseline-results/OFFICIAL-BASELINE-STATUS.md`) already specifies paging-aware measurements: permit and log global swap-ins, reject swap-outs, swap-usage growth, non-nominal thermals, AC loss or power-setting changes. The user explicitly approved that policy historically and has now reconnected AC and agreed to close heavy applications. This fresh diagnostic uses that policy, not a retroactive change to old data. Raw swap counters and deltas remain visible. It does not assert that global swap-ins are cost-free or that the model alone caused them.

Scope: fixed seed-42 512-token input, batch one, greedy 128-token generation, fresh BF16 caches, unchanged native checkpoint/precision. The 2048/4096 cases in the earlier pass remain inconclusive. No comparison to the historical certified baseline is made: this diagnostic has a ten-second policy-quiet preflight and no UI/display/service intervention, unlike the historical controlled cadence.

Before every request and operator block require ten continuous quiet seconds with nominal thermals, AC and no policy-disallowed paging. Timeout after 180 seconds, retain observations and stop. Initial warmup is retained but excluded from statistics.

After one warmup, run two A/A pairs in AB/BA order, with equal preconditions. Require every boundary valid, each pair ratio within 2% of one, relative MAD <=1%, and full relative range <=4%, matching the certified reference's diagnostic screen gates. A failed screen stops; do not rerun statistical failures to obtain a pass. Screen requests never count as baseline or candidate samples.

After the screen, collect three plain-model requests; capture exact native expert intermediates and logits and compare them to a normal-hook control. Preserve selected real inputs for layers 0/12/23 at eight evenly spread decode steps. Run all four dependency-preserving cached-output deletion interventions (bias/SwiGLU, up/gate+activation, down, all experts), each with a retained warmup then three AB/BA pairs. Require exact logits for every replay. These estimate recoverable latency on this trajectory; they are not deployable candidates or universal bounds.

Run a separate serial stage diagnostic over the first 16 decode steps of a full request. Report its inflation against the plain-model baseline; do not treat serial shares as the original GPU critical path.

Use all 24 real cases to compare native experts, both frozen candidates, raw up/gate products, native up/gate+activation, each candidate's extracted up/gate region, down, and bias/SwiGLU alone. Five warmup cycles per lane, then twenty counterbalanced blocks of 24 cases repeated eight times (192 calls/lane/block). Candidate 1's ID cast remains timed. Validate every full and region output before timing. Distinct import packages and snapshot filename guards prevent aliasing; retain extracted region source and hashes.

Keep all timing blocks and all environment flags. Timing interpretations require valid group conditions and the passed A/A screen; no outlier filtering or per-sample retries. Report uncertainty descriptively. Fix the earlier final-identity variable-shadowing bug without changing model math. Freeze source hashes and checkpoint identities; verify after execution. GPU timeline/counter tools remain unavailable, so dispatch gaps, DRAM bandwidth, occupancy and register pressure remain unmeasured.
