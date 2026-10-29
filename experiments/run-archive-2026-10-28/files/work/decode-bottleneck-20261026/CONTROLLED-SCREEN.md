# Decode bottleneck measurements

Status: **failed_diagnostic**. This is a diagnostic, not an acceptance benchmark or candidate verdict.

All 5 requests and 0 operator timing blocks are retained. No outliers were removed.

Environment policy: paging-aware; permit and retain swap-ins, reject swap-outs/growth and non-nominal thermals; 10-second quiet preflight. Flags, including warmups: **swapouts increased during request**. Of 4 non-warmup requests, 0 have boundary flags; 0 of 0 operator groups have flags. Interpret each group's validity below. Warmups stay in the record and are excluded from the planned statistics.

## Unchanged full-model decode

One warmup and three measured requests per context; batch one, native weights, BF16 activations/cache, greedy 128-token generation and fresh caches. Model loading is excluded.

| Context | Decode tok/s | ms/token | Host graph build ms/token | Eval/wait ms/token | Token readback ms/token | Valid |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |

A/A stability screen (excluded from baseline metrics): {"environment_valid": true, "gates": {"max_pair_deviation": 0.02, "max_relative_mad": 0.01, "max_relative_range": 0.04}, "pair_ratios": [1.0302925542823027, 0.9747199337081219], "passed": false, "relative_mad": 0.012853614515444964, "relative_range": 0.03774812968690572}

Measured-request swap activity: {"swap_growth_mib_max_request": 0.0, "swapin_pages_max_request": 700, "swapin_pages_total": 1308, "swapout_pages_total": 0}. These are system-wide counters; they do not identify which process paged or establish that swap-ins had no timing cost.

Eval/wait includes scheduling, GPU execution and waiting. It is not a measurement of GPU idle gaps. The graph-build interval is measured wall time; these host boundaries are not hardware timestamps.

## Optimistic stage-removal replays

Context 512, three alternating paired requests per stage. Required upstream dependencies remain; cached outputs replace the removed computation. Every replay logit is checked against the capture. This is an optimistic intervention on one trajectory, not a deployable candidate or a universal upper bound. The conditional 2× estimate assumes additive stage cost.

| Removed stage | Median saved ms/token | All paired savings ms/token | Median replay speedup | Conditional total speedup for 2× stage | Valid |
| --- | ---: | --- | ---: | ---: | --- |

## Candidate diagnosis on real activations

Layers 0/12/23, eight evenly spread real decode steps per layer, 24 cases × eight cycles = 192 synchronized expert calls per block. Twenty alternating blocks. Candidate 1 includes its routing-ID conversion. These operator times include per-call host/synchronization overhead and cannot be multiplied into a full-model speedup. Isolated substages are non-additive: timing bias/activation alone pays synchronization costs that do not describe its incremental cost inside the larger graph.

| Operator or region | Median ms/call |
| --- | ---: |

| Paired comparison: reference / replacement | Median ratio | Descriptive bootstrap 95% interval | Median saved ms/call |
| --- | ---: | --- | ---: |

Block-order strata (even forward / odd reverse) are retained in audit.json. Inspect them before interpreting small pooled operator gains: a single native call first/last in a block can expose warmup/order effects even with counterbalancing. No order stratum is dropped or used to select a favourable result.

Candidate region functions are diagnostic extractions of each frozen implementation through its `activated` result, excluding only the unchanged final down call. Their source and hashes are retained. Regions and full outputs are checked before timing. Confidence intervals are descriptive across timing blocks, not an acceptance test or a task-quality result.

## Traffic and dispatch limits

Logical weight references are about 3.70 GB per decode token: experts 1.269 GB, attention projections 1.274 GB, vocabulary projection 1.158 GB. Both candidates retain these weight values and logical reads. A simple projection-result traffic model estimates only 2.21 MB/token saved by candidate 1 and 4.42 MB by candidate 2, excluding bias-gather buffers. These are analytical byte counts, not measured DRAM traffic. Caches, re-reads and physical transactions remain unmeasured.

No GPU timeline or bandwidth/occupancy/register counters were available: full Xcode/xctrace is absent. CPU eval/wait cannot establish dispatch bubbles. A megakernel remains a hypothesis; these measurements alone cannot prove that dispatch gaps justify it.

## Integrity and evidence

Controller completed: False; protected sources unchanged: True; checkpoint identity unchanged: True; exact-logit checks: None; real operator checks: 0/0.

Raw evidence: [state.json](controlled-512/state.json), [summary.json](controlled-512/summary.json), [audit.json](controlled-512/audit.json), [serial stages](controlled-512/serial-stages.json), [measurement plan](controlled-512/PLAN.md). The interrupted preparation and first pass remain intact. No candidate was created, promoted or accepted; acceptance remains disabled.
