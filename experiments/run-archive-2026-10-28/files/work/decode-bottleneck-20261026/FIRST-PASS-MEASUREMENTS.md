# Decode bottleneck measurements

Status: **completed_diagnostic**. This is a diagnostic, not an acceptance benchmark or candidate verdict.

All 43 requests and 180 operator timing blocks are retained. No outliers were removed.

Environment policy: strict. Flags, including warmups: **native thermal state is not nominal, swap grew during the request, swapins increased during request, swapouts increased during request**. Of 36 non-warmup requests, 36 have boundary flags; 20 of 20 operator groups have flags. Interpret each group's validity below. Warmups stay in the record and are excluded from the planned statistics.

## Unchanged full-model decode

One warmup and three measured requests per context; batch one, native weights, BF16 activations/cache, greedy 128-token generation and fresh caches. Model loading is excluded.

| Context | Decode tok/s | ms/token | Host graph build ms/token | Eval/wait ms/token | Token readback ms/token | Valid |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2048 | 23.494 | 42.563 | 0.735 | 41.823 | 0.0027 | False |
| 4096 | 21.858 | 45.749 | 1.899 | 43.897 | 0.0046 | False |
| 512 | 23.852 | 41.926 | 0.745 | 41.176 | 0.0024 | False |

Measured-request swap activity: {"swap_growth_mib_max_request": 228.3099999999995, "swapin_pages_max_request": 19608, "swapin_pages_total": 93493, "swapout_pages_total": 72704}. These are system-wide counters; they do not identify which process paged or establish that swap-ins had no timing cost.

Eval/wait includes scheduling, GPU execution and waiting. It is not a measurement of GPU idle gaps. The graph-build interval is measured wall time; these host boundaries are not hardware timestamps.

## Optimistic stage-removal replays

Context 512, three alternating paired requests per stage. Required upstream dependencies remain; cached outputs replace the removed computation. Every replay logit is checked against the capture. This is an optimistic intervention on one trajectory, not a deployable candidate or a universal upper bound. The conditional 2× estimate assumes additive stage cost.

| Removed stage | Median saved ms/token | All paired savings ms/token | Median replay speedup | Conditional total speedup for 2× stage | Valid |
| --- | ---: | --- | ---: | ---: | --- |
| Down product + bias | 8.117 | 8.117, 4.964, 16.637 | 1.2052× | 1.0931× | False |
| Entire expert execution | 17.833 | 20.581, 10.680, 17.833 | 1.5698× | 1.2217× | False |
| Up/gate output biases + SwiGLU | 1.664 | 1.858, 0.134, 1.664 | 1.0396× | 1.0194× | False |
| Up/gate products + biases + SwiGLU | 8.892 | 9.688, 8.892, 6.916 | 1.2555× | 1.1133× | False |

The normal-hook control took 43.918 ms/token versus 41.926 ms/token for the plain baseline. Their +1.993 ms difference also includes run noise and logit retention; it is not a calibrated subtraction. Oracle savings include changed Python graph construction and dispatch work. The impossible cached-output dependency nodes can change scheduling and memory lifetime.

## Candidate diagnosis on real activations

Layers 0/12/23, eight evenly spread real decode steps per layer, 24 cases × eight cycles = 192 synchronized expert calls per block. Twenty alternating blocks. Candidate 1 includes its routing-ID conversion. These operator times include per-call host/synchronization overhead and cannot be multiplied into a full-model speedup. Isolated substages are non-additive: timing bias/activation alone pays synchronization costs that do not describe its incremental cost inside the larger graph.

| Operator or region | Median ms/call |
| --- | ---: |
| candidate1 | 0.8457 |
| candidate1_up_gate | 0.7407 |
| candidate2 | 0.8320 |
| candidate2_up_gate | 0.7066 |
| down | 0.5148 |
| native | 0.8706 |
| pointwise | 0.2253 |
| raw_up_gate | 0.7071 |
| up_gate_activation | 0.7339 |

| Paired comparison: reference / replacement | Median ratio | Descriptive bootstrap 95% interval | Median saved ms/call |
| --- | ---: | --- | ---: |
| native_over_candidate1 | 1.0265× | [1.0024, 1.0658] | 0.0226 |
| native_over_candidate2 | 1.0409× | [1.0231, 1.0714] | 0.0342 |
| up_gate_activation_over_candidate1_up_gate | 0.9915× | [0.9864, 1.0032] | -0.0063 |
| up_gate_activation_over_candidate2_up_gate | 1.0402× | [1.0361, 1.0521] | 0.0286 |
| up_gate_activation_over_raw_up_gate | 1.0386× | [1.0338, 1.0453] | 0.0271 |

Block-order strata (even forward / odd reverse) are retained in audit.json. Inspect them before interpreting small pooled operator gains: a single native call first/last in a block can expose warmup/order effects even with counterbalancing. No order stratum is dropped or used to select a favourable result.

Candidate region functions are diagnostic extractions of each frozen implementation through its `activated` result, excluding only the unchanged final down call. Their source and hashes are retained. Regions and full outputs are checked before timing. Confidence intervals are descriptive across timing blocks, not an acceptance test or a task-quality result.

## Serialized stage diagnostic

Synchronizing stage boundaries gave 101.253 ms/token across the first 16 decode steps, versus 41.926 ms/token for the plain full-request baseline (2.415×). These durations expose serial costs and barrier effects; they are not the uninstrumented GPU critical path.

| Stage | Serial build + eval ms/token |
| --- | ---: |
| activation | 4.560 |
| attention | 22.320 |
| down | 10.832 |
| embedding | 0.181 |
| final_norm | 0.172 |
| gate | 11.144 |
| input_norm | 4.389 |
| lm_head | 11.976 |
| post_norm | 4.150 |
| router | 4.482 |
| up | 10.676 |

An additional 15.301 ms/token was spent evaluating pending graph work before leaf stages, including routing/reduction/residual work. The raw pending attribution and every event are in serial-stages.json.

## Traffic and dispatch limits

Logical weight references are about 3.70 GB per decode token: experts 1.269 GB, attention projections 1.274 GB, vocabulary projection 1.158 GB. Both candidates retain these weight values and logical reads. A simple projection-result traffic model estimates only 2.21 MB/token saved by candidate 1 and 4.42 MB by candidate 2, excluding bias-gather buffers. These are analytical byte counts, not measured DRAM traffic. Caches, re-reads and physical transactions remain unmeasured.

No GPU timeline or bandwidth/occupancy/register counters were available: full Xcode/xctrace is absent. CPU eval/wait cannot establish dispatch bubbles. A megakernel remains a hypothesis; these measurements alone cannot prove that dispatch gaps justify it.

## Integrity and evidence

Controller completed: True; protected sources unchanged: True; checkpoint identity unchanged: False; exact-logit checks: True; real operator checks: 96/96.

Raw evidence: [state.json](controller-repair/state.json), [summary.json](controller-repair/summary.json), [audit.json](controller-repair/audit.json), [serial stages](controller-repair/serial-stages.json), [measurement plan](controller-repair/PLAN.md). The interrupted preparation and first pass remain intact. No candidate was created, promoted or accepted; acceptance remains disabled.

The raw final checkpoint guard recorded false because the controller reused its checkpoint `expected` variable for a numerical expected-output array. An independent post-run dictionary/file-identity comparison returned **True**. The checkpoint did not change according to that comparison. The raw state is retained; see [independent audit](controller-repair/postrun-checkpoint-audit.json). This bookkeeping error is repaired in the quiet pass.
