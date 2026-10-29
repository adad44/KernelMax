# Profiling prepared; full-model measurements pending AC

No next candidate was created. No megakernel performance claim is supported. No full-model profile or deletion ceiling has been measured in this attempt.

The Mac was on battery throughout preparation (last observed 53%). Measured requests require AC. The untimed checkpoint load was interrupted, all evidence was retained, and no profiling process remains running. The initial controller imported candidates through a helper designed for one candidate per process; its shared import name could alias the two implementations. This was caught before any measured request or candidate comparison. The corrected controller uses distinct package names and verifies each implementation filename. The original controller and interrupted load remain unchanged here; the corrected attempt has not started.

The corrected controller is in [controller-repair/profile_decode.py](controller-repair/profile_decode.py); its frozen measurement plan is [controller-repair/PLAN.md](controller-repair/PLAN.md). It provides:

- Fresh-cache 128-token unchanged-model controls at 512/2048/4096 context, retaining host graph-build, eval/wait and readback intervals. Eval/wait includes dispatch, GPU work and waiting; it cannot identify GPU idle gaps.
- Real activations and routing from all 24 layers, with selected cases retained on disk. Capture is compared to an unchanged control using exact logits.
- Dependency-preserving cached-output deletion replays for bias/SwiGLU, up/gate plus activation, down projection, and all experts. These estimate the latency available to recover in this particular replay, rather than assuming a speedup from dispatch counts. Every replay must preserve exact logits. They are optimistic counterfactuals, not deployable optimizations or universal upper bounds.
- Native versus both candidates on identical real activations, including isolated up/gate regions. Candidate 1's required routing-ID cast is included. All samples and environment violations are retained.
- A synchronized stage breakdown with its inflation relative to unchanged decode explicitly reported. Its stage shares must not be presented as the uninstrumented GPU critical path.

Controller validation passed on a small random native-MXFP4 model: exact capture/control logits, six exact-logit replay/instrumentation modes, 304 recorded serial events, and six full/region numerical checks for each separately imported frozen candidate. These checks validate the controller; they are not evidence of full-model performance. See [check results](controller-repair/check_instrumentation.json).

## What the existing evidence establishes

Candidate 1 compiles projection output biases together with an already compiled SwiGLU. It retains all three expert matrix products and native weight values. Its AC operator timing intervals included no change, with one swap-affected baseline sample making the strict overall attempt inconclusive. See [candidate 1 AC results](../first-optimization-20261026/operator-ac-woikuz8m/RESULT.md).

Candidate 2 combines up/gate products and activation in custom Metal, while preserving the native down product. Its corrected AC diagnostic passed environment checks but all three sampled layers' speedup intervals included no change. See [candidate 2 results](../second-optimization-20261026/RESULT.md).

Neither result diagnoses a hardware bottleneck. The operator tests used synthetic activations/routes and synchronized after each expert call; complete decode builds a larger graph and evaluates once per token. Neither experiment measured dispatch gaps, DRAM bandwidth, occupancy or register pressure. Those remain hypotheses, not explanations established by the results.

## Analytical byte accounting, not a timing profile

Using the installed model's dimensions and native formats, one decode token logically references:

| Weight group | Bytes per token |
| --- | ---: |
| Four selected experts × three projections × 24 layers, including MXFP4 scales | 1,269,043,200 |
| BF16 attention projection matrices across 24 layers | 1,274,019,840 |
| BF16 vocabulary projection | 1,158,266,880 |
| Combined | 3,701,329,920 |

A simple projection-result traffic model estimates about 2,211,840 bytes/token saved by candidate 1 and 4,423,680 by candidate 2, excluding bias-gather buffers and other tensors. These savings are approximately 0.060% and 0.120% of the logical weight bytes above. Both candidates still read the same packed expert weights and scales. Cache effects, physical memory transactions, re-reads and actual bandwidth have not been measured; these counts do not prove that decode is bandwidth-bound or quantify the dispatch opportunity.

Attention and the vocabulary projection therefore deserve direct measurement alongside experts. The next hypothesis must follow the full-model measurements, deletion opportunities and real-input component comparisons.

## GPU timeline limitation

Full Xcode and `xctrace` are absent. MLX exposes [Metal frame capture](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html), which the documentation describes analyzing in Xcode. No analyzed GPU timeline exists here. Host eval/wait time must not be reported as dispatch idle time.

The repository and checkpoint identities remained unchanged during the interrupted preparation. Acceptance remains disabled; no verdict, promotion, PR or merge was produced.
