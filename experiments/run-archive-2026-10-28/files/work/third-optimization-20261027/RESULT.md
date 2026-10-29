# Candidate 3: complete compiled native expert graph

No consistent expert-operator improvement established across sampled layers. Park candidate; no promotion.

Candidate: `cand_206746fae66e`; frozen SHA256 `a4b4a69630d5444774119f44e323933a46da64d033779e96f57bb99a7710b901`.

The hypothesis compiles all three original native MXFP4 products, their output biases and GPT-OSS SwiGLU as one graph. Weights/scales/biases are explicit function arguments; routing, precision and prefill stay unchanged. This targets graph-construction overhead, not a custom matrix kernel or a megakernel. The earlier full-model profiling screen remains failed and is not reused as valid timing evidence.

Small fixture checks: 30/30. Protected checkpoint checks: 21/21.

Real decode comparisons, including both ID types: 48/48. Paired timing samples retained: 120.

Twenty alternating AB/BA pairs per layer, eight actual decode cases repeated 32 times: 256 synchronized calls per lane. Five warmup cycles per lane. Model loading and native byte verification are outside timing. The pipeline ran as a temporary launchd Interactive job, equally for baseline and candidate. These are expert-operator wall times, not full-model tokens/s.

| Layer | Baseline ms / 256 calls | Candidate ms / 256 calls | Paired median ratio | Descriptive bootstrap 95% | Valid |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 215.318 | 221.478 | 0.9865× | [0.9495, 1.0057] | True |
| 12 | 221.461 | 221.928 | 0.9972× | [0.9798, 1.0116] | True |
| 23 | 222.557 | 222.945 | 1.0010× | [0.9955, 1.0105] | True |

All samples and order strata are retained in [operator-diagnostic.json](operator-diagnostic.json). Ratios are baseline/candidate; above one means faster candidate. Intervals are descriptive and are not a candidate acceptance verdict. No outliers were removed and no failed statistical result was rerun.

| Layer | Baseline graph build ms/call | Candidate graph build ms/call | Instrumented environment valid |
| ---: | ---: | ---: | --- |
| 0 | 0.00947 | 0.00992 | True |
| 12 | 0.00927 | 0.01021 | True |
| 23 | 0.00929 | 0.00905 | True |

Graph-build measurements are from a separate instrumented pass, excluded from paired wall-time samples. Eval/wait includes dispatch and GPU work; no GPU hardware timeline or DRAM counter is claimed.

No consistent reduction in instrumented graph-construction cost; both lanes remained around 9–10 microseconds per expert call. Compiling the complete graph did not deliver the intended frontend saving in this operator experiment. The exact internal compiler/hardware cause is unmeasured.

Environment policy: AC, nominal thermals, no swap-outs or swap-usage growth, with global swap-ins permitted and retained under the certified reference paging-aware policy. Flags: none.

Protected source hashes unchanged: True; checkpoint file identities unchanged: True; frozen snapshot unchanged: True.

No full-model speedup, held-out quality result or acceptance is established. Acceptance remains disabled. Nothing was committed or pushed, per the user’s instruction.

Evidence: [plan](PLAN.md), [pipeline](pipeline.json), [small checks](tiny-native-checks.json), [native summary](native-correctness-summary.json), [machine-readable result](result.json), [controller call-count verification](call-count-verification.json). All earlier attempts remain intact.
