# Single-sort expert prefill experiment

This candidate tests whether building an inverse permutation by integer scatter
costs less than sorting the permutation a second time. It uses the existing
`run(experts, x, selected_expert_ids)` entry point and `moe_expert_execution`
boundary. The contract, evaluator, reference wrapper and model stay unchanged.

## Hypothesis and scope

MLX-LM 0.31.3 groups token/expert slots with `order = argsort(ids)`, then builds
`inverse = argsort(order)` to restore the original output order. This candidate
builds the same inverse with:

```python
inverse = zeros_like(order)
inverse[order] = arange(order.size, dtype=order.dtype)
```

`order` contains every slot exactly once, even when many expert IDs are equal.
Each destination therefore receives exactly one write. This scatters only `4T`
integers, rather than the much larger expert-output tensor. Native up, gate,
activation and down calls receive the same sorted inputs as the original path.
The final gather and `[1,T,4,H]` output layout stay the same. No routing weights
are combined here. No model parameters, activations, IDs or outputs are cached.

The fast path supports inference with the installed native `SwitchGLU` and
GPT-OSS `SwiGLU`, native `QuantizedSwitchLinear` projections in MXFP4/4-bit/group
32, BF16 `[1,T,H]` activations, and int32/uint32 `[1,T,4]` IDs with `T >= 16`.
Dimensions come from the supplied modules. Decode, training, other formats and
unsupported interfaces delegate to the exact supplied expert callable. Errors
from a selected projection or activation propagate to the caller.

Prediction, declared before native measurement: **0–3% lower selected-expert
prefill latency**, with no intended decode change. Scatter setup and integer
buffer allocation could erase the benefit or cause a regression. This is an
experiment in dispatch overhead; it does not tune matrix tiles or promise the
project's 5% full-model decode goal.

## Files and functions

- `manifest.json`: metadata accepted by the existing candidate validator.
- `wrapper.py`: exports the required `run` function using a snapshot-relative import.
- `implementation.py`: `_inverse_permutation` builds the index lookup table;
  `_run_sorted` calls the supplied native computations; `_supports_native`
  checks interface metadata; `run` chooses the supported path or delegates.
- `test_candidate.py`: CPU development tests with fake projections and NumPy,
  plus two opt-in tests using tiny synthetic MLX expert modules.

## Development tests

Use an existing Python 3.12 environment with NumPy, PyYAML and pytest. No model,
tokenizer, MLX installation or network access is needed for the default tests.
Run from the repository root:

```sh
python -B -m unittest discover -s candidates/cand_521ff575f496 -p test_candidate.py -v
python -B -m pytest -q -p no:cacheprovider
```

Development results (Python 3.12.14, NumPy 2.3.5, PyYAML 6.0.3): the focused
command ran 13 tests, with 11 passing and 2 native tests skipped. The full pytest
suite passed 344 tests and 423 subtests, with 10 opt-in tests skipped. An initial
sandboxed run could not execute `ps` in an existing process-cleanup test; the
same full command passed after granting that OS access. These are CPU results.

The CPU tests compare bytes against an independent two-sort baseline and a
direct original-order fixture. They cover all 21 existing correctness fixture
inputs, token counts 16/17/19/23/512/2048/4096, repeated and unused expert IDs,
noncontiguous inputs, one-sort behavior, input preservation and propagated
component failures. They also verify reference delegation with fake data.
They exercise the private array helpers with NumPy; they do not establish that
the native MLX fast path is correct or faster.

In an existing MLX/MLX-LM environment, enable the tiny native tests:

```sh
KERNELMAX_TEST_MLX=1 python -B -m unittest discover \
  -s candidates/cand_521ff575f496 -p test_candidate.py -v
```

These construct small 64/96-dimensional MXFP4 expert modules with synthetic
weights, assert that the public optimized path actually runs, and compare BF16
outputs exactly. They also check native decode delegation. They never load a
checkpoint. Native tests were not run during development in this checkout.

## Human review and M4 verification

Follow `evaluator/README.md`'s existing reviewed-candidate workflow. Review every
file, then prepare a frozen snapshot without executing it:

```sh
python -B scripts/run_reviewed_candidate.py \
  --candidate candidates/cand_521ff575f496 \
  --work-dir /absolute/controller/prefill-review --prepare-review
```

A human must approve the exact printed SHA256 before using the reviewed worker.
After approval, the existing pinned native environment can run the 21 operator
checks, using only the already installed checkpoint:

```sh
python -B scripts/run_reviewed_candidate.py \
  --candidate candidates/cand_521ff575f496 \
  --work-dir /absolute/controller/prefill-native \
  --native-model /absolute/pinned/checkpoint \
  --approved-sha256 THE_HUMAN_REVIEWED_SHA256 --timeout-seconds 1200
```

Use the repository's checkpoint verification and runtime provenance instructions.
The operator suite uses small token counts, so also test native inputs at
512/2048/4096 tokens before performance measurement. Require exact output values
and order for this permutation-only change; investigate any discrepancy. The
existing provisional numerical tolerances must not be widened.

The current reviewed runner reports correctness only. Alan's trusted M4 timing
harness must compare the original expert callable against `wrapper.run` after
correctness passes. Compile and warm up both paths before measuring. Use the
protected inputs, sampled layers 0/12/23, five warmups and twenty alternating
AB/BA pairs; synchronize only at matching interval boundaries. Retain all
samples, median/p90/MAD, peak memory, power/thermal observations, swap activity,
kernel count and supported/fallback path coverage. Include the scatter and
allocation in the timed candidate call. Confirm unchanged-model A/A stability
before interpreting any small gain. Stop on correctness or environment failures.

Only proceed to full-model timing if expert prefill has a repeatable benefit
above the measured noise, with p90 latency ratio <= 1.02 and peak memory ratio
<= 1.05. Measure TTFT, prefill throughput and decode throughput at all three
locked prompt lengths and 128 generated tokens. A local operator gain alone
cannot establish a full-model speedup or task quality. Acceptance remains
disabled; this candidate supplies no evaluator verdict or benchmark results.

## Implementation references

- [Pinned MLX-LM 0.31.3 SwitchGLU](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/switch_layers.py)
  defines the native sorted expert call sequence.
- [MLX indexed updates](https://ml-explore.github.io/mlx/build/html/usage/indexing.html#in-place-updates)
  describes array assignment; scatter destinations here are a unique permutation.
