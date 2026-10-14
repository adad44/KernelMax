# Week 1 experiment contract

For the current native reference collector, autonomous run controls, independent
audit and protected report workflow, see [scripts/BASELINE.md](../scripts/BASELINE.md).
The scaffold descriptions below include historical deliverable scope; they do
not override the current reference-only implementation. Candidate acceptance
remains disabled.

The contract lives in `kernelmaxxing.yaml`. It pins the upstream model revision,
native MXFP4 expert weights, BF16 activation/KV-cache policy, a 24 GiB M4, and
batch-one workloads of 512/2048/4096 prompt tokens with 128 generated tokens.
The seed is 42, matching the target team's initial workload interface. The input
generator must use the pinned model's vocabulary size and retain the actual
token IDs; synthetic placeholder vocabularies are only for unit tests.

Run the checks from the repository root with Python 3.12 and `uv`:

```bash
uv run --no-project --python 3.12 --with PyYAML==6.0.3 python -B -m evaluator.schemas kernelmaxxing.yaml
uv run --no-project --python 3.12 --with PyYAML==6.0.3 python -B -m unittest discover -s tests -p test_contract.py -v
```

These commands use an isolated uv tool environment without modifying
`pyproject.toml` or `uv.lock`. They do not download model weights. The validator
prints a hash of normalized contract contents; reports must also identify model
file hashes, candidate hashes, and observed runtime versions.

After integrating the team's branches, run both their pytest tests and these
unittest cases through pytest:

```bash
uv run --no-project --python 3.12 --with numpy==2.5.3 --with PyYAML==6.0.3 --with pytest==8.4.2 python -B -m pytest -p no:cacheprovider -q
```

`evaluator/schemas.py` provides immutable, strict data records for the model,
hardware, workload, evaluation policies, candidate identity, correctness checks,
raw performance samples, environment, and verdict. It rejects unknown or missing
fields, invalid types, nonfinite values, and duplicate YAML keys. `AGENTS.md`
defines ownership and required enforcement. Its path policy is not an execution
sandbox; the candidate runtime must implement isolation before untrusted code runs.

## Run reports and failed experiments

The evaluator creates one `EvaluationReport` for each frozen candidate run. The
controller assigns a unique `run_id` (for example, a UUID); the report binds the
candidate identity/hash, contract hash, environment, correctness checks, timings,
run status, errors, and optional final verdict. Repeated candidate/contract
identities in a verdict must match the report. Observed hardware includes the
model identifier, not just chip and memory. Hashes and environment fields are
claims to verify at execution time, not proof supplied by these data records.

`completed`, `interrupted`, and `failed` are terminal run statuses. Interrupted
and failed runs require an error explanation. An early failure may have a null
environment and verdict, and no correctness or timing results. Invalid timing
records may retain empty or unequal sample lists; valid timing records require
at least two complete pairs. Every stored measurement must still be finite and
positive. Describe unsuccessful measurements in errors/detail, never as fake
zero/NaN samples. A raw sample is one trial-level observation for its metric;
the runner must define inter-token aggregation consistently for each trial.

Call `validate_report_against_contract(report, contract)` to get a tuple of
completeness/eligibility issues. It checks contract identity, canonical hardware,
required correctness cases, duplicate results, every workload/metric combination,
validity, and the exact configured trial count. It flags failed checks, execution
errors, non-completed runs, and an `ACCEPTED` claim while acceptance is disabled.

Save the structurally valid report even when issues are returned; failed or
incomplete evidence must not disappear. An invalid acceptance claim can be kept
as an audit artifact but must not become an official accepted result. The future
report writer/verdict implementation must enforce this gate. An empty issue list
does **not** authorize acceptance: actual hash verification, numerical evaluation,
sample pairing/order, timing/environment observations, noise analysis, and
speedup/regression decisions still belong to the protected runtime/evaluator.
The reference-only writer does not implement candidate acceptance.

Acceptance is disabled. The initial 5% speedup target, 2% p90 latency cap, 5%
memory cap, and numerical tolerances are proposals. Calibrate them against the
unchanged M4 baseline, document the evidence, and review a new contract revision
before enabling acceptance. A valid contract is not evidence of calibration.

Timing definitions for the future benchmark runner:

- Load and materialize model weights before timing. Use a fresh KV cache for each
  request and identical saved inputs for baseline and candidate.
- TTFT runs from the first forward call on existing token IDs to the first
  materialized sampled token. It includes prefill and first-token selection;
  model loading and tokenization are excluded.
- Decode throughput measures subsequent generation: 127 token intervals after
  the first token for a 128-token request. Inter-token latency uses those intervals.
- Prefill throughput times prompt processing separately; full-model wall time
  covers the forward start through the final materialized token.
- Synchronize GPU work before and after timing, alternate A/B and B/A pairs,
  retain every sample, and record memory/swap/thermal observations. Invalid
  conditions or improvements within noise cannot support acceptance.

The optimization boundary is selected-expert execution. Routing IDs and the
reference path remain evaluator-controlled. The target adapter and evaluator
must agree on the exact tensor layout and callable before candidate integration.

Candidate execution, isolation and acceptance are subsequent work.

## Candidate-output correctness

`correctness.compare_outputs(reference, candidate, case_name=..., policy=...)`
checks already-produced tensors against the evaluator-owned native reference.
It does not import candidate code, load weights, time inference or issue a verdict.
The target adapter still needs its agreed callable and input layout; the existing
native expert checks use activations `[1, T, H]`, expert IDs `[1, T, K]`, and
outputs `[1, T, K, H]`. This helper checks equality to the supplied reference's
shape/dtype, not whether the caller supplied the correct model, routing or weights.

```python
from evaluator.correctness import compare_outputs

result = compare_outputs(
    reference_output, candidate_output,
    case_name="decode", policy=contract.correctness,
)
# Keep result in EvaluationReport.correctness even when result.passed is False.
```

Use plain NumPy arrays for CPU fixtures or MLX arrays for native outputs. Original
shape, backend and dtype must match, with no broadcasting. MLX BF16 values are
materialized and widened to FP32 for a copied CPU view **after recording the
original dtype**; this does not change model precision. The untimed comparison
uses FP64 arithmetic and checks every element against
`abs(candidate - reference) <= atol + rtol * abs(reference)`. Zero tolerances
require exact equality. Arithmetic overflow fails closed. No tolerances change.
The BF16 host conversion follows [MLX's NumPy guidance](https://ml-explore.github.io/mlx/build/html/usage/numpy.html).

Malformed/nonfinite candidate outputs produce failed `CorrectnessResult` records.
An empty, nonfloating or nonfinite reference raises `InvalidReferenceError`;
record that in the run's errors, rather than blaming a candidate. Backend
execution failures propagate as run errors. These checks are not an execution
sandbox and must remain outside timed intervals. Freeze/verify candidates in
the controller before/after eventual execution.
Freeze the new checker with the protected inputs for future candidate experiments;
do not relabel the historical baseline as if it used this comparison code.

The runner must cover all declared cases and retain subcase/layer details. If
several subchecks share one declared case, aggregate their pass flags with AND;
never overwrite an earlier failure or emit duplicate case names in the report.
The existing dense-BF16 layer-12 discrepancy remains unresolved. Small fixtures
and reference-versus-reference checks do not resolve it or enable acceptance.

CPU tests (no MLX or checkpoint download):

```sh
uv run --no-project --python 3.12 --with numpy==2.5.3 --with PyYAML==6.0.3 \
  python -B -m unittest tests.test_correctness -v
```

To opt into two tiny MLX smoke tests using an **existing** native environment:

```sh
KERNELMAX_TEST_MLX=1 /path/to/existing/environment/bin/python -B \
  -m unittest tests.test_correctness.MLXCorrectnessTests -v
```

## Selected-expert correctness suite

`correctness_suite.run_correctness_suite` builds a declared plan before calling
two **trusted** injected functions with `(layer_index, x, selected_expert_ids)`.
Each function returns the selected-expert output **before** routing-weight
combination. It neither imports submissions nor supplies execution isolation.
Bind the reference to Javi's unchanged expert wrapper using a closure; bind any
future candidate adapter to the verified frozen snapshot in the controller.
Do not pass an untrusted in-process callable to this harness.

```python
from evaluator.correctness_suite import run_correctness_suite

# Trusted native_wrappers maps each sampled layer to its unchanged callable.
def reference(layer_index, x, selected_expert_ids):
    return native_wrappers[layer_index](x, selected_expert_ids)

suite = run_correctness_suite(
    reference, trusted_test_adapter,
    policy=contract.correctness,
    hidden_size=model.args.hidden_size,
    num_experts=model.args.num_local_experts,
    mx=mx,  # Existing mlx.core module; omit only for FP32 NumPy fixtures.
)
# Controller/evaluator retains suite.correctness and suite.errors in its report.
# Any suite.errors means incomplete/failed execution, not numerical rejection.
# No errors plus not suite.passed means completed comparisons with mismatches.
# Do not invent performance samples, a captured environment, or a final verdict.
```

The default plan is all seven contract cases at layers **0, 12, 23** (21 checks).
`correctness_cases` reproduces the small operator counts, expert-ID patterns,
scales and seeds from `baseline_validation.expert_checks`. Seeds stay attached to
case names even if contract case ordering changes. This is not the 512/2048/4096
full-model performance workload; `prefill` here uses 16 operator tokens. Published
deterministic `held_out_inputs` fixtures are not a secret held-out evaluation set.
Additional private coverage needs its own protected, reviewed plan.

Inputs are `[1,T,H]` activations and `[1,T,4]` int32 IDs. Native MLX inputs and
reference outputs must be BF16; CPU fixtures use FP32. Reference outputs must be
`[1,T,4,H]`. Unknown cases, duplicate/empty layer selections and invalid dimensions
are setup errors raised before execution. Explicit `layers=...` is available for
diagnostics, but a smaller plan establishes only that smaller scope; the caller
must bind the reviewed plan and dimensions to protected experiment inputs.

Both callables receive fresh, independently generated identical inputs. The
reference output is copied/materialized before the test runs; metadata and input
value fingerprints detect observed mutations. These checks do **not** detect
change-and-restore attacks or protect weights/files/other process state. They are
not substitutes for isolation, timeouts or protected-input hash verification.
The existing comparison tolerances and model precision remain unchanged.

`SuiteResult` contains `correctness`, `checks`, `errors`, `complete` and `passed`.
`summarize_checks` can aggregate collected `LayerCheck` records against the same
predeclared plan. It retains duplicates, extra checks, missing-check errors,
execution errors and every numerical result. Each case appears once in
`correctness`; its flags AND all expected layer flags. Layer diagnostics, seeds,
results and errors are JSON in `CorrectnessResult.detail`, so ordinary
`EvaluationReport` serialization preserves them. Undeclared case records are
retained under `unexpected_checks` in the first case's detail. Incomplete cases have false
flags **and** coverage/execution errors: never present those as a pure numerical
failure. `complete` means every planned comparison ran once without errors; a
complete suite can still fail numerical checks. Preserve both result and error
tuples, not just the convenience `passed` flag.

Per-check `Exception` failures are stage-labelled (reference/test/comparison)
and collection continues. `KeyboardInterrupt` and other `BaseException`s
propagate; partial-progress persistence/interruption handling belongs to the
controller and is not implemented here. Bad reference outputs skip the test
callable. Setup exceptions must likewise be recorded by the caller.

CPU regression tests, including suite/report round-trips:

```sh
uv run --no-project --python 3.12 --with numpy==2.5.3 --with PyYAML==6.0.3 \
  python -B -m unittest tests.test_correctness_cases tests.test_correctness_suite -v
```

Optional tiny MLX BF16 tests (no checkpoint/weights or native expert wrapper):

```sh
KERNELMAX_TEST_MLX=1 /path/to/existing/environment/bin/python -B \
  -m unittest tests.test_correctness_suite.MLXSuiteTests -v
```

Real-model smoke validation awaits the unchanged reference wrapper: construct
layer closures on the verified pinned model, run reference versus reference with
`mx=mx`, and retain all 21 diagnostics plus errors and source/runtime/model
provenance. That establishes integration only, not candidate accuracy, full-model
correctness, independent matmul accuracy or speedup. No such model run is claimed
by the fake/tiny-MLX tests. The layer-12 dense discrepancy remains unresolved;
acceptance stays disabled. Freeze these new evaluator sources for future runs;
the historical baseline is not rerun or relabelled by this addition.

## Trusted selected-expert adapter

`target.adapter.TrustedExpertAdapter` connects already-loaded, **trusted** expert
callables to the suite's `(layer_index, x, selected_expert_ids)` interface. Bind
Javi's unchanged per-layer reference wrappers when available; the same mapping
interface can use native expert modules directly for a pass-through diagnostic.
It does not load candidate files, replace model modules, reroute tokens, combine
routing weights, convert tensors, requantize weights, or silently fall back.

```python
from evaluator.correctness_suite import run_correctness_suite
from target.adapter import TrustedExpertAdapter

# On an already-loaded and independently verified pinned model:
experts = {i: model.model.layers[i].mlp.experts for i in (0, 12, 23)}
adapter = TrustedExpertAdapter(
    experts, hidden_size=model.args.hidden_size,
    num_experts=model.args.num_local_experts, mx=mx,
)
suite = run_correctness_suite(
    lambda i, x, ids: experts[i](x, ids), adapter,
    policy=contract.correctness, hidden_size=model.args.hidden_size,
    num_experts=model.args.num_local_experts, mx=mx,
)
```

The mapping is copied and read-only; the callable/weight objects are referenced
unchanged, **not** frozen or protected by this class. The controller must verify
checkpoint identity, weight formats, runtime and source provenance separately.
Native mode requires BF16 MLX activations `[1,T,H]`, int32 MLX IDs `[1,T,4]`, and
BF16 MLX output `[1,T,4,H]`. `mx=None` is explicit FP32 NumPy fixture mode. Only
batch one, nonempty token inputs and explicitly bound integer layers are supported.
Inputs must be finite and IDs within the bound expert count. Exact input objects
are forwarded; valid output is returned unchanged. Nonfinite outputs reach the
comparator for a numerical failure, while unsupported interface shapes/dtypes
raise `AdapterError` and remain suite execution errors. Backend errors propagate.

Value validation may synchronize native GPU work: this guarded adapter is for
**untimed correctness**, not a benchmark interval. Keep the same controller-side
mutation checks, recorded errors and isolation requirements as the suite. No
arbitrary agent-generated callable is safe to execute merely because it is
wrapped here. Future candidate execution needs a separate isolated, hash-bound
adapter with timeouts and credential/filesystem restrictions.

Run CPU tests with `python -B -m unittest tests.test_adapter -v` in the evaluator
environment. Opt into the two tiny native-tensor tests with
`KERNELMAX_TEST_MLX=1 .../bin/python -B -m unittest tests.test_adapter.MLXAdapterTests -v`.
They use trusted synthetic expressions, not GPT-OSS weights. Actual pinned-model
pass-through validation requires the existing local checkpoint to be available;
the full Javi-reference/Elina-runner path additionally awaits their implementations.
Even a passing real-model pass-through check would establish wiring only, not
optimized-candidate accuracy, full-model correctness, speedup or acceptance.

## Native reference baseline

See [scripts/BASELINE.md](../scripts/BASELINE.md) for collection, review, audit,
and report commands. `native_baseline.py` drives unchanged MLX-LM synchronously
with fresh caches; `baseline_validation.py` checks stock outputs, native dispatch
invariance and A/A noise. `baseline_report.py` validates and freezes reviewed
reference reports. Candidate execution and acceptance remain unimplemented.
