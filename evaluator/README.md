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

## Orchestrator evaluator handoff (trusted pass-through only)

`evaluator.pass_through.TrustedPassThroughEvaluator` supplies a concrete injected
callback for Elina's runner. Bind the trusted reference and `TrustedExpertAdapter`
once, then call it with the controller's reserved `RunIdentity`, `FrozenCandidate`,
and validated `ExperimentContract`:

```python
from evaluator.pass_through import TrustedPassThroughEvaluator

# reference(layer_index, x, ids) is Javi's unchanged trusted wrapper, or a
# trusted CPU fixture. adapter is the already-bound TrustedExpertAdapter above.
evaluate = TrustedPassThroughEvaluator(reference, adapter)
report = evaluate(run_identity, frozen_candidate, contract)
run_identity.validate_report(report)
budget.observe_result(report)
```

The bridge rejects mismatched candidate IDs/types/hashes or contract hashes
before expert execution. It verifies the frozen snapshot before and after the
suite, calls the fixed seven-case/three-layer suite, and packages every layer
diagnostic and error in an `EvaluationReport`. Snapshot failures produce failed
reports; errors after the suite retain its checks. Numerical mismatches without
execution errors are completed diagnostics with failed correctness flags, not
an invented `REJECTED` verdict. Early identity/setup errors raise to the runner.
Interruptions during the suite propagate to the runner under the existing suite
contract; the runner must record interruption and never claim complete coverage.

The optional keyword hook is `on_diagnostic_passed(report)`. It fires once only
after all 21 trusted pass-through checks and post-suite integrity checks pass.
Hook errors/interruption retain the completed checks in failed/interrupted
reports; snapshot changes made by the hook are also detected. This hook must
**not** be converted to the candidate `CORRECTNESS_PASSED` event: the frozen
submission's implementation has never been run. The existing state machine can
record the terminal `EVALUATOR_RESULT` directly from its correctness phase.
Real candidate correctness progress remains a separate future evaluator API.

Every returned correctness detail identifies `evaluation_scope` as
`trusted_pass_through`, declares the CPU-fixture or MLX-BF16 input mode, and sets
`candidate_code_executed`, `candidate_correctness_established`, and
`performance_measured` to false. Candidate identity is retained for the run's
submission provenance only, not as evidence that its code produced the output.
The bridge always returns empty performance evidence and `verdict=None`.
`environment=None` is explicit absent evidence; supply an independently captured
`EnvironmentRecord` only for an actually verified native environment. No report
files are written, no model is loaded, and no candidate code is imported.

The runner owns event journaling/state/budget; the evaluator owns report content
and any future official report writer. Runner/candidate isolation, timeouts,
credential restrictions, protected-input hashing and real model verification
are still separate requirements. Snapshot checks alone do not provide isolation.

Run the offline handoff tests, including state-machine/budget integration, with:

```sh
uv run --no-project --python 3.12 --with numpy==2.5.3 --with PyYAML==6.0.3 \
  python -B -m unittest tests.test_pass_through -v
```

## Reviewed-candidate execution prototype

`candidate_execution.ReviewedCandidateEvaluator` is distinct from trusted
pass-through diagnostics: it actually imports `wrapper.py` from an explicitly
approved frozen MLX snapshot **in a child process**, then grades the returned
tensors in the parent. The candidate cannot supply its own pass flag or verdict.

This is **not a hostile-code sandbox**. Workers share the user's filesystem and
network permissions. A stripped environment does not block reading credentials
from disk, hashes do not detect change-and-restore attacks, and process-group
cleanup cannot contain descendants that deliberately escape their group. Only
human-reviewed benign snapshots may be approved. Do not point this prototype at
unreviewed agent-generated Python. After any integrity failure, stop the session
and investigate before another run. Stronger isolation remains future work.

### Entry point and approval

`wrapper.py` exports `run(experts, x, selected_expert_ids)`. `experts` is the
unchanged selected layer's native expert callable; return `[1,T,4,H]` before
routing-weight combination. Preserve supplied routing IDs, native MXFP4 weights,
BF16 activations and original output dtype/backend. Helpers use relative imports
(`from .implementation import run`) from the frozen private package, not a mutable
submission directory. The parent and reference worker never import these files.

The existing manifest fields stay unchanged. Approval is an externally supplied
set of exact frozen SHA256 hashes, never a field inside the candidate manifest.
Review **all** files/helpers in that snapshot and its dependencies. Candidate
names, ID reuse and a successful diagnostic are not approval. Test/example code
is deliberately small enough to review; it demonstrates wiring, not optimization.

Prepare a snapshot without executing it:

```sh
python -B scripts/run_reviewed_candidate.py \
  --candidate /absolute/submissions/cand_0123456789ab \
  --work-dir /absolute/controller/run --prepare-review
```

Inspect the printed snapshot, then supply its hash explicitly:

```sh
python -B scripts/run_reviewed_candidate.py \
  --candidate /absolute/submissions/cand_0123456789ab \
  --work-dir /absolute/controller/run --fixture \
  --approved-sha256 THE_REVIEWED_SNAPSHOT_SHA256
```

`--fixture` means NumPy expert expressions, not model evidence. The script uses
the existing single-candidate runner, budget and journal; it does not implement
Elina's multi-submission session or terminal CLI. It freezes into a fresh run
directory and refuses execution if current bytes differ from the approved hash.
Use the existing evaluator environment; no dependencies/weights are installed.

### Worker lifecycle and parent grading

The evaluator declares all seven cases at layers 0/12/23. One reference worker
handles all 21 inputs and exits/reaps **before** one candidate worker starts.
Each native worker loads/materializes its model once on CPU, then calls unchanged
experts with MLX tensors. No model/callable is pickled between processes, and two
model copies are never intentionally resident at once. There is no automatic retry.

Transport is bounded JSON Lines plus base64 numeric buffers, never pickle or
object arrays. Original shape/backend/dtype are recorded before BF16 conversion;
the parent validates both storage and original metadata before comparison.
BF16 and FP32 outputs cannot pass as equal merely because their transport uses
FP32. Every materialized input/output, worker error and partial response is kept.

The parent checks run/candidate/contract identities, coverage, finite values,
shape/dtype and existing tolerances. It verifies snapshots and scoped protected
sources before/after runs, observes input mutations, and retains both per-layer
details and seven aggregate results. Numerical mismatches are completed checks
with false flags; import failures, crashes, timeouts, malformed responses, missing
outputs and integrity failures are execution errors. Verdict stays null and
performance evidence stays empty, including after a successful identity run.

The runner opts into `on_candidate_correctness_passed(report)` only for evaluators
declaring `supports_candidate_correctness=True`. Existing diagnostic/fake evaluator
signatures remain compatible. The new signal is journaled without advancing into
`BENCHMARK`; `on_diagnostic_passed` is never repurposed. It means only that this
reviewed candidate passed the declared operator cases, not full-model/task quality,
performance or acceptance. A later callback/integrity failure invalidates success
and is retained in the final report.

Each worker has a deadline, bounded combined stdout/stderr, bounded result file,
isolated scratch cwd/HOME, an allowlisted environment and process-group cleanup.
Artifacts include requests, raw responses, logs, PIDs, package versions, protected
source hashes and the final evaluator report. They remain on failure; keep them
controller-owned and sanitize before sharing. These controls are POSIX/macOS
reliability measures, not filesystem/network/GPU-memory isolation.

### Required native verification

First verify the existing local `openai/gpt-oss-20b` files against revision
`6cee5e81ee83917806bbde320786a8fb61efebee`, using an independently verified pinned
manifest (for example the preserved official baseline's file hashes). Do not use
a similarly named converted/requantized checkpoint. `--native-model` checks
pinned download metadata; workers verify all supplied SHA256 hashes and native
expert module format. Source, runtime, model-file and review provenance must be
retained together. No downloads occur, and hub access is disabled in workers.

```sh
/path/to/existing/native/environment/bin/python -B scripts/run_reviewed_candidate.py \
  --candidate /absolute/submissions/cand_0123456789ab \
  --work-dir /absolute/controller/native-run \
  --native-model /absolute/pinned/checkpoint \
  --approved-sha256 THE_REVIEWED_SNAPSHOT_SHA256 --timeout-seconds 1200
```

An optional trusted `--expert-factory package.module:function` binds an already
loaded model through `factory(model, mx=mx) -> TrustedExpertAdapter`; a configured
factory failure is an error, not a silent fallback. Until the teammate factory is
available, the explicit default binds unchanged native expert modules directly;
reference output uses Javi's `ReferenceMoE`. No `target/` files are rewritten.

The identity fixture must pass all 21 native comparisons, while the reviewed
incorrect fixture must produce retained numerical failures with no final verdict.
This native validation is a required deliverable, **not** replaced by CPU fixtures
or tiny MLX tests if weights are unavailable. No speedup, full-model correctness,
task-quality pass or acceptance is established. The layer-12 independent dense
diagnostic remains unresolved and tolerances/acceptance gates remain unchanged.

Regression tests: `python -B -m unittest tests.test_candidate_execution -v`.

## Native reference baseline

See [scripts/BASELINE.md](../scripts/BASELINE.md) for collection, review, audit,
and report commands. `native_baseline.py` drives unchanged MLX-LM synchronously
with fresh caches; `baseline_validation.py` checks stock outputs, native dispatch
invariance and A/A noise. `baseline_report.py` validates and freezes reviewed
reference reports. Candidate execution and acceptance remain unimplemented.
