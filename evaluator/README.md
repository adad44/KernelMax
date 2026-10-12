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

## Native reference baseline

See [scripts/BASELINE.md](../scripts/BASELINE.md) for collection, review, audit,
and report commands. `native_baseline.py` drives unchanged MLX-LM synchronously
with fresh caches; `baseline_validation.py` checks stock outputs, native dispatch
invariance and A/A noise. `baseline_report.py` validates and freezes reviewed
reference reports. Candidate execution and acceptance remain unimplemented.
