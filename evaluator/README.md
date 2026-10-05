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
uv run --no-project --python 3.12 --with PyYAML==6.0.3 --with pytest==8.4.2 python -B -m pytest -p no:cacheprovider -q
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
There is no report writer or acceptance algorithm in this PR.

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

This deliverable supplies configuration and record validation. Model execution,
candidate isolation, numerical evaluation, and timing are subsequent work.

## Native baseline calibration collector

`python -B -m evaluator.native_baseline --help` describes the baseline-only
collector. It requires a successful pinned HF CLI checksum verification receipt,
the native checkpoint, and a new output directory. It validates native tensor
headers, hardware, quantized expert representation, BF16 cache, and finite final
logits. It preserves fixed inputs, all token IDs/intervals, warmups, measurements,
hashes, runtime versions, environment snapshots, failures, and median/p90/MAD.

This is an initial calibration **attempt**, not an official `EvaluationReport`
or an acceptance decision. It has no candidate and does not invent paired A/B
samples. The unmodified MLX-LM model is driven synchronously, with fresh caches
and no generation lookahead. Prefill ends at materialized final prompt logits;
TTFT additionally includes greedy argmax. Inter-token trial latency is the median
of 127 intervals; all intervals are retained. Peak memory is the MLX allocator
peak including resident weights, not system-wide memory.

Five warmups and twenty measurements are collected per context size from the
locked contract. Context-size order rotates across rounds. AC power, stable
reported thermal/power state, and no observed swap growth are required; an
invalid request is saved before stopping. Load-time swap is recorded separately.
The collector does not certify background GPU isolation or fine-grained thermal
stability. Human review of the driver and full correctness/noise calibration
remain required before freezing an official reference. Do not run agents or
other inference workloads during measurement. Existing exploratory v0 evidence
is untouched.

Loading uses upstream `load(..., lazy=True)` inside an MLX CPU stream followed
by one-parameter-at-a-time `mx.eval` and synchronization. Native packed-weight
views/slices/contiguous byte copies are built and evaluated on CPU, avoiding
Metal command buffers for SSD-backed layout materialization. Inference resumes
on the unchanged default Metal device after the CPU context exits. All parameters are still
materialized before settling/warmups/timing; no weight values, precision, native
layout conversion, or inference operations are modified. The loading strategy
and progress are retained in the attempt. This is a workaround for a load-time
Metal timeout, not a performance optimization or a confirmed general fix.

The approved persistent-worker protocol is enabled by `--resident-retries 2`:
one verified load, then at most three complete attempts. Only environment-invalid
attempts permit retry; each is retained with a fresh run ID, request caches,
resident preconditioning, all five warmups, and all twenty A/A pairs per size.
GPU errors, correctness errors, noise-calibration failures, or changed protected
inputs stop the session. No samples carry over into a fresh attempt. The session
pointer and failed-attempt hashes bind the winning attempt to its original load.
The untimed controller follows `resident-session.json` across retry directories.
Candidate acceptance remains disabled.

Alan's separately approved `--swap-policy paging-aware` profile permits and
logs system-wide swap-ins while still invalidating swap-outs, growing swap
usage, non-nominal thermal state, or power changes. Strict remains the default.
This policy applies consistently to untimed preconditioning, quiet windows,
warmups, measurements, report validation, and the independent readback audit.
Every request keeps its original counters and paging deltas; no requests are
filtered out. A paging-aware official reference is explicitly labeled and is
not a zero-paging baseline. Future comparisons must use its declared profile.
All workloads, warmup/trial counts, correctness tolerances, noise gates, timing
boundaries, and disabled candidate acceptance are unchanged. Failed strict
attempts remain failed and cannot be reused in this new profile. Quiet-window
timeouts now also retain their observation sequence for diagnosis.

The reviewed quiet-reference procedure adds an optional `--stability-screen`
before fresh official setup. It retains twelve diagnostic requests (two A/A
pairs per size), separate controller/quiet-window receipts, and a predeclared
readiness check (pair deviation <=2%, relative MAD <=1%, full range <=4%). This
small screen is not a confidence interval or proof of repeatability. Failure
stops for diagnosis; no automatic noise-driven retry is allowed. Passing keeps
the same verified model resident, but official priming, all fifteen warmups,
all 120 measurements and the original twenty-pair noise gates are still required.
No diagnostic row is copied into official measurements. The controlled apps may
be hidden to reduce compositor activity without quitting them. A separate,
reviewed MemoryOS pause lease keeps only the three approved jobs unloaded across
attempts through official report freeze; configurations/history are preserved.

After a retained screen failure isolated remaining variability at 4096 tokens,
the next declared environment intervention sleeps the display (not the system)
before checkpoints. CPU-time and IORegistry GPU-client cumulative counters are
recorded before/after requests, outside timing. GPU counters retain their raw,
undocumented units and are diagnostic only; they do not establish frequency or
energy attribution and do not silently add or replace any official validity gate.

The display-sleep screen retained a reproducible order effect at 4096: the second
request of both AB and BA pairs slowed while the first stayed near its prior
quiet throughput. The reviewed `--per-request-control` procedure gives each lane
the same evaluator-owned settling/ten-second quiet window instead of starting
the second lane immediately. It requires 135 official controller/windows (15
warmups plus 120 measured requests), and twelve separate screen windows. Counts,
sample order, original noise gates and precision/correctness are unchanged.
Counter telemetry did not advance even for the running worker in the diagnostic;
these driver counters cannot support an isolation or attribution claim here.
