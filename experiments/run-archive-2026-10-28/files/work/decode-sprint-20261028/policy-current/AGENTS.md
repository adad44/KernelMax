# KernelMax experiment rules

## Scope and ownership

This repository contains both the human-built experiment system and agent-built
optimization candidates. The write restrictions below apply to optimization
runs, not to maintainers explicitly implementing or reviewing the system.

Agents may commit, push, and open or update PRs when authorized, but must never
merge a KernelMax PR. Merging is reserved for the human maintainer.

Maintainers own `kernelmaxxing.yaml`, the reference path, evaluator, tests,
orchestrator, environment setup, and candidate template. Changes to these inputs
require human review. Freeze their hashes before running optimization experiments;
changing them starts a new experiment contract and requires a new baseline.

Use a GPT-6 Luna subagent with high reasoning for running prepared scripts,
monitoring processes, collecting artifacts and concise run reports. The primary
agent handles experiment design, implementation and evidence interpretation.

## Locked workload

Read `kernelmaxxing.yaml` before proposing a candidate. It specifies the pinned
`openai/gpt-oss-20b` checkpoint, native MXFP4 expert weights, BF16 activations and
KV cache, and a 24 GiB M4 with the MLX/Metal backend. Batch size is one; prompt
lengths are 512, 2048, and 4096; timing requires 128 generated tokens.

The Python/MLX/MLX-LM environment and actual model-file hashes still need to be
recorded by environment/model integration. The contract alone is not a frozen
runtime or evidence that inference works. Acceptance is disabled while numerical
tolerances and performance thresholds are provisional and held-out task-quality
evaluation is planned.

## Optimization-agent boundary

An optimization agent may write only within its assigned fresh directory,
`candidates/<candidate_id>/`. IDs follow the candidate team's format:
`cand_` followed by 12 lowercase hexadecimal characters. The controller supplies
the candidate ID; a submission's manifest must match its directory name.

The agent may read the locked contract, reference implementation, and reports
provided by the controller. It may not modify the candidate template or other
candidates. It must not modify root configuration, dependencies, orchestrator,
target, evaluator, tests, reports, or `.kernelmaxxing/` runtime state.

The current candidate manifest fields are `candidate_id`, `name`,
`implementation_type`, and `description`, matching the candidate-pipeline team's
interface. Week 1 optimization submissions are `mlx` or `metal`. The submission
includes `wrapper.py` and `implementation.py` for MLX or `implementation.metal`
for Metal. Describe the hypothesis and assumptions in `description`; never put a
verdict or claimed benchmark result into the submission manifest.

The active `complete_decode` boundary permits complete decode scheduling,
including CPU graph construction overlapped with GPU execution and investigation
of unnecessary waits; graph fusion across normalization, routing, attention and
expert execution; and exact weight-layout transformations, buffer reuse and
KV-cache updates. Preserve checkpoint weight values, native MXFP4 weights, BF16
activations/cache, routing decisions, output shapes/dtypes and generation
behavior. Do not requantize, approximate, skip layers/experts, change the sampler,
reuse a prior request's cache or generate extra tokens to inflate throughput.
Declare layout conversions, scheduling and cache-state changes explicitly.

The existing `moe_expert_execution` adapter remains an expert-only interface;
passing its suite cannot validate a broader candidate. The trusted controller
must define the broader callable and check complete-model logits, routes, token
IDs and cache behavior before timing. Scheduling candidates must account for
the first token, every emitted token and the final GPU drain, with no hidden
lookahead work excluded. Shape-specialized code must reject unsupported inputs
or use an explicitly declared reference fallback, reported by the evaluator.

The 2026-10-27 scope expansion starts a new contract and protected-source
snapshot. Historical baseline and candidate results retain their old identities;
collect fresh matched controls before making claims under this contract.

## Evaluation protocol

The protected evaluator owns inputs, seeds, held-out cases, tolerances, model
installation, warmups, synchronization, sample collection, and verdicts.

Use the contract's two measurement modes explicitly:

- **Exploration:** predeclare request and wall-time bounds, retain every sample,
  flag system-wide swap activity and finish the bounded pass. Swap growth or
  swap-outs alone do not abort it. Preserve completed calibration and numerical
  correctness work with their original identities; a later timing flag does not
  erase a completed result. Use exploratory measurements only to choose
  hypotheses, never to establish a speedup or acceptance.
- **Confirmation:** require fresh matched baseline/candidate comparisons,
  numerical correctness, calibration and strict environment validity before a
  speedup claim. Logged swap-ins may follow the existing paging-aware policy;
  swap-outs or swap growth invalidate confirmation. Do not filter affected
  samples and report the remaining subset as a successful confirmation.

Observe prefill and decode separately. Retain phase boundary/sample times,
memory-pressure observations, free/wired/file-backed/compressed memory, swap
rate and process memory/fault counters when available. These are context and
attribution limits: system-wide counters cannot identify whether model data
paged, process faults are not swap-specific, and file-backed pages are not an
exact reconstruction of Activity Monitor's Cached Files. Apple describes memory
pressure using free memory, swap rate, wired memory and file cached memory:
https://support.apple.com/en-mide/guide/activity-monitor/actmntr1004/mac

Both modes still stop for numerical/build/GPU failures, integrity changes, loss
of required power or nonnominal thermals. Exploration bounds remain mandatory;
retaining swap-affected samples is not permission for an unbounded retry loop.

For the authorized 2026-10-28 three-candidate sprint, use short full-model
exploratory progression: an eight-token teacher-forced comparison under the
existing BF16 tolerances, then three alternating baseline/candidate requests
of128outputs. Record routing/token/cache differences; bit-exact equality is a
diagnostic, not an additional exploratory gate. This short check is not full
correctness or quality certification. Preserve the reference and checkpoint
precision and stop numerical failures. Keep a resident model between candidates
and reuse unchanged fixture evidence; no new broad profiling campaign. Limit
the round to three substantive candidate snapshots (vocabulary integration,
fusedMoEblock, optional composition). A repeated5% full-model improvement still
requires full-length correctness, fresh strict calibration/matched confirmation
and reported uncertainty across the locked prompt lengths. Acceptance/held-out
task-quality gates remain disabled/planned and cannot be inferred from timing.

1. Validate the manifest and candidate boundary without importing candidate code.
2. Freeze and hash candidate files and protected inputs.
3. Run correctness against unchanged MLX-LM using the same checkpoint and precision.
4. Only after correctness passes, run alternating baseline/candidate trials on the
   designated M4. Load the model before timing, start with fresh request caches,
   synchronize GPU work at interval boundaries, and keep agents out of timing.
5. Before final acceptance, require no unacceptable regression on a protected
   held-out task suite using the complete reference and candidate-integrated
   models. The suite, scoring, generation settings and score margin must be
   frozen before candidate evaluation. This gate is planned, not implemented;
   the `held_out_inputs` numerical fixture does not satisfy it.
6. Retain all samples, failed experiments, environment observations, hashes and
   eventual per-example task outputs/scores, including failures and truncations.
7. Return `ACCEPTED`, `REJECTED`, or `INCONCLUSIVE` from the evaluator. The
   orchestrator records that verdict without overriding it.

An invalid build or submission is an execution/validation error, not a speedup.
Noisy or environmentally invalid confirmation timings cannot support `ACCEPTED`. Passing a
small operator test does not establish full-model correctness or performance.
An insufficient task-quality sample or missing task evidence cannot support
`ACCEPTED`. The agent cannot select task examples, scorers or regression margins.
See `evaluator/README.md` for the three acceptance requirements and planned gate.

## Enforcement responsibilities

This document and the contract's lexical path checks describe policy. They do
not sandbox code. Before executing untrusted candidates, the controller needs
filesystem/process isolation, protected-input hash checks before and after runs,
timeouts, and limits on available tools/credentials. A post-run Git diff alone
cannot prevent a write or detect a file changed and then restored.

Only the evaluator writes official reports. Controller/evaluator processes own
runtime state and append-only journal entries. Agents may consume sanitized
reports, but may not delete failed runs, select favorable samples, alter timing
workloads, or grade themselves.
