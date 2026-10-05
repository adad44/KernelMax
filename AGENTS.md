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

## Locked workload

Read `kernelmaxxing.yaml` before proposing a candidate. It specifies the pinned
`openai/gpt-oss-20b` checkpoint, native MXFP4 expert weights, BF16 activations and
KV cache, and a 24 GiB M4 with the MLX/Metal backend. Batch size is one; prompt
lengths are 512, 2048, and 4096; timing requires 128 generated tokens.

The Python/MLX/MLX-LM environment and actual model-file hashes still need to be
recorded by environment/model integration. The contract alone is not a frozen
runtime or evidence that inference works. Acceptance is disabled while numerical
tolerances and performance thresholds are provisional.

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

`moe_expert_execution` is the initial replacement boundary: evaluator-provided
activations and selected expert indices pass through the selected expert
projections and activation. Preserve checkpoint weight values, routing decisions,
output shape, and dtype. The model adapter must define the exact callable and
tensor layout before execution. Any layout conversion must be declared; do not
requantize or replace weights. Shape-specialized code must reject unsupported
inputs or use an explicitly declared reference fallback, reported by the evaluator.

## Evaluation protocol

The protected evaluator owns inputs, seeds, held-out cases, tolerances, model
installation, warmups, synchronization, sample collection, and verdicts.

1. Validate the manifest and candidate boundary without importing candidate code.
2. Freeze and hash candidate files and protected inputs.
3. Run correctness against unchanged MLX-LM using the same checkpoint and precision.
4. Only after correctness passes, run alternating baseline/candidate trials on the
   designated M4. Load the model before timing, start with fresh request caches,
   synchronize GPU work at interval boundaries, and keep agents out of timing.
5. Retain all samples, failed experiments, environment observations, and hashes.
6. Return `ACCEPTED`, `REJECTED`, or `INCONCLUSIVE` from the evaluator. The
   orchestrator records that verdict without overriding it.

An invalid build or submission is an execution/validation error, not a speedup.
Noisy or environmentally invalid timings cannot support `ACCEPTED`. Passing a
small operator test does not establish full-model correctness or performance.

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
