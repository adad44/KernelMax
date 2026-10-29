# First optimization experiment — 2026-10-26

Checkout: `/Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop`
Branch: `candidate/first-optimization-loop`
Base: `3d6acab019e7ef98d14037bc5bd6d06c6da611cd`
Candidate: `cand_b7f7dafae7bb`
Approved snapshot: `04442d2a56c48c328cf812a86bbe0edc45f8736fb3765f223274ec8fd6212949`

## Hypothesis and scope

Compile the native up/gate output-bias additions together with GPT-OSS SwiGLU
for batch-one, one-token, four-expert decode. Preserve the native MXFP4 matmuls,
weights, routing IDs, BF16 activations, and native down projection. Other token
counts explicitly use the reference fallback. Of the protected suite's 21
comparisons, three decode comparisons exercise optimized code; eighteen use the
reference fallback. Weight bandwidth may dominate enough to erase any benefit.

The source is frozen and human approval is recorded in `approval.json`. No
protected source, tolerance, workload, baseline, or acceptance gate was changed.
A separate worktree preserves the original checkout's local changes.

## Evidence stages

1. The approved source passed four tiny native-MXFP4 synthetic comparisons,
   including nonzero biases, large decode activations, and prefill fallbacks.
   These are fixture checks, not checkpoint evidence.
2. Hugging Face verification passed the ten native checkpoint files against
   revision `6cee5e81ee83917806bbde320786a8fb61efebee`. The ten-file view omits
   unused alternative formats and documentation. The first all-format verifier
   was interrupted after it began reading the unused `metal/model.bin`; that
   interrupted attempt is retained and is not counted as verification.
3. The protected evaluator runs separate reference/candidate native workers,
   each verifying actual checkpoint bytes. The merged
   `target.adapter:bind_native_experts` factory is used.
4. `operator_diagnostic.py` is a controller-owned exploratory diagnostic, not an
   official full-model benchmark or evaluator verdict. It requires completed
   native correctness for the exact approved snapshot, rechecks unchanged
   checkpoint file identities, retains all additional inputs/outputs, uses five
   warmup cycles and twenty AB/BA pairs per sampled layer, and retains every
   sample and environment issue. Invalid conditions make interpretation
   inconclusive. It never writes an `ACCEPTED` result.

## Full-model integration remains separate

The existing numerical adapter supplies int32 expert IDs. The installed native
model's top-k operator returns uint32 IDs. A future full-model adapter must
losslessly normalize those IDs to the declared candidate interface, retain that
conversion in its integration description, and include its cost in candidate
timing. No full-model candidate integration or task-quality result is claimed.

Acceptance remains disabled; the held-out task-quality gate is planned. Any
operator diagnostic timing is distinct from the contract's fixed 128-token,
512/2048/4096-prompt full-model comparison. Historical full-model baseline
measurements remain historical evidence and are not matched candidate timings.
