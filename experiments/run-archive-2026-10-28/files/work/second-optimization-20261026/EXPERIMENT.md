# Second optimization candidate — joint up/gate MXFP4 decode

Candidate: `cand_90f0c66e0d03`.
Frozen snapshot SHA256: `93f523c916125738e860b796e6440170d192dcdeb207e327edaa26d826f14272`.
Source checkout: `/Users/alandiaz/Downloads/KernelMaxxxing/kernelmax-first-loop`.
The original candidate and all of its battery/AC observations remain unchanged.

## Hypothesis

One Metal dispatch computes both native up/gate MXFP4 dot products, shares
activation loads, and applies the original biases, clipping and SwiGLU before
calling the unchanged native down projection. No packed weight values, scales,
routing decisions or model modules are replaced or requantized. Both weight
matrices still have to be read. Register pressure and occupancy may outweigh
any dispatch/intermediate savings.

Only batch-one, one-token, four-expert decode is specialized. Larger token
shapes explicitly use the reference fallback. Three protected decode checks
exercise the new kernel; eighteen other protected checks exercise the fallback.

## Planned sequence

1. Human review and approval of the exact frozen Python and Metal sources.
2. Synthetic native-MXFP4 checks with nonzero biases, large activations, K=64,
   K=96 and K=2880, including declared prefill fallbacks. Numerical failure or
   compilation failure stops the experiment before loading native checkpoints.
3. The unchanged protected evaluator runs the seven cases at layers 0/12/23,
   with parent-owned grading and independent byte verification in each worker.
4. Only after a complete protected native pass, run the existing diagnostic
   plan: 96 additional varied-routing decode comparisons, then 20 alternating
   baseline/candidate pairs at each sampled layer (120 retained lane samples).
   Warmups, loading, correctness checks and environment observations stay out
   of timing. Every failure and environment-invalid sample remains retained.

The environment preflight passed on AC. Checkpoint file identities and runtime
versions match the preceding byte-verified native run; the protected correctness
workers will still hash actual checkpoint bytes. All protected sources are
unchanged. The runtime Metal compiler and required SIMD index attributes passed
an independent environment probe; that probe did not import candidate code.

This remains an operator experiment. It cannot establish full-model speedup,
full-model correctness, task quality or acceptance. The historical full-model
baseline is not a matched candidate benchmark. The contract, tolerances,
workload, baseline artifacts and disabled acceptance gates remain unchanged.

## Timing precision adjustment declared before timing

The prior candidate had wide intervals with 32 calls per sample. For this
candidate, retain the same 32 varied routing inputs but repeat eight cycles
per lane sample (256 calls), equally for baseline and candidate. Keep twenty
AB/BA pairs per layer and the strict environment policy. This adjustment was
made before observing any timing from candidate #2; no samples are selected
or discarded. Raw absolute times are not directly comparable to candidate #1.
