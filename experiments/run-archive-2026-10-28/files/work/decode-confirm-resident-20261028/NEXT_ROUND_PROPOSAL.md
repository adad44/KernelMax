# Proposed next round — awaiting user authorization

The current round used all three substantive candidate snapshots. Its single
changed-residency confirmation ended on system-wide swap-outs after a clean
20-pair 512-token A/A calibration, before any candidate confirmation pairs.
No repeatable 5% gain has been demonstrated. This proposal does not authorize
another run, a fourth snapshot, or an alteration of the completed records.

## Objective and budget

Keep the original target: repeatable >=5% full-model decode throughput gain at
512, 2048 and 4096 prompt tokens, then pursue 10–20% through fusion. The latter
is an aspiration, not a forecast. Propose one new round of at most three
substantive snapshots, with no broad profiling campaign. Stop the round on its
candidate/time limits and report an unproven outcome honestly.

## Candidate priorities

1. **Memory-efficient vocabulary layout.** Preserve the existing exact reader
   and checkpoint values, but remove the prototype's simultaneous retention of
   the original vocabulary weights and its 910,739,074-byte sidecar. Declare
   all weight lifetime/layout changes. Native baseline lanes must retain their
   original math; candidate requests must have fresh caches and the same 128
   greedy outputs. Measure preparation costs separately and report peak memory.
   This is a substantive memory-execution candidate and counts toward the new
   budget; it is not disguised as a benchmark-protocol adjustment.
2. **MoE megakernel.** Resolve the existing longer-context routing discrepancy
   first, including native BF16 arithmetic, selection ties and expert-slot
   ordering. Then target normalization, router/top-k, expert up/gate/down,
   weighting and residual work as one block. State launch, synchronization,
   weight-traffic and temporary-buffer savings from existing evidence; require
   a credible several-millisecond opportunity before building. If the full
   fusion cannot preserve native dependencies or precision, record that limit.
3. **Best valid composition with scheduling.** Combine only passing components;
   target CPU/GPU overlap and dependency-safe wait removal where existing
   traces support it. Count the first token, every output and the final drain.
   Avoid an independently speculative whole-model persistent kernel unless a
   concrete resource/dependency design justifies it within this same slot.

## Progression and environment

- Use existing profiles and failure artifacts; permit only a small, targeted
  diagnostic needed to resolve a named uncertainty. Freeze bounds in advance.
- Short 8-output comparison, then the three-pair exploratory screen below.
  Only promising snapshots proceed to full-length correctness at all three
  locked contexts under existing tolerances and route/token/cache checks.
  Reuse only evidence whose source, snapshot, contract and checkpoint identities
  match.
- For exploration, retain and flag swap-affected samples and finish the bounded
  pass. Preserve completed correctness/calibration work. Stop numerical,
  integrity, GPU, power and thermal failures.
- Screen with three alternating full-model 128-output pairs. Promote only a
  promising snapshot that subsequently passes full correctness. Confirmation
  uses fresh strict matched controls,
  20 pairs per context and the declared independent repeat; never filter rows
  or reinterpret failed warmups as confirmation. Freeze any changed residency
  treatment identically for both lanes before collecting data.
- A repeated environment failure calls for a concrete memory/environment change
  or an explicit user decision; it does not trigger an automatic unchanged
  reroll. Do not close user applications without authorization.
- GPT-6 Luna high executes prepared scripts, monitors processes, collects
  artifacts and reports concise outcomes. The primary owns design and code.

Acceptance and held-out task-quality certification remain disabled/planned.
No commits, pushes, PRs, merging or publication are included in this proposal.
