# Search-policy improvement loop

Status: proposed design. This document adds no runtime behavior, agent write
permissions, or enabled acceptance gates. It describes a second phase after
KernelMax can repeatedly run and verify kernel optimization experiments.

## Problem and goal

KernelMax should improve both the kernels it discovers and the strategy it uses
to discover them. Repeated near-duplicate proposals, unproductive retries, and
poor experiment ordering can consume a bounded search budget. Those costs must
be measured before claiming that any fraction of compute is wasted. A rejected
candidate can still provide useful information.

The proposed outer loop learns from recorded searches, proposes a versioned
search policy, compares it against the incumbent on separate searches, and
retains it only after independent confirmation. Its objective is better verified
inference performance within the same total search budget.

| Loop | Proposed changes | Evidence required |
| --- | --- | --- |
| Kernel optimization | MLX/Metal candidate implementations | Protected correctness and performance evaluation |
| Search-policy improvement | Candidate ranking, exploration allocation, duplicate avoidance, and stopping decisions | Repeated complete searches under equal budgets, followed by unseen confirmation |

## Existing foundation and prerequisites

`run_candidate` already connects frozen candidate identities, an injected
evaluator, `ExperimentBudget`, state transitions, and `ExperimentJournal`.
`ExperimentBudget` bounds attempts and controller-classified transient retries;
it does not account for the complete cost of a search. The trusted pass-through
evaluator produces adapter diagnostics, not candidate correctness or speedups.

Before comparing search policies, complete isolated candidate execution, model
integration, calibrated acceptance gates, and repeatable end-to-end searches.
The current contract keeps acceptance disabled with provisional thresholds.
Diagnostic-only reports cannot establish either kernel or policy improvement.

## Policy scope and ownership

Start with a small declarative policy interpreted by trusted controller code.
Keep it immutable for the duration of each search. Initial policy controls are:

- Ranking hypotheses from sanitized training history and profiling summaries.
- Allocating a fixed budget between new hypotheses and refinements.
- Avoiding duplicate proposals using controller-computed identities.
- Choosing to stop or move to another hypothesis within controller limits.

Policies cannot change model weights, routing, precision, kernel boundaries,
correctness checks, task-quality requirements, benchmark inputs, thresholds,
report contents, or the total budget. They cannot retry final evaluator verdicts
or bypass the existing retry rules. Candidate agents retain their current
`candidates/<candidate_id>/` boundary.

Under current `AGENTS.md` and `kernelmaxxing.yaml`, optimization agents cannot
write orchestrator or protected configuration files. A future policy-proposal
interface needs a separately reviewed contract defining its submission area,
schema, read access, and isolation. This proposal does not grant that access.
Any required change to protected inputs follows the existing contract-freeze
and baseline rules; comparisons cannot mix different experiment contracts.

Only trusted code records policy identities, enforces bounds, evaluates results,
and promotes a reviewed policy. Proposing agents cannot grade or activate their
own proposals. Promotion changes the policy for subsequent searches; it never
replaces a policy during a running experiment. Keep the prior version available
for rollback, and journal both promotion and rollback decisions.

## Outer-loop protocol

1. Collect cost and outcome records from completed training searches, retaining
   failures and inconclusive runs. Diagnose a specific search inefficiency.
2. Propose one policy change with a rationale, expected benefit, and policy hash.
3. Freeze the incumbent, challenger, evaluator contract, workload partition,
   generator/model configuration, resource limits, and statistical decision rule.
4. Run complete incumbent and challenger searches on validation tasks using
   paired seeds and equal budgets. Alternate their order on the designated
   machine; avoid concurrent searches and reset controller state between runs.
5. Select a challenger using validation results. Freeze that version before
   independent confirmation on untouched test tasks and seeds.
6. Promote only if confirmation satisfies the predeclared improvement and
   regression rules. Otherwise retain the incumbent and all comparison records.

For example, training history might motivate requiring a profiling step after
repeated failures attributed to the same memory-layout assumption. The challenger
must earn its extra profiling cost back through better verified results. Fewer
failed candidates alone would not establish an improvement.

## Workload separation and information access

Define a search task as a frozen optimization problem with a workload family,
starting implementation, evaluator contract, and allowed candidate boundary.
Each policy comparison runs fresh optimization searches on those tasks; it does
not merely benchmark a shared list of already selected candidates.

Use disjoint training, validation, and test task partitions. Training feedback
may inform policy proposals. Validation feedback may select a challenger. Test
feedback confirms the frozen challenger and cannot feed another revision tested
against that same partition. Precommit the comparison schedule; further revisions
require fresh confirmation tasks or a separately reviewed sequential protocol.

Each inner search still needs its own protected held-out correctness and quality
cases. Those cases serve a different purpose from the outer loop's unseen search
tasks. Policies may see ordinary sanitized feedback within the current search,
but cannot access hidden cases or cross-run test history. Reset policy memory,
journals, candidate pools, and caches according to the comparison protocol;
any shared prior information must be identical and frozen for both policies.

Initially vary inputs within the locked M4, batch-one workload contract. Claims
about other shapes, models, or hardware require explicit contracts and reference
baselines for those tasks. Additional seeds alone do not demonstrate transfer to
new workload families.

## Budget and scoring

Use controller-enforced end-to-end elapsed search time as the initial primary
budget, with equal secondary limits on attempts and generation resources. Count
policy inference, candidate generation, profiling, validation, compilation,
warmups, benchmarking, retries, and failures. Freeze common initialization costs
and include them equally. Reserve confirmation costs for both policies in advance.

This search-level accounting is separate from timed inference intervals, which
continue to exclude model loading and agent activity under the kernel contract.
Stop scheduling when a limit is reached and enforce timeouts for in-flight work.
A result completed after the deadline is retained but cannot improve the score.

For each search, score the best eligible candidate found within budget using
the frozen workload aggregation and primary performance metric. Eligibility
requires all protected correctness, quality, environment, latency, and memory
gates. Use reference performance (speedup 1.0) when a completed valid search
finds no eligible improvement. Never count diagnostic-only reports as eligible.

Compare paired search scores across tasks and seeds. Predeclare workload weights,
repeat counts, minimum practical gain, uncertainty method, and per-task regression
limits before validation. Summarize paired log speedup ratios across tasks, with
task-level uncertainty rather than treating every timing sample as an independent
search. Inadequate coverage or unresolved noise yields an inconclusive comparison.
Numeric promotion thresholds remain to be calibrated before execution.

Also report time to first verified improvement, best-so-far performance over time,
resource consumption, and all failure categories. Acceptance rate is diagnostic;
optimizing it alone can reward easy proposals and discourage useful exploration.
Policy-caused failures consume budget. Environment-invalid searches remain visible
and follow a predeclared symmetric replacement rule rather than being dropped
after inspecting which policy won. Unresolved invalid runs block promotion.

Charge outer-loop proposal and comparison costs separately as well. Report total
cost and the number of future searches needed to recover that investment before
claiming net compute savings. Lower per-search cost alone does not prove a lower
total project cost.

## Evidence to retain

Extend controller-owned records with immutable search IDs, incumbent/challenger
policy hashes, contract and environment identities, task-partition identifiers,
seeds, generator versions, and common starting-state hashes. Record each decision,
its input-summary hash, selected hypothesis, candidate/run IDs, stage durations,
generation usage, remaining budget, and stop reason. Link unchanged evaluator
reports rather than deriving new kernel verdicts in the policy layer.

Retain all search scores, failed and invalid runs, comparison plans, uncertainty
estimates, and promotion/rollback records. Detailed hidden inputs remain evaluator
owned; policy-visible records contain only permitted summaries. Missing cost or
identity records make the comparison inconclusive.

## Implementation sequence

1. Instrument complete-search costs and provenance without changing selection.
2. Add a bounded search-session controller over `run_candidate`, using a fixed
   policy and the existing attempt/retry rules. Prove repeated end-to-end runs.
3. Add a versioned declarative policy interface and a reproducible seeded-random
   baseline. Keep proposal generation separate from policy execution.
4. Build paired comparison and reporting against fake evaluator fixtures first;
   verify deadline enforcement, state isolation, score provenance, failure
   retention, hidden-partition access, and refusal to promote on missing evidence.
5. Calibrate budgets and statistical gates on training/validation searches, then
   run independent confirmation on the designated hardware. Begin with reviewed
   promotion; automate activation only after its own reviewed contract change.

There is no measured policy gain or compute-saving result in this proposal.

## Research context

[SMAC3](https://arxiv.org/abs/2109.09831) provides context for configuring an
algorithm across problem instances. The need for untouched confirmation follows
the selection-bias problem discussed by
[Cawley and Talbot](https://jmlr.csail.mit.edu/beta/papers/v11/cawley10a.html).
Neither source establishes results for KernelMax.
