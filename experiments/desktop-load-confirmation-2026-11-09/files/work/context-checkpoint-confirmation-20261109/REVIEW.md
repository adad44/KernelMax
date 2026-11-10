# Full-workload confirmation using independently verified context checkpoints

This proposal changes how the goal can be evidenced. It requires explicit review
before preparing or running a new confirmation. The previous pass verdicts and
all of their data remain historical records under their original contract.

## Evidence available

The November 8 UUID pass verified the 512 context, then closed e002 with an AC
hard stop before the third 2048 comparison pair. The 2048 control also failed
its frozen A/A test. The partial comparison and failed control are retained.
The exit was verified. That pass is terminal and its full-workload verdict
remains incomplete.

| Context | Receipt | Candidate block medians | Per-block 95% intervals |
| --- | --- | --- | --- |
| 512 | UUID pass e001, `fc3c5ba…18d3e` | 1.134279× and 1.134312× | [1.127581, 1.137397] and [1.120887, 1.142869] |
| 2048 | November 5 pass e001, `1e59c0c…d3cb7` | 1.132078× | [1.129504, 1.134831] |
| 4096 | Fresh independent confirmation needed | | |

The selected receipts each have passing controls, 20 matched pairs per block,
true integrity guards and independently verified worker exit. Their frozen
candidate bundle, original reference, model receipt, native timing runner,
normalized base contract and statistical protocol hashes match. Controller and
metadata hashes remain recorded per pass. The historical pass that contains the
2048 receipt later stopped on AC loss at 4096.

## Proposed evidence rule and timing allocation

1. Pin the two specific receipts above before collecting new data. Reverify
   each against its original journal, raw rows, control, guards and exit record.
   Keep the newer 2048 A/A failure and partial block in the attempt history.
2. Prepare and test a fresh context-scoped controller for one 4096 confirmation.
   Reuse the same candidate, immutable baseline, numerical proof bridge, desktop
   policy, timing definitions, quiet preparation and statistical calculation.
   Freeze the scoped controller and validate its evidence, limits and exit path
   on CPU before requesting launch authorization.
3. Run fresh 4096 qualification and A/A followed by the predeclared matched
   candidate block. Retain all rows and flags, including paging. Keep 5 warmups
   per lane, 20 A/A pairs, 20 candidate pairs and the frozen A/A acceptance rule.
   Proposed bound is one attempt, one worker session, at most 240 start slots
   and 6 active hours. AC, thermal, integrity and numerical hard stops apply.
4. Verify the 4096 receipt, final guards and process exit independently. The
   all-context goal requires passing controls and a valid checkpoint for every
   context. Each of the four required blocks must clear the current 1% milestone
   with median at least 1.01 and 95% lower bound above max(1.01, 1 + A/A noise).
   Report the four block intervals individually and the weakest lower bound.

The resulting claim would describe full-model decode improvements at all three
prompt lengths under their recorded desktop sessions. Report the separate pass
identities, scope and attempt history. Calculate ratios within each block and
pair, with no pooled cross-pass timing rows. A failed 4096 control or incomplete
block leaves the overall goal unverified. Every independently verified positive
checkpoint remains available even below the goal threshold.

The current single-pass requirement is the other available choice. It would
require a new full confirmation covering all three contexts under one pass.
