# Prepared resumable confirmation contract

Status: prepared for review, inactive. The user authorized contract preparation.
No timing allocation, worker launch, new candidate, or historical promotion is
authorized by this bundle. The existing journal is an offline prototype.

The frozen numerical base is `base-contract.yaml`; `protocol.json` is the
controller-owned execution/evidence overlay. Changing either identity requires
a new contract. `reference-bridge.json` proves that the numerical base differs
from the archived original only in experiment identity and the approved desktop
paging policy. The original reference, source archive and catalog are untouched.

## Bounds and resume accounting

The proposal permits at most1,000 model-call starts,3worker epochs (initial plus
2explicit user resumes),20cumulative active hours,20hours per worker and7calendar
days including pauses. Baseline expected starts remain706. Each resume reserves
100starts for a complete fresh A/A group with warmups/primers,20for symmetric
candidate-block warmups/primers,2for an8outputnative/candidate handoff, and4for an
abandoned pair. Total planned envelope is958starts, leaving42within the cap.
The wall cap is finite; this arithmetic is not a guarantee of completion.

Compared with the previous single-worker contract, the global start cap changes
from800to1,000 and active wall from16to20hours. The resident20hour cap stays.
All worker time is charged, including loading, recovery waits, serialization and
cleanup. Only a fully closed, verified user-pause interval is excluded from
active time; the calendar cap still includes it. Budgets never reset on resume.

## Evidence and retention

Twenty complete same-epoch pairs are required per block, with the independent
second512block. A resumed context needs a fresh full control group. Use every
eligible complete pair, preserve all unmatched and failed rows, never pair
across epochs, and never retry a completed failed control through a new epoch.
The maximum A/A uncertainty across all complete controls for the context is used
for its gain threshold. The existing median/bootstrap95% and5% rules stay.

A context checkpoint can finalize after all3full numerical contexts, its required
blocks, valid controls/rows and final contributing-epoch provenance guards pass.
An ongoing worker must explicitly seal the context at a GPU-drained boundary,
verify raw evidence and all identity/storage guards, and commit an immutable
journal-prefix/guard receipt before certification. Mutable live state is not
enough. Closed earlier epochs also require their own final guard receipts.
This allows useful verified gains below5% to survive unrelated later failures.
The full goal still requires all3contexts/all4blocks under this contract. Any
failed context control makes that context inconclusive. Hard stops are terminal
for the pass. Historical evidence is retained under its original policy, without
importing old timing or retroactively changing a failed/incomplete status.

The original catalog remains unchanged. A new resumable-policy catalog would be
created only at activation, linked to the same immutable original reference.
Acceptance remains disabled and held-out task quality remains planned.

## Implementation review before any timing

The production controller must reserve both primer/measured-call slots atomically
before the primer, keeping one GPU call at a time. No journal fsync or artifact
serialization may interpose between primer and measurement. Unused reserved
slots stay charged and actual calls are reported separately. It must implement
atomic raw-artifact publication before journaling hashes, orphan/start accounting, cumulative wall/watchdog limits,
GPU-drained pause handling, source/model validation, fresh controls and a verifier
for context certificates. Pending buffers and partial pairs cannot be invented
or selected by speed. A torn journal or unresolved crash fails closed; recovery
never silently truncates evidence. The tested journal alone is insufficient.

Activation also needs a specific timing basis/operator authorization. A new
resumable storage format alone does not justify rerunning the latest unchanged
512statisticalfailure. This prepared bundle neither launches a model nor removes
that requirement. The current task is contract preparation and review only.
