# Complete-decode confirmation under desktop load

Candidate `cand_57c56c7943e0` combines the fused MoE and packed vocabulary
projection with scheduled decode. The November 9 scoped pass completed the
4096-token confirmation. The per-context audit checks that result alongside
the pinned 512- and 2048-token receipts from earlier sessions with the same
candidate, reference, model receipt, native timing runner and base contract.
It calculates ratios within each block and does not pool timing rows across
sessions.

| Prompt tokens | Matched pairs | Median decode throughput gain | 95% bootstrap interval |
| ---: | ---: | ---: | ---: |
| 512, block 1 | 20 | 13.43% | 12.76%–13.74% |
| 512, block 2 | 20 | 13.43% | 12.09%–14.29% |
| 2048 | 20 | 13.21% | 12.95%–13.48% |
| 4096 | 20 | 13.08% | 12.86%–13.25% |

The baseline-vs-baseline controls, final integrity guards and worker exit
checks passed for these four blocks. The weakest lower bound was 12.09%, so
all four blocks cleared the 5% decode milestone under the per-context audit.
The November 8 partial 2048 comparison remains an unsuccessful attempt: its
control failed, and the run stopped on AC power loss. It contributed no rows
to the published gain. The gains apply to full-model decode under the recorded
desktop workload, including paging. The sessions do not establish single-pass
confirmation, a prefill gain, or candidate acceptance. Acceptance remains
disabled pending the held-out task-quality gate.

The [audit verdict](files/work/context-checkpoint-confirmation-20261109/per-context-verdict.json)
links the receipts by SHA-256. The [4096 receipt](files/work/context-checkpoint-confirmation-20261109/evidence/contexts/4096.json)
and [exit record](files/work/context-checkpoint-confirmation-20261109/post-exit/e001.json)
are retained alongside the earlier [512 receipt](files/work/resumable-boot-uuid-20261107/evidence/contexts/512.json)
and [2048 receipt](files/work/resumable-confirmation-20261105/evidence/contexts/2048.json).

The [manifest](manifest.json) indexes the original source snapshots, receipts,
raw timing rows and failure records. Git carries the latest controller and its
CPU tests, base protocol and selected receipts. Raw rows, generated weights,
numerical arrays and host-specific launch files remain in the original local
workspace. The archived scripts preserve the measured source bytes, including
their original local paths. They are historical snapshots and require the
original runtime and evidence layout to reproduce the audit. The checked-in
example authorization does not authorize a new run.

Verify the published files from the repository root:

```sh
python3 -B scripts/archive_runs.py verify experiments/desktop-load-confirmation-2026-11-09
```

Add `--artifact-root /path/to/KernelMaxxxing` on the original host to also
verify the indexed local files.
