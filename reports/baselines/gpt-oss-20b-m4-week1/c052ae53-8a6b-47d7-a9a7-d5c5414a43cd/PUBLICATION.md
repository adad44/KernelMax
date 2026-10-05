# Publication provenance and limits

This directory publishes the already verified October 4, 2026 reference. No
new GPU measurements were made while packaging the repository or PRs. Original
`raw.json`, `baseline.json`, report README, independent audit, source snapshots,
resident-session receipt and service freeze/restoration receipts are unchanged.
Raw SHA256 is `abad2fd9f73b7c1b187b4ce919eff91f5aad29c5293d9b9f2afa237bbc363192`.
`publication.json` adds hashes and relative locations for the archival additions.

## Measured source versus packaged infrastructure

The collector and protected writer in infrastructure commit
`ce5e0407dc979f0d049e54d034b679f60c97c64d` match the measured snapshot byte-for-byte:

- `evaluator/native_baseline.py`: `8f02036bfb4ec587932b18ad1e65df22dd4132ccfe423cf6d2997f8807bfd6f2`.
- `evaluator/baseline_report.py`: `5568fb9e19db26f9bb9a8f67c591c651c69bb88734e112ff5874f15d9419c321`.
- `scripts/audit_baseline.py` matches the original independent checker: `80aae551ff060a95029f7d80b79ecba85961d1659f25b92bfa83f50e36506486`.

All measured evaluator and installed MLX-LM Python sources are archived under
`protected_sources`; runtime/binary/model identities remain in the original raw
evidence. The upstream MLX-LM snapshot is MIT licensed; its license is included
at `protected_sources/mlx_lm/LICENSE`.

`execution_sources` preserves the original machine-specific controller,
supervisor, durable pipeline, arithmetic checker and service-lease helper.
Four source hashes are bound by `durable-pipeline.json`; the service-lease helper
is additionally archived but was not included in that original four-file source
guard. These historical scripts contain the original absolute paths/UID and
actual approval context. **Do not run them unchanged on another checkout.**
No abandoned renderer-suspension controller or private launch-agent configuration
is published. No weights, virtual environment or MemoryOS history is included.

The portable `scripts/run_native_baseline.py` and controller were introduced for
repository use after this collection. Their CPU regression tests and command
checks do not prove new GPU performance or isolation. They must not be relabeled
as the executable version used to obtain these numbers. Future source/runtime
changes require a newly bound experiment snapshot and fresh paired comparisons.

## Exclusions and retained failures

`historical_attempts` preserves all sixteen earlier official-attempt raw files
byte-for-byte, including strict paging failures, environment-invalid resident
retries, interrupted partial collections, failed stability screens and the
externally interrupted renderer-pause setup. `publication.json` records each
hash, original saved status and recorded row counts. None contributes any
official measurement. The setup interruption retains its original nonterminal
saved status plus a separate `external-interruption.json`; it is not live or
eligible evidence. Exploratory community-weight v0 remains local and unofficial.

The separate passing twelve-request screen is readiness evidence only and stays
excluded. No primary/calibration rows were discarded: notable B/A ratios include
0.889869 at 512 and 1.072633 at 4096. Median-bootstrap 95% intervals contain one
and have widths below the unchanged 0.04 gate. They quantify paired differential
noise, not absence of common-mode drift or OS/GPU contention.

The original dense reconstruction failed large finite activations at layer 12
(max absolute difference 16, max tolerance ratio about 11.85). The full failed
raw artifact is `dense_reference_diagnostic.json`, with its original hash and
summary in `baseline.json`. Numerical tolerances were not widened. The passing
native operator checks establish dispatch invariance only. Candidate acceptance
is still disabled; this is not an ACCEPTED optimization.

## Environment receipts

The original three-service MemoryOS lease stayed paused through collection and
verification. `background-service-lease-at-freeze.json` is the immutable snapshot;
`background-service-lease-after-restoration.json` is the separately archived
restoration receipt, also hash-bound in `publication.json`. Pipeline verification time
`1791155907.327649` precedes restoration time `1791155907.380513`. Live post-run
checks confirmed all three jobs running and backend health. These are historical
receipts, not current process liveness or proof of total background isolation.

Absolute usernames, executable paths, PIDs, OS/version details and genuine
human approval responses are retained as measurement provenance. Process CPU
snapshots use executable names (`ps ... comm`), not full command arguments.
Credential-pattern scanning found no credential-shaped secrets. No provenance
was redacted or rehashed to hide failures.
