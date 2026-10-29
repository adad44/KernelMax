# Local candidate loop

The human has authorized ongoing local candidate implementation, freezing,
correctness checks and timing. Use the controller-owned session configuration
instead of requesting another approval for each snapshot. The configuration
records the actual standing instruction, repository, contract/model revision,
protected source hashes, model location and verification-cache location.
It is supplied explicitly by the controller, never by a candidate manifest.
Do not label this as human review of an exact candidate snapshot.

Run the correctness stage with one command (use the pinned runtime Python):

```sh
python -B scripts/run_candidate_loop.py \
  --session /absolute/path/to/controller-session.json \
  --candidate candidates/cand_0123456789ab \
  --work-dir /absolute/path/to/new-attempt
```

The orchestrator freezes once, records standing authorization for that exact
hash, runs every required native correctness case, and preserves its journal,
worker output, failures and evaluator report. An existing attempt directory is
refused. There is no prepare-review/copy-hash/ask-user cycle. The original
`run_reviewed_candidate.py --approved-sha256` path remains available.

Checkpoint verification hashes the complete pinned manifest on a cache miss.
Subsequent workers and attempts reuse that byte-verification receipt only while
the absolute model path, pinned revision, complete digest manifest, verifier
source hash, and each file's device/inode/size/mtime/ctime match. Workers check
before and after loading; the parent retains its before/after identity guard.
A changed file triggers byte verification, and a digest mismatch blocks the
run. Corrupt/incomplete receipts trigger verification rather than a cache hit.
Keep the cache outside candidate and checkpoint directories and controller-owned.
This is a local authorized-code reliability control, not a sandbox for hostile
submissions. Without explicit `--checkpoint-cache`, independent worker hashing
remains the default.

After correctness passes, use the existing paired timing controller under the
standing authorization. Reuse its completed native correctness report for the
same frozen snapshot and contract; do not launch another correctness session
just to begin timing. Keep real-input checks that cover inputs absent from the
native suite. Avoid redoing the same synthetic check with no code/input change.
Never cache a correctness pass for changed candidate code or reuse old timing
samples as a fresh comparison. This command intentionally stops at correctness;
it does not silently run a workload-specific diagnostic or issue a verdict.

Start from a measured hypothesis and record its possible contribution to total
latency. Run cheap fixtures before native correctness when the implementation
needs them, then operator screening before an expensive full-model comparison.
Park candidates without a reliable operator improvement. Keep all timing
samples, AC/thermal/paging observations, A/A noise gates, synchronization, actual
call-count assertions and AB/BA ordering. A statistical failure stops the pass;
do not reroll it unchanged to get a win. Operator results establish no full-model
speedup. Acceptance, task-quality requirements and publication remain unchanged.

If protected sources or the contract change, the session fails closed. Record a
new source snapshot under the applicable maintainer authorization before more
runs. Existing results retain their original source/contract identities; they
are not recertified by this tooling change. Any formal acceptance comparison
requires fresh matched baseline/candidate evidence under the reviewed harness.

Session fields consumed by the controller:

```json
{
  "schema": 1,
  "authorized": true,
  "repository": "/absolute/path/to/checkout",
  "contract_sha256": "normalized contract hash",
  "model_revision": "pinned revision",
  "allowed_operations": ["candidate_correctness", "operator_timing", "full_model_timing"],
  "acceptance_enabled": false,
  "authorization_basis": "actual user instruction and its experiment scope",
  "protected_sources": {"absolute source path": "sha256"},
  "model_path": "/absolute/path/to/native/checkpoint",
  "checkpoint_cache": "/absolute/controller/path/checkpoint-verification.json"
}
```

Compute `protected_sources` with `scripts.loop_control.controller_sources()`.
The configuration is an authorization record, not evidence of a candidate run.
