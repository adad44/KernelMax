# Candidate submission contract

Optimizer agents may create or edit files **only inside one new direct child
directory of `candidates/`**. Do not edit the orchestrator, target, evaluator,
tests, reports, configuration, or runtime state. The evaluator owns its code,
reports, and `.kernelmaxxing/` state. A submission directory must be a real
directory, not a symlink to another candidate. A candidate must not use symlinks
to read or write files outside its own directory.

## Submission layout

Each submission directory is named with a unique candidate ID, for example
`cand_a1b2c3d4e5f6`. Generate IDs with
`orchestrator.candidate_manager.create_candidate_id()`; do not reuse IDs.
Submit these files in that directory:

| File | Requirement |
| --- | --- |
| `manifest.json` | Required metadata described below |
| `wrapper.py` | Required adapter exposing the agreed candidate entry point |
| `implementation.py` | Required for `mlx` implementations |
| `implementation.metal` | Required for `metal` implementations |

The validator checks the common files plus the implementation file selected by
`implementation_type`. Keep helper code and assets, if needed, inside the same
candidate directory. Do not include generated model weights or modify reference
code.

## Manifest

Copy `candidate-template/manifest.json`, replace the ID placeholder with the
generated ID, and use the same ID for the directory name. The manifest must be
a JSON object containing exactly these fields:

```json
{
  "candidate_id": "cand_a1b2c3d4e5f6",
  "name": "Fused expert dispatch",
  "implementation_type": "mlx",
  "description": "Short explanation of the proposed inference optimization."
}
```

`candidate_id` must match `cand_<12 lowercase hexadecimal characters>` and the
directory name. `name` and `description` must be non-empty strings.
`implementation_type` must be one of `mlx` or `metal`; the required
implementation source file is selected from that value. Unknown or missing
manifest fields are rejected so the contract stays explicit.

Passing submission validation means only that the candidate has the expected
shape. It does not establish correctness, safety, or performance; candidates
must still pass the protected evaluator.

## Freeze before evaluation

The controller must stop the optimization agent and other submission writers
before freezing. Call `freeze_candidate` with a controller-owned snapshot root
outside `candidates/`, inaccessible to agents, for example
`.kernelmaxxing/frozen_candidates`. Evaluate the returned snapshot directory.

```python
from orchestrator.candidate_manager import freeze_candidate, verify_frozen_candidate

frozen = freeze_candidate(
    "cand_a1b2c3d4e5f6",
    frozen_root=".kernelmaxxing/frozen_candidates",
)
verify_frozen_candidate(frozen)
print(frozen.directory, frozen.sha256)
```

Freezing copies every regular file, including nested helpers, into a new
read-only snapshot. It rejects symlinks, hard links, special files, and reused
snapshot IDs. It leaves the original submission unchanged. The returned record
contains an immutable manifest, per-file SHA-256 hashes, and a bundle SHA-256 of
the sorted JSON file-path/hash mapping. Record these hashes in the evaluator's
experiment evidence.

Call `verify_frozen_candidate` before and after evaluation. It rejects changed,
added, or deleted files and linked replacements. File permissions are an
accidental-write guard; process/filesystem isolation must keep agents away from
the snapshot and protected evaluator inputs. Freezing a candidate does not
freeze those protected inputs or run candidate code.

Run the offline tests from the repository root:

```bash
python3 -m unittest tests.test_candidate_boundary -v
```
