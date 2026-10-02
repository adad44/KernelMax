# Candidate submission contract

Optimizer agents may create or edit files **only inside one new direct child
directory of `candidates/`**. Do not edit the orchestrator, target, evaluator,
tests, reports, configuration, or runtime state. The evaluator owns its code,
reports, and `.kernelmaxxing/` state. A candidate must not use symlinks to read
or write files outside its own directory.

## Submission layout

Each submission directory is named with a unique candidate ID, for example
`cand_a1b2c3d4e5f6`. Generate IDs with
`orchestrator.candidate_manager.create_candidate_id()`; do not reuse IDs.
Submit these files in that directory:

| File | Requirement |
| --- | --- |
| `manifest.json` | Required metadata described below |
| `wrapper.py` | Required adapter exposing the agreed candidate entry point |
| `implementation.py` | Required for `python` and `mlx` implementations |
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
`implementation_type` must be one of `python`, `mlx`, or `metal`; the required
implementation source file is selected from that value. Unknown or missing
manifest fields are rejected so the contract stays explicit.

Passing submission validation means only that the candidate has the expected
shape. It does not establish correctness, safety, or performance; candidates
must still pass the protected evaluator.
