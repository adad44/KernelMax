# Week 1 experiment contract

The contract lives in `kernelmaxxing.yaml`. It pins the upstream model revision,
native MXFP4 expert weights, BF16 activation/KV-cache policy, a 24 GiB M4, and
batch-one workloads of 512/2048/4096 prompt tokens with 128 generated tokens.
The seed is 42, matching the target team's initial workload interface. The input
generator must use the pinned model's vocabulary size and retain the actual
token IDs; synthetic placeholder vocabularies are only for unit tests.

Run the checks from the repository root with Python 3.12 and `uv`:

```bash
uv run --no-project --python 3.12 --with PyYAML==6.0.3 python -B -m evaluator.schemas kernelmaxxing.yaml
uv run --no-project --python 3.12 --with PyYAML==6.0.3 python -B -m unittest discover -s tests -p test_contract.py -v
```

These commands use an isolated uv tool environment without modifying
`pyproject.toml` or `uv.lock`. They do not download model weights. The validator
prints a hash of normalized contract contents; reports must also identify model
file hashes, candidate hashes, and observed runtime versions.

After integrating the team's branches, run both their pytest tests and these
unittest cases through pytest:

```bash
uv run --no-project --python 3.12 --with PyYAML==6.0.3 --with pytest==8.4.2 python -B -m pytest -p no:cacheprovider -q
```

`evaluator/schemas.py` provides immutable, strict data records for the model,
hardware, workload, evaluation policies, candidate identity, correctness checks,
raw performance samples, environment, and verdict. It rejects unknown or missing
fields, invalid types, nonfinite values, and duplicate YAML keys. `AGENTS.md`
defines ownership and required enforcement. Its path policy is not an execution
sandbox; the candidate runtime must implement isolation before untrusted code runs.

Acceptance is disabled. The initial 5% speedup target, 2% p90 latency cap, 5%
memory cap, and numerical tolerances are proposals. Calibrate them against the
unchanged M4 baseline, document the evidence, and review a new contract revision
before enabling acceptance. A valid contract is not evidence of calibration.

Timing definitions for the future benchmark runner:

- Load and materialize model weights before timing. Use a fresh KV cache for each
  request and identical saved inputs for baseline and candidate.
- TTFT runs from the first forward call on existing token IDs to the first
  materialized sampled token. It includes prefill and first-token selection;
  model loading and tokenization are excluded.
- Decode throughput measures subsequent generation: 127 token intervals after
  the first token for a 128-token request. Inter-token latency uses those intervals.
- Prefill throughput times prompt processing separately; full-model wall time
  covers the forward start through the final materialized token.
- Synchronize GPU work before and after timing, alternate A/B and B/A pairs,
  retain every sample, and record memory/swap/thermal observations. Invalid
  conditions or improvements within noise cannot support acceptance.

The optimization boundary is selected-expert execution. Routing IDs and the
reference path remain evaluator-controlled. The target adapter and evaluator
must agree on the exact tensor layout and callable before candidate integration.

This deliverable supplies configuration and record validation. Model execution,
candidate isolation, numerical evaluation, and timing are subsequent work.
