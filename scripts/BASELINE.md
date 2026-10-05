# Native M4 reference baseline

This workflow collects unchanged MLX-LM inference, checks correctness and A/A
noise, and freezes a reviewed reference. Candidate acceptance remains disabled.
The workload and provisional tolerances live in `kernelmaxxing.yaml`: native
MXFP4 experts, BF16 activations/cache, batch one, seed 42, 512/2048/4096 prompt
tokens, 128 outputs, five warmups and twenty trials per lane on a 24 GiB M4.

## Setup and collection

Use Python 3.12.12 on Apple Silicon. The requirements pin observed direct
runtime versions; the collector records installed source/binary hashes too.
Obtain the contract's exact HF checkpoint revision separately and supply a real
successful checksum-verification receipt (`revision`, `exit_code`, `result`).
The collector checks shard hashes, tensor headers and native representation.

```sh
python3.12 -m venv .venv-baseline
.venv-baseline/bin/python -m pip install -r scripts/baseline-requirements.txt
.venv-baseline/bin/python -B scripts/run_native_baseline.py \
  --model /path/to/native/gpt-oss-20b \
  --verification /path/to/hf-verification.json \
  --output "$PWD/baseline-runs/fresh-session" --environment-ready
```

Connect AC power, arrange reviewed background-work controls and leave the
machine idle. `--environment-ready` records the caller's confirmation. The
runner does not pause services or apps, approve results or freeze reports.
`--sleep-display` optionally sleeps the display at checkpoints.

Strict swap policy is the default. An explicitly reviewed `--swap-policy paging-aware` permits logged swap-ins; swap-outs/growth, non-nominal thermal
state and power changes invalidate both profiles. `--dense-diagnostic PATH`
attaches a previous diagnostic without treating its failures as passing native
correctness. Native operator checks establish dispatch invariance on sampled
layers, not independent matmul accuracy or complete input coverage.

Loading/materialization and resident priming are untimed. Both A/A lanes
run the identical reference with fresh caches in alternating AB/BA order. Each
request gets a twelve-second controller hold, five-second settle and ten-second
quiet window. Failed attempts and every sample remain on disk. Only environment
failures allow up to two complete retries with the same verified model load;
correctness, GPU, source-change and noise failures stop. Session and checkpoint
idle limits are three hours and 900 seconds.

The synchronous driver measures 127 decode intervals after the first output;
TTFT includes prefill and argmax, while prefill ends at materialized prompt
logits. Peak memory is the MLX allocator peak including weights. These are
controlled-request metrics; retain the same driver/profile for comparisons.
Follow `resident-session.json` to the current attempt across retries. Logs and
supervisor receipts are siblings of the output directory. No scheduled job is
registered; long runs need a caller-managed process that survives UI disconnects.

## Review, audit and freeze

The independent arithmetic audit uses only Python's standard library:

```sh
python3 -B scripts/audit_baseline.py --attempt /path/to/winning/attempt.json
```

After actual review, save a JSON receipt with `approved: true`, the attempt's
`run_id` and `driver_sha256`, `resident_session_root`, and matching `swap_policy`.
Include explicit boolean approvals for each enabled option:
`resident_retries_approved`, `per_request_control_approved`, and
`paging_aware_approved` when applicable.
The writer requires a successful `collection-session.json` for that worker and
rechecks protected inputs. Legacy stability-screen/app-suspension/service-lease
publication is unsupported; use the original frozen writer for historical evidence.

```sh
.venv-baseline/bin/python -B -m evaluator.baseline_report \
  --attempt /path/to/winning/attempt.json \
  --human-review /path/to/run-bound-review.json \
  --destination /path/to/new/frozen-report
python3 -B scripts/audit_baseline.py \
  --attempt /path/to/new/frozen-report/raw.json \
  --report /path/to/new/frozen-report/baseline.json
```

Do not commit weights, virtual environments or generated evidence. Source
changes need a new measurement snapshot; CPU tests do not verify GPU performance.

```sh
uv run --no-project --python 3.12 --with PyYAML==6.0.3 --with pytest==8.4.2 \
  python -B -m pytest -p no:cacheprovider -q
git diff --check
```
