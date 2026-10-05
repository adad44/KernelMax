# Native M4 reference workflow

The evaluator is reference-only: no candidate execution or acceptance. Keep
`kernelmaxxing.yaml` unchanged. Its checkpoint, native MXFP4/BF16 precision,
batch-one workload, seed 42, 128 outputs, five warmups and twenty A/A pairs per
size are fixed. Hardware must match the 24 GiB M4 contract.

## Environment

Use Python 3.12.12 and the observed package versions in
`scripts/baseline-requirements.txt` on Apple Silicon. This pins direct runtime
versions, not every transitive dependency or OS binary. The collector records
and hashes actual installed runtime sources/binaries in every fresh attempt.

```sh
python3.12 -m venv .venv-baseline
.venv-baseline/bin/python -m pip install -r scripts/baseline-requirements.txt
```

Obtain `openai/gpt-oss-20b` revision
`6cee5e81ee83917806bbde320786a8fb61efebee` separately. Never commit weights or
the virtual environment. Supply a real successful HF checksum verification
receipt (`revision`, `exit_code`, and `result`), not a hand-written success
claim. The collector also rehashes all three native safetensors shards and
checks their headers, model configuration and runtime representation.

## Collect, without automatically approving results

Keep AC power connected, arrange explicitly approved background-work controls,
and leave the machine idle. This runner does not stop any services, close apps,
change persistent power settings, forge human approval, or freeze reports.
`--environment-ready` records the caller's confirmation, not isolation proof.
`--sleep-display` explicitly permits display sleep only. Use the same display
and per-request cadence as the reference when making comparisons.

Run from the repository root, substituting actual paths and a fresh output:

```sh
.venv-baseline/bin/python -B scripts/run_native_baseline.py \
  --model /path/to/native/gpt-oss-20b \
  --verification /path/to/hf-verification.json \
  --output "$PWD/baseline-runs/fresh-session" \
  --swap-policy paging-aware --environment-ready --sleep-display \
  --dense-diagnostic /path/to/preserved-failed-dense-attempt.json
```

The paging-aware profile requires the existing explicit protocol review; strict
is the runner's default. Swap-ins are logged/permitted only in paging-aware;
swap-outs/growth and non-nominal thermal or power changes invalidate an attempt.
The optional dense diagnostic stays failed and excluded from passing accuracy
claims; native operator checks establish dispatch invariance only.

The runner uses one model load, autonomous checkpoint acknowledgements, a
separate excluded twelve-request screen, fresh full setup/warmups/measurements,
and identical untimed windows before both lanes. The controller holds twelve
seconds; the evaluator then settles five seconds and checks ten quiet seconds.
Only environment-invalid attempts permit up to two fresh whole-attempt retries.
Correctness, GPU, protected-input or noise failures stop; no statistical rerolls.
There are three-hour session and 900-second checkpoint-idle bounds. All failed
attempts remain on disk and samples never carry over between attempts.

Worker/controller output goes to sibling log files, not a busy UI stream. For
long runs, launch the supervisor outside an app/PTY lifecycle (for example an
explicit temporary user launchd job with absolute Python/script arguments,
repository working directory, `ProcessType=Interactive`, and `KeepAlive=false`).
Do not restart merely because an observer disconnects; inspect the actual job
and `resident-session.json`. The helper itself does not register a scheduled job.

## Review and freeze

Follow `resident-session.json` to the winning attempt; do not treat a retained
failed attempt as the whole session's result. Independent arithmetic auditing
needs only Python's standard library and can also inspect a downloaded frozen
report without the model or MLX:

```sh
python3 -B scripts/audit_baseline.py --attempt /path/to/winning/attempt.json
```

Only after completion and actual human protocol review, bind the review to the
run ID, driver SHA, resident-session root and environment receipt. Do not copy
historical result approval or invent approval strings. The protected writer
validates the supplied review against this project's approved protocol and
checks current protected sources/checkpoint identity; it is not a generic
approval API. It requires the finished supervisor receipt (legacy filename
`memoryos-session.json` even when no services were controlled). Caller-managed
background-service observations/leases must be supplied separately if claiming
the original paused-service profile. No such claim is synthesized by the runner.

```sh
.venv-baseline/bin/python -B -m evaluator.baseline_report \
  --attempt /path/to/winning/attempt.json \
  --human-review /path/to/actual-run-bound-review.json \
  --destination /path/to/new/frozen-report
python3 -B scripts/audit_baseline.py \
  --attempt /path/to/new/frozen-report/raw.json \
  --report /path/to/new/frozen-report/baseline.json
```

Restore any separately paused services only after report verification, following
the actual approved lease. Preserve failed evidence and all restoration receipts.
The published baseline is a historical frozen source/runtime snapshot; packaging
or later source changes must never be relabeled as newly measured performance.
Future optimization comparisons need fresh paired measurements on the approved
snapshot/profile. Candidate acceptance remains disabled.

## Tests

```sh
uv run --no-project --python 3.12 --with PyYAML==6.0.3 --with pytest==8.4.2 \
  python -B -m pytest -p no:cacheprovider -q
git diff --check
```

Unit tests use synthetic fixtures and never prove GPU performance. The measured
report is separate evidence. Machine-specific MemoryOS leases, abandoned UI
suspension controls and exploratory v0 tooling are not runnable repo helpers.
