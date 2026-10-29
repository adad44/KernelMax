#!/bin/zsh
set -euo pipefail
cd /Users/alandiaz/Downloads/KernelMaxxxing/KernelMax
run_stamp=$(date +%Y-%m-%dT%H-%M-%S)
run_output="/Users/alandiaz/Downloads/KernelMaxxxing/baseline-results/official-attempt-${run_stamp}-seed42"
run_log="/Users/alandiaz/Library/Logs/kernelmax-official-baseline-${run_stamp}.log"
echo "Controlled reference baseline attempt: ${run_output}"
echo "Log: ${run_log}"
/usr/bin/caffeinate -i \
  /Users/alandiaz/Downloads/KernelMaxxxing/runtime/gpt-oss/.venv/bin/python -u -B \
  -m evaluator.native_baseline \
  --model /Volumes/BankOfSouls/models/gpt-oss-20b \
  --verification /Users/alandiaz/Downloads/KernelMaxxxing/baseline-tools/native-checkpoint-verification.json \
  --output "${run_output}" --settle-seconds 60 \
  --validate-reference --paired-aa --controlled --rehash-checkpoint \
  --dense-diagnostic /Users/alandiaz/Downloads/KernelMaxxxing/baseline-results/official-attempt-2026-10-03T17-08-11-seed42/attempt.json 2>&1 | tee "${run_log}"
