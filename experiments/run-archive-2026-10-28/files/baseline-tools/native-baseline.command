#!/bin/zsh
set -euo pipefail
cd /Users/alandiaz/Downloads/KernelMaxxxing/KernelMax
run_stamp=$(date +%Y-%m-%dT%H-%M-%S)
run_output="/Users/alandiaz/Downloads/KernelMaxxxing/baseline-results/native-${run_stamp}-seed42"
run_log="/Users/alandiaz/Library/Logs/kernelmax-native-baseline-${run_stamp}.log"
echo "Native baseline calibration attempt: ${run_output}"
echo "Log: ${run_log}"
echo "This run is provisional, not a certified official result. Keep AC power connected."
/usr/bin/caffeinate -i \
  /Users/alandiaz/Downloads/KernelMaxxxing/runtime/gpt-oss/.venv/bin/python -u -B \
  -m evaluator.native_baseline \
  --model /Volumes/BankOfSouls/models/gpt-oss-20b \
  --verification /Users/alandiaz/Downloads/KernelMaxxxing/baseline-tools/native-checkpoint-verification.json \
  --output "${run_output}" \
  --settle-seconds 60 2>&1 | tee "${run_log}"
