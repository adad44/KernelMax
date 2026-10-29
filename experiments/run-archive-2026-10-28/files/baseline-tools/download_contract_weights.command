#!/bin/zsh
set -euo pipefail

HF_BIN="/Users/alandiaz/.local/bin/hf"
MODEL_DIR="/Volumes/BankOfSouls/models/gpt-oss-20b"
REVISION="6cee5e81ee83917806bbde320786a8fb61efebee"

if [[ ! -d /Volumes/BankOfSouls ]]; then
  echo "BankOfSouls is not mounted. No download started."
  exit 1
fi

echo "Starting pinned native GPT-OSS checkpoint download: $REVISION"
"$HF_BIN" download openai/gpt-oss-20b \
  model-00000-of-00002.safetensors \
  model-00001-of-00002.safetensors \
  model-00002-of-00002.safetensors \
  config.json generation_config.json model.safetensors.index.json \
  tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja \
  --revision "$REVISION" --local-dir "$MODEL_DIR" --max-workers 2

for shard in model-00000-of-00002.safetensors model-00001-of-00002.safetensors model-00002-of-00002.safetensors; do
  [[ -s "$MODEL_DIR/$shard" ]] || { echo "Missing native shard: $shard"; exit 1; }
done

echo "DOWNLOAD COMPLETE. Verifying downloaded files against pinned Hub checksums."
# The repo also offers original/ and Metal variants. Missing unused variants
# are allowed; all three required native shards were explicitly checked above.
"$HF_BIN" cache verify openai/gpt-oss-20b \
  --revision "$REVISION" --local-dir "$MODEL_DIR" --format json
echo "CONTRACT CHECKPOINT READY. No benchmark or acceptance verdict has run."
