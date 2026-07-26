#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-/share/project/wuhaiming/spaces/ReLM/outputs/relm-sft-csc_mix/best}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: local checkpoint directory does not exist: $MODEL_PATH" >&2
  echo "Override it with: MODEL_PATH=/absolute/path/to/hf_model bash scripts/run_audit.sh" >&2
  exit 2
fi

python tools/audit_relm_data.py \
  --data_glob 'data/processed/**/*.jsonl' \
  --model_path "$MODEL_PATH" \
  --max_seq_length 128 \
  --output_file outputs/reports/data_audit.json
