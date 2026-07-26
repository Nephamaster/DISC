#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-/share/project/wuhaiming/spaces/ReLM/outputs/relm-sft-csc_mix/best}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: local checkpoint directory does not exist: $MODEL_PATH" >&2
  echo "Override it with: MODEL_PATH=/absolute/path/to/hf_model bash scripts/run_similarity.sh" >&2
  exit 2
fi

python tools/build_disc_similarity.py \
  --model_path "$MODEL_PATH" \
  --dict_dir char-similarity-calculation/dict \
  --output_dir artifacts/similarity/paper \
  --protocol paper \
  --dtype float32 \
  --workers 32 \
  --block_size 64 \
  --resume

python tools/validate_similarity.py \
  --artifact_dir artifacts/similarity/paper \
  --model_path "$MODEL_PATH"
