# DISC: A plug-and-play decoding intervention with similarity of characters for Chinese Spelling Check.

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2412.12863v2)
[![License: Apache](https://img.shields.io/badge/License-Apache-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Abstract

One key characteristic of the Chinese spelling check (CSC) task is that incorrect characters are usually similar to the correct ones in either phonetics or glyph. To accommodate this, previous works usually leverage confusion sets, which suffer from two problems, i.e., difficulty in determining which character pairs to include and lack of probabilities to distinguish items in the set. In this paper, we propose a light-weight plug-and-play DISC (i.e., decoding intervention with similarity of characters) module for CSC models. DISC measures phonetic and glyph similarities between characters and incorporates this similarity information only during the inference phase. This method can be easily integrated into various existing CSC models, such as ReaLiSe, SCOPE, and ReLM, without additional training costs. Experiments on three CSC benchmarks demonstrate that our proposed method significantly improves model performance, approaching and even surpassing the current state-of-the-art models.

## Original source reference

The original archive contains `char-similarity-calculation/merge.py`, but its
vocabulary path is a placeholder and it does not save the matrices used by the
formal experiment. Use the server workflow below for reproducible output.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{qiao2025discplugandplaydecodingintervention,
      title={DISC: Plug-and-Play Decoding Intervention with Similarity of Characters for Chinese Spelling Check}, 
      author={Ziheng Qiao and Houquan Zhou and Yumeng Liu and Zhenghua Li and Min Zhang and Bo Zhang and Chen Li and Ji Zhang and Fei Huang},
      year={2025},
      eprint={2412.12863},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.12863}, 
}
```

## ReLM reproduction data pipeline

The DISC reproduction uses four evaluation datasets: CSCD-NS, rSIGHAN,
LEMON, and ECSpell. Raw files under `data/raw` are converted to JSONL records
with `src` and `tgt` fields before training or evaluation:

```bash
python tools/prepare_data.py \
  --raw_dir data/raw \
  --output_dir data/processed
```

Records whose source and target have different character lengths are removed
before writing the processed files. Text is otherwise preserved without
normalization. CSCD-NS `train.tsv` is for training, `dev.tsv` is for
validation/checkpoint selection, and `test.tsv` is reserved for final
evaluation. `all.tsv` is kept only as a raw-data audit source and is excluded
from the formal pipeline.

The ReLM-compatible baseline helpers are `tools/relm_data.py`,
`tools/audit_relm_data.py`, and `tools/eval_relm.py`. They use the ReLM input
layout `[CLS] src [SEP] [MASK]... [SEP]` and report sentence-level correction
and detection accuracy, precision, recall, F1, and FPR.

## Server deployment workflow

The server environment can install the declared capability set from
`requirements.txt`; the ReLM training environment may already provide the
first five packages.

All checkpoint readers require a local Hugging Face export directory. The
directory must contain `config.json`, tokenizer files such as `vocab.txt` or
`tokenizer.json`, and model weights for inference. The code uses
`local_files_only=True`, so a missing or incorrectly mounted path fails with a
local-path diagnostic instead of being sent to the Hugging Face Hub validator.

The raw-data conversion is deterministic and writes `data_manifest.json` with
input/output SHA-256 values. The current CSCD-NS split is:

```text
data/raw/cscd_ns/train.tsv -> data/processed/cscd_ns/train.jsonl
data/raw/cscd_ns/dev.tsv   -> data/processed/cscd_ns/dev.jsonl
data/raw/cscd_ns/test.tsv  -> data/processed/cscd_ns/test.jsonl
data/raw/cscd_ns/all.tsv   -> audit source only; not consumed by the pipeline
```

Run preprocessing and tokenizer auditing on the server:

```bash
python tools/prepare_data.py --raw_dir data/raw --output_dir data/processed

python tools/audit_relm_data.py \
  --data_glob 'data/processed/**/*.jsonl' \
  --model_path /share/project/wuhaiming/spaces/ReLM/outputs/relm-sft-csc_mix/best \
  --max_seq_length 128 \
  --output_file outputs/data_audit.json
```

The repository wrappers also normalize the working directory and accept an
override for the checkpoint path:

```bash
MODEL_PATH=/share/project/wuhaiming/spaces/ReLM/outputs/relm-sft-csc_mix/best \
  bash scripts/run_audit.sh
```

Before running, verify that the server process can see the exact directory:

```bash
test -d "$MODEL_PATH" && ls -la "$MODEL_PATH"
```

Build and validate the paper-protocol DISC matrices for the exact tokenizer
directory used by the checkpoint. For the streaming ReLM training output,
`--model_path` must point to its exported Hugging Face directory, for example
`checkpoint-60000/hf_model`, not the parent training directory. The matrices
are float32, symmetric, and stored as memory-mapped NumPy files:

```bash
python tools/build_disc_similarity.py \
  --model_path /share/project/wuhaiming/spaces/ReLM/outputs/relm-sft-csc_mix/best \
  --dict_dir char-similarity-calculation/dict \
  --output_dir artifacts/similarity/paper \
  --protocol paper \
  --dtype float32 \
  --workers 32 \
  --block_size 64 \
  --resume

python tools/validate_similarity.py \
  --artifact_dir artifacts/similarity/paper \
  --model_path /share/project/wuhaiming/spaces/ReLM/outputs/relm-sft-csc_mix/best
```

Generate a frozen baseline and a DISC result with identical data order and
sequence handling. `--output_path` is used when one input file is evaluated;
`--prediction_dir` can be used for multiple files. The metrics JSON contains
sentence-level `correction_acc/precision/recall/f1` and
`detection_acc/precision/recall/f1`, with length-mismatched raw records already
removed by preprocessing:

```bash
python tools/eval_relm_disc.py \
  --model_path /share/project/wuhaiming/spaces/ReLM/outputs/relm-sft-csc_mix/best \
  --data_path data/processed/cscd_ns/test.jsonl \
  --disable_disc \
  --max_seq_length 128 \
  --batch_size 256 \
  --output_path outputs/baseline/cscd_ns_test.pred.jsonl \
  --metrics_path outputs/baseline/cscd_ns_test.metrics.json

python tools/eval_relm_disc.py \
  --model_path /share/project/wuhaiming/spaces/ReLM/outputs/relm-sft-csc_mix/best \
  --data_path data/processed/cscd_ns/test.jsonl \
  --similarity_dir artifacts/similarity/paper \
  --alpha 1.1 \
  --beta 0.7 \
  --max_seq_length 128 \
  --batch_size 256 \
  --output_path outputs/disc/cscd_ns_test.pred.jsonl \
  --metrics_path outputs/disc/cscd_ns_test.metrics.json
```

The same entry point accepts all four final evaluation families. Use the
checkpoint appropriate to the experiment line: the fixed 34M checkpoint for
LEMON, the SIGHAN downstream checkpoint for rSIGHAN, the corresponding
ECSpell-domain checkpoint for ECSpell, and the CSCD-NS checkpoint selected on
`dev` when CSCD-NS is used for downstream training. For a line-by-line
baseline/DISC comparison after both runs:

```bash
python tools/compare_predictions.py \
  --baseline outputs/baseline/cscd_ns_test.pred.jsonl \
  --disc outputs/disc/cscd_ns_test.pred.jsonl \
  --output_file outputs/compare/cscd_ns_test.json
```

The old `char-similarity-calculation/merge.py` remains as the original source
reference. It is not the formal runner because its vocabulary path and output
saves are placeholders and its diagonal convention is incompatible with the
paper protocol. The new `tools/build_disc_similarity.py` implements the
paper-protocol path and keeps the old behavior only as the explicit
`repo_compat` audit option.
