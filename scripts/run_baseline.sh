MODEL_PATH="${MODEL_PATH:-/share/project/wuhaiming/spaces/ReLM/outputs/relm-sft-csc_mix/best}"

python tools/eval_relm_disc.py \
    --model_path "$MODEL_PATH" \
    --data_path \
    data/processed/cscd_ns/test.jsonl \
    data/processed/rsighan/rSIGHAN13.jsonl \
    data/processed/rsighan/rSIGHAN14.jsonl \
    data/processed/rsighan/rSIGHAN15.jsonl \
    data/processed/ecspell/test_law.jsonl \
    data/processed/ecspell/test_med.jsonl \
    data/processed/ecspell/test_odw.jsonl \
    data/processed/lemon/car.jsonl \
    data/processed/lemon/cot.jsonl \
    data/processed/lemon/enc.jsonl \
    data/processed/lemon/gam.jsonl \
    data/processed/lemon/mec.jsonl \
    data/processed/lemon/new.jsonl \
    data/processed/lemon/nov.jsonl \
    --disable_disc \
    --max_seq_length 128 \
    --batch_size 256 \
    --prediction_dir outputs/baseline/all_test_predictions \
    --metrics_path outputs/baseline/all_test.metrics.json