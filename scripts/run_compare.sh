mkdir -p outputs/reports/compare

for name in test rSIGHAN13 rSIGHAN14 rSIGHAN15 test_law test_med test_odw car cot enc gam mec new nov; do
  python tools/compare_predictions.py \
    --baseline "outputs/baseline/all_test_predictions/${name}.pred.jsonl" \
    --disc "outputs/disc/all_test_predictions/${name}.pred.jsonl" \
    --output_file "outputs/reports/compare/${name}.json"
done