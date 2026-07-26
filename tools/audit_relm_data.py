"""Audit prepared DISC JSONL data against a ReLM tokenizer."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from tqdm.auto import tqdm
try:
    from relm_data import DataFormatError, FeatureStats, make_feature, read_json_pair
except ImportError:  # Supports ``python -m tools.audit_relm_data``.
    from tools.relm_data import DataFormatError, FeatureStats, make_feature, read_json_pair

try:
    from hf_local import load_local_tokenizer
except ImportError:  # Supports ``python -m tools.audit_relm_data``.
    from tools.hf_local import load_local_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_glob", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--output_file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(path for path in glob.glob(args.data_glob, recursive=True) if Path(path).is_file())
    if not files:
        raise FileNotFoundError(f"No files matched --data_glob: {args.data_glob}")
    tokenizer, model_dir = load_local_tokenizer(args.model_path)
    stats = FeatureStats()
    total = 0
    invalid_examples: list[str] = []

    for filename in files:
        with open(filename, encoding="utf-8") as handle:
            for line_number, line in enumerate(tqdm(handle, desc=Path(filename).name, unit="lines"), 1):
                total += 1
                try:
                    src, tgt = read_json_pair(line.strip(), line_number)
                except DataFormatError as exc:
                    stats.invalid_records += 1
                    if len(invalid_examples) < 10:
                        invalid_examples.append(str(exc))
                    continue
                make_feature(src, tgt, tokenizer, args.max_seq_length, stats)

    report = {
        "files": files,
        "model_path": str(model_dir),
        "total_records": total,
        "max_seq_length": args.max_seq_length,
        "stats": stats.as_dict(),
        "invalid_examples": invalid_examples,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output_file:
        output = Path(args.output_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
