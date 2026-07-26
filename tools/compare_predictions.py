"""Compare frozen baseline and DISC prediction JSONL files line by line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from metrics import sentence_metrics
except ImportError:  # Supports ``python -m tools.compare_predictions``.
    from tools.metrics import sentence_metrics


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not all(isinstance(row.get(key), str) for key in ("src", "tgt", "pred")):
                raise ValueError(f"Prediction row missing string src/tgt/pred at {path}:{line_number}")
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--disc", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--examples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = read_rows(Path(args.baseline))
    disc = read_rows(Path(args.disc))
    if len(baseline) != len(disc):
        raise ValueError(f"Prediction counts differ: {len(baseline)} vs {len(disc)}")
    for index, (left, right) in enumerate(zip(baseline, disc)):
        if (left["src"], left["tgt"]) != (right["src"], right["tgt"]):
            raise ValueError(f"Input order/content differs at row {index}")

    srcs = [row["src"] for row in baseline]
    tgts = [row["tgt"] for row in baseline]
    baseline_preds = [row["pred"] for row in baseline]
    disc_preds = [row["pred"] for row in disc]
    changed = [
        {
            "id": index,
            "src": left["src"],
            "tgt": left["tgt"],
            "baseline": left["pred"],
            "disc": right["pred"],
        }
        for index, (left, right) in enumerate(zip(baseline, disc))
        if left["pred"] != right["pred"]
    ]
    payload = {
        "baseline": str(Path(args.baseline).resolve()),
        "disc": str(Path(args.disc).resolve()),
        "samples": len(baseline),
        "changed_prediction_rows": len(changed),
        "baseline_metrics": sentence_metrics(srcs, tgts, baseline_preds),
        "disc_metrics": sentence_metrics(srcs, tgts, disc_preds),
        "examples": changed[: max(0, args.examples)],
    }
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
