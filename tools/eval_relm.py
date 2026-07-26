"""Evaluate a ReLM checkpoint with optional paper-protocol DISC decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

try:
    from relm_data import DataFormatError, encode_tokens, read_json_pair
except ImportError:  # Supports ``python -m tools.eval_relm``.
    from tools.relm_data import DataFormatError, encode_tokens, read_json_pair

try:
    from metrics import normalize_width_tokens, sentence_metrics
except ImportError:  # Supports ``python -m tools.eval_relm``.
    from tools.metrics import normalize_width_tokens, sentence_metrics

try:
    from disc_decoder import DISCDecoder
except ImportError:  # Supports ``python -m tools.eval_relm``.
    from tools.disc_decoder import DISCDecoder

try:
    from hf_local import load_local_bert, load_local_tokenizer
except ImportError:  # Supports ``python -m tools.eval_relm``.
    from tools.hf_local import load_local_bert, load_local_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True, nargs="+")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_file", default=None, help="Combined metrics/report JSON path.")
    parser.add_argument("--metrics_path", default=None, help="Alias for the metrics/report JSON path.")
    parser.add_argument(
        "--output_path",
        default=None,
        help="Prediction JSONL path for a single --data_path; kept for the markdown CLI.",
    )
    parser.add_argument("--prediction_dir", default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument(
        "--similarity_dir",
        default=None,
        help="Paper-protocol DISC artifact directory; omit only with --disable_disc.",
    )
    parser.add_argument("--alpha", type=float, default=1.1)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--copy_punishment", type=float, default=0.0)
    parser.add_argument("--disable_disc", action="store_true")
    parser.add_argument(
        "--state_dict",
        default=None,
        help="Optional compatible plain-Bert state dict when model_path only supplies config/tokenizer.",
    )
    return parser.parse_args()


def read_pairs(path: Path) -> tuple[list[tuple[list[str], list[str]]], int]:
    pairs = []
    format_dropped = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                source, target = read_json_pair(line.strip(), line_number)
            except DataFormatError:
                format_dropped += 1
                continue
            pairs.append((list(source), list(target)))
    return pairs, format_dropped


def metric_slot_tokens(tokenizer, token_ids: list[int]) -> list[str]:
    """Decode fixed slots for metrics without losing tokenizer-level identity."""

    tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
    return normalize_width_tokens(
        [token[2:] if token.startswith("##") else token for token in tokens]
    )


def build_records(pairs, tokenizer, max_seq_length: int):
    slots = (max_seq_length - 3) // 2
    cls_id, sep_id, mask_id, pad_id = (
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.mask_token_id,
        tokenizer.pad_token_id,
    )
    records = []
    dropped = {"length_mismatch": 0, "token_mismatch": 0, "seq_length": 0}
    for source_tokens, target_tokens in pairs:
        if len(source_tokens) != len(target_tokens):
            dropped["length_mismatch"] += 1
            continue
        source_ids = encode_tokens(tokenizer, source_tokens)
        target_ids = encode_tokens(tokenizer, target_tokens)
        if len(source_ids) != len(target_ids):
            dropped["token_mismatch"] += 1
            continue
        if len(source_ids) > slots:
            dropped["seq_length"] += 1
            continue
        input_ids = [cls_id] + source_ids + [sep_id] + [mask_id] * len(target_ids) + [sep_id]
        input_ids += [pad_id] * (max_seq_length - len(input_ids))
        records.append(
            {
                "src_tokens": source_tokens,
                "tgt_tokens": target_tokens,
                "input_ids": input_ids,
                "source_ids": source_ids,
                "target_ids": target_ids,
                "source_length": len(source_ids),
                "target_length": len(target_ids),
            }
        )
    return records, dropped


def _load_state_dict(model, path: str) -> None:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported state dict payload: {path}")
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in payload.items()
    }
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "The supplied state_dict is not compatible with plain BertForMaskedLM. "
            "Prompt-wrapper ReLM .bin checkpoints need their original ReLM loader."
        ) from exc


def predict_records(model, tokenizer, records, batch_size: int, device, decoder=None):
    metric_srcs, metric_tgts, metric_preds = [], [], []
    prediction_rows = []
    pad_id = tokenizer.pad_token_id
    coverage_start = (
        (decoder.covered_positions, decoder.total_positions) if decoder is not None else None
    )
    model.eval()
    description = "ReLM + DISC" if decoder is not None else "ReLM baseline"
    for start in tqdm(range(0, len(records), batch_size), desc=description, unit="batch"):
        chunk = records[start : start + batch_size]
        input_ids = torch.tensor([record["input_ids"] for record in chunk], dtype=torch.long, device=device)
        attention_mask = input_ids.ne(pad_id).long()
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        if decoder is not None:
            max_target_length = max(record["target_length"] for record in chunk)
            vocab_size = logits.shape[-1]
            target_logits = logits.new_zeros((len(chunk), max_target_length, vocab_size))
            target_source_ids = torch.full(
                (len(chunk), max_target_length), pad_id, dtype=torch.long, device=device
            )
            target_valid = torch.zeros(
                (len(chunk), max_target_length), dtype=torch.bool, device=device
            )
            for index, record in enumerate(chunk):
                target_start = record["source_length"] + 2
                target_end = target_start + record["target_length"]
                length = record["target_length"]
                target_logits[index, :length] = logits[index, target_start:target_end]
                target_source_ids[index, :length] = torch.tensor(
                    record["source_ids"], dtype=torch.long, device=device
                )
                target_valid[index, :length] = True
            adjusted_ids = decoder.adjust(target_logits, target_source_ids, target_valid)

        for index, record in enumerate(chunk):
            if decoder is None:
                row = logits[index]
                target_start = record["source_length"] + 2
                target_end = target_start + record["target_length"]
                predicted_ids = row[target_start:target_end].argmax(dim=-1).tolist()
            else:
                predicted_ids = adjusted_ids[index, : record["target_length"]].tolist()
            predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids)
            src_tokens = normalize_width_tokens(record["src_tokens"])
            tgt_tokens = normalize_width_tokens(record["tgt_tokens"])
            predicted_tokens = normalize_width_tokens(predicted_tokens)
            metric_srcs.append(metric_slot_tokens(tokenizer, record["source_ids"]))
            metric_tgts.append(metric_slot_tokens(tokenizer, record["target_ids"]))
            metric_preds.append(metric_slot_tokens(tokenizer, predicted_ids))
            prediction_rows.append(
                {
                    "src": "".join(src_tokens),
                    "tgt": "".join(tgt_tokens),
                    "pred": "".join(predicted_tokens),
                    "src_tokens": src_tokens,
                    "tgt_tokens": tgt_tokens,
                    "pred_tokens": predicted_tokens,
                }
            )
    metrics = sentence_metrics(metric_srcs, metric_tgts, metric_preds)
    if decoder is not None and coverage_start is not None:
        covered_delta = decoder.covered_positions - coverage_start[0]
        total_delta = decoder.total_positions - coverage_start[1]
        metrics["disc_coverage"] = covered_delta / total_delta if total_delta else 0.0
    return metrics, prediction_rows


def main() -> None:
    args = parse_args()
    report_path = args.metrics_path or args.output_file
    if report_path is None:
        raise ValueError("Provide --metrics_path or --output_file")
    if args.output_path and len(args.data_path) != 1:
        raise ValueError("--output_path requires exactly one --data_path")
    if not args.disable_disc and not args.similarity_dir:
        raise ValueError("Provide --similarity_dir for DISC, or pass --disable_disc for the baseline")
    if not 0.0 <= args.beta <= 1.0:
        raise ValueError("--beta must be in [0, 1]")
    if args.alpha < 0.0 or args.copy_punishment < 0.0:
        raise ValueError("--alpha and --copy_punishment must be non-negative")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    tokenizer, model_dir = load_local_tokenizer(args.model_path)
    model, model_dir = load_local_bert(model_dir)
    model = model.to(device)
    if args.state_dict:
        _load_state_dict(model, args.state_dict)
    decoder = None
    if not args.disable_disc:
        decoder = DISCDecoder(
            args.similarity_dir,
            device=device,
            alpha=args.alpha,
            beta=args.beta,
            copy_punishment=args.copy_punishment,
        )
        decoder.validate_model_path(model_dir)
        if decoder.metadata["num_vocab_tokens"] != model.config.vocab_size:
            raise ValueError(
                "Similarity artifact vocab size does not match the ReLM checkpoint: "
                f"{decoder.metadata['num_vocab_tokens']} vs {model.config.vocab_size}"
            )
    results = {}
    single_predictions = None
    prediction_dir = Path(args.prediction_dir) if args.prediction_dir else None

    for raw_path in args.data_path:
        path = Path(raw_path)
        pairs, format_dropped = read_pairs(path)
        records, filtered = build_records(pairs, tokenizer, args.max_seq_length)
        metrics, predictions = predict_records(
            model, tokenizer, records, args.batch_size, device, decoder=decoder
        )
        metrics["dropped"] = {"format": format_dropped, **filtered}
        results[path.stem] = metrics
        if prediction_dir is not None:
            prediction_dir.mkdir(parents=True, exist_ok=True)
            output_path = prediction_dir / f"{path.stem}.pred.jsonl"
            with output_path.open("w", encoding="utf-8", newline="\n") as handle:
                for index, row in enumerate(predictions):
                    handle.write(json.dumps({"id": index, **row}, ensure_ascii=False) + "\n")
        if args.output_path:
            single_predictions = predictions

    payload = {
        "model_path": str(model_dir),
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "method": "relm_disc" if decoder is not None else "relm_baseline",
        "similarity_dir": str(Path(args.similarity_dir).resolve()) if args.similarity_dir else None,
        "alpha": args.alpha,
        "beta": args.beta,
        "copy_punishment": args.copy_punishment,
        "results": results,
    }
    if args.output_path is not None and single_predictions is not None:
        prediction_output = Path(args.output_path)
        prediction_output.parent.mkdir(parents=True, exist_ok=True)
        with prediction_output.open("w", encoding="utf-8", newline="\n") as handle:
            for index, row in enumerate(single_predictions):
                handle.write(json.dumps({"id": index, **row}, ensure_ascii=False) + "\n")

    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
