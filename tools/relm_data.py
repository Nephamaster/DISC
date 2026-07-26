"""Reusable ReLM data helpers copied from the ReLM reproduction.

DISC keeps the helpers small and format-focused: raw conversion is handled by
``prepare_data.py`` and produces JSONL records with ``src``/``tgt`` fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

import torch


class DataFormatError(ValueError):
    """Raised when a JSONL pair cannot be interpreted as a CSC example."""


def read_json_pair(line: str, line_number: int) -> tuple[str, str]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise DataFormatError(f"invalid JSON at line {line_number}: {exc}") from exc
    if not isinstance(record, dict) or "src" not in record or "tgt" not in record:
        raise DataFormatError(f"line {line_number} must contain JSON fields 'src' and 'tgt'")
    source, target = record["src"], record["tgt"]
    if not isinstance(source, str) or not isinstance(target, str):
        raise DataFormatError(f"line {line_number} has non-string src/tgt fields")
    return source, target


def encode_tokens(tokenizer, pieces: Sequence[str]) -> list[int]:
    """Encode already character-tokenized text without adding specials."""

    encoded = tokenizer(
        list(pieces),
        add_special_tokens=False,
        is_split_into_words=True,
        truncation=False,
    )
    return list(encoded["input_ids"])


def encode_text(tokenizer, text: str) -> list[int]:
    return encode_tokens(tokenizer, list(text))


@dataclass
class FeatureStats:
    accepted: int = 0
    invalid_records: int = 0
    empty_records: int = 0
    length_mismatch: int = 0
    token_mismatch: int = 0
    too_long: int = 0
    unknown_tokens: int = 0
    total_tokens: int = 0

    def as_dict(self) -> dict[str, int | float]:
        return {
            "accepted": self.accepted,
            "invalid_records": self.invalid_records,
            "dropped_empty": self.empty_records,
            "dropped_length_mismatch": self.length_mismatch,
            "dropped_token_mismatch": self.token_mismatch,
            "dropped_seq_length": self.too_long,
            "unknown_tokens": self.unknown_tokens,
            "total_tokens": self.total_tokens,
            "unk_rate": self.unknown_tokens / self.total_tokens if self.total_tokens else 0.0,
        }


def make_feature(src: str, tgt: str, tokenizer, max_seq_length: int, stats: FeatureStats | None = None):
    """Build the standard ReLM source-plus-target-mask input.

    This mirrors ReLM's ``tools/relm_data.py``.  It returns ``None`` for a
    record that cannot be represented by the fixed-length ReLM template.
    """

    stats = stats or FeatureStats()
    if not src or not tgt:
        stats.empty_records += 1
        return None
    if len(src) != len(tgt):
        stats.length_mismatch += 1
        return None

    src_ids = encode_text(tokenizer, src)
    tgt_ids = encode_text(tokenizer, tgt)
    stats.unknown_tokens += sum(token_id == tokenizer.unk_token_id for token_id in src_ids + tgt_ids)
    stats.total_tokens += len(src_ids) + len(tgt_ids)
    if len(src_ids) != len(tgt_ids):
        stats.token_mismatch += 1
        return None

    slots = (max_seq_length - 3) // 2
    if len(src_ids) > slots:
        stats.too_long += 1
        return None

    ids = (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id, tokenizer.pad_token_id)
    if any(value is None for value in ids):
        raise ValueError("Tokenizer must define CLS, SEP, MASK and PAD token IDs")
    cls_id, sep_id, mask_id, pad_id = ids
    source_start = 1
    separator_index = source_start + len(src_ids)
    target_start = separator_index + 1
    sequence = [cls_id] + src_ids + [sep_id] + [mask_id] * len(tgt_ids) + [sep_id]
    labels = [-100] * len(sequence)
    labels[target_start : target_start + len(tgt_ids)] = tgt_ids
    attention_mask = [1] * len(sequence)
    padding = max_seq_length - len(sequence)
    sequence += [pad_id] * padding
    attention_mask += [0] * padding
    labels += [-100] * padding
    stats.accepted += 1
    return {
        "input_ids": torch.tensor(sequence, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "source_length": len(src_ids),
        "target_length": len(tgt_ids),
    }
