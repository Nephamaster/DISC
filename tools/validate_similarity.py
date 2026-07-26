"""Validate a DISC similarity artifact and its tokenizer alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from artifact_utils import tokenizer_hash
except ImportError:  # Supports ``python -m tools.validate_similarity``.
    from tools.artifact_utils import tokenizer_hash

try:
    from hf_local import load_local_tokenizer
except ImportError:  # Supports ``python -m tools.validate_similarity``.
    from tools.hf_local import load_local_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    tokenizer, model_dir = load_local_tokenizer(args.model_path)
    if metadata["vocab_sha256"] != tokenizer_hash(model_dir):
        raise ValueError("Tokenizer/model hash does not match similarity metadata")
    token_ids = metadata["char_token_ids"]
    tokens = metadata["char_tokens"]
    if len(token_ids) != len(set(token_ids)) or len(tokens) != len(set(tokens)):
        raise ValueError("Character IDs or tokens contain duplicates")
    if len(token_ids) != metadata["num_disc_chars"] or len(tokens) != metadata["num_disc_chars"]:
        raise ValueError("Character metadata count is inconsistent")
    current_ids, current_tokens = [], []
    vocab = tokenizer.get_vocab()
    for token_id, token in sorted(((value, key) for key, value in vocab.items()), key=lambda item: item[0]):
        if len(token) == 1 and "\u4e00" <= token <= "\u9fff":
            current_ids.append(token_id)
            current_tokens.append(token)
    if metadata["protocol"] == "paper" and (current_ids != token_ids or current_tokens != tokens):
        raise ValueError("Tokenizer single-CJK token order does not match similarity metadata")

    phonetic = np.load(artifact_dir / "phonetic.npy", mmap_mode="r")
    glyph = np.load(artifact_dir / "glyph.npy", mmap_mode="r")
    if phonetic.shape != glyph.shape or phonetic.shape != (len(token_ids), len(token_ids)):
        raise ValueError("Similarity matrices are not matching square matrices")
    if phonetic.dtype != np.float32 or glyph.dtype != np.float32:
        raise ValueError("Similarity matrices must use float32")
    for name, matrix in (("phonetic", phonetic), ("glyph", glyph)):
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} contains NaN or Inf")
        if float(matrix.min()) < -args.atol or float(matrix.max()) > 1.0 + args.atol:
            raise ValueError(f"{name} contains values outside [0, 1]")
        if not np.allclose(matrix, matrix.T, atol=args.atol):
            raise ValueError(f"{name} is not symmetric")
        if metadata["protocol"] == "paper" and not np.allclose(np.diag(matrix), 1.0, atol=args.atol):
            raise ValueError(f"{name} diagonal is not all 1.0")

    report = {
        "artifact_dir": str(artifact_dir.resolve()),
        "protocol": metadata["protocol"],
        "num_disc_chars": len(token_ids),
        "phonetic_shape": list(phonetic.shape),
        "glyph_shape": list(glyph.shape),
        "valid": True,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
