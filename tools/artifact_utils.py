"""Small hashing helpers shared by DISC artifact builders and consumers."""

from __future__ import annotations

import hashlib
from pathlib import Path


TOKENIZER_FILE_NAMES = (
    "vocab.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "sentencepiece.bpe.model",
    "spiece.model",
    "merges.txt",
)


def hash_path(path: str | Path) -> str:
    """Hash a file/directory deterministically, including relative names."""

    path = Path(path)
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(
        item for item in path.rglob("*") if item.is_file()
    )
    for item in paths:
        digest.update(
            str(item.relative_to(path.parent if path.is_file() else path)).encode("utf-8")
        )
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def tokenizer_hash(model_path: str | Path) -> str:
    """Hash tokenizer/config files while ignoring model weights.

    A similarity matrix depends on the tokenizer vocabulary and special-token
    configuration, not on fine-tuned model weights. This permits one artifact
    to be reused by checkpoints that share the same tokenizer.
    """

    path = Path(model_path)
    if path.is_file():
        return hash_path(path)
    files = [path / name for name in TOKENIZER_FILE_NAMES if (path / name).is_file()]
    config = path / "config.json"
    if config.is_file():
        files.append(config)
    if not files:
        raise FileNotFoundError(f"No tokenizer files found under {path}")
    digest = hashlib.sha256()
    for item in sorted(files):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()
