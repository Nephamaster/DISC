"""Strict local loading helpers for ReLM Hugging Face exports."""

from __future__ import annotations

import os
from pathlib import Path


TOKENIZER_FILES = (
    "tokenizer.json",
    "vocab.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
)
WEIGHT_FILES = (
    "pytorch_model.bin",
    "model.safetensors",
    "pytorch_model.bin.index.json",
    "model.safetensors.index.json",
)


def resolve_local_model_dir(model_path: str | os.PathLike[str], require_weights: bool = False) -> Path:
    """Resolve and validate a local HF export before calling Transformers.

    Without this check, a missing or incorrectly mounted absolute path is
    passed to ``from_pretrained`` and Hugging Face attempts to validate it as
    a Hub repository name, producing a misleading ``HFValidationError``.
    """

    raw = os.path.expandvars(os.path.expanduser(os.fspath(model_path)))
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve(strict=False)
    if not path.is_dir():
        raise FileNotFoundError(
            "Local ReLM checkpoint directory was not found:\n"
            f"  model_path={path}\n"
            f"  cwd={Path.cwd()}\n"
            "Transformers would otherwise treat this value as a Hugging Face repo id. "
            "Check the server mount, checkpoint number, and the trailing hf_model directory."
        )
    if not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"Invalid local ReLM checkpoint: missing config.json under {path}"
        )
    if not any((path / name).is_file() for name in TOKENIZER_FILES):
        raise FileNotFoundError(
            f"Invalid local ReLM checkpoint: no tokenizer file found under {path}; "
            f"expected one of {TOKENIZER_FILES}"
        )
    if require_weights and not any((path / name).is_file() for name in WEIGHT_FILES):
        raise FileNotFoundError(
            f"Invalid local ReLM checkpoint: no model weight file found under {path}; "
            f"expected one of {WEIGHT_FILES}"
        )
    return path


def load_local_tokenizer(model_path: str | os.PathLike[str]):
    """Load a tokenizer only from a validated local directory."""

    from transformers import AutoTokenizer

    path = resolve_local_model_dir(model_path, require_weights=False)
    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        use_fast=True,
        local_files_only=True,
    )
    return tokenizer, path


def load_local_bert(model_path: str | os.PathLike[str]):
    """Load a plain BertForMaskedLM only from a validated local directory."""

    from transformers import BertForMaskedLM

    path = resolve_local_model_dir(model_path, require_weights=True)
    model = BertForMaskedLM.from_pretrained(
        str(path),
        local_files_only=True,
    )
    return model, path
