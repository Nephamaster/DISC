"""Build resumable paper-protocol DISC similarity matrices."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
try:
    from disc_similarity import CharFeature, glyph_similarity, load_features, phonetic_similarity
except ImportError:  # Supports ``python -m tools.build_disc_similarity``.
    from tools.disc_similarity import CharFeature, glyph_similarity, load_features, phonetic_similarity

try:
    from artifact_utils import hash_path, tokenizer_hash
except ImportError:  # Supports ``python -m tools.build_disc_similarity``.
    from tools.artifact_utils import hash_path, tokenizer_hash

try:
    from hf_local import load_local_tokenizer
except ImportError:  # Supports ``python -m tools.build_disc_similarity``.
    from tools.hf_local import load_local_tokenizer


_WORKER_FEATURES: list[CharFeature] | None = None
_WORKER_PROTOCOL: str = "paper"


def _init_worker(features: list[CharFeature], protocol: str) -> None:
    global _WORKER_FEATURES, _WORKER_PROTOCOL
    _WORKER_FEATURES = features
    _WORKER_PROTOCOL = protocol


def _compute_block(task: tuple[int, int]) -> tuple[int, np.ndarray, np.ndarray]:
    start, end = task
    if _WORKER_FEATURES is None:
        raise RuntimeError("Similarity worker was not initialized")
    features = _WORKER_FEATURES
    size = len(features)
    phonetic = np.zeros((end - start, size), dtype=np.float32)
    glyph = np.zeros((end - start, size), dtype=np.float32)
    for local, i in enumerate(range(start, end)):
        for j in range(i, size):
            if i == j:
                phonetic[local, j] = 1.0
                glyph[local, j] = 1.0
            else:
                phonetic[local, j] = phonetic_similarity(features[i], features[j])
                glyph[local, j] = glyph_similarity(features[i], features[j])
    return start, phonetic, glyph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dict_dir", default="char-similarity-calculation/dict")
    parser.add_argument("--output_dir", default="artifacts/similarity/paper")
    parser.add_argument("--protocol", choices=("paper", "repo_compat"), default="paper")
    parser.add_argument("--dtype", choices=("float32",), default="float32")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def tokenizer_characters(tokenizer, protocol: str) -> tuple[list[int], list[str]]:
    vocab = tokenizer.get_vocab()
    if protocol == "repo_compat":
        ids = list(range(670, min(7992, len(tokenizer))))
        tokens = tokenizer.convert_ids_to_tokens(ids)
        return ids, tokens
    entries = sorted(((token_id, token) for token, token_id in vocab.items()), key=lambda item: item[0])
    selected = [
        (token_id, token)
        for token_id, token in entries
        if len(token) == 1 and "\u4e00" <= token <= "\u9fff"
    ]
    return [item[0] for item in selected], [item[1] for item in selected]


def save_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def open_matrix(path: Path, shape: tuple[int, int], resume: bool) -> np.memmap:
    if path.exists() and not resume:
        path.unlink()
    if path.exists():
        matrix = np.lib.format.open_memmap(path, mode="r+")
        if matrix.shape != shape or matrix.dtype != np.float32:
            raise ValueError(f"Existing matrix has incompatible shape/dtype: {path}")
        return matrix
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    matrix[:] = 0.0
    matrix.flush()
    return matrix


def mirror_upper(matrix: np.memmap) -> None:
    for row in range(matrix.shape[0]):
        matrix[row + 1 :, row] = matrix[row, row + 1 :]
    matrix.flush()


def apply_repo_compat_diagonal(matrix: np.memmap) -> None:
    for row in range(matrix.shape[0]):
        values = np.concatenate((matrix[row, :row], matrix[row, row + 1 :]))
        matrix[row, row] = float(values.max()) if values.size else 0.0
    matrix.flush()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.block_size < 1:
        raise ValueError("--workers and --block_size must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dict_dir = Path(args.dict_dir)
    tokenizer, model_dir = load_local_tokenizer(args.model_path)
    token_ids, characters = tokenizer_characters(tokenizer, args.protocol)
    if len(token_ids) != len(set(token_ids)):
        raise ValueError("Tokenizer character IDs are not unique")
    if len(characters) != len(set(characters)):
        raise ValueError("Tokenizer character tokens are not unique")

    metadata_path = output_dir / "metadata.json"
    feature_path = output_dir / "features.json"
    state_path = output_dir / "build_state.json"
    tokenizer_digest = tokenizer_hash(model_dir)
    metadata = {
        "protocol": args.protocol,
        "tokenizer_path": str(model_dir),
        "tokenizer_sha256": tokenizer_digest,
        "vocab_sha256": tokenizer_digest,
        "num_vocab_tokens": len(tokenizer),
        "num_disc_chars": len(characters),
        "char_token_ids": token_ids,
        "char_tokens": characters,
        "dict_dir": str(dict_dir.resolve()),
        "dict_sha256": hash_path(dict_dir),
        "dtype": args.dtype,
        "self_similarity": 1.0 if args.protocol == "paper" else "row_max_except_self",
    }

    if args.resume and metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("protocol", "vocab_sha256", "num_disc_chars", "char_token_ids", "dict_sha256"):
            if previous.get(key) != metadata.get(key):
                raise ValueError(f"Resume metadata mismatch for {key}")
    else:
        save_json(metadata_path, metadata)

    if args.resume and feature_path.exists():
        feature_payload = json.loads(feature_path.read_text(encoding="utf-8"))
        features = [CharFeature(**item) for item in feature_payload]
        if [feature.char for feature in features] != characters:
            raise ValueError("Feature cache character order does not match tokenizer")
    else:
        features = load_features(characters, dict_dir)
        save_json(feature_path, [feature.as_dict() for feature in features])

    size = len(features)
    shape = (size, size)
    phonetic_path = output_dir / "phonetic.npy"
    glyph_path = output_dir / "glyph.npy"
    phonetic = open_matrix(phonetic_path, shape, args.resume)
    glyph = open_matrix(glyph_path, shape, args.resume)
    block_count = math.ceil(size / args.block_size)
    completed: set[int] = set()
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("shape") != list(shape) or state.get("block_size") != args.block_size:
            raise ValueError("Resume build state shape/block size mismatch")
        completed = set(state.get("completed_blocks", []))

    tasks = [
        (block, block * args.block_size, min(size, (block + 1) * args.block_size))
        for block in range(block_count)
        if block not in completed
    ]

    def consume(result: tuple[int, np.ndarray, np.ndarray]) -> None:
        start, phonetic_block, glyph_block = result
        phonetic[start : start + len(phonetic_block)] = phonetic_block
        glyph[start : start + len(glyph_block)] = glyph_block
        phonetic.flush()
        glyph.flush()
        completed.add(start // args.block_size)
        save_json(
            state_path,
            {
                "shape": list(shape),
                "block_size": args.block_size,
                "completed_blocks": sorted(completed),
                "total_blocks": block_count,
            },
        )

    if args.workers == 1:
        _init_worker(features, args.protocol)
        iterator = (_compute_block((start, end)) for _, start, end in tasks)
        for result in tqdm(iterator, total=len(tasks), desc="build similarity", unit="block"):
            consume(result)
    else:
        context = __import__("multiprocessing").get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(features, args.protocol),
        ) as executor:
            iterator = executor.map(_compute_block, [(start, end) for _, start, end in tasks])
            for result in tqdm(iterator, total=len(tasks), desc="build similarity", unit="block"):
                consume(result)

    mirror_upper(phonetic)
    mirror_upper(glyph)
    if args.protocol == "repo_compat":
        apply_repo_compat_diagonal(phonetic)
        apply_repo_compat_diagonal(glyph)
    else:
        np.fill_diagonal(phonetic, 1.0)
        np.fill_diagonal(glyph, 1.0)
    phonetic.flush()
    glyph.flush()
    metadata["completed_blocks"] = block_count
    save_json(metadata_path, metadata)
    print(json.dumps({"output_dir": str(output_dir.resolve()), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
