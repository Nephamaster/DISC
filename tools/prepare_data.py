"""Prepare the four DISC evaluation datasets for ReLM.

The raw files remain authoritative.  This script only converts their container
formats to JSONL records with ``src`` and ``tgt`` fields and drops records whose
source and target have different character lengths.  It does not normalize
Unicode, remove spaces/punctuation, or rewrite the sentence text.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Iterator, TextIO


DATASETS = ("cscd_ns", "rsighan", "lemon", "ecspell")
LEMON_NAMES = ("car", "cot", "enc", "gam", "mec", "new", "nov")
RSIGHAN_YEARS = (13, 14, 15)
ECSPELL_DOMAINS = ("law", "med", "odw")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def read_labeled_pairs(path: Path) -> Iterator[tuple[str, str]]:
    """Read ``label<TAB>source<TAB>target`` files."""

    with open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"Expected 3 tab-separated fields at {path}:{line_number}")
            yield fields[1], fields[2]


def read_json_pairs(path: Path) -> Iterator[tuple[str, str]]:
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            source = row.get("source", row.get("src")) if isinstance(row, dict) else None
            target = row.get("target", row.get("tgt")) if isinstance(row, dict) else None
            if not isinstance(source, str) or not isinstance(target, str):
                raise ValueError(f"Missing string source/target at {path}:{line_number}")
            yield source, target


def read_lemon_pairs(path: Path) -> Iterator[tuple[str, str]]:
    """Read JSONL LEMON records without changing their sentence text."""

    yield from read_json_pairs(path)


def _valid_rows(
    pairs: Iterable[tuple[str, str]],
    dataset: str,
    subset: str,
) -> tuple[list[dict[str, str]], Counter]:
    stats = Counter(input_rows=0, accepted=0, dropped_empty=0, dropped_length_mismatch=0)
    rows: list[dict[str, str]] = []
    for index, (source, target) in enumerate(pairs, 1):
        stats["input_rows"] += 1
        if not source or not target:
            stats["dropped_empty"] += 1
            continue
        if len(source) != len(target):
            stats["dropped_length_mismatch"] += 1
            continue
        rows.append(
            {
                "id": f"{dataset}_{subset}_{index:06d}",
                "dataset": dataset,
                "subset": subset,
                "src": source,
                "tgt": target,
            }
        )
        stats["accepted"] += 1
    return rows, stats


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_one(
    source_path: Path,
    output_path: Path,
    dataset: str,
    subset: str,
    reader: Callable[[Path], Iterator[tuple[str, str]]],
) -> dict:
    rows, stats = _valid_rows(reader(source_path), dataset, subset)
    write_rows(output_path, rows)
    return {
        **dict(stats),
        "dataset": dataset,
        "subset": subset,
        "source": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "output": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
    }


def find_existing(*paths: Path) -> Path:
    for path in paths:
        if path.is_dir():
            return path
    joined = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"None of the expected raw directories exists: {joined}")


def prepare_cscd_ns(raw_dir: Path, output_dir: Path) -> list[dict]:
    source_dir = raw_dir / "cscd_ns"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing CSCD-NS raw directory: {source_dir}")
    results = []
    # ``all.tsv`` is retained as a raw-data audit source, but is deliberately
    # not consumed here: using it would make the formal train/dev/test split
    # ambiguous and could leak the held-out test rows.
    for split in ("train", "dev", "test"):
        source_path = source_dir / f"{split}.tsv"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing CSCD-NS file: {source_path}")
        results.append(
            prepare_one(
                source_path,
                output_dir / "cscd_ns" / f"{split}.jsonl",
                "cscd_ns",
                split,
                read_labeled_pairs,
            )
        )
    return results


def prepare_rsighan(raw_dir: Path, output_dir: Path) -> list[dict]:
    source_dir = find_existing(raw_dir / "rsighan", raw_dir / "sighan")
    results = []
    for year in RSIGHAN_YEARS:
        official = source_dir / f"rSIGHAN{year}.jsonl"
        test = source_dir / f"sighan{year}_test.jsonl"
        train = source_dir / f"sighan{year}_train.jsonl"
        if official.is_file():
            results.append(
                prepare_one(
                    official,
                    output_dir / "rsighan" / f"rSIGHAN{year}.jsonl",
                    "rsighan",
                    f"rSIGHAN{year}",
                    read_json_pairs,
                )
            )
        elif test.is_file():
            results.append(
                prepare_one(
                    test,
                    output_dir / "rsighan" / f"rSIGHAN{year}.jsonl",
                    "rsighan",
                    f"rSIGHAN{year}",
                    read_json_pairs,
                )
            )
            if train.is_file():
                results.append(
                    prepare_one(
                        train,
                        output_dir / "rsighan" / f"train_sighan{year}.jsonl",
                        "rsighan",
                        f"train_sighan{year}",
                        read_json_pairs,
                    )
                )
        else:
            raise FileNotFoundError(f"Missing rSIGHAN{year} raw file under {source_dir}")

    wang = source_dir / "data.jsonl.gz"
    if wang.is_file():
        results.append(
            prepare_one(
                wang,
                output_dir / "rsighan" / "train_wang271k.jsonl",
                "rsighan",
                "train_wang271k",
                read_json_pairs,
            )
        )
    return results


def prepare_lemon(raw_dir: Path, output_dir: Path) -> list[dict]:
    source_dir = find_existing(raw_dir / "lemon", raw_dir / "lemon_v2")
    results = []
    for name in LEMON_NAMES:
        source_path = source_dir / f"{name}.jsonl"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing LEMON raw file: {source_path}")
        results.append(
            prepare_one(
                source_path,
                output_dir / "lemon" / f"{name}.jsonl",
                "lemon",
                name,
                read_lemon_pairs,
            )
        )
    return results


def prepare_ecspell(raw_dir: Path, output_dir: Path) -> list[dict]:
    source_dir = raw_dir / "ecspell"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing ECSpell raw directory: {source_dir}")
    results = []
    for domain in ECSPELL_DOMAINS:
        for split in ("train", "test"):
            source_path = source_dir / f"{domain}_{split}.csv"
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing ECSpell raw file: {source_path}")
            results.append(
                prepare_one(
                    source_path,
                    output_dir / "ecspell" / f"{split}_{domain}.jsonl",
                    "ecspell",
                    f"{split}_{domain}",
                    read_labeled_pairs,
                )
            )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    builders = {
        "cscd_ns": prepare_cscd_ns,
        "rsighan": prepare_rsighan,
        "lemon": prepare_lemon,
        "ecspell": prepare_ecspell,
    }
    results: list[dict] = []
    for dataset in args.datasets:
        results.extend(builders[dataset](raw_dir, output_dir))
    manifest = {
        "raw_dir": str(raw_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "datasets": args.datasets,
        "files": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
