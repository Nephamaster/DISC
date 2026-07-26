"""Paper-protocol character phonetic and glyph similarity functions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from pypinyin import Style, pinyin
from pypinyin.contrib.tone_convert import to_normal


def weighted_edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    """Levenshtein distance with delete/insert/replace costs 1/1/2."""

    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, 1):
        current = [i]
        for j, right_item in enumerate(right, 1):
            delete = previous[j] + 1
            insert = current[j - 1] + 1
            replace = previous[j - 1] + (0 if left_item == right_item else 2)
            current.append(min(delete, insert, replace))
        previous = current
    return previous[-1]


def weighted_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    value = 1.0 - weighted_edit_distance(left, right) / (len(left) + len(right))
    return max(0.0, min(1.0, value))


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for j, right_item in enumerate(right, 1):
            current.append(previous[j - 1] + 1 if left_item == right_item else max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def lcs_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return 0.0
    return lcs_length(left, right) / max(len(left), len(right))


def _read_dict(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class CharFeature:
    char: str
    pinyin: tuple[str, ...]
    four_corner: str
    structure: str
    components: tuple[str, ...]
    structure_four_corner: tuple[str, ...]
    stroke_order: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def _pinyin_values(char: str) -> tuple[str, ...]:
    values = pinyin(char, style=Style.TONE3, heteronym=True, errors="default")[0]
    normalized = {to_normal(value) for value in values if value}
    return tuple(sorted(normalized))


def make_feature(
    char: str,
    four_corner_dict: dict[str, str],
    structure_dict: dict[str, str],
    chaizi_dict: dict[str, list[str]],
    order_dict: dict[str, str],
) -> CharFeature:
    four_corner = str(four_corner_dict.get(char, ""))[:4]
    structure = str(structure_dict.get(char, ""))
    components = tuple([char] if structure == "0" else chaizi_dict.get(char, [char]))
    structure_four_corner = tuple(
        f"{structure}:{str(four_corner_dict.get(component, ''))[:4]}" for component in components
    )
    stroke_order = tuple(str(order_dict.get(char, "")))
    return CharFeature(
        char=char,
        pinyin=_pinyin_values(char),
        four_corner=four_corner,
        structure=structure,
        components=components,
        structure_four_corner=structure_four_corner,
        stroke_order=stroke_order,
    )


def phonetic_similarity(left: CharFeature, right: CharFeature) -> float:
    if left.char == right.char:
        return 1.0
    return max(
        (weighted_similarity(list(left_py), list(right_py)) for left_py in left.pinyin for right_py in right.pinyin),
        default=0.0,
    )


def glyph_components(left: CharFeature, right: CharFeature) -> tuple[float, float, float, float]:
    if left.char == right.char:
        return 1.0, 1.0, 1.0, 1.0
    ordinary_four = 0.0
    if left.four_corner and right.four_corner:
        ordinary_four = sum(a == b for a, b in zip(left.four_corner[:4], right.four_corner[:4])) / 4.0
    structure_four = weighted_similarity(left.structure_four_corner, right.structure_four_corner)
    stroke_edit = weighted_similarity(left.stroke_order, right.stroke_order)
    stroke_lcs = lcs_similarity(left.stroke_order, right.stroke_order)
    return ordinary_four, structure_four, stroke_edit, stroke_lcs


def glyph_similarity(left: CharFeature, right: CharFeature) -> float:
    return sum(glyph_components(left, right)) / 4.0


def load_features(characters: Sequence[str], dict_dir: Path) -> list[CharFeature]:
    four_corner_dict = _read_dict(dict_dir / "fourconer.dict")
    structure_dict = _read_dict(dict_dir / "structure.dict")
    chaizi_dict = _read_dict(dict_dir / "chaizi.dict")
    order_dict = _read_dict(dict_dir / "order.dict")
    return [
        make_feature(char, four_corner_dict, structure_dict, chaizi_dict, order_dict)
        for char in characters
    ]
