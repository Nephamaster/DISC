"""Dependency-free sentence-level CSC metrics for the DISC evaluator."""

from __future__ import annotations

import unicodedata


def normalize_width_token(token: str) -> str:
    """Convert one full-width token to its half-width form when safe."""

    normalized = unicodedata.normalize("NFKC", token)
    return normalized if len(normalized) == 1 else token


def normalize_width_tokens(tokens) -> list[str]:
    """Normalize width without expanding the number of prediction slots."""

    return [normalize_width_token(token) for token in tokens]


def sentence_metrics(srcs, tgts, preds) -> dict[str, float | int]:
    """Return the agreed sentence-level correction/detection metrics.

    Correction accuracy is exact ``pred == tgt`` over all sentences. Detection
    accuracy is exact equality of the changed-position sets over all sentences.
    Detection P/R/F1 use exact changed-position-set matches as true positives.
    """

    srcs = [normalize_width_tokens(src) for src in srcs]
    tgts = [normalize_width_tokens(tgt) for tgt in tgts]
    preds = [normalize_width_tokens(pred) for pred in preds]

    total = len(srcs)
    gold_positive = sum(src != tgt for src, tgt in zip(srcs, tgts))
    gold_negative = total - gold_positive
    predicted_positive = sum(src != pred for src, pred in zip(srcs, preds))
    correction_tp = sum(src != tgt and pred == tgt for src, tgt, pred in zip(srcs, tgts, preds))
    correction_fp = sum(src == tgt and pred != src for src, tgt, pred in zip(srcs, tgts, preds))

    detection_tp = 0
    detection_fp = 0
    detection_tn = 0
    detection_predicted_positive = 0
    for src, tgt, pred in zip(srcs, tgts, preds):
        gold_positions = {i for i, (source, target) in enumerate(zip(src, tgt)) if source != target}
        pred_positions = {i for i, (source, prediction) in enumerate(zip(src, pred)) if source != prediction}
        if pred_positions:
            detection_predicted_positive += 1
        if gold_positions and pred_positions == gold_positions:
            detection_tp += 1
        elif not gold_positions and pred_positions:
            detection_fp += 1
        elif not gold_positions and not pred_positions:
            detection_tn += 1

    correction_precision = correction_tp / predicted_positive if predicted_positive else 0.0
    correction_recall = correction_tp / gold_positive if gold_positive else 0.0
    correction_f1 = (
        2 * correction_precision * correction_recall / (correction_precision + correction_recall)
        if correction_precision + correction_recall
        else 0.0
    )
    detection_precision = detection_tp / detection_predicted_positive if detection_predicted_positive else 0.0
    detection_recall = detection_tp / gold_positive if gold_positive else 0.0
    detection_f1 = (
        2 * detection_precision * detection_recall / (detection_precision + detection_recall)
        if detection_precision + detection_recall
        else 0.0
    )
    percent = lambda value: value * 100.0
    return {
        "samples": total,
        "positive_samples": gold_positive,
        "negative_samples": gold_negative,
        "correction_acc": percent(sum(pred == tgt for pred, tgt in zip(preds, tgts)) / total) if total else 0.0,
        "correction_precision": percent(correction_precision),
        "correction_recall": percent(correction_recall),
        "correction_f1": percent(correction_f1),
        "detection_acc": percent((detection_tp + detection_tn) / total) if total else 0.0,
        "detection_precision": percent(detection_precision),
        "detection_recall": percent(detection_recall),
        "detection_f1": percent(detection_f1),
        "fpr": percent(correction_fp / gold_negative) if gold_negative else 0.0,
        "correction_tp": correction_tp,
        "correction_fp": correction_fp,
        "detection_tp": detection_tp,
        "detection_fp": detection_fp,
        "detection_tn": detection_tn,
    }
