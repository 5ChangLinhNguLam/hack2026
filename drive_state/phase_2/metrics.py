"""Dependency-light classification metrics used by training and EDA."""

from __future__ import annotations

from typing import Sequence

from .data.eye_sequences import IGNORE_INDEX, PHASE_NAMES


def phase_classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, object]:
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have equal length")
    pairs = [
        (int(target), int(prediction))
        for target, prediction in zip(targets, predictions, strict=True)
        if int(target) != ignore_index
    ]
    confusion = [[0 for _ in PHASE_NAMES] for _ in PHASE_NAMES]
    for target, prediction in pairs:
        if target not in range(len(PHASE_NAMES)) or prediction not in range(len(PHASE_NAMES)):
            raise ValueError("phase target and prediction values must be in [0, 3]")
        confusion[target][prediction] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    f1_scores: list[float] = []
    for index, name in enumerate(PHASE_NAMES):
        true_positive = confusion[index][index]
        false_positive = sum(confusion[row][index] for row in range(len(PHASE_NAMES))) - true_positive
        false_negative = sum(confusion[index]) - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(confusion[index])
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_scores.append(f1)

    transition_scores = [
        float(per_class[name]["f1"]) for name in ("closing", "opening")
    ]
    return {
        "support": len(pairs),
        "confusion_matrix": confusion,
        "per_class": per_class,
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "transition_macro_f1": sum(transition_scores) / len(transition_scores),
    }
