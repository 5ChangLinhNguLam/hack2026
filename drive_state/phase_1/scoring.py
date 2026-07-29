"""The official Challenge 2 metric, reimplemented.

Mirrors ``package_starterkit/team_kit/evaluation.py::compute_challenge2_metrics``
so that a local number means the same thing as a leaderboard number::

    composite = 100 * (0.5 * accuracy + 0.5 * macro_f1)

The one detail that is easy to get wrong: `macro_f1` averages per-class F1 only
over the classes **present in that trip's ground truth**. Predicting a class
the trip does not contain adds no zero to the macro average — it only costs
accuracy and the true class's recall. Most trips hold one or two states, so
that asymmetry is worth a lot; it is why the rules are ordered by risk but the
thresholds sit conservative.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from drive_state.phase_1.states import DRIVER_STATE_CLASSES


@dataclass(slots=True)
class TripScore:
    trip_id: str
    n_frames: int
    accuracy: float
    macro_f1: float
    composite: float
    per_class_f1: dict[str, float] = field(default_factory=dict)
    present_classes: tuple[str, ...] = ()
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "trip_id": self.trip_id,
            "n_frames": self.n_frames,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "composite": round(self.composite, 1),
            "per_class_f1": {k: round(v, 4) for k, v in self.per_class_f1.items()},
            "present_classes": list(self.present_classes),
        }


def score_trip(
    trip_id: str,
    predicted: Mapping[int, str],
    truth: Mapping[int, str],
) -> TripScore:
    """Score one trip over the frames present in both mappings."""
    frame_ids = sorted(set(predicted) & set(truth))
    if not frame_ids:
        raise ValueError(f"No overlapping frame ids for {trip_id}")

    pairs = [(predicted[fid], truth[fid]) for fid in frame_ids]
    n = len(pairs)
    accuracy = sum(1 for p, g in pairs if p == g) / n

    per_class_f1: dict[str, float] = {}
    for cls in DRIVER_STATE_CLASSES:
        tp = sum(1 for p, g in pairs if p == cls and g == cls)
        fp = sum(1 for p, g in pairs if p == cls and g != cls)
        fn = sum(1 for p, g in pairs if p != cls and g == cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class_f1[cls] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )

    present = tuple(c for c in DRIVER_STATE_CLASSES if any(g == c for _, g in pairs))
    macro_f1 = sum(per_class_f1[c] for c in present) / len(present) if present else 0.0

    confusion: dict[tuple[str, str], int] = {}
    for p, g in pairs:
        confusion[(g, p)] = confusion.get((g, p), 0) + 1

    return TripScore(
        trip_id=trip_id,
        n_frames=n,
        accuracy=accuracy,
        macro_f1=macro_f1,
        composite=100.0 * (0.5 * accuracy + 0.5 * macro_f1),
        per_class_f1=per_class_f1,
        present_classes=present,
        confusion=confusion,
    )


def overall_composite(scores: list[TripScore]) -> float:
    """The leaderboard figure: the unweighted mean of per-trip composites.

    Unweighted, so a 600-frame trip counts as much as an 1,800-frame one --
    that is what the organiser's report does.
    """
    return sum(s.composite for s in scores) / len(scores) if scores else 0.0


def format_confusion(scores: list[TripScore]) -> str:
    """Pooled truth-vs-prediction table across trips, as aligned text."""
    pooled: dict[tuple[str, str], int] = {}
    for score in scores:
        for key, count in score.confusion.items():
            pooled[key] = pooled.get(key, 0) + count

    width = max(len(c) for c in DRIVER_STATE_CLASSES) + 2
    header = " " * (width + 8) + "".join(f"{c:>{width}}" for c in DRIVER_STATE_CLASSES)
    lines = [header, " " * (width + 8) + "-" * (width * len(DRIVER_STATE_CLASSES))]
    for truth in DRIVER_STATE_CLASSES:
        total = sum(pooled.get((truth, p), 0) for p in DRIVER_STATE_CLASSES)
        if total == 0:
            continue
        cells = "".join(f"{pooled.get((truth, p), 0):>{width}}" for p in DRIVER_STATE_CLASSES)
        lines.append(f"true {truth:<{width}} |{cells}   (n={total})")
    return "\n".join(lines)
