"""Scoring a threshold set against the Practice trips.

Two numbers come out of here and they mean different things:

* **fitted** — thresholds chosen on all six trips, then scored on all six. This
  is the optimistic number. Six trips over six subjects is far too small to
  fit on honestly, and three of the six hold one state for their whole
  duration.
* **leave-one-trip-out** — thresholds refitted with one trip withheld, then
  scored on that trip, repeated six times. Nothing a trip contributed can
  influence its own score, so this is the number to quote.

Report both, and expect the gap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from drive_state.phase_1.classifier import TUNABLE_KNOBS, ClassifierConfig, predict_trip
from drive_state.phase_1.features import FrameFeatures, read_features_csv
from drive_state.phase_1.practice import Trip
from drive_state.phase_1.scoring import TripScore, overall_composite, score_trip


@dataclass(slots=True)
class TripData:
    """A trip's ground truth paired with its cached features."""

    trip: Trip
    features: list[FrameFeatures]

    @property
    def trip_id(self) -> str:
        return self.trip.trip_id


def load_trip_data(trips: Sequence[Trip], features_dir: str | Path) -> list[TripData]:
    features_dir = Path(features_dir)
    out = []
    for trip in trips:
        path = features_dir / f"{trip.trip_id}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"No cached features for {trip.trip_id}: {path}. Run `challenge extract` first."
            )
        out.append(TripData(trip=trip, features=read_features_csv(path)))
    return out


def evaluate(data: Sequence[TripData], config: ClassifierConfig) -> list[TripScore]:
    """Score one fixed threshold set across every trip."""
    return [
        score_trip(item.trip_id, predict_trip(item.features, config), item.trip.states())
        for item in data
    ]


@dataclass(slots=True)
class EvaluationReport:
    per_trip: list[TripScore]
    overall: float
    config: ClassifierConfig

    def to_dict(self) -> dict[str, object]:
        return {
            "overall_composite": round(self.overall, 1),
            "per_trip": [s.to_dict() for s in self.per_trip],
            "config": {k: getattr(self.config, k) for k in TUNABLE_KNOBS},
        }


def evaluate_report(data: Sequence[TripData], config: ClassifierConfig) -> EvaluationReport:
    scores = evaluate(data, config)
    return EvaluationReport(per_trip=scores, overall=overall_composite(scores), config=config)
