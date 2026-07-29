"""Challenge 2 driver state — Phase 1.

Per-frame classification into `alert | drowsy | yawning | distracted |
microsleep` from the cabin camera: MediaPipe face landmarks for the eyes and
mouth, a YOLO11 COCO detector for the phone, and thresholds over a time window.

    python -m drive_state.phase_1.cli evaluate --dataset data
    python -m drive_state.phase_1.cli demo --trip data/T01-Sample

Scores 97.5 leave-one-trip-out on the six Practice trips (99.4 fitted), and
94.8 running causally with a trailing window. `docs/drive_state.md` has the
per-trip numbers and the reasoning behind each threshold.

Replays through `tripkit.TripReplayer` when available, falling back to this
package's own loader otherwise.
"""

from drive_state.phase_1.classifier import (
    ClassifierConfig,
    classify_sequence,
    predict_trip,
)
from drive_state.phase_1.practice import Trip, discover_trips, load_trip
from drive_state.phase_1.scoring import overall_composite, score_trip
from drive_state.phase_1.states import (
    DRIVER_STATE_CLASSES,
    EYE_STATES,
    HEAD_POSES,
    MOUTH_STATES,
    state_from_signals,
)

__all__ = [
    "DRIVER_STATE_CLASSES",
    "EYE_STATES",
    "HEAD_POSES",
    "MOUTH_STATES",
    "ClassifierConfig",
    "Trip",
    "classify_sequence",
    "discover_trips",
    "load_trip",
    "overall_composite",
    "predict_trip",
    "score_trip",
    "state_from_signals",
]
