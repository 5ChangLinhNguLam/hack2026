"""Trip folder -> submission CSV.

The submission format is ``frame_id,timestamp,predicted_driver_state``, one row
per frame. Columns belonging to the other challenges are omitted entirely
rather than filled with placeholders — the organiser's parser treats a blank
`predicted_driver_state` as "not attempted", and junk in a column it does read
would be scored.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from drive_state.phase_1.classifier import ClassifierConfig, predict_trip
from drive_state.phase_1.extract import extract_trip
from drive_state.phase_1.features import FrameFeatures, read_features_csv
from drive_state.phase_1.practice import Trip, load_trip

SUBMISSION_FIELDS = ("frame_id", "timestamp", "predicted_driver_state")


def write_submission(
    path: str | Path,
    rows: Sequence[FrameFeatures],
    states: dict[int, str],
    fps: float,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUBMISSION_FIELDS)
        for row in rows:
            writer.writerow([row.frame_id, f"{row.frame_id / fps:.3f}", states[row.frame_id]])
    return path


def predict_trip_folder(
    trip_root: str | Path,
    out_path: str | Path,
    *,
    config: ClassifierConfig | None = None,
    features_dir: str | Path | None = None,
    model_path: str | Path = "models/face_landmarker.task",
) -> Path:
    """Predict one trip, reusing cached features when they exist.

    `features_dir` is both the cache to read and the place a fresh extraction
    is written, so repeated runs over the same trip skip landmarking.
    """
    trip: Trip = load_trip(trip_root)
    config = config or ClassifierConfig()

    cached = Path(features_dir) / f"{trip.trip_id}.csv" if features_dir else None
    if cached is not None and cached.exists():
        rows = read_features_csv(cached)
    else:
        target = cached.parent if cached else Path(out_path).parent / "features"
        rows = read_features_csv(extract_trip(trip, target, model_path=model_path))

    return write_submission(out_path, rows, predict_trip(rows, config), trip.fps)
