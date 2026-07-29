"""Batch feature extraction: Practice trips -> one features CSV per trip."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from drive_state.phase_1.features import (
    FaceFeatureExtractor,
    FrameFeatures,
    write_features_csv,
)
from drive_state.phase_1.phone import DEFAULT_LABELS, DEFAULT_MODEL, create_phone_detector
from drive_state.phase_1.practice import Trip, discover_trips, imread_bgr, load_trip


def extract_trip(
    trip: Trip,
    out_dir: str | Path,
    *,
    model_path: str | Path = "models/face_landmarker.task",
    phone_model: str | Path | None = DEFAULT_MODEL,
    phone_labels: str | Path = DEFAULT_LABELS,
    phone_stride: int = 5,
    progress: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Landmark every frame of `trip` and write `<out_dir>/<trip_id>.csv`.

    `phone_model=None` disables phone detection, which costs roughly 20
    composite points — see `phone.py`. It exists for running on a machine
    without the ONNX model, not as a default worth choosing.
    """
    paths = trip.frame_paths()
    rows: list[FrameFeatures] = []
    phone_detector = create_phone_detector(phone_model, phone_labels, stride=phone_stride)

    with FaceFeatureExtractor(model_path) as extractor:
        for index, path in enumerate(paths):
            frame_id = int(path.stem.split("_")[-1])
            image = imread_bgr(path)
            timestamp = frame_id / trip.fps
            phone = (
                phone_detector.detect(image, frame_id, timestamp)
                if phone_detector is not None
                else None
            )
            rows.append(extractor.extract(image, frame_id, timestamp, phone))
            if progress is not None and (index % 50 == 0 or index == len(paths) - 1):
                progress(trip.trip_id, index + 1, len(paths))

    return write_features_csv(Path(out_dir) / f"{trip.trip_id}.csv", rows)


def extract_dataset(
    dataset_root: str | Path,
    out_dir: str | Path,
    *,
    model_path: str | Path = "models/face_landmarker.task",
    phone_model: str | Path | None = DEFAULT_MODEL,
    phone_labels: str | Path = DEFAULT_LABELS,
    phone_stride: int = 5,
    trip_ids: list[str] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> list[Path]:
    root = Path(dataset_root)
    trips = (
        [load_trip(root / trip_id) for trip_id in trip_ids] if trip_ids else discover_trips(root)
    )
    return [
        extract_trip(
            trip,
            out_dir,
            model_path=model_path,
            phone_model=phone_model,
            phone_labels=phone_labels,
            phone_stride=phone_stride,
            progress=progress,
        )
        for trip in trips
    ]
