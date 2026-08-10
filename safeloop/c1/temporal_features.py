"""Runtime-safe temporal features for the C1 TTC model.

This module deliberately contains no label loader.  ``build_trip_features``
accepts only the lean one-camera frame contract, a cache produced from
``image_2``, camera calibration, and the existing detector/tracker outputs.
Ground-truth TTC is joined later by the training-only code in
``train_temporal``.
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np

from .tracker import MonocularTTCTracker, TrackerConfig
from .types import Detection, TrackRisk


MAX_TRACK_FEATURES = 3
GLOBAL_FEATURE_DIM = 9
TRACK_FEATURE_DIM = 13
FEATURE_DIM = GLOBAL_FEATURE_DIM + MAX_TRACK_FEATURES * TRACK_FEATURE_DIM
FEATURE_NAMES: tuple[str, ...] = (
    "valid",
    "delta_t_s",
    "ego_speed_kmh",
    "ego_longitudinal_accel",
    "ego_lateral_accel",
    "detector_update",
    "detection_count",
    "active_track_count",
    "max_detection_confidence",
) + tuple(
    f"track_{slot}_{name}"
    for slot in range(MAX_TRACK_FEATURES)
    for name in (
        "center_x",
        "center_y",
        "width",
        "height",
        "area",
        "confidence",
        "is_person",
        "is_two_wheeler",
        "is_vehicle",
        "bbox_velocity_x",
        "bbox_velocity_y",
        "log_area_rate",
        "association_age",
    )
)

if len(FEATURE_NAMES) != FEATURE_DIM:  # pragma: no cover - import-time invariant
    raise AssertionError("Temporal feature schema has an inconsistent size")


class LeanInputFrame(Protocol):
    """Narrow runtime frame contract; intentionally excludes every label."""

    frame_id: int
    timestamp: float
    ego: Mapping[str, float]


class LeanTrip(Protocol):
    trip_id: str
    n_frames: int

    def frame(self, frame_id: int) -> LeanInputFrame: ...


@dataclass(frozen=True)
class CameraGeometry:
    width: int
    height: int
    focal_x_px: float
    focal_y_px: float
    principal_x_px: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera width/height must be positive")
        if self.focal_x_px <= 0 or self.focal_y_px <= 0:
            raise ValueError("Camera focal lengths must be positive")


@dataclass(frozen=True)
class DetectionCache:
    trip_id: str
    stride: int
    source_camera: str
    rows: Mapping[int, tuple[Detection, ...]]
    timestamps: Mapping[int, float]

    def __post_init__(self) -> None:
        if self.stride < 1:
            raise ValueError("Detection-cache stride must be >= 1")
        if self.source_camera != "image_2":
            raise ValueError(
                f"C1 temporal features require image_2, got {self.source_camera!r}"
            )


@dataclass(frozen=True)
class TemporalTripFeatures:
    """Label-free features and physics outputs for one complete trip."""

    trip_id: str
    frame_ids: np.ndarray
    timestamps: np.ndarray
    features: np.ndarray
    physics_ttc_s: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.frame_ids)
        if self.frame_ids.ndim != 1 or self.timestamps.shape != (n,):
            raise ValueError("frame_ids and timestamps must be one-dimensional")
        if self.features.shape != (n, FEATURE_DIM):
            raise ValueError(
                f"features must have shape ({n}, {FEATURE_DIM}), got "
                f"{self.features.shape}"
            )
        if self.physics_ttc_s.shape != (n,):
            raise ValueError("physics_ttc_s must have one value per frame")
        if len(np.unique(self.frame_ids)) != n:
            raise ValueError(f"Duplicate frame IDs in runtime features for {self.trip_id}")
        if n and not np.array_equal(self.frame_ids, np.arange(n, dtype=np.int64)):
            raise ValueError(
                f"Runtime features for {self.trip_id} must contain contiguous IDs 0..{n - 1}"
            )
        if not np.isfinite(self.features).all():
            raise ValueError(f"Non-finite temporal features in {self.trip_id}")

    @property
    def n_frames(self) -> int:
        return len(self.frame_ids)


def load_camera_geometry(trip_dir: str | Path) -> CameraGeometry:
    """Read only the public monocular calibration file (never labels/GT)."""

    path = Path(trip_dir) / "kitti" / "calibration_info.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing camera calibration: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    matrix = document["K_left"]
    return CameraGeometry(
        width=int(document["image_width"]),
        height=int(document["image_height"]),
        focal_x_px=float(matrix[0][0]),
        focal_y_px=float(matrix[1][1]),
        principal_x_px=float(matrix[0][2]),
    )


def load_detection_cache(path: str | Path) -> DetectionCache:
    """Load and validate an ``image_2`` cache without touching the dataset."""

    cache_path = Path(path)
    opener = gzip.open if cache_path.suffix == ".gz" else open
    with opener(cache_path, "rt", encoding="utf-8") as stream:
        document = json.load(stream)

    stride = int(document.get("stride", 0))
    rows: dict[int, tuple[Detection, ...]] = {}
    timestamps: dict[int, float] = {}
    for raw_row in document.get("rows") or []:
        frame_id = int(raw_row["frame_id"])
        if frame_id in rows:
            raise ValueError(f"Duplicate detector frame {frame_id} in {cache_path}")
        parsed: list[Detection] = []
        for raw_detection in raw_row.get("detections") or []:
            bbox = tuple(float(value) for value in raw_detection["bbox"])
            if len(bbox) != 4:
                raise ValueError(f"Invalid bbox at frame {frame_id} in {cache_path}")
            parsed.append(
                Detection(
                    class_id=int(raw_detection["class_id"]),
                    label=str(raw_detection["label"]),
                    confidence=float(raw_detection["confidence"]),
                    bbox=bbox,  # type: ignore[arg-type]
                )
            )
        rows[frame_id] = tuple(parsed)
        timestamps[frame_id] = float(raw_row["timestamp"])

    return DetectionCache(
        trip_id=str(document["trip_id"]),
        stride=stride,
        source_camera=str(document.get("source_camera", "")),
        rows=rows,
        timestamps=timestamps,
    )


def baseline_tracker(geometry: CameraGeometry) -> MonocularTTCTracker:
    """Construct the frozen physics-v2 tracker used by the 55.2 baseline."""

    return MonocularTTCTracker(
        TrackerConfig(
            history_size=12,
            min_history=2,
            enable_range_ttc=True,
            min_range_history=3,
            range_recent_observations=6,
            min_range_decreasing_fraction=0.80,
            max_range_slope_relative_mad=0.50,
            min_range_box_height_px=20.0,
        ),
        focal_y_px=geometry.focal_y_px,
        focal_x_px=geometry.focal_x_px,
        principal_x_px=geometry.principal_x_px,
    )


def _class_indicators(label: str) -> tuple[float, float, float]:
    if label == "person":
        return 1.0, 0.0, 0.0
    if label in {"bicycle", "motorcycle"}:
        return 0.0, 1.0, 0.0
    return 0.0, 0.0, 1.0


def _track_features(
    risk: TrackRisk,
    geometry: CameraGeometry,
    association: tuple[float, float, float, float],
) -> tuple[float, ...]:
    x1, y1, x2, y2 = risk.bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    person, two_wheeler, vehicle = _class_indicators(risk.label)
    return (
        ((x1 + x2) * 0.5) / geometry.width,
        ((y1 + y2) * 0.5) / geometry.height,
        width / geometry.width,
        height / geometry.height,
        (width * height) / (geometry.width * geometry.height),
        float(np.clip(risk.confidence, 0.0, 1.0)),
        person,
        two_wheeler,
        vehicle,
        *association,
    )


def frame_feature_vector(
    *,
    timestamp: float,
    previous_timestamp: float | None,
    ego: Mapping[str, float],
    detector_update: bool,
    detection_count: int,
    risks: Sequence[TrackRisk],
    geometry: CameraGeometry,
    association_features: Mapping[int, tuple[float, float, float, float]] | None = None,
) -> np.ndarray:
    """Convert current detector/tracker state into one finite feature row."""

    delta_t = 0.0 if previous_timestamp is None else timestamp - previous_timestamp
    if delta_t < 0.0:
        raise ValueError("Timestamps must be monotonic")
    row = np.zeros(FEATURE_DIM, dtype=np.float32)
    row[:GLOBAL_FEATURE_DIM] = (
        1.0,
        float(np.clip(delta_t, 0.0, 1.0)),
        float(np.clip(float(ego.get("speed_kmh", 0.0)) / 150.0, 0.0, 2.0)),
        float(np.clip(float(ego.get("longitudinal_accel", 0.0)) / 10.0, -2.0, 2.0)),
        float(np.clip(float(ego.get("lateral_accel", 0.0)) / 10.0, -2.0, 2.0)),
        float(detector_update),
        float(np.clip(detection_count / 20.0, 0.0, 1.0)),
        float(np.clip(len(risks) / 20.0, 0.0, 1.0)),
        max((float(risk.confidence) for risk in risks), default=0.0),
    )

    # Sorting uses only current geometry/confidence.  It does not use physics
    # TTC, collision relevance, class-calibrated range or range trends, so the
    # temporal-only ablation remains independent from constants fitted on the
    # six labelled practice trips.
    ranked = sorted(
        risks,
        key=lambda risk: (
            max(0.0, risk.bbox[2] - risk.bbox[0])
            * max(0.0, risk.bbox[3] - risk.bbox[1]),
            risk.confidence,
            -risk.track_id,
        ),
        reverse=True,
    )
    for slot, risk in enumerate(ranked[:MAX_TRACK_FEATURES]):
        start = GLOBAL_FEATURE_DIM + slot * TRACK_FEATURE_DIM
        association = (association_features or {}).get(
            risk.track_id, (0.0, 0.0, 0.0, 0.0)
        )
        row[start : start + TRACK_FEATURE_DIM] = _track_features(
            risk, geometry, association
        )
    if not np.isfinite(row).all():
        raise ValueError("Temporal runtime feature extraction produced non-finite data")
    return row


def _validated_detector_frames(cache: DetectionCache, n_frames: int) -> set[int]:
    expected = set(range(0, n_frames, cache.stride))
    actual = set(cache.rows)
    if actual != expected:
        missing = sorted(expected - actual)[:10]
        unexpected = sorted(actual - expected)[:10]
        raise ValueError(
            f"Incomplete detection cache for {cache.trip_id}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return expected


def _bbox_association_features(
    risks: Sequence[TrackRisk],
    *,
    timestamp: float,
    geometry: CameraGeometry,
    detector_update: bool,
    detector_states: dict[int, tuple[float, float, float, float, int]],
    cached_motion: dict[int, tuple[float, float, float, float]],
) -> Mapping[int, tuple[float, float, float, float]]:
    """Causal bbox motion/persistence from tracker association IDs only."""

    if not detector_update:
        return cached_motion
    for risk in risks:
        x1, y1, x2, y2 = risk.bbox
        center_x = (x1 + x2) * 0.5 / geometry.width
        center_y = (y1 + y2) * 0.5 / geometry.height
        area = max(1.0, (x2 - x1) * (y2 - y1))
        log_area = math.log(area / (geometry.width * geometry.height))
        previous = detector_states.get(risk.track_id)
        velocity_x = velocity_y = area_rate = 0.0
        age = 1
        if previous is not None:
            previous_time, previous_x, previous_y, previous_area, previous_age = previous
            delta_t = timestamp - previous_time
            if delta_t > 1e-4:
                velocity_x = (center_x - previous_x) / delta_t
                velocity_y = (center_y - previous_y) / delta_t
                area_rate = (log_area - previous_area) / delta_t
            age = previous_age + 1
        motion = (
            float(np.clip(velocity_x, -5.0, 5.0)),
            float(np.clip(velocity_y, -5.0, 5.0)),
            float(np.clip(area_rate, -10.0, 10.0)),
            float(np.clip(age / 20.0, 0.0, 1.0)),
        )
        detector_states[risk.track_id] = (
            timestamp,
            center_x,
            center_y,
            log_area,
            age,
        )
        cached_motion[risk.track_id] = motion
    return cached_motion


def _currently_observed_risks(
    risks: Sequence[TrackRisk], detections: Sequence[Detection]
) -> tuple[TrackRisk, ...]:
    """Remove physics-dependent coast tracks from temporal-only features.

    On a detector update, matched/new tracker risks carry the exact bbox of a
    current detection.  An unmatched coast risk retains an older bbox and is
    useful to physics, but its presence can depend on range/lateral TTC.  The
    temporal feature stream therefore keeps only currently observed boxes.
    """

    observed: list[TrackRisk] = []
    for risk in risks:
        if any(
            max(abs(left - right) for left, right in zip(risk.bbox, detection.bbox))
            <= 1e-6
            for detection in detections
        ):
            observed.append(risk)
    return tuple(observed)


def build_trip_features(
    loader: LeanTrip,
    cache: DetectionCache,
    geometry: CameraGeometry,
    *,
    confidence_threshold: float = 0.25,
    tracker: MonocularTTCTracker | None = None,
) -> TemporalTripFeatures:
    """Replay cached detections into label-free causal runtime features."""

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if loader.trip_id != cache.trip_id:
        raise ValueError(
            f"Trip mismatch: lean loader={loader.trip_id}, cache={cache.trip_id}"
        )
    detector_frames = _validated_detector_frames(cache, loader.n_frames)
    active_tracker = tracker or baseline_tracker(geometry)
    active_tracker.reset()

    frame_ids = np.empty(loader.n_frames, dtype=np.int64)
    timestamps = np.empty(loader.n_frames, dtype=np.float64)
    features = np.empty((loader.n_frames, FEATURE_DIM), dtype=np.float32)
    physics_ttc = np.full(loader.n_frames, np.inf, dtype=np.float32)
    previous_timestamp: float | None = None
    detector_states: dict[int, tuple[float, float, float, float, int]] = {}
    cached_motion: dict[int, tuple[float, float, float, float]] = {}
    last_observed_risks: tuple[TrackRisk, ...] = ()

    for index in range(loader.n_frames):
        frame = loader.frame(index)
        frame_id = int(frame.frame_id)
        if frame_id != index:
            raise ValueError(
                f"{loader.trip_id}: expected frame_id={index}, got {frame_id}"
            )
        timestamp = float(frame.timestamp)
        ego = frame.ego or {}
        common = {
            "timestamp": timestamp,
            "image_shape": (geometry.height, geometry.width),
            "ego_speed_kmh": float(ego.get("speed_kmh", 0.0)),
        }
        detector_update = frame_id in detector_frames
        detections: tuple[Detection, ...] = ()
        if detector_update:
            cached_timestamp = cache.timestamps[frame_id]
            if not math.isclose(cached_timestamp, timestamp, abs_tol=1e-3):
                raise ValueError(
                    f"{loader.trip_id} frame {frame_id}: cache timestamp "
                    f"{cached_timestamp} != telemetry timestamp {timestamp}"
                )
            detections = tuple(
                detection
                for detection in cache.rows[frame_id]
                if detection.confidence >= confidence_threshold
            )
            risks = active_tracker.update(detections, **common)
            last_observed_risks = _currently_observed_risks(risks, detections)
        else:
            risks = active_tracker.predict(**common)

        finite_ttc = [
            risk.predicted_ttc_s
            for risk in risks
            if math.isfinite(risk.predicted_ttc_s)
        ]
        frame_ids[index] = frame_id
        timestamps[index] = timestamp
        physics_ttc[index] = min(finite_ttc, default=float("inf"))
        association_features = _bbox_association_features(
            last_observed_risks,
            timestamp=timestamp,
            geometry=geometry,
            detector_update=detector_update,
            detector_states=detector_states,
            cached_motion=cached_motion,
        )
        features[index] = frame_feature_vector(
            timestamp=timestamp,
            previous_timestamp=previous_timestamp,
            ego=ego,
            detector_update=detector_update,
            detection_count=len(detections),
            risks=last_observed_risks,
            geometry=geometry,
            association_features=association_features,
        )
        previous_timestamp = timestamp

    return TemporalTripFeatures(
        trip_id=loader.trip_id,
        frame_ids=frame_ids,
        timestamps=timestamps,
        features=features,
        physics_ttc_s=physics_ttc,
    )


def locate_detection_cache(
    cache_dir: str | Path,
    trip_id: str,
    *,
    pattern: str = "{trip_id}.stride3.conf020.json.gz",
) -> Path:
    """Resolve one explicit cache file and reject ambiguous glob results."""

    rendered = pattern.format(trip_id=trip_id)
    root = Path(cache_dir)
    matches = sorted(root.glob(rendered))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one detection cache for {trip_id} matching "
            f"{root / rendered}, found {len(matches)}"
        )
    return matches[0]


def inverse_ttc(values: np.ndarray, *, minimum_ttc_s: float = 0.1) -> np.ndarray:
    """Stable inverse TTC with ``inf`` mapped to zero."""

    values = np.asarray(values, dtype=np.float32)
    out = np.zeros_like(values, dtype=np.float32)
    finite = np.isfinite(values)
    out[finite] = 1.0 / np.maximum(values[finite], minimum_ttc_s)
    return out


def concatenate_frames(
    trips: Iterable[TemporalTripFeatures],
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience helper used to fit train-only normalization."""

    materialized = list(trips)
    if not materialized:
        raise ValueError("At least one trip is required")
    return (
        np.concatenate([trip.features for trip in materialized], axis=0),
        np.concatenate([trip.physics_ttc_s for trip in materialized], axis=0),
    )
