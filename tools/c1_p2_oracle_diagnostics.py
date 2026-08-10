#!/usr/bin/env python3
"""Offline-only C1 failure analysis and oracle ladder.

This program is deliberately kept under ``tools/`` rather than the
``safeloop.c1`` runtime package.  It reads practice-only fields such as
``targets``, KITTI ``label_2`` and ground-truth TTC, so importing it from an
inference path would be a data leak.  The production detector/tracker never
imports this file.

The five ladder rows are diagnostic interventions, not deployable models:

1. frozen detector + tracker prediction;
2. current detections split into ground-truth identity-specific tracker banks
   (oracle association; unmatched detector tracks are retained);
3. current detector/tracker with only the ground-truth collision target chosen
   (oracle target selection);
4. projected KITTI cuboids at the frozen detector cadence + current tracker;
5. causal finite-difference TTC from privileged target range history.  This
   rung never reads target velocity or any TTC label to make a prediction.

All label-bearing inputs remain inside this process and every generated
prediction is written below an ignored output directory (``runs/c1`` by
default).  The official evaluator is only hashed and imported for its public
metric function; it is never edited.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safeloop.c1.prediction_preflight import preflight_prediction_csv
from safeloop.c1.temporal_features import (
    CameraGeometry,
    baseline_tracker,
    load_camera_geometry,
    load_detection_cache,
)
from safeloop.c1.tracker import MonocularTTCTracker
from safeloop.c1.types import BBox, Detection, TrackRisk
from team_kit.evaluation import compute_trip_metrics


EXPECTED_FRAMES = 600
DEFAULT_TRIPS = tuple(f"T0{index}-Sample" for index in range(1, 7))
DEFAULT_DIAGNOSTIC_TRIPS = ("T01-Sample", "T02-Sample", "T06-Sample")
LADDER_VARIANTS = (
    "detector_tracker_current",
    "detector_oracle_association",
    "detector_oracle_target_selection",
    "oracle_projected_bbox_tracker",
    "oracle_range_upper_bound",
)
SUPPLEMENTAL_VARIANTS = ("current_selection_oracle_range",)
FAILURE_FLAGS = (
    "detector_missed_object",
    "projection_unmatched_ambiguous",
    "detection_without_track",
    "association_corruption",
    "identity_switch",
    "track_fragmentation",
    "track_coasting",
    "track_lost",
    "wrong_collision_target",
    "selected_target_mapping_unknown",
    "range_or_ttc_jump",
    "gt_mae_critical_prediction_not_critical",
    "gt_critical_prediction_not_critical",
    "prediction_critical_false",
)
OFFICIAL_EVALUATOR = ROOT / "team_kit" / "evaluation.py"
EXPECTED_EVALUATOR_SHA256 = (
    "674320460240797c2cb3a2b60815629c1cc1dcdcd9cc17ad52988c3dd353065b"
)


@dataclass(frozen=True)
class KittiObject:
    """One practice-only KITTI object with a projected image_2 box."""

    target_id: int
    kitti_label: str
    runtime_label: str
    bbox: BBox
    height_m: float
    width_m: float
    length_m: float
    location_x_m: float
    location_z_m: float
    longitudinal_distance_m: float
    lateral_distance_m: float
    in_collision_cone: bool


@dataclass(frozen=True)
class OfflineFrame:
    frame_id: int
    timestamp: float
    ego_speed_kmh: float
    ground_truth_ttc_s: float
    targets: tuple[Mapping[str, Any], ...]
    objects: tuple[KittiObject, ...]


@dataclass(frozen=True)
class Assignment:
    detection_index: int
    object_index: int
    iou: float


@dataclass(frozen=True)
class ReplayRow:
    frame_id: int
    timestamp: float
    ground_truth_ttc_s: float
    baseline_ttc_s: float
    current_replay_ttc_s: float
    oracle_association_ttc_s: float
    oracle_target_ttc_s: float
    oracle_bbox_ttc_s: float
    oracle_range_ttc_s: float
    current_selection_oracle_range_ttc_s: float
    oracle_target_id: int | None
    selected_track_id: int | None
    selected_target_id: int | None
    selected_range_m: float
    detector_update: bool
    detector_target_seen: bool
    detector_target_seen_iou_005: bool
    detector_target_seen_iou_020: bool
    projected_object_count: int
    matched_object_count_iou_005: int
    matched_object_count_iou_010: int
    matched_object_count_iou_020: int
    projection_ambiguous_object_count: int
    matched_class_pairs: str
    target_detection_count: int
    target_track_id: int | None
    target_track_missed_updates: int | None
    detector_missed_object: bool
    projection_unmatched_ambiguous: bool
    detection_without_track: bool
    association_corruption: bool
    identity_switch: bool
    track_fragmentation: bool
    track_coasting: bool
    track_lost: bool
    wrong_collision_target: bool
    selected_target_mapping_unknown: bool
    range_jump: bool
    ttc_jump: bool
    ttc_finite_toggle: bool
    range_or_ttc_jump: bool
    gt_mae_critical_prediction_not_critical: bool
    gt_critical_prediction_not_critical: bool
    prediction_critical_false: bool


def _finite_number(value: Any, *, default: float = float("inf")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if not math.isnan(result) else default


def _open_trip_document(trip_dir: Path) -> Mapping[str, Any]:
    """Read the full practice JSON only inside this offline tool."""

    trip_id = trip_dir.name
    compressed = trip_dir / f"{trip_id}.json.gz"
    plain = trip_dir / f"{trip_id}.json"
    if compressed.is_file():
        with gzip.open(compressed, "rt", encoding="utf-8") as stream:
            document = json.load(stream)
    elif plain.is_file():
        with plain.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    else:
        # Some extracted kits contain a directory named ``<trip>.json``.
        # Never mistake that shadow directory for a JSON file.
        raise FileNotFoundError(
            f"Missing practice JSON file for {trip_id}; checked {compressed} and {plain}"
        )
    if not isinstance(document, Mapping):
        raise ValueError(f"{trip_id}: trip document must be an object")
    return document


def parse_kitti_label(line: str) -> tuple[str, float, float, float, float, float, float, float]:
    """Parse the fields needed to project a standard KITTI label line."""

    fields = line.split()
    if len(fields) < 15:
        raise ValueError(f"Malformed KITTI label: {line!r}")
    label = fields[0]
    height, width, length = (float(value) for value in fields[8:11])
    location_x, location_y, location_z = (float(value) for value in fields[11:14])
    rotation_y = float(fields[14])
    if min(height, width, length) <= 0.0:
        raise ValueError(f"Non-positive KITTI cuboid dimensions: {line!r}")
    return (
        label,
        height,
        width,
        length,
        location_x,
        location_y,
        location_z,
        rotation_y,
    )


def project_kitti_cuboid(
    line: str,
    projection: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> tuple[str, BBox] | None:
    """Project a KITTI 3-D cuboid into a clipped image_2 bounding box."""

    (
        label,
        height,
        width,
        length,
        x,
        y,
        z,
        rotation_y,
    ) = parse_kitti_label(line)
    if projection.shape != (3, 4):
        raise ValueError(f"Projection matrix must be 3x4, got {projection.shape}")

    x_corners = np.asarray(
        [length / 2, length / 2, -length / 2, -length / 2] * 2,
        dtype=np.float64,
    )
    y_corners = np.asarray([0.0] * 4 + [-height] * 4, dtype=np.float64)
    z_corners = np.asarray(
        [width / 2, -width / 2, -width / 2, width / 2] * 2,
        dtype=np.float64,
    )
    cosine, sine = math.cos(rotation_y), math.sin(rotation_y)
    rotation = np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )
    corners = rotation @ np.vstack((x_corners, y_corners, z_corners))
    corners += np.asarray([[x], [y], [z]], dtype=np.float64)
    if np.any(corners[2] <= 1e-3):
        return None
    homogeneous = np.vstack((corners, np.ones((1, corners.shape[1]))))
    pixels = projection @ homogeneous
    pixels[:2] /= pixels[2:3]
    x1 = max(0.0, float(np.min(pixels[0])))
    y1 = max(0.0, float(np.min(pixels[1])))
    x2 = min(float(image_width - 1), float(np.max(pixels[0])))
    y2 = min(float(image_height - 1), float(np.max(pixels[1])))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return label, (x1, y1, x2, y2)


def _runtime_label(kitti_label: str) -> tuple[int, str]:
    mapping = {
        "Pedestrian": (0, "person"),
        "Person_sitting": (0, "person"),
        "Cyclist": (1, "bicycle"),
        "Car": (2, "car"),
        "Van": (2, "car"),
        "Truck": (7, "truck"),
        "Tram": (5, "bus"),
    }
    return mapping.get(kitti_label, (2, "car"))


def _bbox_iou(left: BBox, right: BBox) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def match_detections_to_objects(
    detections: Sequence[Detection],
    objects: Sequence[KittiObject],
    *,
    min_iou: float = 0.10,
) -> tuple[Assignment, ...]:
    """Maximum-IoU one-to-one assignment used only to construct the oracle."""

    if not detections or not objects:
        return ()
    scores = np.asarray(
        [[_bbox_iou(detection.bbox, obj.bbox) for obj in objects] for detection in detections],
        dtype=np.float64,
    )
    try:
        from scipy.optimize import linear_sum_assignment

        detection_indices, object_indices = linear_sum_assignment(scores, maximize=True)
        candidates = zip(detection_indices.tolist(), object_indices.tolist())
    except (ImportError, TypeError):  # pragma: no cover - old/minimal environments
        # The locked training environment has SciPy.  This deterministic
        # fallback keeps geometry/unit tests usable in the lean baseline env.
        ranked = sorted(
            (
                (float(scores[di, oi]), di, oi)
                for di in range(len(detections))
                for oi in range(len(objects))
            ),
            reverse=True,
        )
        used_detections: set[int] = set()
        used_objects: set[int] = set()
        greedy: list[tuple[int, int]] = []
        for _, detection_index, object_index in ranked:
            if detection_index in used_detections or object_index in used_objects:
                continue
            used_detections.add(detection_index)
            used_objects.add(object_index)
            greedy.append((detection_index, object_index))
        candidates = greedy

    return tuple(
        Assignment(
            detection_index=int(detection_index),
            object_index=int(object_index),
            iou=float(scores[detection_index, object_index]),
        )
        for detection_index, object_index in candidates
        if scores[detection_index, object_index] >= min_iou
    )


def _load_projection(trip_dir: Path, frame_id: int) -> np.ndarray:
    path = trip_dir / "kitti" / "calib" / f"{frame_id:06d}.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("P2:"):
            values = np.asarray([float(value) for value in line.partition(":")[2].split()])
            if values.size != 12:
                break
            return values.reshape(3, 4)
    raise ValueError(f"Missing valid P2 in {path}")


def _match_label_target(
    *,
    kitti_label: str,
    location_x_m: float,
    location_z_m: float,
    targets: Sequence[Mapping[str, Any]],
    tolerance_m: float = 0.05,
) -> Mapping[str, Any]:
    expected_target_class = (
        "walker"
        if kitti_label in {"Pedestrian", "Person_sitting"}
        else "bike"
        if kitti_label == "Cyclist"
        else "vehicle"
    )
    candidates = []
    for target in targets:
        if str(target.get("target_class")) != expected_target_class:
            continue
        lateral = _finite_number(target.get("lateral_distance"))
        longitudinal = _finite_number(target.get("longitudinal_distance"))
        distance = math.hypot(location_x_m - lateral, location_z_m - longitudinal)
        candidates.append((distance, int(target["target_id"]), target))
    if not candidates:
        raise ValueError("KITTI label has no JSON target candidate")
    candidates.sort(key=lambda item: (item[0], item[1]))
    distance, _, target = candidates[0]
    if distance > tolerance_m:
        raise ValueError(
            f"KITTI/target coordinate mismatch: nearest distance={distance:.3f} m"
        )
    if len(candidates) > 1 and candidates[1][0] - distance < 0.50:
        raise ValueError(
            "Ambiguous KITTI/target coordinate join: "
            f"best={distance:.3f} m, second={candidates[1][0]:.3f} m"
        )
    return target


def load_offline_trip(trip_dir: Path) -> tuple[OfflineFrame, ...]:
    """Load practice-only GT, targets and projected boxes with strict IDs."""

    document = _open_trip_document(trip_dir)
    raw_frames = document.get("frames")
    if not isinstance(raw_frames, list) or len(raw_frames) != EXPECTED_FRAMES:
        raise ValueError(
            f"{trip_dir.name}: expected {EXPECTED_FRAMES} GT frames, "
            f"found {len(raw_frames) if isinstance(raw_frames, list) else 'invalid'}"
        )
    geometry = load_camera_geometry(trip_dir)
    output: list[OfflineFrame] = []
    seen_ids: set[int] = set()
    for index, raw in enumerate(raw_frames):
        frame_id = int(raw["frame_id"])
        if frame_id != index or frame_id in seen_ids:
            raise ValueError(
                f"{trip_dir.name}: expected unique contiguous frame_id={index}, got {frame_id}"
            )
        seen_ids.add(frame_id)
        targets = tuple(raw.get("targets") or ())
        label_path = trip_dir / "kitti" / "label_2" / f"{frame_id:06d}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        projection = _load_projection(trip_dir, frame_id)
        objects: list[KittiObject] = []
        used_targets: set[int] = set()
        for line in lines:
            parsed = parse_kitti_label(line)
            projected = project_kitti_cuboid(
                line,
                projection,
                image_width=geometry.width,
                image_height=geometry.height,
            )
            if projected is None:
                continue
            kitti_label, bbox = projected
            target = _match_label_target(
                kitti_label=kitti_label,
                location_x_m=parsed[4],
                location_z_m=parsed[6],
                targets=targets,
            )
            target_id = int(target["target_id"])
            if target_id in used_targets:
                raise ValueError(
                    f"{trip_dir.name} frame {frame_id}: duplicate projected target {target_id}"
                )
            used_targets.add(target_id)
            _, runtime_label = _runtime_label(kitti_label)
            objects.append(
                KittiObject(
                    target_id=target_id,
                    kitti_label=kitti_label,
                    runtime_label=runtime_label,
                    bbox=bbox,
                    height_m=parsed[1],
                    width_m=parsed[2],
                    length_m=parsed[3],
                    location_x_m=parsed[4],
                    location_z_m=parsed[6],
                    longitudinal_distance_m=_finite_number(target.get("longitudinal_distance")),
                    lateral_distance_m=_finite_number(target.get("lateral_distance")),
                    in_collision_cone=bool(target.get("in_collision_cone")),
                )
            )
        ego = raw.get("ego") or {}
        output.append(
            OfflineFrame(
                frame_id=frame_id,
                timestamp=float(raw.get("timestamp", frame_id / 20.0)),
                ego_speed_kmh=float(ego.get("speed_kmh") or 0.0),
                ground_truth_ttc_s=_finite_number(raw.get("min_ttc")),
                targets=targets,
                objects=tuple(objects),
            )
        )
    return tuple(output)


def _minimum_risk(risks: Iterable[TrackRisk]) -> TrackRisk | None:
    finite = [risk for risk in risks if math.isfinite(risk.predicted_ttc_s)]
    return min(finite, key=lambda risk: (risk.predicted_ttc_s, risk.track_id)) if finite else None


def _minimum_ttc(risks: Iterable[TrackRisk]) -> float:
    selected = _minimum_risk(risks)
    return selected.predicted_ttc_s if selected is not None else float("inf")


def _same_bbox(left: BBox, right: BBox, *, tolerance: float = 1e-6) -> bool:
    return max(abs(a - b) for a, b in zip(left, right)) <= tolerance


def _risk_for_detection(
    risks: Sequence[TrackRisk], detection: Detection
) -> TrackRisk | None:
    candidates = [risk for risk in risks if _same_bbox(risk.bbox, detection.bbox)]
    return min(candidates, key=lambda risk: risk.track_id) if candidates else None


def _oracle_target_id(frame: OfflineFrame) -> int | None:
    """Privileged target selector for ladder rung 3 and later.

    This intentionally uses the practice-only collision-cone flag and TTC
    label.  It is never called by production inference.  The selection leak is
    explicit because this rung measures target-selection headroom.
    """

    candidates: list[tuple[float, int]] = []
    for target in frame.targets:
        if not bool(target.get("in_collision_cone")):
            continue
        ttc = _finite_number(target.get("ttc_2d"))
        if math.isfinite(ttc):
            candidates.append((ttc, int(target["target_id"])))
    return min(candidates)[1] if candidates else None


def _object_for_target(frame: OfflineFrame, target_id: int | None) -> KittiObject | None:
    if target_id is None:
        return None
    return next((obj for obj in frame.objects if obj.target_id == target_id), None)


def _plausible_projection_match(
    target: KittiObject,
    detections: Sequence[Detection],
) -> bool:
    """Flag projection ambiguity without claiming a definitive detector miss.

    CARLA/KITTI cuboids are systematically offset from visible silhouettes in
    a few late T01 frames.  A class-compatible detection within 2.5 projected
    box diagonals is therefore reported as ambiguous, not a detector failure.
    """

    tx = (target.bbox[0] + target.bbox[2]) * 0.5
    ty = (target.bbox[1] + target.bbox[3]) * 0.5
    diagonal = max(
        12.0,
        math.hypot(target.bbox[2] - target.bbox[0], target.bbox[3] - target.bbox[1]),
    )
    target_group = (
        "person"
        if target.runtime_label == "person"
        else "two_wheeler"
        if target.runtime_label in {"bicycle", "motorcycle"}
        else "vehicle"
    )
    for detection in detections:
        detection_group = (
            "person"
            if detection.label == "person"
            else "two_wheeler"
            if detection.label in {"bicycle", "motorcycle"}
            else "vehicle"
        )
        # Cyclists in these synthetic images are frequently split into person
        # and motorcycle detections, so either component is plausible.
        compatible = target_group == detection_group or (
            target_group == "two_wheeler" and detection_group == "person"
        )
        if not compatible:
            continue
        dx = (detection.bbox[0] + detection.bbox[2]) * 0.5 - tx
        dy = (detection.bbox[1] + detection.bbox[3]) * 0.5 - ty
        if math.hypot(dx, dy) <= 2.5 * diagonal:
            return True
    return False


class OracleAssociationTracker(MonocularTTCTracker):
    """Current tracker with only target-to-detection association overridden.

    Before an object receives a confident projected-box identity, this tracker
    behaves exactly like the frozen greedy tracker and therefore preserves its
    accumulated history.  Once identified, a surviving track is forced to
    consume future detections with the same target ID.  Unmatched detections
    and false positives continue through the ordinary greedy association.
    """

    def __init__(self, geometry: CameraGeometry) -> None:
        frozen = baseline_tracker(geometry)
        super().__init__(
            frozen.config,
            focal_y_px=frozen.focal_y_px,
            focal_x_px=frozen.focal_x_px,
            principal_x_px=frozen.principal_x_px,
        )
        self._pending_target_ids: tuple[int | None, ...] = ()
        self._target_by_track: dict[int, int] = {}

    def _match(
        self,
        detections: Sequence[Detection],
        image_shape: tuple[int, int],
    ) -> list[tuple[int, int]]:
        target_detection = {
            target_id: detection_index
            for detection_index, target_id in enumerate(self._pending_target_ids)
            if target_id is not None
        }
        forced: list[tuple[int, int]] = []
        reserved_tracks: set[int] = set()
        reserved_detections: set[int] = set()
        for track_index, track in enumerate(self._tracks):
            target_id = self._target_by_track.get(track.track_id)
            if target_id is None:
                continue
            reserved_tracks.add(track_index)
            detection_index = target_detection.get(target_id)
            if detection_index is not None and detection_index not in reserved_detections:
                forced.append((track_index, detection_index))
                reserved_detections.add(detection_index)

        baseline_matches = super()._match(detections, image_shape)
        output = list(forced)
        used_tracks = {track_index for track_index, _ in forced}
        used_detections = {detection_index for _, detection_index in forced}
        for track_index, detection_index in baseline_matches:
            if (
                track_index in used_tracks
                or track_index in reserved_tracks
                or detection_index in used_detections
            ):
                continue
            output.append((track_index, detection_index))
            used_tracks.add(track_index)
            used_detections.add(detection_index)
        return output

    def update_with_identities(
        self,
        detections: Sequence[Detection],
        target_ids: Sequence[int | None],
        **kwargs: Any,
    ) -> list[TrackRisk]:
        if len(detections) != len(target_ids):
            raise ValueError("Oracle target IDs must align one-to-one with detections")
        self._pending_target_ids = tuple(target_ids)
        try:
            risks = super().update(detections, **kwargs)
        finally:
            self._pending_target_ids = ()

        active_ids = {int(track.track_id) for track in self._tracks}
        self._target_by_track = {
            track_id: target_id
            for track_id, target_id in self._target_by_track.items()
            if track_id in active_ids
        }
        for detection, target_id in zip(detections, target_ids):
            matching_track = next(
                (
                    track
                    for track in self._tracks
                    if track.missed == 0 and _same_bbox(track.bbox, detection.bbox)
                ),
                None,
            )
            if matching_track is None:
                continue
            self._target_by_track.pop(int(matching_track.track_id), None)
            if target_id is not None:
                # A privileged identity belongs to one live track only.
                self._target_by_track = {
                    track_id: mapped
                    for track_id, mapped in self._target_by_track.items()
                    if mapped != target_id
                }
                self._target_by_track[int(matching_track.track_id)] = int(target_id)
        return risks


class CausalOracleRange:
    """TTC from privileged range history, without any velocity/TTC label.

    The input range at frame *t* is allowed because this is an offline upper
    bound.  Closing speed is the median of all pairwise past-to-current range
    slopes over a causal window.  A fixed 4.5 m ego length plus the KITTI actor
    length defines centre-to-contact distance.
    """

    def __init__(
        self,
        *,
        history_size: int = 12,
        min_history: int = 3,
        ego_length_m: float = 4.5,
        min_closing_speed_mps: float = 0.25,
        max_ttc_s: float = 99.0,
    ) -> None:
        if history_size < min_history or min_history < 2:
            raise ValueError("Causal oracle range requires history_size >= min_history >= 2")
        self.history_size = history_size
        self.min_history = min_history
        self.ego_length_m = ego_length_m
        self.min_closing_speed_mps = min_closing_speed_mps
        self.max_ttc_s = max_ttc_s
        self._history: dict[int, list[tuple[float, float]]] = defaultdict(list)

    def observe(self, frame: OfflineFrame) -> None:
        # JSON targets expose range before an actor becomes label_2-visible.
        # Consume only current/past longitudinal range here; velocity and all
        # TTC fields are deliberately ignored.
        for target in frame.targets:
            target_id = int(target["target_id"])
            longitudinal_range = _finite_number(target.get("longitudinal_distance"))
            if not math.isfinite(longitudinal_range):
                continue
            history = self._history[target_id]
            history.append((frame.timestamp, longitudinal_range))
            del history[:-self.history_size]

    def predict(self, frame: OfflineFrame, selected_target_id: int | None) -> float:
        if selected_target_id is None:
            return float("inf")
        history = self._history.get(selected_target_id, [])
        if len(history) < self.min_history:
            return float("inf")
        slopes: list[float] = []
        for left_index in range(len(history) - 1):
            for right_index in range(left_index + 1, len(history)):
                delta_t = history[right_index][0] - history[left_index][0]
                if delta_t > 1e-6:
                    slopes.append(
                        (history[right_index][1] - history[left_index][1]) / delta_t
                    )
        if not slopes:
            return float("inf")
        _, centre_range = history[-1]
        current_object = _object_for_target(frame, selected_target_id)
        if current_object is not None:
            actor_length = current_object.length_m
        else:
            raw_target = next(
                (
                    target
                    for target in frame.targets
                    if int(target["target_id"]) == selected_target_id
                ),
                None,
            )
            target_class = str((raw_target or {}).get("target_class", "vehicle"))
            actor_length = {"walker": 0.6, "bike": 1.8, "vehicle": 4.0}.get(
                target_class, 4.0
            )
        contact_envelope = 0.5 * (self.ego_length_m + actor_length)
        gap = centre_range - contact_envelope
        if gap <= 0.0:
            return 0.1
        closing_speed = -float(np.median(np.asarray(slopes, dtype=np.float64)))
        if closing_speed < self.min_closing_speed_mps:
            return float("inf")
        candidate = gap / closing_speed
        return max(0.1, candidate) if candidate <= self.max_ttc_s else float("inf")

    def update(self, frame: OfflineFrame, selected_target_id: int | None) -> float:
        self.observe(frame)
        return self.predict(frame, selected_target_id)


def load_frozen_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read authoritative ladder rung 1 after strict preflight."""

    preflight_prediction_csv(path, expected_frames=EXPECTED_FRAMES)
    timestamps = np.empty(EXPECTED_FRAMES, dtype=np.float64)
    values = np.full(EXPECTED_FRAMES, np.inf, dtype=np.float64)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            frame_id = int(row["frame_id"])
            timestamps[frame_id] = float(row["timestamp"])
            values[frame_id] = _finite_number(row["predicted_ttc"])
    return timestamps, values


def _projected_detection(obj: KittiObject) -> Detection:
    class_id, label = _runtime_label(obj.kitti_label)
    return Detection(class_id=class_id, label=label, confidence=1.0, bbox=obj.bbox)


def replay_trip(
    frames: Sequence[OfflineFrame],
    *,
    geometry: CameraGeometry,
    cache_path: Path,
    baseline_csv: Path,
    confidence_threshold: float = 0.25,
    assignment_iou: float = 0.10,
) -> tuple[ReplayRow, ...]:
    if len(frames) != EXPECTED_FRAMES:
        raise ValueError(f"Expected {EXPECTED_FRAMES} offline frames, found {len(frames)}")
    cache = load_detection_cache(cache_path)
    timestamps, frozen_baseline = load_frozen_predictions(baseline_csv)
    if cache.trip_id != baseline_csv.stem:
        raise ValueError(f"Cache/baseline trip mismatch: {cache.trip_id} vs {baseline_csv.stem}")
    expected_detector_frames = set(range(0, EXPECTED_FRAMES, cache.stride))
    if set(cache.rows) != expected_detector_frames:
        raise ValueError(f"{cache.trip_id}: detector cache is not complete at stride {cache.stride}")

    current_tracker = baseline_tracker(geometry)
    current_tracker.reset()
    oracle_association_tracker = OracleAssociationTracker(geometry)
    oracle_association_tracker.reset()
    # One-factor rung 4: projected boxes replace matched YOLO boxes while
    # unmatched YOLO detections/false positives remain in the current tracker.
    privileged_current_tracker = baseline_tracker(geometry)
    privileged_current_tracker.reset()
    causal_range = CausalOracleRange()

    track_target: dict[int, int | None] = {}
    last_target_observation: dict[int, tuple[int, int]] = {}
    last_track_state: dict[int, tuple[bool, bool, bool]] = {}
    previous_state_by_track: dict[int, tuple[float, float, float]] = {}
    rows: list[ReplayRow] = []

    for frame in frames:
        frame_id = frame.frame_id
        if not math.isclose(frame.timestamp, timestamps[frame_id], abs_tol=1e-3):
            raise ValueError(
                f"{cache.trip_id} frame {frame_id}: baseline timestamp mismatch "
                f"{timestamps[frame_id]} vs {frame.timestamp}"
            )
        detector_update = frame_id in expected_detector_frames
        detections: tuple[Detection, ...] = ()
        assignments: tuple[Assignment, ...] = ()
        assignments_005: tuple[Assignment, ...] = ()
        assignments_020: tuple[Assignment, ...] = ()
        if detector_update:
            detections = tuple(
                detection
                for detection in cache.rows[frame_id]
                if detection.confidence >= confidence_threshold
            )
            assignments = match_detections_to_objects(
                detections, frame.objects, min_iou=assignment_iou
            )
            assignments_005 = match_detections_to_objects(
                detections, frame.objects, min_iou=0.05
            )
            assignments_020 = match_detections_to_objects(
                detections, frame.objects, min_iou=0.20
            )

        common = {
            "timestamp": frame.timestamp,
            "image_shape": (geometry.height, geometry.width),
            "ego_speed_kmh": frame.ego_speed_kmh,
        }
        if detector_update:
            current_risks = tuple(current_tracker.update(detections, **common))
        else:
            current_risks = tuple(current_tracker.predict(**common))

        assignment_by_detection = {
            assignment.detection_index: frame.objects[assignment.object_index].target_id
            for assignment in assignments
        }
        observed_track_by_target: dict[int, int] = {}
        if detector_update:
            for detection_index, detection in enumerate(detections):
                risk = _risk_for_detection(current_risks, detection)
                if risk is None:
                    continue
                target_id = assignment_by_detection.get(detection_index)
                track_target[risk.track_id] = target_id
                if target_id is not None:
                    observed_track_by_target[target_id] = risk.track_id
            active_ids = {
                int(track.track_id)
                for track in getattr(current_tracker, "_tracks", ())
            }
            track_target = {
                track_id: target_id
                for track_id, target_id in track_target.items()
                if track_id in active_ids
            }

        if detector_update:
            association_target_ids = tuple(
                assignment_by_detection.get(index) for index in range(len(detections))
            )
            association_risks = tuple(
                oracle_association_tracker.update_with_identities(
                    detections, association_target_ids, **common
                )
            )
        else:
            association_risks = tuple(oracle_association_tracker.predict(**common))
        # This is a one-factor intervention: every unmatched detection and FP
        # stays in the same tracker; only confidently identified associations
        # are forced to remain on their live target track.
        oracle_association_ttc = _minimum_ttc(association_risks)

        oracle_target_id = _oracle_target_id(frame)
        # One-factor rung 3: retain the current tracker and TTC estimator, then
        # select only a confidently mapped GT collision target.
        oracle_target_ttc = (
            _minimum_ttc(
                risk
                for risk in current_risks
                if track_target.get(risk.track_id) == oracle_target_id
            )
            if oracle_target_id is not None
            else float("inf")
        )

        projected_by_target = {
            obj.target_id: _projected_detection(obj) for obj in frame.objects
        } if detector_update else {}
        if detector_update:
            matched_detection_indices = {item.detection_index for item in assignments}
            privileged_detections = tuple(projected_by_target.values()) + tuple(
                detection
                for index, detection in enumerate(detections)
                if index not in matched_detection_indices
            )
            privileged_risks = tuple(
                privileged_current_tracker.update(privileged_detections, **common)
            )
        else:
            privileged_risks = tuple(privileged_current_tracker.predict(**common))
        oracle_bbox_ttc = _minimum_ttc(privileged_risks)

        oracle_range_ttc = causal_range.update(frame, oracle_target_id)
        selected_risk = _minimum_risk(current_risks)
        selected_track_id = selected_risk.track_id if selected_risk is not None else None
        mapping_known = selected_track_id is not None and selected_track_id in track_target
        selected_target_id = track_target.get(selected_track_id) if mapping_known else None
        selected_range = (
            selected_risk.estimated_distance_m if selected_risk is not None else float("inf")
        )
        current_selection_oracle_range_ttc = causal_range.predict(
            frame, selected_target_id if mapping_known else None
        )

        target_object = _object_for_target(frame, oracle_target_id)
        target_object_indices = {
            index for index, obj in enumerate(frame.objects) if obj.target_id == oracle_target_id
        }
        target_assignments = [
            item for item in assignments if item.object_index in target_object_indices
        ]
        target_assignments_005 = [
            item for item in assignments_005 if item.object_index in target_object_indices
        ]
        target_assignments_020 = [
            item for item in assignments_020 if item.object_index in target_object_indices
        ]
        assigned_object_indices = {item.object_index for item in assignments}
        assigned_object_indices_005 = {item.object_index for item in assignments_005}
        ambiguous_object_count = 0
        if detector_update:
            for object_index, obj in enumerate(frame.objects):
                if object_index in assigned_object_indices:
                    continue
                if (
                    object_index in assigned_object_indices_005
                    or _plausible_projection_match(obj, detections)
                ):
                    ambiguous_object_count += 1
        matched_class_pairs = ";".join(
            sorted(
                f"{frame.objects[item.object_index].kitti_label}->{detections[item.detection_index].label}"
                for item in assignments
            )
        )
        detector_target_seen = bool(target_assignments)
        detector_target_seen_005 = bool(target_assignments_005)
        detector_target_seen_020 = bool(target_assignments_020)
        ambiguous = bool(
            detector_update
            and target_object is not None
            and not detector_target_seen
            and (
                detector_target_seen_005
                or _plausible_projection_match(target_object, detections)
            )
        )
        detector_missed = bool(
            detector_update
            and target_object is not None
            and not detector_target_seen
            and not ambiguous
        )
        target_track_id = observed_track_by_target.get(oracle_target_id) \
            if oracle_target_id is not None else None
        detection_without_track = bool(detector_target_seen and target_track_id is None)

        identity_switch = False
        fragmentation = False
        if detector_update:
            # Audit every confidently projected actor identity, not only the
            # current min-TTC target.  This captures class/association churn
            # before and outside danger frames.
            for observed_target_id, observed_track_id in observed_track_by_target.items():
                previous = last_target_observation.get(observed_target_id)
                if previous is not None and previous[1] != observed_track_id:
                    gap_cycles = (frame_id - previous[0]) // cache.stride
                    identity_switch = identity_switch or gap_cycles <= 6
                    fragmentation = fragmentation or gap_cycles > 6
                last_target_observation[observed_target_id] = (
                    frame_id,
                    observed_track_id,
                )

        active_missed = {
            int(track.track_id): int(track.missed)
            for track in getattr(current_tracker, "_tracks", ())
        }
        mapped_target_tracks = [
            track_id
            for track_id, mapped_target in track_target.items()
            if mapped_target == oracle_target_id and track_id in active_missed
        ] if oracle_target_id is not None else []
        target_track_missed: int | None = None
        if mapped_target_tracks:
            best_track = min(mapped_target_tracks, key=lambda track_id: (active_missed[track_id], track_id))
            target_track_missed = active_missed[best_track]
            if target_track_id is None:
                target_track_id = best_track
        if detector_update and oracle_target_id is not None:
            previous_observation = last_target_observation.get(oracle_target_id)
            previous_track_id = previous_observation[1] if previous_observation else None
            previous_track_active = (
                previous_track_id is not None and previous_track_id in active_missed
            )
            corruption = bool(
                not detector_target_seen
                and previous_track_active
                and active_missed[previous_track_id] == 0
                and track_target.get(previous_track_id) != oracle_target_id
            )
            coast = bool(
                not detector_target_seen
                and previous_track_active
                and active_missed[previous_track_id] >= 1
            )
            lost = bool(
                not detector_target_seen
                and previous_observation is not None
                and not previous_track_active
            )
            last_track_state[oracle_target_id] = (corruption, coast, lost)
        corruption, coast, lost = last_track_state.get(
            oracle_target_id, (False, False, False)
        ) if oracle_target_id is not None else (False, False, False)
        if detector_target_seen:
            corruption = coast = lost = False

        baseline_ttc = float(frozen_baseline[frame_id])
        gt_danger = frame.ground_truth_ttc_s < 2.0
        pred_danger = baseline_ttc < 2.0
        false_negative = gt_danger and not pred_danger
        false_positive = pred_danger and not gt_danger
        mae_critical_miss = (
            frame.ground_truth_ttc_s < 3.0 and not baseline_ttc < 3.0
        )
        selected_mapping_unknown = bool(
            math.isfinite(baseline_ttc)
            and (
                oracle_target_id is None
                or not mapping_known
                or selected_target_id is None
            )
        )
        wrong_target = False
        if (
            math.isfinite(baseline_ttc)
            and oracle_target_id is not None
            and mapping_known
            and selected_target_id is not None
        ):
            wrong_target = selected_target_id != oracle_target_id

        ttc_jump = False
        ttc_finite_toggle = False
        range_jump = False
        if detector_update:
            for risk in current_risks:
                previous_state = previous_state_by_track.get(risk.track_id)
                if previous_state is None:
                    previous_state_by_track[risk.track_id] = (
                        frame.timestamp,
                        risk.predicted_ttc_s,
                        risk.estimated_distance_m,
                    )
                    continue
                old_timestamp, old_ttc, old_range = previous_state
                delta_t = frame.timestamp - old_timestamp
                if delta_t > 1e-6:
                    old_finite = math.isfinite(old_ttc)
                    new_finite = math.isfinite(risk.predicted_ttc_s)
                    ttc_finite_toggle = ttc_finite_toggle or old_finite != new_finite
                    if old_finite and new_finite:
                        # A constant-velocity TTC should count down by delta_t.
                        ttc_jump = ttc_jump or abs(
                            (risk.predicted_ttc_s - old_ttc) + delta_t
                        ) > 1.0
                    if math.isfinite(old_range) and math.isfinite(risk.estimated_distance_m):
                        range_jump = range_jump or (
                            abs(risk.estimated_distance_m - old_range) / delta_t > 45.0
                        )
                previous_state_by_track[risk.track_id] = (
                    frame.timestamp,
                    risk.predicted_ttc_s,
                    risk.estimated_distance_m,
                )

        rows.append(
            ReplayRow(
                frame_id=frame_id,
                timestamp=frame.timestamp,
                ground_truth_ttc_s=frame.ground_truth_ttc_s,
                baseline_ttc_s=baseline_ttc,
                current_replay_ttc_s=_minimum_ttc(current_risks),
                oracle_association_ttc_s=oracle_association_ttc,
                oracle_target_ttc_s=oracle_target_ttc,
                oracle_bbox_ttc_s=oracle_bbox_ttc,
                oracle_range_ttc_s=oracle_range_ttc,
                current_selection_oracle_range_ttc_s=current_selection_oracle_range_ttc,
                oracle_target_id=oracle_target_id,
                selected_track_id=selected_track_id,
                selected_target_id=selected_target_id,
                selected_range_m=selected_range,
                detector_update=detector_update,
                detector_target_seen=detector_target_seen,
                detector_target_seen_iou_005=detector_target_seen_005,
                detector_target_seen_iou_020=detector_target_seen_020,
                projected_object_count=len(frame.objects) if detector_update else 0,
                matched_object_count_iou_005=len({item.object_index for item in assignments_005}),
                matched_object_count_iou_010=len(assigned_object_indices),
                matched_object_count_iou_020=len({item.object_index for item in assignments_020}),
                projection_ambiguous_object_count=ambiguous_object_count,
                matched_class_pairs=matched_class_pairs,
                target_detection_count=len(target_assignments),
                target_track_id=target_track_id,
                target_track_missed_updates=target_track_missed,
                detector_missed_object=detector_missed,
                projection_unmatched_ambiguous=ambiguous,
                detection_without_track=detection_without_track,
                association_corruption=corruption,
                identity_switch=identity_switch,
                track_fragmentation=fragmentation,
                track_coasting=coast,
                track_lost=lost,
                wrong_collision_target=wrong_target,
                selected_target_mapping_unknown=selected_mapping_unknown,
                range_jump=range_jump,
                ttc_jump=ttc_jump,
                ttc_finite_toggle=ttc_finite_toggle,
                range_or_ttc_jump=ttc_jump or ttc_finite_toggle or range_jump,
                gt_mae_critical_prediction_not_critical=mae_critical_miss,
                gt_critical_prediction_not_critical=false_negative,
                prediction_critical_false=false_positive,
            )
        )
    return tuple(rows)


def _ttc_csv_value(value: float) -> str:
    return "inf" if not math.isfinite(value) else f"{value:.6f}".rstrip("0").rstrip(".")


def _variant_values(rows: Sequence[ReplayRow]) -> Mapping[str, tuple[float, ...]]:
    return {
        "detector_tracker_current": tuple(row.baseline_ttc_s for row in rows),
        "detector_oracle_association": tuple(
            row.oracle_association_ttc_s for row in rows
        ),
        "detector_oracle_target_selection": tuple(
            row.oracle_target_ttc_s for row in rows
        ),
        "oracle_projected_bbox_tracker": tuple(row.oracle_bbox_ttc_s for row in rows),
        "oracle_range_upper_bound": tuple(row.oracle_range_ttc_s for row in rows),
        "current_selection_oracle_range": tuple(
            row.current_selection_oracle_range_ttc_s for row in rows
        ),
    }


def write_prediction_csv(
    path: Path,
    rows: Sequence[ReplayRow],
    values: Sequence[float],
) -> Mapping[str, object]:
    if len(rows) != EXPECTED_FRAMES or len(values) != EXPECTED_FRAMES:
        raise ValueError("Prediction writer requires exactly 600 frames")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("frame_id", "timestamp", "predicted_ttc")
        )
        writer.writeheader()
        for row, value in zip(rows, values):
            writer.writerow(
                {
                    "frame_id": row.frame_id,
                    "timestamp": f"{row.timestamp:.6f}".rstrip("0").rstrip("."),
                    "predicted_ttc": _ttc_csv_value(float(value)),
                }
            )
    return preflight_prediction_csv(path, expected_frames=EXPECTED_FRAMES).to_dict()


def contiguous_segments(
    trip_id: str,
    rows: Sequence[ReplayRow],
    flag: str,
) -> tuple[dict[str, object], ...]:
    """Return strictly contiguous true runs; intentionally never bridges gaps."""

    if flag not in FAILURE_FLAGS:
        raise ValueError(f"Unknown diagnostic flag: {flag}")
    segments: list[dict[str, object]] = []
    start: int | None = None
    for index in range(len(rows) + 1):
        enabled = index < len(rows) and bool(getattr(rows[index], flag))
        if enabled and start is None:
            start = index
        if not enabled and start is not None:
            selected = rows[start:index]
            finite_gt = [row.ground_truth_ttc_s for row in selected if math.isfinite(row.ground_truth_ttc_s)]
            finite_pred = [row.baseline_ttc_s for row in selected if math.isfinite(row.baseline_ttc_s)]
            segments.append(
                {
                    "trip_id": trip_id,
                    "category": flag,
                    "start_frame": selected[0].frame_id,
                    "end_frame": selected[-1].frame_id,
                    "n_frames": len(selected),
                    "start_timestamp": selected[0].timestamp,
                    "end_timestamp": selected[-1].timestamp,
                    "min_ground_truth_ttc_s": min(finite_gt) if finite_gt else None,
                    "min_baseline_ttc_s": min(finite_pred) if finite_pred else None,
                }
            )
            start = None
    return tuple(segments)


def primary_failure_cause(row: ReplayRow) -> str:
    """Assign one non-overlapping root cause; FN/FP remain separate outcomes."""

    critical_large_error = bool(
        row.ground_truth_ttc_s < 3.0
        and (
            not math.isfinite(row.baseline_ttc_s)
            or abs(row.baseline_ttc_s - row.ground_truth_ttc_s) > 1.0
        )
    )
    if not (
        row.gt_critical_prediction_not_critical
        or row.prediction_critical_false
        or row.gt_mae_critical_prediction_not_critical
        or critical_large_error
    ):
        return ""
    hierarchy = (
        (row.detector_missed_object, "detector_missed_object"),
        (row.projection_unmatched_ambiguous, "projection_unmatched_ambiguous"),
        (row.detection_without_track, "detection_without_track"),
        (row.association_corruption, "association_corruption"),
        (row.track_lost, "track_lost"),
        (row.track_coasting, "track_coasting"),
        (row.identity_switch, "identity_switch"),
        (row.track_fragmentation, "track_fragmentation"),
        (
            row.gt_critical_prediction_not_critical
            and not math.isfinite(row.baseline_ttc_s),
            "no_finite_collision_target",
        ),
        (row.wrong_collision_target, "wrong_collision_target"),
        (row.selected_target_mapping_unknown, "selected_target_mapping_unknown"),
        (row.range_or_ttc_jump, "range_or_ttc_jump"),
    )
    return next((name for enabled, name in hierarchy if enabled), "")


def primary_failure_segments(
    trip_id: str, rows: Sequence[ReplayRow]
) -> tuple[dict[str, object], ...]:
    segments: list[dict[str, object]] = []
    start = 0
    while start < len(rows):
        cause = primary_failure_cause(rows[start])
        if not cause:
            start += 1
            continue
        end = start + 1
        while end < len(rows) and primary_failure_cause(rows[end]) == cause:
            end += 1
        selected = rows[start:end]
        finite_gt = [row.ground_truth_ttc_s for row in selected if math.isfinite(row.ground_truth_ttc_s)]
        finite_pred = [row.baseline_ttc_s for row in selected if math.isfinite(row.baseline_ttc_s)]
        segments.append(
            {
                "trip_id": trip_id,
                "category": f"primary_failure:{cause}",
                "start_frame": selected[0].frame_id,
                "end_frame": selected[-1].frame_id,
                "n_frames": len(selected),
                "start_timestamp": selected[0].timestamp,
                "end_timestamp": selected[-1].timestamp,
                "min_ground_truth_ttc_s": min(finite_gt) if finite_gt else None,
                "min_baseline_ttc_s": min(finite_pred) if finite_pred else None,
            }
        )
        start = end
    return tuple(segments)


def _write_diagnostic_frames(path: Path, trip_id: str, rows: Sequence[ReplayRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("trip_id", "primary_failure_cause", "categories") + tuple(asdict(rows[0]).keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = asdict(row)
            for key, value in list(values.items()):
                if isinstance(value, float) and not math.isfinite(value):
                    values[key] = "inf"
            categories = ";".join(flag for flag in FAILURE_FLAGS if getattr(row, flag))
            writer.writerow(
                {
                    "trip_id": trip_id,
                    "primary_failure_cause": primary_failure_cause(row),
                    "categories": categories,
                    **values,
                }
            )


def _trip_metrics(trip_id: str, rows: Sequence[ReplayRow], values: Sequence[float]) -> Any:
    pairs = {
        row.frame_id: (float(value), row.ground_truth_ttc_s)
        for row, value in zip(rows, values)
    }
    return compute_trip_metrics(trip_id, pairs)


def _macro_metrics(per_trip: Sequence[Any]) -> dict[str, object]:
    valid_mae = [metric.mae_critical for metric in per_trip if metric.mae_critical >= 0]
    return {
        "n_trips": len(per_trip),
        "mae_critical": round(float(np.mean(valid_mae)), 3) if valid_mae else None,
        "inv_ttc_mae": round(float(np.mean([m.inv_ttc_mae for m in per_trip])), 4),
        "f1": round(float(np.mean([m.f1 for m in per_trip])), 3),
        "composite_score": round(float(np.mean([m.composite_score for m in per_trip])), 1),
    }


def _trip_diagnostic_summary(
    trip_dir: Path,
    rows: Sequence[ReplayRow],
    frames: Sequence[OfflineFrame],
) -> dict[str, object]:
    target_update_rows = [
        row for row in rows if row.detector_update and row.oracle_target_id is not None
    ]
    denominator = len(target_update_rows)
    detector_rows = [row for row in rows if row.detector_update]
    projected_total = sum(row.projected_object_count for row in detector_rows)
    class_pairs: Counter[str] = Counter()
    for row in detector_rows:
        class_pairs.update(pair for pair in row.matched_class_pairs.split(";") if pair)
    finite_mask_disagreement = sum(
        math.isfinite(row.baseline_ttc_s) != math.isfinite(row.current_replay_ttc_s)
        for row in rows
    )
    jointly_finite_delta = [
        abs(row.baseline_ttc_s - row.current_replay_ttc_s)
        for row in rows
        if math.isfinite(row.baseline_ttc_s) and math.isfinite(row.current_replay_ttc_s)
    ]
    join_errors: list[float] = []
    join_margins: list[float] = []
    for frame in frames:
        for obj in frame.objects:
            join_errors.append(
                math.hypot(
                    obj.location_x_m - obj.lateral_distance_m,
                    obj.location_z_m - obj.longitudinal_distance_m,
                )
            )
            target_class = (
                "walker"
                if obj.kitti_label in {"Pedestrian", "Person_sitting"}
                else "bike"
                if obj.kitti_label == "Cyclist"
                else "vehicle"
            )
            alternatives = sorted(
                math.hypot(
                    obj.location_x_m - _finite_number(target.get("lateral_distance")),
                    obj.location_z_m - _finite_number(target.get("longitudinal_distance")),
                )
                for target in frame.targets
                if str(target.get("target_class")) == target_class
                and int(target["target_id"]) != obj.target_id
            )
            if alternatives:
                join_margins.append(alternatives[0] - join_errors[-1])
    depths = list((trip_dir / "kitti" / "depth").glob("*.npy"))
    return {
        "n_frames": len(rows),
        "n_mae_critical_gt_lt_3s": sum(row.ground_truth_ttc_s < 3.0 for row in rows),
        "n_f1_danger_gt_lt_2s": sum(row.ground_truth_ttc_s < 2.0 for row in rows),
        "failure_frame_counts": {
            flag: sum(bool(getattr(row, flag)) for row in rows) for flag in FAILURE_FLAGS
        },
        "primary_failure_frame_counts": dict(
            sorted(
                (
                    (cause, sum(primary_failure_cause(row) == cause for row in rows))
                    for cause in {primary_failure_cause(row) for row in rows}
                    if cause
                ),
                key=lambda item: item[0],
            )
        ),
        "jump_frame_counts": {
            "range_jump_over_45_mps": sum(row.range_jump for row in rows),
            "ttc_countdown_residual_over_1s": sum(row.ttc_jump for row in rows),
            "ttc_finite_infinite_toggle": sum(row.ttc_finite_toggle for row in rows),
        },
        "frozen_vs_current_cache_replay": {
            "finite_mask_disagreement_frames": finite_mask_disagreement,
            "jointly_finite_frames": len(jointly_finite_delta),
            "mean_absolute_delta_s": (
                round(float(np.mean(jointly_finite_delta)), 6)
                if jointly_finite_delta else None
            ),
            "max_absolute_delta_s": (
                round(max(jointly_finite_delta), 6) if jointly_finite_delta else None
            ),
        },
        "target_detector_updates": denominator,
        "target_match_coverage": {
            "iou_0.05": (
                round(sum(row.detector_target_seen_iou_005 for row in target_update_rows) / denominator, 4)
                if denominator else None
            ),
            "iou_0.10_primary": (
                round(sum(row.detector_target_seen for row in target_update_rows) / denominator, 4)
                if denominator else None
            ),
            "iou_0.20": (
                round(sum(row.detector_target_seen_iou_020 for row in target_update_rows) / denominator, 4)
                if denominator else None
            ),
        },
        "all_projected_object_match_coverage": {
            "projected_objects_on_detector_updates": projected_total,
            "iou_0.05": (
                round(sum(row.matched_object_count_iou_005 for row in detector_rows) / projected_total, 4)
                if projected_total else None
            ),
            "iou_0.10_primary": (
                round(sum(row.matched_object_count_iou_010 for row in detector_rows) / projected_total, 4)
                if projected_total else None
            ),
            "iou_0.20": (
                round(sum(row.matched_object_count_iou_020 for row in detector_rows) / projected_total, 4)
                if projected_total else None
            ),
            "projection_ambiguous_objects_iou_0.10": sum(
                row.projection_ambiguous_object_count for row in detector_rows
            ),
            "matched_kitti_to_yolo_class_pairs": dict(sorted(class_pairs.items())),
        },
        "label_target_join": {
            "joined_objects": len(join_errors),
            "maximum_coordinate_error_m": round(max(join_errors), 6) if join_errors else None,
            "minimum_second_candidate_margin_m": (
                round(min(join_margins), 6) if join_margins else None
            ),
            "required_max_error_m": 0.05,
            "required_min_margin_m": 0.50,
        },
        "schema_integrity": {
            "image_2_frames": len(list((trip_dir / "kitti" / "image_2").glob("*.png")))
            + len(list((trip_dir / "kitti" / "image_2").glob("*.jpg"))),
            "label_2_frames": len(list((trip_dir / "kitti" / "label_2").glob("*.txt"))),
            "calibration_frames": len(list((trip_dir / "kitti" / "calib").glob("*.txt"))),
            "depth_frames_diagnostic_only": len(depths),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_analysis(
    *,
    data_root: Path,
    cache_dir: Path,
    baseline_dir: Path,
    output_dir: Path,
    ladder_trips: Sequence[str] = DEFAULT_TRIPS,
    diagnostic_trips: Sequence[str] = DEFAULT_DIAGNOSTIC_TRIPS,
    confidence_threshold: float = 0.25,
) -> Mapping[str, object]:
    evaluator_sha_before = _sha256(OFFICIAL_EVALUATOR)
    if evaluator_sha_before != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError(
            "Official evaluator SHA gate failed before diagnostics: "
            f"{evaluator_sha_before} != {EXPECTED_EVALUATOR_SHA256}"
        )
    if tuple(ladder_trips) != DEFAULT_TRIPS:
        raise ValueError(
            "Comparable oracle ladder requires exactly T01-Sample..T06-Sample"
        )
    if not set(diagnostic_trips).issubset(set(ladder_trips)):
        raise ValueError("Diagnostic trips must be a subset of ladder trips")

    output_dir.mkdir(parents=True, exist_ok=True)
    trip_rows: dict[str, tuple[ReplayRow, ...]] = {}
    summaries: dict[str, object] = {}
    all_segments: list[dict[str, object]] = []
    for trip_id in ladder_trips:
        trip_dir = data_root / trip_id
        frames = load_offline_trip(trip_dir)
        geometry = load_camera_geometry(trip_dir)
        cache_path = cache_dir / f"{trip_id}.stride3.conf020.json.gz"
        baseline_csv = baseline_dir / f"{trip_id}.csv"
        rows = replay_trip(
            frames,
            geometry=geometry,
            cache_path=cache_path,
            baseline_csv=baseline_csv,
            confidence_threshold=confidence_threshold,
        )
        if len(rows) != EXPECTED_FRAMES or [row.frame_id for row in rows] != list(range(EXPECTED_FRAMES)):
            raise RuntimeError(f"{trip_id}: diagnostic replay did not produce exact 0..599")
        trip_rows[trip_id] = rows
        summaries[trip_id] = _trip_diagnostic_summary(trip_dir, rows, frames)
        mask_disagreement = sum(
            math.isfinite(row.baseline_ttc_s) != math.isfinite(row.current_replay_ttc_s)
            for row in rows
        )
        joint_deltas = [
            abs(row.baseline_ttc_s - row.current_replay_ttc_s)
            for row in rows
            if math.isfinite(row.baseline_ttc_s) and math.isfinite(row.current_replay_ttc_s)
        ]
        if (
            mask_disagreement > 1
            or (joint_deltas and float(np.mean(joint_deltas)) > 0.01)
            or (joint_deltas and max(joint_deltas) > 0.05)
        ):
            raise RuntimeError(
                f"{trip_id}: frozen/current cache replay parity gate failed: "
                f"mask={mask_disagreement}, mean={float(np.mean(joint_deltas)) if joint_deltas else 0.0}, "
                f"max={max(joint_deltas) if joint_deltas else 0.0}"
            )
        if trip_id in diagnostic_trips:
            _write_diagnostic_frames(
                output_dir / "diagnostics" / f"{trip_id}-frames.csv", trip_id, rows
            )
            for flag in FAILURE_FLAGS:
                all_segments.extend(contiguous_segments(trip_id, rows, flag))
            all_segments.extend(primary_failure_segments(trip_id, rows))

    all_prediction_variants = LADDER_VARIANTS + SUPPLEMENTAL_VARIANTS
    preflight: dict[str, list[Mapping[str, object]]] = {
        variant: [] for variant in all_prediction_variants
    }
    metric_rows: dict[str, list[Any]] = {
        variant: [] for variant in all_prediction_variants
    }
    for trip_id, rows in trip_rows.items():
        for variant, values in _variant_values(rows).items():
            output_csv = output_dir / "predictions" / variant / f"{trip_id}.csv"
            preflight[variant].append(write_prediction_csv(output_csv, rows, values))
            metric_rows[variant].append(_trip_metrics(trip_id, rows, values))

    expected_csv_names = {f"{trip_id}.csv" for trip_id in ladder_trips}
    for variant in all_prediction_variants:
        variant_dir = output_dir / "predictions" / variant
        actual_csv_names = {path.name for path in variant_dir.glob("*.csv")}
        if actual_csv_names != expected_csv_names:
            raise RuntimeError(
                f"{variant}: expected exactly {sorted(expected_csv_names)}, "
                f"found {sorted(actual_csv_names)}"
            )
        if len(preflight[variant]) != 6 or sum(item["rows"] for item in preflight[variant]) != 3600:
            raise RuntimeError(f"{variant}: preflight total is not exactly 6 x 600")

    ladder: dict[str, object] = {}
    for variant in LADDER_VARIANTS:
        ladder[variant] = {
            "n_predictions": sum(item["rows"] for item in preflight[variant]),
            "preflight": preflight[variant],
            "per_trip": [asdict(metric) for metric in metric_rows[variant]],
            "macro": _macro_metrics(metric_rows[variant]),
        }
    supplemental = {
        variant: {
            "n_predictions": sum(item["rows"] for item in preflight[variant]),
            "preflight": preflight[variant],
            "per_trip": [asdict(metric) for metric in metric_rows[variant]],
            "macro": _macro_metrics(metric_rows[variant]),
        }
        for variant in SUPPLEMENTAL_VARIANTS
    }
    baseline_score = ladder["detector_tracker_current"]["macro"]["composite_score"]  # type: ignore[index]
    if baseline_score != 55.2:
        raise RuntimeError(
            f"Frozen authoritative baseline gate failed: {baseline_score} != 55.2"
        )

    segments_dir = output_dir / "diagnostics"
    segments_dir.mkdir(parents=True, exist_ok=True)
    with (segments_dir / "segments.csv").open("w", encoding="utf-8", newline="") as stream:
        segment_fields = (
            "trip_id",
            "category",
            "start_frame",
            "end_frame",
            "n_frames",
            "start_timestamp",
            "end_timestamp",
            "min_ground_truth_ttc_s",
            "min_baseline_ttc_s",
        )
        writer = csv.DictWriter(stream, fieldnames=segment_fields)
        writer.writeheader()
        writer.writerows(all_segments)

    exact_ceiling_metrics = [
        _trip_metrics(
            trip_id,
            rows,
            [row.ground_truth_ttc_s for row in rows],
        )
        for trip_id, rows in trip_rows.items()
    ]
    evaluator_sha_after = _sha256(OFFICIAL_EVALUATOR)
    if evaluator_sha_after != evaluator_sha_before:
        raise RuntimeError("Official evaluator changed while diagnostics ran")
    report: dict[str, object] = {
        "protocol": "P2 offline development diagnostic; not external held-out evaluation",
        "official_evaluator_sha256_before": evaluator_sha_before,
        "official_evaluator_sha256_after": evaluator_sha_after,
        "configuration": {
            "expected_frames_per_trip": EXPECTED_FRAMES,
            "ladder_trips": list(ladder_trips),
            "diagnostic_trips": list(diagnostic_trips),
            "detector_source_camera": "image_2",
            "detector_stride": 3,
            "detector_confidence_threshold": confidence_threshold,
            "primary_projection_match_iou": 0.10,
            "projection_match_sensitivity_iou": [0.05, 0.20],
            "identity_switch_max_gap_detector_cycles": 6,
            "range_upper_bound": {
                "causal": True,
                "history_frames": 12,
                "minimum_history_frames": 3,
                "robust_slope": "median all pairwise slopes",
                "ego_length_m": 4.5,
                "forbidden_range_estimator_inputs": [
                    "rel_velocity",
                    "closing_speed",
                    "ttc_simple",
                    "ttc_2d",
                    "min_ttc",
                ],
            },
        },
        "ladder_semantics": {
            "detector_tracker_current": "byte-for-byte frozen authoritative 55.2 CSV values; replay only supplies metadata",
            "detector_oracle_association": "one-factor current tracker with confidently identified target associations forced; unmatched YOLO tracks/false positives retained",
            "detector_oracle_target_selection": "one-factor current detector/tracker/TTC plus GT collision-cone/min-TTC target selection (offline only)",
            "oracle_projected_bbox_tracker": "one-factor privileged projected-cuboid replacements plus current tracker/min-risk selection and unmatched YOLO retained; not a pure detector oracle",
            "oracle_range_upper_bound": "GT target selection plus causal robust slope of privileged longitudinal range; no velocity/TTC label used for prediction",
        },
        "ladder": ladder,
        "supplemental_one_factor": supplemental,
        "diagnostics": summaries,
        "segments": all_segments,
        "label_exact_ceiling_not_a_ladder_prediction": {
            "description": "GT-vs-GT metric ceiling for interpreting scale only; no prediction CSV is written",
            "macro": _macro_metrics(exact_ceiling_metrics),
        },
        "caveats": [
            "All results are development cross-validation diagnostics on six previously inspected practice trips.",
            "KITTI label_2 stores zero 2-D boxes; cuboids are projected with P2 and can be visibly misregistered.",
            "Projection-unmatched but trajectory/class-plausible detections are ambiguous, not counted as definitive detector misses.",
            "Per-frame IoU association is privileged and uncertain; sensitivity at IoU 0.05/0.10/0.20 is reported.",
            "Oracle association is forced only after a projected IoU identity is confident; projection-unmatched observations retain current greedy behavior.",
            "Detector-miss and identity-switch flags are sparse detector-update events at stride 3; only coast/lost states persist between updates.",
            "KITTI rotation_y is zero in these samples and labels provide no occlusion/truncation-valid 2-D silhouette, limiting projected-box attribution.",
            "Depth exists only every fifth frame and is inventoried for diagnostics; no ladder prediction reads depth arrays.",
            "MAE-critical uses GT TTC <3 s; F1 danger uses GT TTC <2 s.",
        ],
    }
    with (output_dir / "report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with (segments_dir / "segments.json").open("w", encoding="utf-8") as stream:
        json.dump(all_segments, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def _parse_trip_list(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline P2 C1 failure analysis and oracle ladder (practice GT only)."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "predictions" / "c1_baseline_recheck" / "detection_cache",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=ROOT / "predictions" / "c1_baseline_recheck" / "six_samples",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "runs" / "c1" / "p2_oracle_ladder"
    )
    parser.add_argument(
        "--diagnostic-trips",
        type=_parse_trip_list,
        default=DEFAULT_DIAGNOSTIC_TRIPS,
        help="Comma-separated subset; default T01-Sample,T02-Sample,T06-Sample",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    args = parser.parse_args(argv)
    try:
        report = run_analysis(
            data_root=args.data_root,
            cache_dir=args.cache_dir,
            baseline_dir=args.baseline_dir,
            output_dir=args.output_dir,
            diagnostic_trips=args.diagnostic_trips,
            confidence_threshold=args.confidence_threshold,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"P2 diagnostic failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "evaluator_sha256": report["official_evaluator_sha256_after"],
                "ladder_macro": {
                    variant: report["ladder"][variant]["macro"]  # type: ignore[index]
                    for variant in LADDER_VARIANTS
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
