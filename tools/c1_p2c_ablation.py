#!/usr/bin/env python3
"""Locked, leakage-separated six-rung C1 P2-C development ablation.

Inference consumes only the lean ``image_2`` manifest, camera calibration,
ego kinematics and cached ``image_2`` detector outputs.  All 6 x 6 x 600
prediction and diagnostic rows must be written and preflighted before this
module imports either the official evaluator or the offline target audit.

The physics and accepted P2-B rows are replay-parity gates followed by
byte-exact copies of their frozen CSVs.  The remaining four rows are fresh,
causal P2-C inference.  Generated predictions, diagnostics and reports belong
under ignored artifact directories and are never source-control inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safeloop.c1.lean_loader import C1LeanTripLoader
from safeloop.c1.prediction_preflight import preflight_prediction_csv
from safeloop.c1.temporal_features import (
    CameraGeometry,
    DetectionCache,
    load_camera_geometry,
    load_detection_cache,
    locate_detection_cache,
)
from safeloop.c1.types import Detection


EXPECTED_FRAMES = 600
TRIP_IDS = tuple(f"T0{index}-Sample" for index in range(1, 7))
VARIANT_VALUES = (
    "physics",
    "p2b_full",
    "p2b_ground_plane",
    "p2b_scale_expansion",
    "p2b_class_prior",
    "p2c_full_fusion",
)
FROZEN_VARIANTS = frozenset(("physics", "p2b_full"))
FRESH_P2C_VARIANTS = VARIANT_VALUES[2:]
TARGET_AUDIT_VARIANTS = VARIANT_VALUES[1:]
EXPECTED_PREDICTION_ROWS = len(VARIANT_VALUES) * len(TRIP_IDS) * EXPECTED_FRAMES
OFFICIAL_EVALUATOR = ROOT / "team_kit" / "evaluation.py"
EXPECTED_EVALUATOR_SHA256 = (
    "674320460240797c2cb3a2b60815629c1cc1dcdcd9cc17ad52988c3dd353065b"
)
DEFAULT_CACHE_PATTERN = "{trip_id}.stride3.conf020.json.gz"
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_P2B_FULL_DIR = (
    ROOT
    / "predictions/c1_p2b_ablation_final_kf_fix/"
    "selector_association_robust_range_fallback_hysteresis"
)
REPORT_SCHEMA = "safeloop.c1.p2c_ablation.v1"
SUCCESS_REASON_CODES = frozenset(("", "valid", "ok"))
PHYSICS_MAX_MASK_MISMATCHES = 1
PHYSICS_MAX_PARITY_DELTA_S = 0.02
P2B_MAX_MASK_MISMATCHES = 0
P2B_MAX_PARITY_DELTA_S = 0.02
TARGET_LIFECYCLE_FIELDS = (
    "target_switch_count",
    "target_acquisition_count",
    "target_drop_count",
    "target_reacquisition_count",
    "target_identity_change_after_gap_count",
)


BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class DetectionStep:
    detections: tuple[Detection, ...]
    detector_update: bool
    detector_latency_ms: float | None = None


class DetectionSource(Protocol):
    def step(
        self,
        frame_id: int,
        timestamp: float,
        image_bgr: np.ndarray,
    ) -> DetectionStep: ...


class CachedDetectionSource:
    """Validated image_2 cache used for reproducible post-detector replay."""

    def __init__(
        self,
        cache: DetectionCache,
        *,
        n_frames: int,
        confidence_threshold: float,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        expected = set(range(0, n_frames, cache.stride))
        actual = set(cache.rows)
        if actual != expected:
            raise ValueError(
                f"{cache.trip_id}: incomplete detector cache; "
                f"missing={sorted(expected - actual)[:10]}, "
                f"unexpected={sorted(actual - expected)[:10]}"
            )
        self.cache = cache
        self.confidence_threshold = confidence_threshold

    def step(
        self,
        frame_id: int,
        timestamp: float,
        image_bgr: np.ndarray,
    ) -> DetectionStep:
        del image_bgr
        if frame_id not in self.cache.rows:
            return DetectionStep((), False)
        cached_timestamp = float(self.cache.timestamps[frame_id])
        if not math.isclose(cached_timestamp, timestamp, abs_tol=1e-3):
            raise ValueError(
                f"{self.cache.trip_id} frame {frame_id}: cache timestamp "
                f"{cached_timestamp} != manifest timestamp {timestamp}"
            )
        detections = tuple(
            detection
            for detection in self.cache.rows[frame_id]
            if detection.confidence >= self.confidence_threshold
        )
        return DetectionStep(detections, True)


class RuntimeResult(Protocol):
    predicted_ttc_s: float
    primary_track_id: int | None
    primary_bbox: BBox | None
    dangerous_track_ids: Sequence[int]
    target_switched: bool
    held_by_hysteresis: bool
    warning: bool
    invalid_reason: str
    ttc_source: str
    lane_source: str
    lane_confidence: float
    candidate_count: int
    downstream_latency_ms: float
    estimator_reason: str
    estimator_uncertainty_s: float
    estimator_sources: object


class RuntimeInstance(Protocol):
    def step(
        self,
        image_bgr: np.ndarray,
        detections: Sequence[Detection],
        *,
        detector_update: bool,
        timestamp: float,
        ego_speed_kmh: float,
    ) -> RuntimeResult: ...


RuntimeFactory = Callable[[CameraGeometry, object], RuntimeInstance]


@dataclass(frozen=True, slots=True)
class RuntimeRow:
    trip_id: str
    variant: str
    frame_id: int
    timestamp: float
    predicted_ttc_s: float
    runtime_predicted_ttc_s: float
    primary_track_id: int | None
    primary_bbox: BBox | None
    dangerous_track_ids: tuple[int, ...]
    target_switched: bool
    held_by_hysteresis: bool
    warning: bool
    warning_ttc_consistent: bool
    primary_marked_dangerous: bool
    primary_danger_ttc_consistent: bool
    invalid_reason: str
    ttc_source: str
    lane_source: str
    lane_confidence: float
    candidate_count: int
    estimator_reason: str
    estimator_uncertainty_s: float
    estimator_sources: str
    detector_update: bool
    detection_count: int
    ttc_jump: bool
    ttc_finite_toggle: bool
    warning_toggle: bool
    downstream_latency_ms: float
    detector_latency_ms: float | None


@dataclass(frozen=True, slots=True)
class InferenceCompletion:
    """Capability token proving that the full label-free replay exists."""

    variants: tuple[str, ...]
    trips: tuple[str, ...]
    prediction_rows: int
    prediction_root: Path
    diagnostics_root: Path
    summaries: Mapping[str, Mapping[str, object]]

    def require_complete(self) -> None:
        if self.variants != VARIANT_VALUES or self.trips != TRIP_IDS:
            raise RuntimeError("evaluation requires the locked P2-C variant/trip order")
        if self.prediction_rows != EXPECTED_PREDICTION_ROWS:
            raise RuntimeError(
                "evaluation requires exactly 6 x 6 x 600 = 21,600 predictions"
            )


DIAGNOSTIC_COLUMNS = (
    "trip_id",
    "variant",
    "frame_id",
    "timestamp",
    "predicted_ttc",
    "runtime_predicted_ttc",
    "primary_track_id",
    "primary_bbox_x1",
    "primary_bbox_y1",
    "primary_bbox_x2",
    "primary_bbox_y2",
    "dangerous_track_ids",
    "target_switched",
    "held_by_hysteresis",
    "warning",
    "warning_ttc_consistent",
    "primary_marked_dangerous",
    "primary_danger_ttc_consistent",
    "invalid_reason",
    "ttc_source",
    "lane_source",
    "lane_confidence",
    "candidate_count",
    "estimator_reason",
    "estimator_uncertainty_s",
    "estimator_sources",
    "detector_update",
    "detection_count",
    "ttc_jump",
    "ttc_finite_toggle",
    "warning_toggle",
    "downstream_latency_ms",
    "detector_latency_ms",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_evaluator_hash() -> str:
    actual = sha256_file(OFFICIAL_EVALUATOR)
    if actual != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError(
            f"official evaluator SHA-256 mismatch: {actual} != "
            f"{EXPECTED_EVALUATOR_SHA256}"
        )
    return actual


def _ttc_text(value: float) -> str:
    return "inf" if not math.isfinite(value) else f"{value:.9f}".rstrip("0").rstrip(".")


def read_prediction_rows(path: Path) -> tuple[tuple[int, float, float], ...]:
    preflight_prediction_csv(path, expected_frames=EXPECTED_FRAMES)
    output: list[tuple[int, float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            output.append(
                (int(row["frame_id"]), float(row["timestamp"]), float(row["predicted_ttc"]))
            )
    if tuple(frame_id for frame_id, _, _ in output) != tuple(range(EXPECTED_FRAMES)):
        raise RuntimeError(f"{path}: prediction rows are not ordered 0..599")
    return tuple(output)


def _atomic_prediction_csv(path: Path, rows: Sequence[RuntimeRow]) -> None:
    if len(rows) != EXPECTED_FRAMES:
        raise ValueError("prediction writer requires exactly 600 rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".csv", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("frame_id", "timestamp", "predicted_ttc")
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "frame_id": row.frame_id,
                        "timestamp": _ttc_text(row.timestamp),
                        "predicted_ttc": _ttc_text(row.predicted_ttc_s),
                    }
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    preflight_prediction_csv(path, expected_frames=EXPECTED_FRAMES)


def copy_frozen_prediction(source: Path, destination: Path) -> str:
    """Copy one accepted CSV byte-for-byte after its caller passes parity."""

    preflight_prediction_csv(source, expected_frames=EXPECTED_FRAMES)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    source_hash = sha256_file(source)
    if sha256_file(destination) != source_hash:
        raise RuntimeError(f"{destination}: frozen prediction copy is not byte-exact")
    preflight_prediction_csv(destination, expected_frames=EXPECTED_FRAMES)
    return source_hash


def _normalize_bbox(value: object, track_id: int | None) -> BBox | None:
    if track_id is None:
        if value is not None:
            raise ValueError("primary_bbox must be None when primary_track_id is None")
        return None
    # Hysteresis may deliberately retain a primary identity for a bounded
    # interval after its live track disappears.  No current bbox exists in
    # that state; preserve it as unknown coverage instead of inventing or
    # reusing stale geometry.
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("primary target requires a four-value bbox")
    if len(value) != 4:
        raise ValueError("primary_bbox must contain exactly four coordinates")
    bbox = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in bbox):
        raise ValueError("primary_bbox must be finite")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError("primary_bbox must have positive area")
    return bbox  # type: ignore[return-value]


def _normalize_sources(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return ";".join(str(item) for item in value)
    raise ValueError("estimator_sources must be a string, mapping or sequence")


def _runtime_row(
    *,
    trip_id: str,
    variant: str,
    frame_id: int,
    timestamp: float,
    result: RuntimeResult,
    detector_step: DetectionStep,
    published_ttc_s: float,
    previous: RuntimeRow | None,
) -> RuntimeRow:
    runtime_ttc = float(result.predicted_ttc_s)
    published_ttc = float(published_ttc_s)
    track_id = (
        None if result.primary_track_id is None else int(result.primary_track_id)
    )
    try:
        bbox = _normalize_bbox(result.primary_bbox, track_id)
    except ValueError as exc:
        raise ValueError(f"{variant}/{trip_id}/{frame_id}: {exc}") from exc
    invalid_reason = str(result.invalid_reason)
    if track_id is not None and not math.isfinite(published_ttc):
        if invalid_reason.strip().lower() in SUCCESS_REASON_CODES:
            raise RuntimeError(
                f"{variant}/{trip_id}/{frame_id}: non-finite primary TTC has "
                f"success reason {invalid_reason!r}"
            )
    uncertainty_s = float(result.estimator_uncertainty_s)
    if math.isnan(uncertainty_s) or uncertainty_s < 0.0:
        raise ValueError(
            f"{variant}/{trip_id}/{frame_id}: estimator uncertainty must be "
            "non-negative or inf"
        )
    warning = published_ttc < 2.0
    warning_ttc_consistent = bool(result.warning) == warning
    if not warning_ttc_consistent:
        raise RuntimeError(
            f"{variant}/{trip_id}/{frame_id}: warning and published TTC disagree"
        )
    dangerous_track_ids = tuple(int(item) for item in result.dangerous_track_ids)
    if len(set(dangerous_track_ids)) != len(dangerous_track_ids):
        raise RuntimeError(
            f"{variant}/{trip_id}/{frame_id}: dangerous_track_ids contains duplicates"
        )
    primary_marked_dangerous = (
        track_id is not None and track_id in dangerous_track_ids
    )
    primary_danger_ttc_consistent = not primary_marked_dangerous or (
        math.isfinite(published_ttc) and published_ttc < 3.0
    )
    if not primary_danger_ttc_consistent and variant in FRESH_P2C_VARIANTS:
        raise RuntimeError(
            f"{variant}/{trip_id}/{frame_id}: primary track {track_id} remains "
            f"dangerous with safe/non-finite published TTC {published_ttc}"
        )
    finite = math.isfinite(published_ttc)
    finite_toggle = (
        previous is not None
        and finite != math.isfinite(previous.predicted_ttc_s)
    )
    warning_toggle = previous is not None and warning != previous.warning
    jump = False
    if previous is not None and finite and math.isfinite(previous.predicted_ttc_s):
        elapsed = max(0.0, timestamp - previous.timestamp)
        expected = max(0.0, previous.predicted_ttc_s - elapsed)
        jump = abs(published_ttc - expected) > 1.0
    latency = float(result.downstream_latency_ms)
    if not math.isfinite(latency) or latency < 0.0:
        raise ValueError("downstream latency must be finite and non-negative")
    return RuntimeRow(
        trip_id=trip_id,
        variant=variant,
        frame_id=frame_id,
        timestamp=timestamp,
        predicted_ttc_s=published_ttc,
        runtime_predicted_ttc_s=runtime_ttc,
        primary_track_id=track_id,
        primary_bbox=bbox,
        dangerous_track_ids=dangerous_track_ids,
        target_switched=bool(result.target_switched),
        held_by_hysteresis=bool(result.held_by_hysteresis),
        warning=warning,
        warning_ttc_consistent=warning_ttc_consistent,
        primary_marked_dangerous=primary_marked_dangerous,
        primary_danger_ttc_consistent=primary_danger_ttc_consistent,
        invalid_reason=invalid_reason,
        ttc_source=str(result.ttc_source),
        lane_source=str(result.lane_source),
        lane_confidence=float(result.lane_confidence),
        candidate_count=int(result.candidate_count),
        estimator_reason=str(result.estimator_reason),
        estimator_uncertainty_s=uncertainty_s,
        estimator_sources=_normalize_sources(result.estimator_sources),
        detector_update=detector_step.detector_update,
        detection_count=len(detector_step.detections),
        ttc_jump=jump,
        ttc_finite_toggle=finite_toggle,
        warning_toggle=warning_toggle,
        downstream_latency_ms=latency,
        detector_latency_ms=detector_step.detector_latency_ms,
    )


def replay_runtime(
    loader: C1LeanTripLoader,
    runtime: RuntimeInstance,
    variant: str,
    detection_source: DetectionSource,
    *,
    published_override: Sequence[float] | None = None,
) -> tuple[RuntimeRow, ...]:
    """Run one causal trip without importing or opening any label source."""

    if published_override is not None and len(published_override) != loader.n_frames:
        raise ValueError("published override must align one-to-one with the trip")
    rows: list[RuntimeRow] = []
    for index in range(loader.n_frames):
        frame = loader.frame(index)
        if frame.frame_id != index:
            raise ValueError(
                f"{loader.trip_id}: expected frame_id {index}, got {frame.frame_id}"
            )
        timestamp = float(frame.timestamp)
        image_bgr = frame.left()
        detector_step = detection_source.step(index, timestamp, image_bgr)
        result = runtime.step(
            image_bgr,
            detector_step.detections,
            detector_update=detector_step.detector_update,
            timestamp=timestamp,
            ego_speed_kmh=float(frame.ego.get("speed_kmh", 0.0)),
        )
        published = (
            float(published_override[index])
            if published_override is not None
            else float(result.predicted_ttc_s)
        )
        rows.append(
            _runtime_row(
                trip_id=loader.trip_id,
                variant=variant,
                frame_id=index,
                timestamp=timestamp,
                result=result,
                detector_step=detector_step,
                published_ttc_s=published,
                previous=rows[-1] if rows else None,
            )
        )
    return tuple(rows)


def frozen_runtime_parity(
    rows: Sequence[RuntimeRow],
    authoritative: Sequence[tuple[int, float, float]],
    *,
    max_mask_mismatches: int,
    max_finite_delta_s: float,
) -> dict[str, object]:
    """Framewise replay gate before a frozen prediction row may be copied."""

    if len(rows) != EXPECTED_FRAMES or len(authoritative) != EXPECTED_FRAMES:
        raise RuntimeError("frozen parity requires exactly 600 aligned frames")
    mask_mismatches: list[int] = []
    finite_deltas: list[float] = []
    for row, (frame_id, timestamp, expected_ttc) in zip(rows, authoritative):
        if row.frame_id != frame_id or not math.isclose(
            row.timestamp, timestamp, abs_tol=1e-3
        ):
            raise RuntimeError("frozen runtime/prediction frame alignment mismatch")
        runtime_ttc = row.runtime_predicted_ttc_s
        if math.isfinite(runtime_ttc) != math.isfinite(expected_ttc):
            mask_mismatches.append(frame_id)
        elif math.isfinite(runtime_ttc):
            finite_deltas.append(abs(runtime_ttc - expected_ttc))
    maximum_delta = max(finite_deltas, default=0.0)
    if len(mask_mismatches) > max_mask_mismatches or maximum_delta > max_finite_delta_s:
        raise RuntimeError(
            "frozen runtime parity failed: "
            f"mask_mismatches={mask_mismatches[:10]}, "
            f"max_finite_delta_s={maximum_delta:.9f}"
        )
    return {
        "checked_frames": EXPECTED_FRAMES,
        "finite_mask_mismatch_count": len(mask_mismatches),
        "finite_mask_mismatch_frame_ids": mask_mismatches,
        "max_jointly_finite_abs_delta_s": round(maximum_delta, 9),
        "allowed_mask_mismatches": max_mask_mismatches,
        "allowed_finite_abs_delta_s": max_finite_delta_s,
    }


def write_diagnostic_csv(path: Path, rows: Sequence[RuntimeRow]) -> None:
    if len(rows) != EXPECTED_FRAMES:
        raise ValueError("diagnostic writer requires exactly 600 rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=DIAGNOSTIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            bbox = row.primary_bbox or (None, None, None, None)
            writer.writerow(
                {
                    "trip_id": row.trip_id,
                    "variant": row.variant,
                    "frame_id": row.frame_id,
                    "timestamp": _ttc_text(row.timestamp),
                    "predicted_ttc": _ttc_text(row.predicted_ttc_s),
                    "runtime_predicted_ttc": _ttc_text(
                        row.runtime_predicted_ttc_s
                    ),
                    "primary_track_id": (
                        "" if row.primary_track_id is None else row.primary_track_id
                    ),
                    "primary_bbox_x1": "" if bbox[0] is None else bbox[0],
                    "primary_bbox_y1": "" if bbox[1] is None else bbox[1],
                    "primary_bbox_x2": "" if bbox[2] is None else bbox[2],
                    "primary_bbox_y2": "" if bbox[3] is None else bbox[3],
                    "dangerous_track_ids": ";".join(
                        str(item) for item in row.dangerous_track_ids
                    ),
                    "target_switched": int(row.target_switched),
                    "held_by_hysteresis": int(row.held_by_hysteresis),
                    "warning": int(row.warning),
                    "warning_ttc_consistent": int(row.warning_ttc_consistent),
                    "primary_marked_dangerous": int(
                        row.primary_marked_dangerous
                    ),
                    "primary_danger_ttc_consistent": int(
                        row.primary_danger_ttc_consistent
                    ),
                    "invalid_reason": row.invalid_reason,
                    "ttc_source": row.ttc_source,
                    "lane_source": row.lane_source,
                    "lane_confidence": f"{row.lane_confidence:.9f}",
                    "candidate_count": row.candidate_count,
                    "estimator_reason": row.estimator_reason,
                    "estimator_uncertainty_s": _ttc_text(
                        row.estimator_uncertainty_s
                    ),
                    "estimator_sources": row.estimator_sources,
                    "detector_update": int(row.detector_update),
                    "detection_count": row.detection_count,
                    "ttc_jump": int(row.ttc_jump),
                    "ttc_finite_toggle": int(row.ttc_finite_toggle),
                    "warning_toggle": int(row.warning_toggle),
                    "downstream_latency_ms": f"{row.downstream_latency_ms:.9f}",
                    "detector_latency_ms": (
                        ""
                        if row.detector_latency_ms is None
                        else f"{row.detector_latency_ms:.9f}"
                    ),
                }
            )


def _target_lifecycle_counts(rows: Sequence[RuntimeRow]) -> dict[str, int]:
    counts = {field: 0 for field in TARGET_LIFECYCLE_FIELDS}
    previous: int | None = None
    most_recent: int | None = None
    for row in rows:
        current = row.primary_track_id
        if previous is not None and current is not None and current != previous:
            counts["target_switch_count"] += 1
        elif previous is None and current is not None:
            counts["target_acquisition_count"] += 1
            if most_recent is not None:
                counts["target_reacquisition_count"] += 1
                if current != most_recent:
                    counts["target_identity_change_after_gap_count"] += 1
        elif previous is not None and current is None:
            counts["target_drop_count"] += 1
        if current is not None:
            most_recent = current
        previous = current
    return counts


def summarize_runtime(rows: Sequence[RuntimeRow]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot summarize empty runtime rows")
    latency = np.asarray([row.downstream_latency_ms for row in rows], dtype=float)
    lifecycle = _target_lifecycle_counts(rows)
    if sum(int(row.target_switched) for row in rows) != lifecycle["target_switch_count"]:
        raise RuntimeError("target_switched flags disagree with primary track IDs")
    lane_counts = Counter(row.lane_source for row in rows)
    reason_counts = Counter(row.estimator_reason for row in rows)
    source_counts = Counter(row.estimator_sources for row in rows)
    finite_uncertainties = [
        row.estimator_uncertainty_s
        for row in rows
        if math.isfinite(row.estimator_uncertainty_s)
    ]
    return {
        "n_frames": len(rows),
        "timing_scope": "cached replay; post-detector downstream CPU only",
        "downstream_cpu_fps": round(
            1000.0 / max(float(np.mean(latency)), 1e-9), 3
        ),
        "downstream_mean_latency_ms": round(float(np.mean(latency)), 3),
        "downstream_p95_latency_ms": round(float(np.percentile(latency, 95)), 3),
        **lifecycle,
        "lane_source_counts": dict(sorted(lane_counts.items())),
        "estimator_reason_counts": dict(sorted(reason_counts.items())),
        "estimator_source_counts": dict(sorted(source_counts.items())),
        "estimator_uncertainty_finite_count": len(finite_uncertainties),
        "estimator_uncertainty_infinite_count": (
            len(rows) - len(finite_uncertainties)
        ),
        "estimator_uncertainty_mean_s": (
            round(float(np.mean(finite_uncertainties)), 6)
            if finite_uncertainties
            else None
        ),
        "estimator_uncertainty_p95_s": (
            round(float(np.percentile(finite_uncertainties, 95)), 6)
            if finite_uncertainties
            else None
        ),
        "nonfinite_prediction_count": sum(
            not math.isfinite(row.predicted_ttc_s) for row in rows
        ),
        "invalid_primary_ttc_count": sum(
            row.primary_track_id is not None
            and not math.isfinite(row.predicted_ttc_s)
            for row in rows
        ),
        "no_primary_target_count": sum(
            row.primary_track_id is None for row in rows
        ),
        "ttc_jump_count": sum(row.ttc_jump for row in rows),
        "ttc_finite_toggle_count": sum(row.ttc_finite_toggle for row in rows),
        "warning_toggle_count": sum(row.warning_toggle for row in rows),
        "warning_ttc_consistency_violation_count": sum(
            not row.warning_ttc_consistent for row in rows
        ),
        "primary_danger_ttc_consistency_violation_count": sum(
            not row.primary_danger_ttc_consistent for row in rows
        ),
    }


def _load_runtime_api() -> tuple[type[Any], Mapping[str, object]]:
    """Load deployable code without importing any label-bearing module."""

    module = importlib.import_module("safeloop.c1.p2c_runtime")
    enum_type = getattr(module, "P2CVariant")
    runtime_type = getattr(module, "P2CDeployableRuntime")
    variants = tuple(enum_type)
    values = tuple(str(variant.value) for variant in variants)
    if values != VARIANT_VALUES:
        raise RuntimeError(
            f"P2-C runtime variants {values} do not match locked protocol "
            f"{VARIANT_VALUES}"
        )
    return runtime_type, {str(variant.value): variant for variant in variants}


def _assert_prediction_layout(prediction_root: Path) -> None:
    expected_files = {f"{trip_id}.csv" for trip_id in TRIP_IDS}
    actual_directories = {
        path.name for path in prediction_root.iterdir() if path.is_dir()
    }
    unexpected = actual_directories - set(VARIANT_VALUES)
    if unexpected:
        raise RuntimeError(
            f"unexpected P2-C prediction directories: {sorted(unexpected)}"
        )
    rows = 0
    for variant in VARIANT_VALUES:
        directory = prediction_root / variant
        actual_files = {path.name for path in directory.glob("*.csv")}
        if actual_files != expected_files:
            raise RuntimeError(
                f"{variant}: expected {sorted(expected_files)}, "
                f"found {sorted(actual_files)}"
            )
        for trip_id in TRIP_IDS:
            report = preflight_prediction_csv(
                directory / f"{trip_id}.csv", expected_frames=EXPECTED_FRAMES
            )
            rows += int(report.rows)
    if rows != EXPECTED_PREDICTION_ROWS:
        raise RuntimeError(f"P2-C prediction layout has {rows} != 21,600 rows")


def run_all_inference(
    *,
    data_root: Path,
    manifest_dir: Path,
    cache_dir: Path,
    physics_dir: Path,
    accepted_p2b_full_dir: Path,
    prediction_root: Path,
    diagnostics_root: Path,
    cache_pattern: str = DEFAULT_CACHE_PATTERN,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> InferenceCompletion:
    """Materialize all 21,600 rows before returning an evaluation token."""

    runtime_type, runtime_variants = _load_runtime_api()
    prediction_root.mkdir(parents=True, exist_ok=True)
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, object]] = {}
    prediction_rows = 0
    frozen_roots = {
        "physics": physics_dir,
        "p2b_full": accepted_p2b_full_dir,
    }

    for variant in VARIANT_VALUES:
        trip_summaries: dict[str, object] = {}
        for trip_id in TRIP_IDS:
            trip_dir = data_root / trip_id
            loader = C1LeanTripLoader(
                trip_dir, manifest_dir / f"{trip_id}.json.gz"
            )
            if loader.n_frames != EXPECTED_FRAMES:
                raise RuntimeError(
                    f"{trip_id}: expected 600 manifest frames, got {loader.n_frames}"
                )
            loader.check_image_2_integrity().require_ok()
            geometry = load_camera_geometry(trip_dir)
            cache = load_detection_cache(
                locate_detection_cache(cache_dir, trip_id, pattern=cache_pattern)
            )
            if cache.trip_id != trip_id:
                raise ValueError(f"cache trip {cache.trip_id} != {trip_id}")
            source = CachedDetectionSource(
                cache,
                n_frames=EXPECTED_FRAMES,
                confidence_threshold=confidence_threshold,
            )
            accepted: tuple[tuple[int, float, float], ...] | None = None
            accepted_path: Path | None = None
            if variant in FROZEN_VARIANTS:
                accepted_path = frozen_roots[variant] / f"{trip_id}.csv"
                accepted = read_prediction_rows(accepted_path)
                for frame_id, timestamp, _ in accepted:
                    if not math.isclose(
                        loader.frame(frame_id).timestamp, timestamp, abs_tol=1e-3
                    ):
                        raise ValueError(
                            f"{variant}/{trip_id}/{frame_id}: frozen timestamp "
                            "does not match lean manifest"
                        )
            runtime = runtime_type(geometry, runtime_variants[variant])
            rows = replay_runtime(
                loader,
                runtime,
                variant,
                source,
                published_override=(
                    tuple(row[2] for row in accepted) if accepted is not None else None
                ),
            )
            if len(rows) != EXPECTED_FRAMES or tuple(
                row.frame_id for row in rows
            ) != tuple(range(EXPECTED_FRAMES)):
                raise RuntimeError(f"{variant}/{trip_id}: replay is not exact 0..599")

            parity: dict[str, object] | None = None
            output_path = prediction_root / variant / f"{trip_id}.csv"
            frozen_hash: str | None = None
            if accepted is not None and accepted_path is not None:
                parity = frozen_runtime_parity(
                    rows,
                    accepted,
                    max_mask_mismatches=(
                        PHYSICS_MAX_MASK_MISMATCHES
                        if variant == "physics"
                        else P2B_MAX_MASK_MISMATCHES
                    ),
                    max_finite_delta_s=(
                        PHYSICS_MAX_PARITY_DELTA_S
                        if variant == "physics"
                        else P2B_MAX_PARITY_DELTA_S
                    ),
                )
                frozen_hash = copy_frozen_prediction(accepted_path, output_path)
            else:
                _atomic_prediction_csv(output_path, rows)
            write_diagnostic_csv(
                diagnostics_root / variant / f"{trip_id}.csv", rows
            )
            summary = summarize_runtime(rows)
            if parity is not None:
                summary["frozen_runtime_parity"] = parity
                summary["frozen_prediction_sha256"] = frozen_hash
            trip_summaries[trip_id] = summary
            prediction_rows += len(rows)
        summaries[variant] = trip_summaries

    _assert_prediction_layout(prediction_root)
    completion = InferenceCompletion(
        variants=VARIANT_VALUES,
        trips=TRIP_IDS,
        prediction_rows=prediction_rows,
        prediction_root=prediction_root,
        diagnostics_root=diagnostics_root,
        summaries=summaries,
    )
    completion.require_complete()
    return completion


def confusion_counts(
    pairs: Mapping[int, tuple[float, float]],
    *,
    danger_threshold_s: float = 2.0,
) -> dict[str, int]:
    predicted = {
        frame_id for frame_id, (pred, _) in pairs.items() if pred < danger_threshold_s
    }
    actual = {
        frame_id for frame_id, (_, truth) in pairs.items() if truth < danger_threshold_s
    }
    return {
        "tp": len(predicted & actual),
        "fp": len(predicted - actual),
        "fn": len(actual - predicted),
        "tn": len(pairs) - len(predicted | actual),
    }


def macro_metrics(per_trip: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(per_trip) != len(TRIP_IDS):
        raise ValueError("P2-C macro metrics require exactly six trips")
    valid_mae = [
        float(row["mae_critical"])
        for row in per_trip
        if float(row["mae_critical"]) >= 0.0
    ]
    return {
        "n_trips": len(per_trip),
        "n_frames": sum(int(row["n_frames"]) for row in per_trip),
        "mae_critical": round(float(np.mean(valid_mae)), 3) if valid_mae else None,
        "inv_ttc_mae": round(
            float(np.mean([float(row["inv_ttc_mae"]) for row in per_trip])), 4
        ),
        "precision": round(
            float(np.mean([float(row["precision"]) for row in per_trip])), 3
        ),
        "recall": round(
            float(np.mean([float(row["recall"]) for row in per_trip])), 3
        ),
        "f1": round(
            float(np.mean([float(row["f1"]) for row in per_trip])), 3
        ),
        "composite_score": round(
            float(np.mean([float(row["composite_score"]) for row in per_trip])),
            1,
        ),
    }


def invalid_primary_ttc_audit(
    diagnostic_path: Path,
    truth_ttc: Mapping[int, float],
) -> dict[str, object]:
    """Offline-only disjoint GT-zone audit for non-finite primary TTC."""

    zones = Counter({"danger": 0, "critical": 0, "non_critical": 0})
    reasons: Counter[str] = Counter()
    frame_ids: list[int] = []
    with diagnostic_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != DIAGNOSTIC_COLUMNS:
            raise RuntimeError(f"unexpected diagnostic schema: {diagnostic_path}")
        for row in reader:
            frame_id = int(row["frame_id"])
            frame_ids.append(frame_id)
            predicted = float(row["predicted_ttc"])
            if not row["primary_track_id"].strip() or math.isfinite(predicted):
                continue
            truth = float(truth_ttc[frame_id])
            zone = (
                "danger"
                if truth < 2.0
                else "critical"
                if truth < 3.0
                else "non_critical"
            )
            reason = row["invalid_reason"].strip()
            if reason.lower() in SUCCESS_REASON_CODES:
                raise RuntimeError(
                    f"{diagnostic_path} frame {frame_id}: invalid TTC has "
                    f"success reason {reason!r}"
                )
            zones[zone] += 1
            reasons[reason] += 1
    if frame_ids != list(range(EXPECTED_FRAMES)):
        raise RuntimeError(f"{diagnostic_path}: diagnostics are not exact 0..599")
    total = sum(zones.values())
    return {
        "danger": zones["danger"],
        "critical": zones["critical"],
        "non_critical": zones["non_critical"],
        "total": total,
        "reason_counts": dict(sorted(reasons.items())),
    }


def aggregate_invalid_audits(
    per_trip: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if tuple(per_trip) != TRIP_IDS:
        raise ValueError("invalid TTC aggregation requires locked trip order")
    reasons: Counter[str] = Counter()
    pooled: dict[str, int] = {}
    for field in ("danger", "critical", "non_critical", "total"):
        pooled[field] = sum(int(report[field]) for report in per_trip.values())
    for report in per_trip.values():
        raw_reasons = report["reason_counts"]
        if not isinstance(raw_reasons, Mapping):
            raise TypeError("invalid TTC reason_counts must be a mapping")
        reasons.update({str(key): int(value) for key, value in raw_reasons.items()})
    return {
        "per_trip": dict(per_trip),
        "pooled": {**pooled, "reason_counts": dict(sorted(reasons.items()))},
        "macro_mean_per_trip": {
            field: pooled[field] / len(TRIP_IDS)
            for field in ("danger", "critical", "non_critical", "total")
        },
    }


def _read_variant_diagnostics(
    diagnostics_root: Path, variant: str
) -> tuple[tuple[float, ...], Mapping[str, Mapping[str, object]]]:
    latency: list[float] = []
    rows_by_trip: dict[str, dict[str, object]] = {}
    for trip_id in TRIP_IDS:
        path = diagnostics_root / variant / f"{trip_id}.csv"
        trip_latency: list[float] = []
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != DIAGNOSTIC_COLUMNS:
                raise RuntimeError(f"unexpected diagnostic schema: {path}")
            ids: list[int] = []
            for row in reader:
                if row["trip_id"] != trip_id or row["variant"] != variant:
                    raise RuntimeError(f"diagnostic identity mismatch: {path}")
                ids.append(int(row["frame_id"]))
                value = float(row["downstream_latency_ms"])
                if not math.isfinite(value) or value < 0.0:
                    raise RuntimeError(f"invalid downstream latency: {path}")
                trip_latency.append(value)
        if ids != list(range(EXPECTED_FRAMES)):
            raise RuntimeError(f"{path}: diagnostics are not exact 0..599")
        latency.extend(trip_latency)
        rows_by_trip[trip_id] = {
            "n_frames": len(trip_latency),
            "downstream_mean_latency_ms": float(np.mean(trip_latency)),
            "downstream_p95_latency_ms": float(np.percentile(trip_latency, 95)),
        }
    if len(latency) != len(TRIP_IDS) * EXPECTED_FRAMES:
        raise RuntimeError("pooled P2-C diagnostics do not contain 3,600 rows")
    return tuple(latency), rows_by_trip


def aggregate_runtime_summaries(
    per_trip: Mapping[str, Mapping[str, object]],
    downstream_latency_ms: Sequence[float],
) -> dict[str, object]:
    if tuple(per_trip) != TRIP_IDS:
        raise ValueError("runtime aggregation requires locked trip order")
    expected = len(TRIP_IDS) * EXPECTED_FRAMES
    latency = np.asarray(downstream_latency_ms, dtype=float)
    if latency.shape != (expected,) or not np.isfinite(latency).all():
        raise RuntimeError("runtime aggregation requires 3,600 finite timings")

    def sum_field(field: str) -> int:
        return sum(int(report[field]) for report in per_trip.values())

    def merge_counter(field: str) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for report in per_trip.values():
            values = report[field]
            if not isinstance(values, Mapping):
                raise TypeError(f"{field} must be a mapping")
            counter.update({str(key): int(value) for key, value in values.items()})
        return dict(sorted(counter.items()))

    return {
        "n_frames": expected,
        "timing_scope": "cached replay; post-detector downstream CPU only",
        "downstream_cpu_fps": round(
            1000.0 / max(float(np.mean(latency)), 1e-9), 3
        ),
        "downstream_mean_latency_ms": round(float(np.mean(latency)), 3),
        "downstream_p95_latency_ms": round(float(np.percentile(latency, 95)), 3),
        **{field: sum_field(field) for field in TARGET_LIFECYCLE_FIELDS},
        "lane_source_counts": merge_counter("lane_source_counts"),
        "estimator_reason_counts": merge_counter("estimator_reason_counts"),
        "estimator_source_counts": merge_counter("estimator_source_counts"),
        "nonfinite_prediction_count": sum_field("nonfinite_prediction_count"),
        "invalid_primary_ttc_count": sum_field("invalid_primary_ttc_count"),
        "no_primary_target_count": sum_field("no_primary_target_count"),
        "ttc_jump_count": sum_field("ttc_jump_count"),
        "ttc_finite_toggle_count": sum_field("ttc_finite_toggle_count"),
        "warning_toggle_count": sum_field("warning_toggle_count"),
        "warning_ttc_consistency_violation_count": sum_field(
            "warning_ttc_consistency_violation_count"
        ),
        "primary_danger_ttc_consistency_violation_count": sum_field(
            "primary_danger_ttc_consistency_violation_count"
        ),
    }


def build_acceptance(variants: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    p2b = variants["p2b_full"]
    full = variants["p2c_full_fusion"]
    physics_macro = float(variants["physics"]["macro"]["composite_score"])  # type: ignore[index]
    p2b_macro = float(p2b["macro"]["composite_score"])  # type: ignore[index]
    full_macro = float(full["macro"]["composite_score"])  # type: ignore[index]
    if physics_macro != 55.2:
        raise RuntimeError(f"frozen physics score drifted: {physics_macro} != 55.2")
    if p2b_macro != 60.7:
        raise RuntimeError(f"accepted P2-B score drifted: {p2b_macro} != 60.7")
    p2b_trips = {
        str(row["trip_id"]): row for row in p2b["per_trip"]  # type: ignore[index]
    }
    full_trips = {
        str(row["trip_id"]): row for row in full["per_trip"]  # type: ignore[index]
    }
    trip_delta = {
        trip_id: round(
            float(full_trips[trip_id]["composite_score"])
            - float(p2b_trips[trip_id]["composite_score"]),
            1,
        )
        for trip_id in TRIP_IDS
    }
    t02_p2b_inv = float(p2b_trips["T02-Sample"]["inv_ttc_mae"])
    t02_full_inv = float(full_trips["T02-Sample"]["inv_ttc_mae"])
    t06_confusion = full["fp_fn"]["T06-Sample"]  # type: ignore[index]
    p2b_invalid = p2b["invalid_primary_ttc"]["pooled"]  # type: ignore[index]
    full_invalid = full["invalid_primary_ttc"]["pooled"]  # type: ignore[index]
    full_runtime = full["runtime"]  # type: ignore[index]
    warning_consistency_violations = {
        variant: int(
            report["runtime"]["warning_ttc_consistency_violation_count"]  # type: ignore[index]
        )
        for variant, report in variants.items()
    }
    primary_danger_consistency_violations = {
        variant: int(
            report["runtime"][  # type: ignore[index]
                "primary_danger_ttc_consistency_violation_count"
            ]
        )
        for variant, report in variants.items()
    }
    invalid_danger_p2b = int(p2b_invalid["danger"])
    invalid_danger_full = int(full_invalid["danger"])
    invalid_danger_delta = invalid_danger_full - invalid_danger_p2b
    gates = {
        "macro_strictly_above_60_7": full_macro > 60.7,
        "preferred_macro_at_least_65": full_macro >= 65.0,
        "no_trip_declines_more_than_5": all(
            delta >= -5.0 for delta in trip_delta.values()
        ),
        "t02_inverse_ttc_improved": t02_full_inv < t02_p2b_inv,
        "t06_false_negatives_at_most_1": int(t06_confusion["fn"]) <= 1,
        "danger_invalid_ttc_not_increased": invalid_danger_delta <= 0,
        "post_detector_p95_under_25_ms": (
            float(full_runtime["downstream_p95_latency_ms"]) < 25.0
        ),
        "warning_ttc_consistent_all_rungs": all(
            count == 0 for count in warning_consistency_violations.values()
        ),
        "primary_danger_ttc_consistent_p2c_rungs": all(
            primary_danger_consistency_violations[variant] == 0
            for variant in FRESH_P2C_VARIANTS
        ),
    }
    required = tuple(key for key in gates if key != "preferred_macro_at_least_65")
    return {
        "physics_macro_composite": physics_macro,
        "p2b_macro_composite": p2b_macro,
        "p2c_full_macro_composite": full_macro,
        "macro_delta_vs_p2b": round(full_macro - p2b_macro, 1),
        "trip_composite_delta_vs_p2b": trip_delta,
        "t02_inverse_ttc": {
            "p2b_full": t02_p2b_inv,
            "p2c_full_fusion": t02_full_inv,
            "improvement": round(t02_p2b_inv - t02_full_inv, 4),
        },
        "t06_false_negatives": int(t06_confusion["fn"]),
        "danger_invalid_ttc": {
            "definition": (
                "primary target exists, published TTC is non-finite, and "
                "offline GT TTC is <2 seconds"
            ),
            "p2b_full": invalid_danger_p2b,
            "p2c_full_fusion": invalid_danger_full,
            "delta": invalid_danger_delta,
            "gate_pass": invalid_danger_delta <= 0,
        },
        "warning_ttc_consistency_violations": warning_consistency_violations,
        "primary_danger_ttc_consistency_violations": (
            primary_danger_consistency_violations
        ),
        "legacy_p2b_primary_danger_consistency_caveat": (
            "accepted P2-B is reported but not blocked by its legacy dangerous-ID "
            "semantics; all four estimator P2-C rungs fail closed"
        ),
        "p2c_full_post_detector_p95_latency_ms": float(
            full_runtime["downstream_p95_latency_ms"]
        ),
        **gates,
        "required_gate_pass": all(bool(gates[key]) for key in required),
    }


def evaluate_predictions(
    completion: InferenceCompletion,
    *,
    data_root: Path,
) -> dict[str, object]:
    """Late label-bearing boundary, callable only with a complete token."""

    completion.require_complete()
    _assert_prediction_layout(completion.prediction_root)

    # These imports are intentionally delayed until all 21,600 causal rows
    # exist.  The target tool transitively imports practice-only oracle code.
    evaluator = importlib.import_module("team_kit.evaluation")
    target_audit = importlib.import_module("tools.c1_p2c_target_audit")
    truth_by_trip = {
        trip_id: evaluator.load_ground_truth_from_trip(data_root / trip_id)
        for trip_id in TRIP_IDS
    }
    variants: dict[str, dict[str, object]] = {}
    for variant in VARIANT_VALUES:
        per_trip: list[dict[str, object]] = []
        confusions: dict[str, dict[str, int]] = {}
        invalid_by_trip: dict[str, Mapping[str, object]] = {}
        for trip_id in TRIP_IDS:
            predictions = evaluator.load_predictions(
                completion.prediction_root / variant / f"{trip_id}.csv"
            )
            truth = truth_by_trip[trip_id]
            if set(predictions) != set(range(EXPECTED_FRAMES)):
                raise RuntimeError(f"{variant}/{trip_id}: evaluator did not load 600 rows")
            if set(truth.ttc) != set(range(EXPECTED_FRAMES)):
                raise RuntimeError(f"{trip_id}: trusted TTC is not exact 0..599")
            pairs = {
                frame_id: (
                    predictions[frame_id].predicted_ttc,
                    truth.ttc[frame_id],
                )
                for frame_id in range(EXPECTED_FRAMES)
            }
            metric = asdict(evaluator.compute_trip_metrics(trip_id, pairs))
            if int(metric["n_frames"]) != EXPECTED_FRAMES:
                raise RuntimeError(f"{variant}/{trip_id}: official metric row mismatch")
            per_trip.append(metric)
            confusions[trip_id] = confusion_counts(pairs)
            invalid_by_trip[trip_id] = invalid_primary_ttc_audit(
                completion.diagnostics_root / variant / f"{trip_id}.csv",
                truth.ttc,
            )
        latency, _ = _read_variant_diagnostics(
            completion.diagnostics_root, variant
        )
        variants[variant] = {
            "n_predictions": sum(int(row["n_frames"]) for row in per_trip),
            "per_trip": per_trip,
            "macro": macro_metrics(per_trip),
            "fp_fn": confusions,
            "invalid_primary_ttc": aggregate_invalid_audits(invalid_by_trip),
            "runtime_per_trip": completion.summaries[variant],
            "runtime": aggregate_runtime_summaries(
                completion.summaries[variant], latency
            ),
        }

    target_reports: dict[str, object] = {}
    for variant in TARGET_AUDIT_VARIANTS:
        audit = target_audit.run_audit(
            data_root=data_root,
            diagnostics_root=completion.diagnostics_root / variant,
            trips=TRIP_IDS,
        )
        target_reports[variant] = audit
        variants[variant]["target_audit"] = audit
    return {
        "variants": variants,
        "target_audit_variants": target_reports,
        "acceptance": build_acceptance(variants),
    }


def _atomic_json(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".json", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_ablation(args: argparse.Namespace) -> Mapping[str, object]:
    evaluator_sha_before = require_evaluator_hash()
    completion = run_all_inference(
        data_root=args.data_root,
        manifest_dir=args.manifest_dir,
        cache_dir=args.cache_dir,
        physics_dir=args.physics_dir,
        accepted_p2b_full_dir=args.accepted_p2b_full_dir,
        prediction_root=args.prediction_root,
        diagnostics_root=args.run_root / "diagnostics",
        cache_pattern=args.cache_pattern,
        confidence_threshold=args.confidence_threshold,
    )
    evaluation = evaluate_predictions(completion, data_root=args.data_root)
    evaluator_sha_after = require_evaluator_hash()
    if evaluator_sha_after != evaluator_sha_before:
        raise RuntimeError("official evaluator changed during P2-C ablation")
    report: dict[str, object] = {
        "report_schema": REPORT_SCHEMA,
        "protocol": "P2-C development score; not external held-out evaluation",
        "runtime_input_boundary": (
            "sanitized manifest + image_2 + image_2 detector cache + camera "
            "calibration + ego kinematics; frozen CSV only for physics/P2-B "
            "published rows after runtime parity"
        ),
        "label_boundary": (
            "official evaluator, trip GT and offline projected-target audit are "
            "loaded only after all 21,600 predictions and diagnostics pass preflight"
        ),
        "runtime_consistency_protocol": {
            "warning": (
                "fail closed on every rung unless warning is exactly equivalent "
                "to published TTC <2 seconds"
            ),
            "primary_danger": (
                "fail closed on all four estimator P2-C rungs if the primary "
                "track remains in dangerous_track_ids while published TTC is "
                "non-finite or >=3 seconds"
            ),
            "accepted_p2b_caveat": (
                "legacy primary-danger inconsistencies are counted and reported "
                "for the frozen P2-B row but do not alter its accepted CSV"
            ),
            "invalid_danger": (
                "offline count where a primary exists, published TTC is non-finite, "
                "and GT TTC is <2 seconds"
            ),
        },
        "official_evaluator_sha256_before": evaluator_sha_before,
        "official_evaluator_sha256_after": evaluator_sha_after,
        "variant_order": list(completion.variants),
        "trips": list(completion.trips),
        "prediction_rows": completion.prediction_rows,
        **evaluation,
    }
    _atomic_json(args.run_root / "report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the locked six-rung causal C1 P2-C development ablation."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--manifest-dir", type=Path, default=ROOT / "runs/c1/input_manifests"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "predictions/c1_baseline_recheck/detection_cache",
    )
    parser.add_argument(
        "--physics-dir",
        type=Path,
        default=ROOT / "predictions/c1_baseline_recheck/six_samples",
    )
    parser.add_argument(
        "--accepted-p2b-full-dir", type=Path, default=DEFAULT_P2B_FULL_DIR
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=ROOT / "predictions/c1_p2c_ablation",
    )
    parser.add_argument(
        "--run-root", type=Path, default=ROOT / "runs/c1/p2c_ablation"
    )
    parser.add_argument("--cache-pattern", default=DEFAULT_CACHE_PATTERN)
    parser.add_argument(
        "--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_ablation(args)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"P2-C ablation failed: {exc}", file=sys.stderr)
        return 2
    compact = {
        variant: values["macro"]
        for variant, values in report["variants"].items()  # type: ignore[union-attr]
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
