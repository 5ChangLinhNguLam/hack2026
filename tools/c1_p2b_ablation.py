#!/usr/bin/env python3
"""Locked, leakage-separated development ablation for C1 P2-B.

The runtime phase consumes only sanitized manifests, ``image_2``, the
``image_2`` detector cache, camera calibration, and the frozen authoritative
physics predictions.  It writes prediction and causal diagnostic CSVs before
the evaluation phase is allowed to import the official evaluator or open any
label-bearing trip record.

Cached replay timing is explicitly post-detector.  ``--live-yolo-cpu-benchmark``
uses the same frame/runtime path to add an optional detector-inclusive CPU
measurement; it does not replace the reproducible cached ablation predictions.
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
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safeloop.c1.detector import OpenCVDnnYoloDetector
from safeloop.c1.lean_loader import C1LeanTripLoader
from safeloop.c1.p2b_runtime import (
    LOCKED_P2_VARIANTS,
    P2DeployableRuntime,
    P2FrameResult,
    P2Variant,
)
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
DIAGNOSTIC_TRIPS = frozenset(("T01-Sample", "T02-Sample", "T06-Sample"))
ABLATION_VARIANTS = (
    P2Variant.PHYSICS_CURRENT,
    P2Variant.CORRIDOR_SELECTOR,
    P2Variant.SELECTOR_CLASS_HISTORY,
    P2Variant.SELECTOR_ROBUST_RANGE,
    P2Variant.FULL,
)
OFFICIAL_EVALUATOR = ROOT / "team_kit" / "evaluation.py"
EXPECTED_EVALUATOR_SHA256 = (
    "674320460240797c2cb3a2b60815629c1cc1dcdcd9cc17ad52988c3dd353065b"
)
DEFAULT_CACHE_PATTERN = "{trip_id}.stride3.conf020.json.gz"
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
REPORT_SCHEMA = "safeloop.c1.p2b_ablation.v2"
SHARED_DETECTOR_TIMING_SCHEMA = "safeloop.c1.shared_detector_timing.v1"
TARGET_LIFECYCLE_COUNT_DEFINITIONS: Mapping[str, str] = {
    "target_switch_count": (
        "direct ID change between consecutive frames where both primary "
        "target IDs are non-null"
    ),
    "target_acquisition_count": (
        "transition from no primary target to a primary target, including "
        "the first acquisition in a trip"
    ),
    "target_drop_count": (
        "transition from a primary target to no primary target on the next frame"
    ),
    "target_reacquisition_count": (
        "transition from no primary target to a primary target after the trip "
        "has previously had a primary target"
    ),
    "target_identity_change_after_gap_count": (
        "reacquisition whose primary ID differs from the most recent non-null "
        "primary ID before the gap"
    ),
}
CORRIDOR_SOURCES = ("lane", "fixed")
LANE_SOURCE_COVERAGE_DEFINITION = (
    "per-source frame count divided by all frames in the same trip or pooled "
    "runtime; 'fixed' is the calibrated fallback corridor"
)


@dataclass(frozen=True, slots=True)
class DetectionStep:
    detections: tuple[Detection, ...]
    detector_update: bool
    detector_latency_ms: float | None


class DetectionSource(Protocol):
    """One-camera detector boundary shared by cached and live timing paths."""

    label: str

    def step(
        self,
        frame_id: int,
        timestamp: float,
        image_bgr: np.ndarray,
    ) -> DetectionStep: ...


class CachedDetectionSource:
    """Read an already validated ``image_2`` cache; detector time is excluded."""

    label = "cached_image_2_post_detector"

    def __init__(
        self,
        cache: DetectionCache,
        *,
        n_frames: int,
        confidence_threshold: float,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence threshold must be in [0, 1]")
        expected = set(range(0, n_frames, cache.stride))
        actual = set(cache.rows)
        if actual != expected:
            raise ValueError(
                f"{cache.trip_id}: incomplete cache: "
                f"missing={sorted(expected - actual)[:10]}, "
                f"unexpected={sorted(actual - expected)[:10]}"
            )
        self._cache = cache
        self._threshold = confidence_threshold

    def step(
        self,
        frame_id: int,
        timestamp: float,
        image_bgr: np.ndarray,
    ) -> DetectionStep:
        del image_bgr
        if frame_id not in self._cache.rows:
            return DetectionStep((), False, None)
        cached_timestamp = self._cache.timestamps[frame_id]
        if not math.isclose(cached_timestamp, timestamp, abs_tol=1e-3):
            raise ValueError(
                f"{self._cache.trip_id} frame {frame_id}: cache timestamp "
                f"{cached_timestamp} != manifest timestamp {timestamp}"
            )
        detections = tuple(
            detection
            for detection in self._cache.rows[frame_id]
            if detection.confidence >= self._threshold
        )
        return DetectionStep(detections, True, None)


class LiveYoloCpuSource:
    """Optional CPU detector source used only for an inference benchmark."""

    label = "live_image_2_yolo_cpu"

    def __init__(self, detector: OpenCVDnnYoloDetector, *, stride: int) -> None:
        if stride < 1:
            raise ValueError("live detector stride must be >= 1")
        if detector.device != "cpu":
            raise ValueError("live CPU benchmark requires a CPU detector")
        self._detector = detector
        self._stride = stride

    def step(
        self,
        frame_id: int,
        timestamp: float,
        image_bgr: np.ndarray,
    ) -> DetectionStep:
        del timestamp
        if frame_id % self._stride:
            return DetectionStep((), False, 0.0)
        started = time.perf_counter()
        detections = tuple(self._detector.detect(image_bgr))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return DetectionStep(detections, True, elapsed_ms)


@dataclass(frozen=True, slots=True)
class RuntimeRow:
    trip_id: str
    variant: str
    frame_id: int
    timestamp: float
    predicted_ttc_s: float
    primary_track_id: int | None
    dangerous_track_ids: tuple[int, ...]
    target_switched: bool
    held_by_hysteresis: bool
    warning: bool
    invalid_reason: str
    ttc_source: str
    lane_source: str
    lane_confidence: float
    candidate_count: int
    detector_update: bool
    detection_count: int
    ttc_jump: bool
    ttc_finite_toggle: bool
    warning_toggle: bool
    downstream_latency_ms: float
    detector_latency_ms: float | None


@dataclass(frozen=True, slots=True)
class InferenceCompletion:
    """Capability token proving that all predictions exist before scoring."""

    variants: tuple[str, ...]
    trips: tuple[str, ...]
    prediction_rows: int
    prediction_root: Path
    diagnostics_root: Path
    summaries: Mapping[str, Mapping[str, object]]

    def require_complete(self) -> None:
        expected_variants = tuple(variant.value for variant in ABLATION_VARIANTS)
        if self.variants != expected_variants or self.trips != TRIP_IDS:
            raise RuntimeError("evaluation requires the locked five-variant inference")
        if self.prediction_rows != len(ABLATION_VARIANTS) * len(TRIP_IDS) * EXPECTED_FRAMES:
            raise RuntimeError("evaluation requires exactly 5 x 6 x 600 predictions")


@dataclass(frozen=True, slots=True)
class SharedDetectorTiming:
    """One warm-up-excluded live YOLO CPU sample shared by every rung."""

    trip_id: str
    source_frames: int
    stride: int
    warmup_detector_updates: int
    measured_detector_updates: int
    latency_ms: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.measured_detector_updates != len(self.latency_ms):
            raise ValueError("detector timing count does not match latency samples")
        if not self.latency_ms or not all(
            math.isfinite(value) and value >= 0.0 for value in self.latency_ms
        ):
            raise ValueError("detector timing sample must be finite and non-empty")

    def report(self) -> dict[str, object]:
        values = np.asarray(self.latency_ms, dtype=float)
        return {
            "schema": SHARED_DETECTOR_TIMING_SCHEMA,
            "trip_id": self.trip_id,
            "source_frames": self.source_frames,
            "stride": self.stride,
            "warmup_detector_updates_excluded": self.warmup_detector_updates,
            "measured_detector_updates": self.measured_detector_updates,
            # Full-precision samples remain inline and in measurement order so
            # the recycled detector-inclusive mean and p95 can be reproduced
            # exactly from report.json without an untracked sidecar.
            "latency_sample_unit": "ms",
            "latency_sample_order": (
                "detector-update measurement order after excluded warm-up"
            ),
            "latency_samples_ms": [float(value) for value in self.latency_ms],
            "detector_cpu_fps": round(
                1000.0 / max(float(np.mean(values)), 1e-9), 3
            ),
            "detector_mean_latency_ms": round(float(np.mean(values)), 3),
            "detector_p95_latency_ms": round(float(np.percentile(values, 95)), 3),
            "method": (
                "live YOLO ONNX/OpenCV CPU on image_2 detector-update frames; "
                "model construction, warm-up updates, and image decode excluded"
            ),
        }


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
    if not math.isfinite(value):
        return "inf"
    return f"{value:.9f}".rstrip("0").rstrip(".")


def read_prediction_rows(path: Path) -> tuple[tuple[int, float, float], ...]:
    """Read a preflighted prediction file without accessing trip labels."""

    preflight_prediction_csv(path, expected_frames=EXPECTED_FRAMES)
    rows: list[tuple[int, float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            rows.append(
                (
                    int(raw["frame_id"]),
                    float(raw["timestamp"]),
                    float(raw["predicted_ttc"]),
                )
            )
    if tuple(frame_id for frame_id, _, _ in rows) != tuple(range(EXPECTED_FRAMES)):
        raise RuntimeError(f"{path}: authoritative rows are not ordered 0..599")
    return tuple(rows)


def _atomic_prediction_csv(path: Path, rows: Sequence[RuntimeRow]) -> None:
    if len(rows) != EXPECTED_FRAMES:
        raise ValueError("prediction writer requires exactly 600 runtime rows")
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


DIAGNOSTIC_COLUMNS = (
    "trip_id",
    "variant",
    "frame_id",
    "timestamp",
    "predicted_ttc",
    "primary_track_id",
    "dangerous_track_ids",
    "target_switched",
    "held_by_hysteresis",
    "warning",
    "invalid_reason",
    "ttc_source",
    "lane_source",
    "lane_confidence",
    "candidate_count",
    "detector_update",
    "detection_count",
    "ttc_jump",
    "ttc_finite_toggle",
    "warning_toggle",
    "downstream_latency_ms",
    "detector_latency_ms",
)


def write_diagnostic_csv(path: Path, rows: Sequence[RuntimeRow]) -> None:
    """Write causal runtime diagnostics separately from submission columns."""

    if not rows:
        raise ValueError("cannot write empty runtime diagnostics")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=DIAGNOSTIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "trip_id": row.trip_id,
                    "variant": row.variant,
                    "frame_id": row.frame_id,
                    "timestamp": _ttc_text(row.timestamp),
                    "predicted_ttc": _ttc_text(row.predicted_ttc_s),
                    "primary_track_id": "" if row.primary_track_id is None else row.primary_track_id,
                    "dangerous_track_ids": ";".join(map(str, row.dangerous_track_ids)),
                    "target_switched": int(row.target_switched),
                    "held_by_hysteresis": int(row.held_by_hysteresis),
                    "warning": int(row.warning),
                    "invalid_reason": row.invalid_reason,
                    "ttc_source": row.ttc_source,
                    "lane_source": row.lane_source,
                    "lane_confidence": f"{row.lane_confidence:.6f}",
                    "candidate_count": row.candidate_count,
                    "detector_update": int(row.detector_update),
                    "detection_count": row.detection_count,
                    "ttc_jump": int(row.ttc_jump),
                    "ttc_finite_toggle": int(row.ttc_finite_toggle),
                    "warning_toggle": int(row.warning_toggle),
                    "downstream_latency_ms": f"{row.downstream_latency_ms:.6f}",
                    "detector_latency_ms": (
                        "" if row.detector_latency_ms is None
                        else f"{row.detector_latency_ms:.6f}"
                    ),
                }
            )


def _runtime_row(
    *,
    trip_id: str,
    variant: P2Variant,
    frame_id: int,
    timestamp: float,
    result: P2FrameResult,
    detector_step: DetectionStep,
    published_ttc_s: float,
    previous: RuntimeRow | None,
) -> RuntimeRow:
    finite = math.isfinite(published_ttc_s)
    finite_toggle = previous is not None and finite != math.isfinite(previous.predicted_ttc_s)
    warning = published_ttc_s < 2.0
    warning_toggle = previous is not None and warning != previous.warning
    jump = False
    if previous is not None and finite and math.isfinite(previous.predicted_ttc_s):
        delta_t = max(0.0, timestamp - previous.timestamp)
        expected = max(0.0, previous.predicted_ttc_s - delta_t)
        jump = abs(published_ttc_s - expected) > 1.0
    invalid_reason = result.invalid_reason
    if not finite and variant == P2Variant.PHYSICS_CURRENT:
        invalid_reason = "no_finite_physics_ttc"
    return RuntimeRow(
        trip_id=trip_id,
        variant=variant.value,
        frame_id=frame_id,
        timestamp=timestamp,
        predicted_ttc_s=published_ttc_s,
        primary_track_id=result.primary_track_id,
        dangerous_track_ids=result.dangerous_track_ids,
        target_switched=result.target_switched,
        held_by_hysteresis=result.held_by_hysteresis,
        warning=warning,
        invalid_reason=invalid_reason,
        ttc_source=result.ttc_source,
        lane_source=result.lane_source,
        lane_confidence=result.lane_confidence,
        candidate_count=result.candidate_count,
        detector_update=detector_step.detector_update,
        detection_count=len(detector_step.detections),
        ttc_jump=jump,
        ttc_finite_toggle=finite_toggle,
        warning_toggle=warning_toggle,
        downstream_latency_ms=result.downstream_latency_ms,
        detector_latency_ms=detector_step.detector_latency_ms,
    )


def replay_runtime(
    loader: C1LeanTripLoader,
    geometry: CameraGeometry,
    variant: P2Variant,
    detection_source: DetectionSource,
    *,
    published_override: Sequence[float] | None = None,
    frame_limit: int | None = None,
) -> tuple[RuntimeRow, ...]:
    """Run the shared causal image/runtime path without opening trip labels."""

    stop = loader.n_frames if frame_limit is None else min(loader.n_frames, frame_limit)
    if stop <= 0:
        raise ValueError("runtime replay requires at least one frame")
    if published_override is not None and len(published_override) != stop:
        raise ValueError("published override must have one value per replayed frame")
    runtime = P2DeployableRuntime(geometry, variant)
    rows: list[RuntimeRow] = []
    for index in range(stop):
        frame = loader.frame(index)
        if frame.frame_id != index:
            raise ValueError(
                f"{loader.trip_id}: expected frame_id {index}, got {frame.frame_id}"
            )
        image_bgr = frame.left()
        timestamp = float(frame.timestamp)
        step = detection_source.step(frame.frame_id, timestamp, image_bgr)
        result = runtime.step(
            image_bgr,
            step.detections,
            detector_update=step.detector_update,
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
                frame_id=frame.frame_id,
                timestamp=timestamp,
                result=result,
                detector_step=step,
                published_ttc_s=published,
                previous=rows[-1] if rows else None,
            )
        )
    return tuple(rows)


def physics_runtime_parity(
    rows: Sequence[RuntimeRow],
    authoritative: Sequence[tuple[int, float, float]],
) -> dict[str, object]:
    """Gate cached physics replay before publishing the frozen CSV row."""

    if len(rows) != EXPECTED_FRAMES or len(authoritative) != EXPECTED_FRAMES:
        raise RuntimeError("physics parity requires exactly 600 rows")
    finite_mask_mismatches: list[int] = []
    finite_deltas: list[float] = []
    for row, (frame_id, timestamp, expected_ttc) in zip(rows, authoritative):
        if row.frame_id != frame_id or not math.isclose(
            row.timestamp, timestamp, abs_tol=1e-3
        ):
            raise RuntimeError("physics runtime/authoritative frame alignment mismatch")
        runtime_finite = math.isfinite(row.predicted_ttc_s)
        expected_finite = math.isfinite(expected_ttc)
        if runtime_finite != expected_finite:
            finite_mask_mismatches.append(frame_id)
        elif runtime_finite:
            finite_deltas.append(abs(row.predicted_ttc_s - expected_ttc))
    maximum_delta = max(finite_deltas, default=0.0)
    # One accepted horizon-edge mask difference is documented for the frozen
    # replay; any broader or material drift invalidates attribution.
    if len(finite_mask_mismatches) > 1 or maximum_delta > 0.02:
        raise RuntimeError(
            "physics replay drifted from the authoritative row: "
            f"mask_mismatches={finite_mask_mismatches[:10]}, "
            f"max_finite_delta={maximum_delta:.6f}"
        )
    return {
        "checked_frames": EXPECTED_FRAMES,
        "finite_mask_mismatch_count": len(finite_mask_mismatches),
        "finite_mask_mismatch_frame_ids": finite_mask_mismatches,
        "max_jointly_finite_abs_delta_s": round(maximum_delta, 6),
    }


def summarize_runtime(rows: Sequence[RuntimeRow], *, primary_supported: bool) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot summarize empty runtime rows")
    downstream = np.asarray([row.downstream_latency_ms for row in rows], dtype=float)
    detector = np.asarray(
        [
            row.detector_latency_ms
            for row in rows
            if row.detector_update and row.detector_latency_ms is not None
        ],
        dtype=float,
    )
    pipeline = np.asarray(
        [
            row.downstream_latency_ms + (row.detector_latency_ms or 0.0)
            for row in rows
        ],
        dtype=float,
    )
    mean_downstream = float(np.mean(downstream))
    lifecycle = _target_lifecycle_counts(rows) if primary_supported else {
        name: None for name in TARGET_LIFECYCLE_COUNT_DEFINITIONS
    }
    lane_source_counts, lane_source_coverage = (
        _lane_source_summary(rows) if primary_supported else (None, None)
    )
    if primary_supported:
        flagged_switches = sum(row.target_switched for row in rows)
        if flagged_switches != lifecycle["target_switch_count"]:
            raise RuntimeError(
                "target_switched flags do not match consecutive non-null "
                "primary-ID changes"
            )
    summary: dict[str, object] = {
        "n_frames": len(rows),
        "timing_scope": (
            "live image_2 detector plus downstream CPU"
            if detector.size
            else "cached replay; post-detector downstream CPU only"
        ),
        "downstream_cpu_fps": round(1000.0 / max(mean_downstream, 1e-9), 3),
        "downstream_mean_latency_ms": round(mean_downstream, 3),
        "downstream_p95_latency_ms": round(float(np.percentile(downstream, 95)), 3),
        **lifecycle,
        "lane_source_counts": lane_source_counts,
        "lane_source_coverage": lane_source_coverage,
        "nonfinite_prediction_count": sum(
            not math.isfinite(row.predicted_ttc_s) for row in rows
        ),
        "invalid_ttc_count": (
            sum(
                row.primary_track_id is not None
                and not math.isfinite(row.predicted_ttc_s)
                for row in rows
            )
            if primary_supported else None
        ),
        "no_primary_target_count": (
            sum(row.primary_track_id is None for row in rows)
            if primary_supported else None
        ),
        "ttc_jump_count": sum(row.ttc_jump for row in rows),
        "ttc_finite_toggle_count": sum(row.ttc_finite_toggle for row in rows),
        "warning_toggle_count": sum(row.warning_toggle for row in rows),
    }
    if detector.size:
        summary.update(
            detector_update_count=int(detector.size),
            detector_cpu_fps=round(1000.0 / max(float(np.mean(detector)), 1e-9), 3),
            detector_mean_latency_ms=round(float(np.mean(detector)), 3),
            detector_p95_latency_ms=round(float(np.percentile(detector, 95)), 3),
            pipeline_compute_fps=round(
                1000.0 / max(float(np.mean(pipeline)), 1e-9), 3
            ),
            pipeline_compute_p95_latency_ms=round(
                float(np.percentile(pipeline, 95)), 3
            ),
        )
    return summary


def _target_lifecycle_counts(rows: Sequence[RuntimeRow]) -> dict[str, int]:
    """Count primary-target lifecycle transitions within one trip.

    A virtual no-primary state precedes frame zero.  Direct switches remain
    deliberately distinct from identity changes after an intervening gap.
    """

    previous_id: int | None = None
    most_recent_non_null_id: int | None = None
    counts = {name: 0 for name in TARGET_LIFECYCLE_COUNT_DEFINITIONS}
    for row in rows:
        current_id = row.primary_track_id
        if previous_id is not None and current_id is not None:
            if current_id != previous_id:
                counts["target_switch_count"] += 1
        elif previous_id is None and current_id is not None:
            counts["target_acquisition_count"] += 1
            if most_recent_non_null_id is not None:
                counts["target_reacquisition_count"] += 1
                if current_id != most_recent_non_null_id:
                    counts["target_identity_change_after_gap_count"] += 1
        elif previous_id is not None and current_id is None:
            counts["target_drop_count"] += 1

        if current_id is not None:
            most_recent_non_null_id = current_id
        previous_id = current_id
    return counts


def _lane_source_summary(
    rows: Sequence[RuntimeRow],
) -> tuple[dict[str, int], dict[str, float]]:
    """Count lane-backed versus calibrated fixed corridor frames exactly."""

    counts = {source: 0 for source in CORRIDOR_SOURCES}
    for row in rows:
        if row.lane_source not in counts:
            raise RuntimeError(f"unexpected deployable corridor source: {row.lane_source}")
        counts[row.lane_source] += 1
    coverage = {source: count / len(rows) for source, count in counts.items()}
    return counts, coverage


def _validate_locked_protocol() -> None:
    actual = tuple(LOCKED_P2_VARIANTS)
    if actual != ABLATION_VARIANTS:
        raise RuntimeError(
            "P2-B runtime variants changed; update the protocol explicitly before rerunning"
        )


def _assert_prediction_layout(prediction_root: Path) -> tuple[Mapping[str, object], ...]:
    expected_names = {f"{trip_id}.csv" for trip_id in TRIP_IDS}
    reports: list[Mapping[str, object]] = []
    expected_directories = {variant.value for variant in ABLATION_VARIANTS}
    actual_directories = {path.name for path in prediction_root.iterdir() if path.is_dir()}
    unexpected_directories = actual_directories - expected_directories
    if unexpected_directories:
        raise RuntimeError(
            f"unexpected prediction variant directories: {sorted(unexpected_directories)}"
        )
    for variant in ABLATION_VARIANTS:
        directory = prediction_root / variant.value
        actual_names = {path.name for path in directory.glob("*.csv")}
        if actual_names != expected_names:
            raise RuntimeError(
                f"{variant.value}: expected exactly {sorted(expected_names)}, "
                f"found {sorted(actual_names)}"
            )
        for trip_id in TRIP_IDS:
            reports.append(
                preflight_prediction_csv(
                    directory / f"{trip_id}.csv", expected_frames=EXPECTED_FRAMES
                ).to_dict()
            )
    if len(reports) != 30 or sum(int(item["rows"]) for item in reports) != 18000:
        raise RuntimeError("prediction layout is not exactly 5 x 6 x 600")
    return tuple(reports)


def run_all_inference(
    *,
    data_root: Path,
    manifest_dir: Path,
    cache_dir: Path,
    baseline_dir: Path,
    prediction_root: Path,
    diagnostics_root: Path,
    cache_pattern: str = DEFAULT_CACHE_PATTERN,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> InferenceCompletion:
    """Complete all five variants sequentially before returning a score token."""

    _validate_locked_protocol()
    prediction_root.mkdir(parents=True, exist_ok=True)
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, object]] = {}
    prediction_rows = 0

    # Variant-major order is deliberate: the ablation always runs 1 -> 5.
    for variant in ABLATION_VARIANTS:
        variant_summaries: dict[str, object] = {}
        for trip_id in TRIP_IDS:
            trip_dir = data_root / trip_id
            manifest_path = manifest_dir / f"{trip_id}.json.gz"
            loader = C1LeanTripLoader(trip_dir, manifest_path)
            if loader.n_frames != EXPECTED_FRAMES:
                raise RuntimeError(
                    f"{trip_id}: expected {EXPECTED_FRAMES} manifest frames, "
                    f"found {loader.n_frames}"
                )
            loader.check_image_2_integrity().require_ok()
            geometry = load_camera_geometry(trip_dir)
            cache_path = locate_detection_cache(
                cache_dir, trip_id, pattern=cache_pattern
            )
            cache = load_detection_cache(cache_path)
            if cache.trip_id != trip_id:
                raise ValueError(f"cache trip {cache.trip_id} != {trip_id}")
            source = CachedDetectionSource(
                cache,
                n_frames=loader.n_frames,
                confidence_threshold=confidence_threshold,
            )

            authoritative_path = baseline_dir / f"{trip_id}.csv"
            authoritative: tuple[tuple[int, float, float], ...] | None = None
            if variant == P2Variant.PHYSICS_CURRENT:
                authoritative = read_prediction_rows(authoritative_path)
                for frame_id, timestamp, _ in authoritative:
                    manifest_frame = loader.frame(frame_id)
                    if not math.isclose(
                        timestamp, manifest_frame.timestamp, abs_tol=1e-3
                    ):
                        raise ValueError(
                            f"{trip_id} frame {frame_id}: baseline timestamp "
                            f"{timestamp} != manifest timestamp "
                            f"{manifest_frame.timestamp}"
                        )
            rows = replay_runtime(
                loader, geometry, variant, source
            )
            if len(rows) != EXPECTED_FRAMES or tuple(row.frame_id for row in rows) != tuple(range(EXPECTED_FRAMES)):
                raise RuntimeError(f"{variant.value}/{trip_id}: replay is not exact 0..599")

            output_path = prediction_root / variant.value / f"{trip_id}.csv"
            if variant == P2Variant.PHYSICS_CURRENT:
                if authoritative is None:  # pragma: no cover - structural invariant
                    raise AssertionError("physics baseline rows were not loaded")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if output_path.resolve() != authoritative_path.resolve():
                    shutil.copyfile(authoritative_path, output_path)
                if sha256_file(output_path) != sha256_file(authoritative_path):
                    raise RuntimeError(f"{trip_id}: physics baseline copy is not byte-exact")
                preflight_prediction_csv(output_path, expected_frames=EXPECTED_FRAMES)
            else:
                _atomic_prediction_csv(output_path, rows)
            write_diagnostic_csv(
                diagnostics_root / variant.value / f"{trip_id}.csv", rows
            )
            runtime_summary = summarize_runtime(
                rows, primary_supported=variant != P2Variant.PHYSICS_CURRENT
            )
            if variant == P2Variant.PHYSICS_CURRENT:
                if authoritative is None:  # pragma: no cover - structural invariant
                    raise AssertionError("physics baseline rows were not loaded")
                runtime_summary["authoritative_prediction_parity"] = (
                    physics_runtime_parity(rows, authoritative)
                )
            variant_summaries[trip_id] = runtime_summary
            prediction_rows += len(rows)
        summaries[variant.value] = variant_summaries

    _assert_prediction_layout(prediction_root)
    completion = InferenceCompletion(
        variants=tuple(variant.value for variant in ABLATION_VARIANTS),
        trips=TRIP_IDS,
        prediction_rows=prediction_rows,
        prediction_root=prediction_root,
        diagnostics_root=diagnostics_root,
        summaries=summaries,
    )
    completion.require_complete()
    return completion


def confusion_counts(
    pairs: Mapping[int, tuple[float, float]], *, danger_threshold_s: float = 2.0
) -> dict[str, int]:
    pred_danger = {frame_id for frame_id, (pred, _) in pairs.items() if pred < danger_threshold_s}
    true_danger = {frame_id for frame_id, (_, truth) in pairs.items() if truth < danger_threshold_s}
    return {
        "tp": len(pred_danger & true_danger),
        "fp": len(pred_danger - true_danger),
        "fn": len(true_danger - pred_danger),
        "tn": len(pairs) - len(pred_danger | true_danger),
    }


def macro_metrics(per_trip: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(per_trip) != len(TRIP_IDS):
        raise ValueError("macro metrics require exactly six trips")
    valid_mae = [float(row["mae_critical"]) for row in per_trip if float(row["mae_critical"]) >= 0.0]
    return {
        "n_trips": len(per_trip),
        "n_frames": sum(int(row["n_frames"]) for row in per_trip),
        "mae_critical": round(float(np.mean(valid_mae)), 3) if valid_mae else None,
        "f1": round(float(np.mean([float(row["f1"]) for row in per_trip])), 3),
        "composite_score": round(
            float(np.mean([float(row["composite_score"]) for row in per_trip])), 1
        ),
    }


def _aggregate_runtime_summaries(
    per_trip: Mapping[str, Mapping[str, object]],
    *,
    downstream_latency_ms: Sequence[float],
) -> dict[str, object]:
    frames = sum(int(row["n_frames"]) for row in per_trip.values())
    if frames != len(TRIP_IDS) * EXPECTED_FRAMES:
        raise RuntimeError("runtime diagnostics do not contain exactly 3,600 frames")
    latency = np.asarray(downstream_latency_ms, dtype=float)
    if latency.shape != (frames,) or not np.isfinite(latency).all() or np.any(latency < 0.0):
        raise RuntimeError("pooled downstream timing must contain 3,600 finite rows")
    weighted_mean_ms = float(np.mean(latency))

    def sum_optional_count(name: str) -> int | None:
        if any(row[name] is None for row in per_trip.values()):
            return None
        return sum(int(row[name]) for row in per_trip.values())

    lane_counts: dict[str, int] | None
    if any(row["lane_source_counts"] is None for row in per_trip.values()):
        lane_counts = None
    else:
        lane_counts = {source: 0 for source in CORRIDOR_SOURCES}
        for row in per_trip.values():
            trip_counts = row["lane_source_counts"]
            if not isinstance(trip_counts, Mapping) or set(trip_counts) != set(
                CORRIDOR_SOURCES
            ):
                raise RuntimeError("unexpected per-trip lane source count schema")
            for source in CORRIDOR_SOURCES:
                lane_counts[source] += int(trip_counts[source])
        if sum(lane_counts.values()) != frames:
            raise RuntimeError("pooled lane source counts do not total 3,600 frames")
    lane_coverage = (
        None
        if lane_counts is None
        else {source: count / frames for source, count in lane_counts.items()}
    )

    return {
        "n_frames": frames,
        "timing_scope": "cached replay; post-detector downstream CPU only",
        "downstream_cpu_fps": round(1000.0 / max(weighted_mean_ms, 1e-9), 3),
        "downstream_p95_latency_ms": round(
            float(np.percentile(latency, 95)), 3
        ),
        "target_switch_count": sum_optional_count("target_switch_count"),
        "target_acquisition_count": sum_optional_count(
            "target_acquisition_count"
        ),
        "target_drop_count": sum_optional_count("target_drop_count"),
        "target_reacquisition_count": sum_optional_count(
            "target_reacquisition_count"
        ),
        "target_identity_change_after_gap_count": sum_optional_count(
            "target_identity_change_after_gap_count"
        ),
        "lane_source_counts": lane_counts,
        "lane_source_coverage": lane_coverage,
        "nonfinite_prediction_count": sum(
            int(row["nonfinite_prediction_count"]) for row in per_trip.values()
        ),
        "invalid_ttc_count": (
            None
            if any(row["invalid_ttc_count"] is None for row in per_trip.values())
            else sum(int(row["invalid_ttc_count"]) for row in per_trip.values())
        ),
        "no_primary_target_count": (
            None
            if any(row["no_primary_target_count"] is None for row in per_trip.values())
            else sum(int(row["no_primary_target_count"]) for row in per_trip.values())
        ),
        "ttc_jump_count": sum(int(row["ttc_jump_count"]) for row in per_trip.values()),
        "ttc_finite_toggle_count": sum(int(row["ttc_finite_toggle_count"]) for row in per_trip.values()),
        "warning_toggle_count": sum(int(row["warning_toggle_count"]) for row in per_trip.values()),
    }


def _read_diagnostic_timing(
    diagnostics_root: Path, variant: P2Variant
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    """Read exact pooled causal timings without touching evaluation labels."""

    latencies: list[float] = []
    detector_updates: list[bool] = []
    for trip_id in TRIP_IDS:
        path = diagnostics_root / variant.value / f"{trip_id}.csv"
        if not path.is_file():
            raise RuntimeError(f"missing runtime diagnostic CSV: {path}")
        frame_ids: list[int] = []
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != DIAGNOSTIC_COLUMNS:
                raise RuntimeError(f"unexpected runtime diagnostic schema: {path}")
            for row in reader:
                if row["trip_id"] != trip_id or row["variant"] != variant.value:
                    raise RuntimeError(f"runtime diagnostic identity mismatch: {path}")
                frame_ids.append(int(row["frame_id"]))
                latency = float(row["downstream_latency_ms"])
                if not math.isfinite(latency) or latency < 0.0:
                    raise RuntimeError(f"non-finite runtime timing in {path}")
                latencies.append(latency)
                detector_updates.append(row["detector_update"] == "1")
        if frame_ids != list(range(EXPECTED_FRAMES)):
            raise RuntimeError(f"runtime diagnostics are not exact 0..599: {path}")
    if len(latencies) != len(TRIP_IDS) * EXPECTED_FRAMES:
        raise RuntimeError("runtime diagnostics do not total exactly 3,600 rows")
    return tuple(latencies), tuple(detector_updates)


def estimate_end_to_end_cpu_timing(
    downstream_latency_ms: Sequence[float],
    detector_updates: Sequence[bool],
    detector_sample_latency_ms: Sequence[float],
) -> dict[str, object]:
    """Combine cached downstream rows with one shared live detector sample."""

    if len(downstream_latency_ms) != len(detector_updates) or not downstream_latency_ms:
        raise ValueError("downstream latency and detector cadence must align")
    sample = tuple(float(value) for value in detector_sample_latency_ms)
    if not sample or not all(math.isfinite(value) and value >= 0.0 for value in sample):
        raise ValueError("detector sample must be finite and non-empty")
    combined: list[float] = []
    sample_index = 0
    for downstream, update in zip(downstream_latency_ms, detector_updates):
        value = float(downstream)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("downstream timings must be finite and non-negative")
        if update:
            value += sample[sample_index % len(sample)]
            sample_index += 1
        combined.append(value)
    values = np.asarray(combined, dtype=float)
    return {
        "n_source_frames": len(combined),
        "detector_update_frames": sample_index,
        "estimated_end_to_end_cpu_fps": round(
            1000.0 / max(float(np.mean(values)), 1e-9), 3
        ),
        "estimated_end_to_end_mean_latency_ms": round(float(np.mean(values)), 3),
        "estimated_end_to_end_p95_latency_ms": round(
            float(np.percentile(values, 95)), 3
        ),
        "method": (
            "estimate = cached per-frame downstream CPU latency + one shared "
            "warm-up-excluded live YOLO CPU latency sample recycled in recorded "
            "order on detector-update frames; raw samples are persisted at "
            "shared_live_yolo_cpu_timing.latency_samples_ms; image decode and "
            "I/O excluded; per-frame downstream latency and detector-update "
            "cadence remain in diagnostics/<variant>/<trip>.csv"
        ),
    }


def add_shared_end_to_end_estimates(
    evaluation: Mapping[str, object],
    completion: InferenceCompletion,
    detector_timing: SharedDetectorTiming,
) -> None:
    variants = evaluation["variants"]
    if not isinstance(variants, dict):
        raise TypeError("evaluation variants must be mutable for timing annotation")
    for variant in ABLATION_VARIANTS:
        downstream, detector_updates = _read_diagnostic_timing(
            completion.diagnostics_root, variant
        )
        variant_report = variants[variant.value]
        if not isinstance(variant_report, dict):
            raise TypeError("variant report must be mutable")
        variant_report["estimated_end_to_end_cpu"] = estimate_end_to_end_cpu_timing(
            downstream, detector_updates, detector_timing.latency_ms
        )


def evaluate_predictions(completion: InferenceCompletion, *, data_root: Path) -> dict[str, object]:
    """Offline scoring phase; this is the only label-bearing call boundary."""

    completion.require_complete()
    _assert_prediction_layout(completion.prediction_root)

    # Deliberately late: no official evaluator or trusted label loader is
    # imported until every causal prediction has been materialized/preflighted.
    evaluator = importlib.import_module("team_kit.evaluation")
    variants: dict[str, object] = {}
    pairs_by_variant_trip: dict[tuple[str, str], Mapping[int, tuple[float, float]]] = {}
    for variant in ABLATION_VARIANTS:
        per_trip: list[dict[str, object]] = []
        selected_confusions: dict[str, Mapping[str, int]] = {}
        for trip_id in TRIP_IDS:
            prediction_path = completion.prediction_root / variant.value / f"{trip_id}.csv"
            predictions = evaluator.load_predictions(prediction_path)
            truth = evaluator.load_ground_truth_from_trip(data_root / trip_id)
            if set(predictions) != set(range(EXPECTED_FRAMES)):
                raise RuntimeError(f"{variant.value}/{trip_id}: evaluator did not load 600 predictions")
            if set(truth.ttc) != set(range(EXPECTED_FRAMES)):
                raise RuntimeError(f"{trip_id}: trusted TTC does not contain exact IDs 0..599")
            pairs = {
                frame_id: (predictions[frame_id].predicted_ttc, truth.ttc[frame_id])
                for frame_id in range(EXPECTED_FRAMES)
            }
            pairs_by_variant_trip[(variant.value, trip_id)] = pairs
            metric = asdict(evaluator.compute_trip_metrics(trip_id, pairs))
            if int(metric["n_frames"]) != EXPECTED_FRAMES:
                raise RuntimeError(f"{variant.value}/{trip_id}: official metric count != 600")
            per_trip.append(metric)
            if trip_id in DIAGNOSTIC_TRIPS:
                selected_confusions[trip_id] = confusion_counts(pairs)
        downstream_latency, _ = _read_diagnostic_timing(
            completion.diagnostics_root, variant
        )
        variants[variant.value] = {
            "n_predictions": sum(int(row["n_frames"]) for row in per_trip),
            "per_trip": per_trip,
            "macro": macro_metrics(per_trip),
            "fp_fn": selected_confusions,
            "runtime_per_trip": completion.summaries[variant.value],
            "runtime": _aggregate_runtime_summaries(
                completion.summaries[variant.value],
                downstream_latency_ms=downstream_latency,
            ),
        }

    baseline = variants[P2Variant.PHYSICS_CURRENT.value]
    full = variants[P2Variant.FULL.value]
    baseline_macro = float(baseline["macro"]["composite_score"])  # type: ignore[index]
    full_macro = float(full["macro"]["composite_score"])  # type: ignore[index]
    if baseline_macro != 55.2:
        raise RuntimeError(f"authoritative physics baseline gate failed: {baseline_macro} != 55.2")
    trip_deltas: dict[str, float] = {}
    baseline_trip = {row["trip_id"]: row for row in baseline["per_trip"]}  # type: ignore[index]
    full_trip = {row["trip_id"]: row for row in full["per_trip"]}  # type: ignore[index]
    for trip_id in TRIP_IDS:
        trip_deltas[trip_id] = round(
            float(full_trip[trip_id]["composite_score"])
            - float(baseline_trip[trip_id]["composite_score"]),
            1,
        )

    t06_base = pairs_by_variant_trip[(P2Variant.PHYSICS_CURRENT.value, "T06-Sample")]
    t06_full = pairs_by_variant_trip[(P2Variant.FULL.value, "T06-Sample")]
    relevant_frames = tuple(range(420, 428))
    baseline_missed = tuple(
        frame_id
        for frame_id in relevant_frames
        if t06_base[frame_id][1] < 2.0 and not t06_base[frame_id][0] < 2.0
    )
    full_missed = tuple(
        frame_id
        for frame_id in relevant_frames
        if t06_full[frame_id][1] < 2.0 and not t06_full[frame_id][0] < 2.0
    )
    recovered = tuple(frame_id for frame_id in baseline_missed if frame_id not in full_missed)
    baseline_fp = baseline["fp_fn"]  # type: ignore[index]
    full_fp = full["fp_fn"]  # type: ignore[index]
    return {
        "variants": variants,
        "acceptance": {
            "baseline_macro_composite": baseline_macro,
            "full_macro_composite": full_macro,
            "macro_delta": round(full_macro - baseline_macro, 1),
            "macro_gain_at_least_3": full_macro - baseline_macro >= 3.0,
            "t01_t02_false_positive_delta": {
                trip_id: int(full_fp[trip_id]["fp"]) - int(baseline_fp[trip_id]["fp"])
                for trip_id in ("T01-Sample", "T02-Sample")
            },
            "t01_t02_false_positives_both_reduced": all(
                int(full_fp[trip_id]["fp"]) < int(baseline_fp[trip_id]["fp"])
                for trip_id in ("T01-Sample", "T02-Sample")
            ),
            "t06_frames_420_427": {
                "baseline_missed_danger_frames": baseline_missed,
                "full_missed_danger_frames": full_missed,
                "recovered_frames": recovered,
            },
            "trip_composite_delta": trip_deltas,
            "trips_declining_more_than_5": tuple(
                trip_id for trip_id, delta in trip_deltas.items() if delta < -5.0
            ),
        },
    }


def sample_live_yolo_cpu_detector(
    *,
    data_root: Path,
    manifest_dir: Path,
    trip_id: str,
    frames: int,
    stride: int,
    warmup_detector_updates: int,
    model_path: Path,
    labels_path: Path,
) -> SharedDetectorTiming:
    """Measure one detector-only CPU sample, excluding explicit warm-up calls."""

    if trip_id not in TRIP_IDS:
        raise ValueError(f"benchmark trip must be one of {TRIP_IDS}")
    if not 1 <= frames <= EXPECTED_FRAMES:
        raise ValueError("benchmark frames must be in [1, 600]")
    if warmup_detector_updates < 1:
        raise ValueError("benchmark requires at least one excluded warm-up update")
    trip_dir = data_root / trip_id
    loader = C1LeanTripLoader(trip_dir, manifest_dir / f"{trip_id}.json.gz")
    loader.check_image_2_integrity().require_ok()
    detector = OpenCVDnnYoloDetector(
        model_path,
        labels_path,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        device="cpu",
    )
    source = LiveYoloCpuSource(detector, stride=stride)
    measured: list[float] = []
    update_index = 0
    for frame_id in range(frames):
        if frame_id % stride:
            continue
        frame = loader.frame(frame_id)
        step = source.step(frame_id, frame.timestamp, frame.left())
        if not step.detector_update or step.detector_latency_ms is None:
            raise RuntimeError("live detector timing source skipped an update frame")
        if update_index >= warmup_detector_updates:
            measured.append(step.detector_latency_ms)
        update_index += 1
    if not measured:
        raise ValueError(
            "benchmark frame window is too short after excluding detector warm-up"
        )
    return SharedDetectorTiming(
        trip_id=trip_id,
        source_frames=frames,
        stride=stride,
        warmup_detector_updates=warmup_detector_updates,
        measured_detector_updates=len(measured),
        latency_ms=tuple(measured),
    )


def _atomic_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".json", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
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
        baseline_dir=args.baseline_dir,
        prediction_root=args.prediction_root,
        diagnostics_root=args.run_root / "diagnostics",
        cache_pattern=args.cache_pattern,
        confidence_threshold=args.confidence_threshold,
    )
    detector_timing: SharedDetectorTiming | None = None
    if args.live_yolo_cpu_benchmark:
        detector_timing = sample_live_yolo_cpu_detector(
            data_root=args.data_root,
            manifest_dir=args.manifest_dir,
            trip_id=args.benchmark_trip,
            frames=args.benchmark_frames,
            stride=args.benchmark_stride,
            warmup_detector_updates=args.benchmark_warmup_updates,
            model_path=args.model,
            labels_path=args.labels,
        )

    # This call occurs only after all inference (and optional label-free live
    # timing) completed successfully.
    evaluation = evaluate_predictions(completion, data_root=args.data_root)
    if detector_timing is not None:
        add_shared_end_to_end_estimates(evaluation, completion, detector_timing)
    evaluator_sha_after = require_evaluator_hash()
    if evaluator_sha_after != evaluator_sha_before:
        raise RuntimeError("official evaluator changed while the ablation ran")
    report: dict[str, object] = {
        "report_schema": REPORT_SCHEMA,
        "protocol": "P2-B development score; not external held-out evaluation",
        "runtime_input_boundary": (
            "sanitized manifest + image_2 + image_2 detector cache + camera "
            "calibration; authoritative CSV only for the frozen physics row"
        ),
        "cached_timing_scope": "post-detector downstream CPU",
        "official_evaluator_sha256_before": evaluator_sha_before,
        "official_evaluator_sha256_after": evaluator_sha_after,
        "variant_order": list(completion.variants),
        "trips": list(completion.trips),
        "prediction_rows": completion.prediction_rows,
        "target_lifecycle_count_definitions": dict(
            TARGET_LIFECYCLE_COUNT_DEFINITIONS
        ),
        "lane_source_coverage_definition": LANE_SOURCE_COVERAGE_DEFINITION,
        **evaluation,
    }
    if detector_timing is not None:
        report["shared_live_yolo_cpu_timing"] = detector_timing.report()
    _atomic_json(args.run_root / "report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the locked five-row causal C1 P2-B development ablation."
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
        "--baseline-dir",
        type=Path,
        default=ROOT / "predictions/c1_baseline_recheck/six_samples",
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=ROOT / "predictions/c1_p2b_ablation",
    )
    parser.add_argument(
        "--run-root", type=Path, default=ROOT / "runs/c1/p2b_ablation"
    )
    parser.add_argument("--cache-pattern", default=DEFAULT_CACHE_PATTERN)
    parser.add_argument(
        "--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD
    )
    parser.add_argument("--live-yolo-cpu-benchmark", action="store_true")
    parser.add_argument("--benchmark-trip", default="T01-Sample", choices=TRIP_IDS)
    parser.add_argument("--benchmark-frames", type=int, default=120)
    parser.add_argument("--benchmark-stride", type=int, default=3)
    parser.add_argument("--benchmark-warmup-updates", type=int, default=3)
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolo11s.onnx")
    parser.add_argument(
        "--labels", type=Path, default=ROOT / "models/driver-objects.labels"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_ablation(args)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"P2-B ablation failed: {exc}", file=sys.stderr)
        return 2
    compact = {
        variant: row["macro"]
        for variant, row in report["variants"].items()  # type: ignore[union-attr]
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
