#!/usr/bin/env python3
"""Actual synchronous CPU/CUDA benchmark for the locked C1 P2-C runtime.

The benchmark is deliberately label-free.  It reads only a sanitized C1
manifest, ``image_2``, camera calibration, and causal ego speed.  Accuracy is
locked before this tool is run; this module neither imports nor edits the
official evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import cv2
import numpy as np

from safeloop.c1.detector import OpenCVDnnYoloDetector
from safeloop.c1.lean_loader import C1LeanTripLoader
from safeloop.c1.ort_detector import OnnxRuntimeYoloDetector
from safeloop.c1.p2b_tracker import _hungarian
from safeloop.c1.temporal_features import CameraGeometry, load_camera_geometry
from safeloop.c1.types import Detection


SCHEMA = "safeloop.c1.p2c_gpu_benchmark.v1"
EXPECTED_FRAMES = 600
DEFAULT_STRIDE = 3
DEFAULT_CONFIDENCE = 0.25
DEFAULT_NMS = 0.45
CONFIDENCE_ABS_TOLERANCE = 1e-4
BBOX_COORD_ABS_TOLERANCE_PX = 0.1
IOU_TOLERANCE = 0.999
ROOT = Path(__file__).resolve().parents[1]


class DetectorLike(Protocol):
    device: str

    def detect(self, image_bgr: np.ndarray) -> list[Detection]: ...


class FrameLike(Protocol):
    frame_id: int
    timestamp: float
    ego: Mapping[str, float]

    def left(self) -> np.ndarray: ...


class TripLike(Protocol):
    trip_id: str
    n_frames: int

    def frame(self, index: int) -> FrameLike: ...


class RuntimeResultLike(Protocol):
    predicted_ttc_s: float
    warning: bool
    primary_track_id: int | None


class RuntimeLike(Protocol):
    def reset(self) -> None: ...

    def step(
        self,
        image_bgr: np.ndarray,
        detections: Sequence[Detection],
        *,
        detector_update: bool,
        timestamp: float,
        ego_speed_kmh: float,
    ) -> RuntimeResultLike: ...


@dataclass(frozen=True, slots=True)
class WarmupReport:
    decode_ms: float
    call_latency_ms: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.call_latency_ms:
            raise ValueError("warm-up must contain the cold first detector call")
        values = (self.decode_ms, *self.call_latency_ms)
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("warm-up latency must be finite and non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "image_decode_ms": self.decode_ms,
            "cold_first_call_ms": self.call_latency_ms[0],
            "subsequent_warmup_call_ms": list(self.call_latency_ms[1:]),
            "calls": len(self.call_latency_ms),
            "excluded_from_measured_pass": True,
        }


@dataclass(frozen=True, slots=True)
class FrameSample:
    frame_id: int
    timestamp: float
    decode_ms: float
    detector_ms: float | None
    runtime_ms: float
    end_to_end_ms: float
    detector_update: bool
    detections: tuple[Detection, ...]
    predicted_ttc_s: float
    warning: bool
    primary_track_id: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "decode_ms": self.decode_ms,
            "detector_ms": self.detector_ms,
            "runtime_ms": self.runtime_ms,
            "end_to_end_ms": self.end_to_end_ms,
            "detector_update": self.detector_update,
            "detection_count": len(self.detections),
            "detections": [
                {
                    "class_id": item.class_id,
                    "label": item.label,
                    "confidence": item.confidence,
                    "bbox": list(item.bbox),
                }
                for item in self.detections
            ],
            "predicted_ttc_s": (
                self.predicted_ttc_s
                if math.isfinite(self.predicted_ttc_s)
                else None
            ),
            "predicted_ttc_finite": math.isfinite(self.predicted_ttc_s),
            "warning": self.warning,
            "primary_track_id": self.primary_track_id,
        }


@dataclass(frozen=True, slots=True)
class BackendPass:
    backend: str
    trip_id: str
    stride: int
    wall_time_s: float
    samples: tuple[FrameSample, ...]

    def __post_init__(self) -> None:
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if not math.isfinite(self.wall_time_s) or self.wall_time_s <= 0.0:
            raise ValueError("wall time must be finite and positive")
        ids = tuple(item.frame_id for item in self.samples)
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.trip_id}: duplicate benchmark frame IDs")
        if ids != tuple(range(len(ids))):
            raise ValueError(f"{self.trip_id}: frame IDs must be exact 0..N-1")
        expected_updates = tuple(range(0, len(ids), self.stride))
        actual_updates = tuple(
            item.frame_id for item in self.samples if item.detector_update
        )
        if actual_updates != expected_updates:
            raise ValueError(f"{self.trip_id}: detector-update schedule changed")

    def summary(self) -> dict[str, object]:
        end_to_end = np.asarray(
            [item.end_to_end_ms for item in self.samples], dtype=np.float64
        )
        decode = np.asarray(
            [item.decode_ms for item in self.samples], dtype=np.float64
        )
        runtime = np.asarray(
            [item.runtime_ms for item in self.samples], dtype=np.float64
        )
        detector = np.asarray(
            [
                item.detector_ms
                for item in self.samples
                if item.detector_ms is not None
            ],
            dtype=np.float64,
        )
        return {
            "backend": self.backend,
            "trip_id": self.trip_id,
            "frames": len(self.samples),
            "detector_updates": int(detector.size),
            "wall_time_s": self.wall_time_s,
            "whole_pass_wall_fps": len(self.samples) / self.wall_time_s,
            "per_frame_mean_ms": float(np.mean(end_to_end)),
            "per_frame_p95_ms": float(np.percentile(end_to_end, 95)),
            "decode_mean_ms": float(np.mean(decode)),
            "decode_p95_ms": float(np.percentile(decode, 95)),
            "detector_update_mean_ms": float(np.mean(detector)),
            "detector_update_p95_ms": float(np.percentile(detector, 95)),
            "runtime_mean_ms": float(np.mean(runtime)),
            "runtime_p95_ms": float(np.percentile(runtime, 95)),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary(),
            "raw_frame_samples": [item.to_dict() for item in self.samples],
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def warm_up_detector(
    trip: TripLike,
    detector: DetectorLike,
    *,
    calls: int,
    clock: Callable[[], float] = time.perf_counter,
) -> WarmupReport:
    """Measure the cold first call and exclude all warm-up from the pass."""

    if calls < 1:
        raise ValueError("warm-up calls must be >= 1")
    decode_started = clock()
    image = trip.frame(0).left()
    decode_ms = (clock() - decode_started) * 1000.0
    latency: list[float] = []
    for _ in range(calls):
        started = clock()
        detector.detect(image)
        latency.append((clock() - started) * 1000.0)
    return WarmupReport(decode_ms, tuple(latency))


def run_backend_pass(
    trip: TripLike,
    detector: DetectorLike,
    runtime: RuntimeLike,
    *,
    backend: str,
    stride: int,
    expected_frames: int | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> BackendPass:
    """Run a real synchronous decode -> detector -> P2-C pass."""

    if stride < 1:
        raise ValueError("stride must be >= 1")
    if expected_frames is not None and trip.n_frames != expected_frames:
        raise ValueError(
            f"{trip.trip_id}: expected {expected_frames} frames, found {trip.n_frames}"
        )
    runtime.reset()
    samples: list[FrameSample] = []
    pass_started = clock()
    for index in range(trip.n_frames):
        frame_started = clock()
        frame = trip.frame(index)
        image = frame.left()
        decoded = clock()
        detector_update = frame.frame_id % stride == 0
        detector_ms: float | None = None
        if detector_update:
            detections = tuple(detector.detect(image))
            detector_done = clock()
            detector_ms = (detector_done - decoded) * 1000.0
        else:
            detections = ()
            detector_done = decoded
        result = runtime.step(
            image,
            detections,
            detector_update=detector_update,
            timestamp=float(frame.timestamp),
            ego_speed_kmh=float(frame.ego.get("speed_kmh") or 0.0),
        )
        finished = clock()
        samples.append(
            FrameSample(
                frame_id=frame.frame_id,
                timestamp=float(frame.timestamp),
                decode_ms=(decoded - frame_started) * 1000.0,
                detector_ms=detector_ms,
                runtime_ms=(finished - detector_done) * 1000.0,
                end_to_end_ms=(finished - frame_started) * 1000.0,
                detector_update=detector_update,
                detections=detections,
                predicted_ttc_s=float(result.predicted_ttc_s),
                warning=bool(result.warning),
                primary_track_id=result.primary_track_id,
            )
        )
    wall_time_s = clock() - pass_started
    return BackendPass(backend, trip.trip_id, stride, wall_time_s, tuple(samples))


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def compare_detection_sets(
    reference: Sequence[Detection],
    candidate: Sequence[Detection],
    *,
    confidence_tolerance: float = CONFIDENCE_ABS_TOLERANCE,
    bbox_tolerance_px: float = BBOX_COORD_ABS_TOLERANCE_PX,
    iou_tolerance: float = IOU_TOLERANCE,
) -> dict[str, object]:
    """Hungarian same-label parity for one detector-update frame."""

    if confidence_tolerance < 0.0 or bbox_tolerance_px < 0.0:
        raise ValueError("absolute parity tolerances must be non-negative")
    if not 0.0 <= iou_tolerance <= 1.0:
        raise ValueError("IoU tolerance must be in [0, 1]")
    pairs: list[dict[str, object]] = []
    unmatched_reference: list[int] = []
    unmatched_candidate: list[int] = []
    labels = sorted({item.label for item in (*reference, *candidate)})
    for label in labels:
        left_indices = [index for index, item in enumerate(reference) if item.label == label]
        right_indices = [index for index, item in enumerate(candidate) if item.label == label]
        if not left_indices:
            unmatched_candidate.extend(right_indices)
            continue
        if not right_indices:
            unmatched_reference.extend(left_indices)
            continue
        cost = np.asarray(
            [
                [
                    1.0
                    - _bbox_iou(reference[left].bbox, candidate[right].bbox)
                    for right in right_indices
                ]
                for left in left_indices
            ],
            dtype=np.float64,
        )
        assignments = _hungarian(cost)
        used_left: set[int] = set()
        used_right: set[int] = set()
        for local_left, local_right in assignments:
            left = left_indices[local_left]
            right = right_indices[local_right]
            used_left.add(left)
            used_right.add(right)
            iou = _bbox_iou(reference[left].bbox, candidate[right].bbox)
            confidence_delta = abs(
                reference[left].confidence - candidate[right].confidence
            )
            bbox_delta = max(
                abs(a - b)
                for a, b in zip(reference[left].bbox, candidate[right].bbox)
            )
            geometry_ok = bbox_delta <= bbox_tolerance_px or iou >= iou_tolerance
            pair_ok = confidence_delta <= confidence_tolerance and geometry_ok
            pairs.append(
                {
                    "reference_index": left,
                    "candidate_index": right,
                    "label": label,
                    "iou": iou,
                    "confidence_abs_delta": confidence_delta,
                    "bbox_coordinate_max_abs_delta_px": bbox_delta,
                    "passed": pair_ok,
                }
            )
        unmatched_reference.extend(index for index in left_indices if index not in used_left)
        unmatched_candidate.extend(index for index in right_indices if index not in used_right)
    exact_count_and_labels = Counter(item.label for item in reference) == Counter(
        item.label for item in candidate
    )
    passed = (
        exact_count_and_labels
        and not unmatched_reference
        and not unmatched_candidate
        and all(bool(item["passed"]) for item in pairs)
    )
    return {
        "passed": passed,
        "exact_count_and_labels": exact_count_and_labels,
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "unmatched_reference_indices": sorted(unmatched_reference),
        "unmatched_candidate_indices": sorted(unmatched_candidate),
        "matches": pairs,
    }


def compare_backend_passes(
    reference: BackendPass,
    candidate: BackendPass,
) -> dict[str, object]:
    """Compare detector outputs and downstream P2-C decisions frame-by-frame."""

    if reference.trip_id != candidate.trip_id:
        raise ValueError("backend passes must belong to the same trip")
    if reference.stride != candidate.stride:
        raise ValueError("backend passes must use the same detector stride")
    if len(reference.samples) != len(candidate.samples):
        raise ValueError("backend passes must contain the same number of frames")
    detector_frames: list[dict[str, object]] = []
    output_frames: list[dict[str, object]] = []
    finite_ttc_deltas: list[float] = []
    for left, right in zip(reference.samples, candidate.samples):
        if left.frame_id != right.frame_id or not math.isclose(
            left.timestamp, right.timestamp, abs_tol=1e-9
        ):
            raise ValueError("backend frame/timestamp alignment changed")
        if left.detector_update != right.detector_update:
            raise ValueError("backend detector-update schedule changed")
        if left.detector_update:
            comparison = compare_detection_sets(left.detections, right.detections)
            detector_frames.append({"frame_id": left.frame_id, **comparison})
        left_finite = math.isfinite(left.predicted_ttc_s)
        right_finite = math.isfinite(right.predicted_ttc_s)
        ttc_delta: float | None = None
        if left_finite and right_finite:
            ttc_delta = abs(left.predicted_ttc_s - right.predicted_ttc_s)
            finite_ttc_deltas.append(ttc_delta)
        output_frames.append(
            {
                "frame_id": left.frame_id,
                "warning_equal": left.warning == right.warning,
                "finite_ttc_equal": left_finite == right_finite,
                "primary_track_id_equal": (
                    left.primary_track_id == right.primary_track_id
                ),
                "finite_ttc_abs_delta_s": ttc_delta,
            }
        )
    detector_passed = all(bool(item["passed"]) for item in detector_frames)
    warning_mismatches = sum(not bool(item["warning_equal"]) for item in output_frames)
    finite_mismatches = sum(not bool(item["finite_ttc_equal"]) for item in output_frames)
    primary_mismatches = sum(
        not bool(item["primary_track_id_equal"]) for item in output_frames
    )
    output_passed = not (warning_mismatches or finite_mismatches or primary_mismatches)
    pair_rows = [
        pair
        for frame in detector_frames
        for pair in frame["matches"]  # type: ignore[union-attr]
    ]
    return {
        "trip_id": reference.trip_id,
        "reference_backend": reference.backend,
        "candidate_backend": candidate.backend,
        "passed": detector_passed and output_passed,
        "detector": {
            "passed": detector_passed,
            "update_frames": len(detector_frames),
            "matched_detections": len(pair_rows),
            "unmatched_reference": sum(
                len(item["unmatched_reference_indices"])  # type: ignore[arg-type]
                for item in detector_frames
            ),
            "unmatched_candidate": sum(
                len(item["unmatched_candidate_indices"])  # type: ignore[arg-type]
                for item in detector_frames
            ),
            "max_confidence_abs_delta": max(
                (float(item["confidence_abs_delta"]) for item in pair_rows),
                default=0.0,
            ),
            "max_bbox_coordinate_abs_delta_px": max(
                (
                    float(item["bbox_coordinate_max_abs_delta_px"])
                    for item in pair_rows
                ),
                default=0.0,
            ),
            "minimum_iou": min(
                (float(item["iou"]) for item in pair_rows), default=1.0
            ),
            "raw_update_comparisons": detector_frames,
        },
        "runtime_output": {
            "passed": output_passed,
            "warning_mismatches": warning_mismatches,
            "finite_ttc_mismatches": finite_mismatches,
            "primary_track_id_mismatches": primary_mismatches,
            "max_finite_ttc_abs_delta_s": max(finite_ttc_deltas, default=0.0),
            "p95_finite_ttc_abs_delta_s": (
                float(np.percentile(finite_ttc_deltas, 95))
                if finite_ttc_deltas
                else 0.0
            ),
            "raw_frame_comparisons": output_frames,
        },
    }


def aggregate_passes(passes: Sequence[BackendPass]) -> dict[str, object]:
    if not passes:
        raise ValueError("aggregate requires at least one backend pass")
    backend = passes[0].backend
    if any(item.backend != backend for item in passes):
        raise ValueError("aggregate backend labels must match")
    frames = [sample for item in passes for sample in item.samples]
    latency = np.asarray([item.end_to_end_ms for item in frames], dtype=np.float64)
    detector = np.asarray(
        [item.detector_ms for item in frames if item.detector_ms is not None],
        dtype=np.float64,
    )
    wall = sum(item.wall_time_s for item in passes)
    return {
        "backend": backend,
        "trips": len(passes),
        "frames": len(frames),
        "detector_updates": int(detector.size),
        "summed_whole_pass_wall_time_s": wall,
        "whole_pass_wall_fps": len(frames) / wall,
        "per_frame_mean_ms": float(np.mean(latency)),
        "per_frame_p95_ms": float(np.percentile(latency, 95)),
        "detector_update_mean_ms": float(np.mean(detector)),
        "detector_update_p95_ms": float(np.percentile(detector, 95)),
    }


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


def _nvidia_smi() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return [{"error": str(exc)}]
    rows: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            rows.append(
                {
                    "name": fields[0],
                    "memory_total_mib": int(fields[1]),
                    "driver_version": fields[2],
                }
            )
    return rows


def _build_full_runtime(geometry: CameraGeometry) -> RuntimeLike:
    # Explicit import: the benchmark cannot silently fall back to P2-B.
    from safeloop.c1.p2c_runtime import P2CDeployableRuntime, P2CVariant

    return P2CDeployableRuntime(geometry, P2CVariant.FULL_FUSION)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Actual synchronous image_2 decode + unchanged YOLO + P2-C full "
            "CPU/CUDA benchmark; no evaluator or ground truth is read."
        )
    )
    parser.add_argument("trip_dirs", nargs="+", type=Path)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolo11s.onnx")
    parser.add_argument(
        "--labels", type=Path, default=ROOT / "models/driver-objects.labels"
    )
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--nms", type=float, default=DEFAULT_NMS)
    parser.add_argument("--warmup-calls", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.stride < 1 or args.warmup_calls < 1:
        print("stride and warmup-calls must be >= 1", file=sys.stderr)
        return 2
    if not math.isclose(
        args.confidence, DEFAULT_CONFIDENCE, rel_tol=0.0, abs_tol=1e-12
    ):
        print(
            "confidence is locked to 0.25 to match the P2-C accuracy protocol",
            file=sys.stderr,
        )
        return 2
    if args.output.exists() and not args.force:
        print(f"output already exists: {args.output}; pass --force to replace", file=sys.stderr)
        return 2
    try:
        loaders: list[C1LeanTripLoader] = []
        geometries: list[CameraGeometry] = []
        for trip_dir in args.trip_dirs:
            loader = C1LeanTripLoader(
                trip_dir, args.manifest_dir / f"{trip_dir.name}.json.gz"
            )
            loader.check_image_2_integrity().require_ok()
            if loader.n_frames != EXPECTED_FRAMES:
                raise ValueError(
                    f"{loader.trip_id}: expected {EXPECTED_FRAMES} frames, "
                    f"found {loader.n_frames}"
                )
            loaders.append(loader)
            geometries.append(load_camera_geometry(trip_dir))

        cpu_construct_started = time.perf_counter()
        cpu_detector = OpenCVDnnYoloDetector(
            args.model,
            args.labels,
            confidence_threshold=args.confidence,
            nms_threshold=args.nms,
            device="cpu",
        )
        cpu_construct_ms = (time.perf_counter() - cpu_construct_started) * 1000.0
        cuda_construct_started = time.perf_counter()
        cuda_detector = OnnxRuntimeYoloDetector(
            args.model,
            args.labels,
            confidence_threshold=args.confidence,
            nms_threshold=args.nms,
            device="cuda",
        )
        cuda_construct_ms = (time.perf_counter() - cuda_construct_started) * 1000.0
        if (
            cuda_detector.device != "cuda"
            or cuda_detector.execution_provider != "CUDAExecutionProvider"
            or cuda_detector.session.get_providers()[0] != "CUDAExecutionProvider"
        ):
            raise RuntimeError("CUDAExecutionProvider fallback detected")

        cpu_warmup = warm_up_detector(
            loaders[0], cpu_detector, calls=args.warmup_calls
        )
        cuda_warmup = warm_up_detector(
            loaders[0], cuda_detector, calls=args.warmup_calls
        )
        cpu_passes: list[BackendPass] = []
        cuda_passes: list[BackendPass] = []
        execution_order: list[dict[str, object]] = []
        for trip_index, (loader, geometry) in enumerate(zip(loaders, geometries)):
            # Alternating first backend reduces page-cache/order bias while
            # preserving identical frame order inside each measured pass.
            order = ("opencv_cpu", "onnxruntime_cuda")
            if trip_index % 2:
                order = tuple(reversed(order))
            execution_order.append({"trip_id": loader.trip_id, "order": list(order)})
            for backend in order:
                if backend == "opencv_cpu":
                    cpu_passes.append(
                        run_backend_pass(
                            loader,
                            cpu_detector,
                            _build_full_runtime(geometry),
                            backend=backend,
                            stride=args.stride,
                            expected_frames=EXPECTED_FRAMES,
                        )
                    )
                else:
                    cuda_passes.append(
                        run_backend_pass(
                            loader,
                            cuda_detector,
                            _build_full_runtime(geometry),
                            backend=backend,
                            stride=args.stride,
                            expected_frames=EXPECTED_FRAMES,
                        )
                    )
        cpu_by_trip = {item.trip_id: item for item in cpu_passes}
        cuda_by_trip = {item.trip_id: item for item in cuda_passes}
        parity = [
            compare_backend_passes(cpu_by_trip[loader.trip_id], cuda_by_trip[loader.trip_id])
            for loader in loaders
        ]

        import onnxruntime as ort

        report: dict[str, object] = {
            "schema": SCHEMA,
            "method": {
                "synchronous": True,
                "per_frame_boundary": (
                    "C1InputFrame lookup + image_2 cv2 decode + detector on stride "
                    "+ explicit P2-C FULL_FUSION runtime"
                ),
                "whole_pass_fps_is_measured_not_estimated": True,
                "warmup_excluded": True,
                "backend_order_balanced_by_trip": True,
                "stride": args.stride,
                "confidence_threshold": args.confidence,
                "nms_threshold": args.nms,
                "expected_frames_per_trip": EXPECTED_FRAMES,
                "execution_order": execution_order,
            },
            "artifacts": {
                "model_filename": args.model.name,
                "model_sha256": sha256_file(args.model),
                "labels_filename": args.labels.name,
                "labels_sha256": sha256_file(args.labels),
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "onnxruntime": ort.__version__,
                "onnxruntime_device": ort.get_device(),
                "onnxruntime_available_providers": ort.get_available_providers(),
                "onnxruntime_active_providers": cuda_detector.session.get_providers(),
                "onnxruntime_provider_options": (
                    cuda_detector.session.get_provider_options()
                ),
                "gpus": _nvidia_smi(),
            },
            "initialization": {
                "opencv_cpu_constructor_ms": cpu_construct_ms,
                "onnxruntime_cuda_constructor_ms": cuda_construct_ms,
                "opencv_cpu_warmup": cpu_warmup.to_dict(),
                "onnxruntime_cuda_warmup": cuda_warmup.to_dict(),
            },
            "backend_aggregate": {
                "opencv_cpu": aggregate_passes(cpu_passes),
                "onnxruntime_cuda": aggregate_passes(cuda_passes),
            },
            "passes": {
                "opencv_cpu": {item.trip_id: item.to_dict() for item in cpu_passes},
                "onnxruntime_cuda": {
                    item.trip_id: item.to_dict() for item in cuda_passes
                },
            },
            "parity": {
                "passed": all(bool(item["passed"]) for item in parity),
                "tolerances": {
                    "confidence_abs": CONFIDENCE_ABS_TOLERANCE,
                    "bbox_coordinate_abs_px": BBOX_COORD_ABS_TOLERANCE_PX,
                    "iou_alternative": IOU_TOLERANCE,
                    "unmatched_allowed": 0,
                    "warning_finite_primary_require_exact": True,
                },
                "per_trip": parity,
            },
        }
        _atomic_json(args.output, report)
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"P2-C GPU benchmark failed: {exc}", file=sys.stderr)
        return 3
    print(json.dumps({
        "output": str(args.output),
        "parity_passed": report["parity"]["passed"],  # type: ignore[index]
        "backend_aggregate": report["backend_aggregate"],
    }, indent=2, ensure_ascii=False))
    return 0 if report["parity"]["passed"] else 4  # type: ignore[index]


if __name__ == "__main__":
    raise SystemExit(main())
