from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace

import numpy as np

from safeloop.c1.types import Detection
from tools.c1_p2c_gpu_benchmark import (
    DEFAULT_CONFIDENCE,
    BackendPass,
    FrameSample,
    _build_parser,
    aggregate_passes,
    compare_backend_passes,
    compare_detection_sets,
    main,
    run_backend_pass,
    warm_up_detector,
)


class _Frame:
    def __init__(self, frame_id: int, *, left_calls: list[int]) -> None:
        self.frame_id = frame_id
        self.timestamp = frame_id * 0.05
        self.ego = {"speed_kmh": 36.0}
        self._left_calls = left_calls

    def left(self) -> np.ndarray:
        self._left_calls.append(self.frame_id)
        return np.full((12, 16, 3), self.frame_id, dtype=np.uint8)


class _Trip:
    trip_id = "synthetic-trip"

    def __init__(self, frames: int) -> None:
        self.n_frames = frames
        self.left_calls: list[int] = []

    def frame(self, index: int) -> _Frame:
        return _Frame(index, left_calls=self.left_calls)


class _Detector:
    device = "cpu"

    def __init__(self, *, perturbation: float = 0.0) -> None:
        self.calls: list[int] = []
        self.perturbation = perturbation

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        frame_id = int(image_bgr[0, 0, 0])
        self.calls.append(frame_id)
        shift = self.perturbation
        return [
            Detection(
                2,
                "car",
                0.8 + min(shift, 5e-5),
                (2.0 + shift, 2.0, 8.0 + shift, 10.0),
            )
        ]


@dataclass(frozen=True)
class _Result:
    predicted_ttc_s: float
    warning: bool
    primary_track_id: int | None


class _Runtime:
    def __init__(self) -> None:
        self.steps: list[tuple[bool, int, float]] = []
        self.resets = 0
        self._primary: int | None = None

    def reset(self) -> None:
        self.resets += 1
        self._primary = None

    def step(
        self,
        _image_bgr: np.ndarray,
        detections: tuple[Detection, ...],
        *,
        detector_update: bool,
        timestamp: float,
        ego_speed_kmh: float,
    ) -> _Result:
        self.steps.append((detector_update, len(detections), ego_speed_kmh))
        if detections:
            self._primary = 1
        ttc = 1.5 if self._primary is not None else float("inf")
        return _Result(ttc, ttc < 2.0, self._primary)


def test_cli_defaults_to_locked_accuracy_confidence_and_rejects_override(
    tmp_path, capsys
) -> None:
    parsed = _build_parser().parse_args(
        [
            "trip",
            "--manifest-dir",
            "manifests",
            "--output",
            str(tmp_path / "report.json"),
        ]
    )
    assert DEFAULT_CONFIDENCE == 0.25
    assert parsed.confidence == DEFAULT_CONFIDENCE

    result = main(
        [
            "trip",
            "--manifest-dir",
            "manifests",
            "--output",
            str(tmp_path / "report.json"),
            "--confidence",
            "0.20",
        ]
    )
    assert result == 2
    assert "locked to 0.25" in capsys.readouterr().err


def test_warmup_records_cold_call_separately_and_is_not_a_pass() -> None:
    trip = _Trip(4)
    detector = _Detector()

    report = warm_up_detector(trip, detector, calls=3)

    assert detector.calls == [0, 0, 0]
    assert trip.left_calls == [0]
    document = report.to_dict()
    assert document["calls"] == 3
    assert document["excluded_from_measured_pass"] is True
    assert len(document["subsequent_warmup_call_ms"]) == 2


def test_backend_pass_measures_decode_detector_and_runtime_synchronously() -> None:
    trip = _Trip(6)
    detector = _Detector()
    runtime = _Runtime()

    result = run_backend_pass(
        trip,
        detector,
        runtime,
        backend="fake_cpu",
        stride=2,
        expected_frames=6,
    )

    assert runtime.resets == 1
    assert trip.left_calls == list(range(6))
    assert detector.calls == [0, 2, 4]
    assert [item.detector_update for item in result.samples] == [
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert [len(item.detections) for item in result.samples] == [1, 0, 1, 0, 1, 0]
    assert all(item.decode_ms >= 0.0 for item in result.samples)
    assert all(item.runtime_ms >= 0.0 for item in result.samples)
    assert all(item.end_to_end_ms >= item.decode_ms for item in result.samples)
    summary = result.summary()
    assert summary["frames"] == 6
    assert summary["detector_updates"] == 3
    assert summary["whole_pass_wall_fps"] > 0.0
    assert summary["per_frame_p95_ms"] >= 0.0


def test_hungarian_same_label_parity_accepts_tiny_backend_noise_and_reordering() -> None:
    reference = [
        Detection(0, "person", 0.75, (10.0, 10.0, 30.0, 50.0)),
        Detection(2, "car", 0.90, (100.0, 80.0, 220.0, 200.0)),
        Detection(0, "person", 0.65, (300.0, 90.0, 340.0, 180.0)),
    ]
    candidate = [
        Detection(0, "person", 0.65001, (300.01, 90.0, 340.01, 180.0)),
        Detection(0, "person", 0.74999, (10.01, 10.0, 30.01, 50.0)),
        Detection(2, "car", 0.90001, (100.01, 80.0, 220.01, 200.0)),
    ]

    report = compare_detection_sets(reference, candidate)

    assert report["passed"] is True
    assert report["exact_count_and_labels"] is True
    assert report["unmatched_reference_indices"] == []
    assert report["unmatched_candidate_indices"] == []
    assert len(report["matches"]) == 3


def test_detector_parity_rejects_unmatched_or_out_of_tolerance_output() -> None:
    reference = [Detection(2, "car", 0.9, (0.0, 0.0, 10.0, 10.0))]
    wrong_geometry = [Detection(2, "car", 0.9, (2.0, 0.0, 12.0, 10.0))]
    wrong_label = [Detection(0, "person", 0.9, (0.0, 0.0, 10.0, 10.0))]

    assert compare_detection_sets(reference, wrong_geometry)["passed"] is False
    label_report = compare_detection_sets(reference, wrong_label)
    assert label_report["passed"] is False
    assert label_report["unmatched_reference_indices"] == [0]
    assert label_report["unmatched_candidate_indices"] == [0]


def test_backend_parity_checks_warning_finite_ttc_and_primary_track() -> None:
    trip = _Trip(6)
    reference = run_backend_pass(
        trip, _Detector(), _Runtime(), backend="opencv_cpu", stride=2
    )
    candidate = run_backend_pass(
        trip,
        _Detector(perturbation=1e-5),
        _Runtime(),
        backend="onnxruntime_cuda",
        stride=2,
    )

    passing = compare_backend_passes(reference, candidate)
    assert passing["passed"] is True
    assert passing["detector"]["unmatched_reference"] == 0
    assert passing["runtime_output"]["warning_mismatches"] == 0

    changed_sample = replace(
        candidate.samples[3],
        predicted_ttc_s=float("inf"),
        warning=False,
        primary_track_id=99,
    )
    changed = BackendPass(
        candidate.backend,
        candidate.trip_id,
        candidate.stride,
        candidate.wall_time_s,
        (*candidate.samples[:3], changed_sample, *candidate.samples[4:]),
    )
    failing = compare_backend_passes(reference, changed)
    assert failing["passed"] is False
    assert failing["runtime_output"]["warning_mismatches"] == 1
    assert failing["runtime_output"]["finite_ttc_mismatches"] == 1
    assert failing["runtime_output"]["primary_track_id_mismatches"] == 1


def test_raw_samples_are_strict_json_and_aggregate_uses_measured_wall_time() -> None:
    trip = _Trip(3)
    first = run_backend_pass(
        trip, _Detector(), _Runtime(), backend="opencv_cpu", stride=2
    )
    second = BackendPass(
        first.backend,
        "synthetic-trip-2",
        first.stride,
        first.wall_time_s,
        tuple(replace(item, frame_id=index) for index, item in enumerate(first.samples)),
    )

    document = first.to_dict()
    json.dumps(document, allow_nan=False)
    assert document["raw_frame_samples"][0]["detections"]
    assert document["raw_frame_samples"][0]["predicted_ttc_finite"] is True
    assert document["raw_frame_samples"][0]["predicted_ttc_s"] == 1.5

    aggregate = aggregate_passes((first, second))
    assert aggregate["frames"] == 6
    assert math.isclose(
        aggregate["whole_pass_wall_fps"], 6 / (2 * first.wall_time_s)
    )
