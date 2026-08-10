from __future__ import annotations

import csv
from argparse import Namespace
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from safeloop.c1.p2b_runtime import LOCKED_P2_VARIANTS, P2Variant
from safeloop.c1.temporal_features import DetectionCache
from safeloop.c1.types import Detection
from tools import c1_p2b_ablation as ablation


def _runtime_summary(*, primary_supported: bool = True) -> dict[str, object]:
    summary: dict[str, object] = {
        "n_frames": 600,
        "downstream_mean_latency_ms": 2.0,
        "downstream_p95_latency_ms": 3.0,
        "target_switch_count": 1,
        "target_acquisition_count": 4,
        "target_drop_count": 3,
        "target_reacquisition_count": 3,
        "target_identity_change_after_gap_count": 2,
        "lane_source_counts": {"lane": 300, "fixed": 300},
        "lane_source_coverage": {"lane": 0.5, "fixed": 0.5},
        "nonfinite_prediction_count": 2,
        "invalid_ttc_count": 2,
        "no_primary_target_count": 3,
        "ttc_jump_count": 4,
        "ttc_finite_toggle_count": 5,
        "warning_toggle_count": 6,
    }
    if not primary_supported:
        for name in ablation.TARGET_LIFECYCLE_COUNT_DEFINITIONS:
            summary[name] = None
        summary["lane_source_counts"] = None
        summary["lane_source_coverage"] = None
        summary["invalid_ttc_count"] = None
        summary["no_primary_target_count"] = None
    return summary


def _complete_token(tmp_path: Path) -> ablation.InferenceCompletion:
    summaries = {
        variant.value: {
            trip_id: _runtime_summary(
                primary_supported=variant != P2Variant.PHYSICS_CURRENT
            )
            for trip_id in ablation.TRIP_IDS
        }
        for variant in ablation.ABLATION_VARIANTS
    }
    return ablation.InferenceCompletion(
        variants=tuple(variant.value for variant in ablation.ABLATION_VARIANTS),
        trips=ablation.TRIP_IDS,
        prediction_rows=18000,
        prediction_root=tmp_path / "predictions",
        diagnostics_root=tmp_path / "diagnostics",
        summaries=summaries,
    )


def _write_prediction(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame_id", "timestamp", "predicted_ttc"))
        for frame_id in range(600):
            writer.writerow((frame_id, frame_id * 0.05, "inf"))


def test_ablation_order_is_exactly_the_locked_five_variants() -> None:
    assert ablation.ABLATION_VARIANTS == (
        P2Variant.PHYSICS_CURRENT,
        P2Variant.CORRIDOR_SELECTOR,
        P2Variant.SELECTOR_CLASS_HISTORY,
        P2Variant.SELECTOR_ROBUST_RANGE,
        P2Variant.FULL,
    )
    assert tuple(LOCKED_P2_VARIANTS) == ablation.ABLATION_VARIANTS


def test_cached_source_is_complete_timestamp_checked_and_confidence_filtered() -> None:
    high = Detection(2, "car", 0.9, (0.0, 0.0, 10.0, 10.0))
    low = Detection(0, "person", 0.2, (20.0, 0.0, 30.0, 10.0))
    cache = DetectionCache(
        trip_id="T01-Sample",
        stride=3,
        source_camera="image_2",
        rows={0: (high, low), 3: (high,)},
        timestamps={0: 0.0, 3: 0.15},
    )
    source = ablation.CachedDetectionSource(
        cache, n_frames=6, confidence_threshold=0.25
    )

    update = source.step(0, 0.0, object())  # image is intentionally unused by a cache
    coast = source.step(1, 0.05, object())

    assert update.detector_update
    assert update.detections == (high,)
    assert update.detector_latency_ms is None
    assert coast == ablation.DetectionStep((), False, None)
    with pytest.raises(ValueError, match="cache timestamp"):
        source.step(3, 9.0, object())

    incomplete = DetectionCache(
        trip_id="T01-Sample",
        stride=3,
        source_camera="image_2",
        rows={0: (high,)},
        timestamps={0: 0.0},
    )
    with pytest.raises(ValueError, match="incomplete cache"):
        ablation.CachedDetectionSource(
            incomplete, n_frames=6, confidence_threshold=0.25
        )


def test_evaluation_token_rejects_partial_inference_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = ablation.InferenceCompletion(
        variants=(P2Variant.PHYSICS_CURRENT.value,),
        trips=ablation.TRIP_IDS,
        prediction_rows=3600,
        prediction_root=tmp_path,
        diagnostics_root=tmp_path,
        summaries={},
    )
    imported = False

    def forbidden_import(name: str) -> object:
        nonlocal imported
        imported = True
        raise AssertionError(name)

    monkeypatch.setattr(ablation.importlib, "import_module", forbidden_import)
    with pytest.raises(RuntimeError, match="locked five-variant inference"):
        ablation.evaluate_predictions(token, data_root=tmp_path)
    assert not imported


def test_run_ablation_finishes_inference_before_offline_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    token = _complete_token(tmp_path)

    def fake_inference(**kwargs: object) -> ablation.InferenceCompletion:
        del kwargs
        events.append("inference")
        return token

    def fake_evaluation(
        completion: ablation.InferenceCompletion, *, data_root: Path
    ) -> dict[str, object]:
        del completion, data_root
        assert events == ["hash", "inference"]
        events.append("labels_and_metrics")
        return {"variants": {}, "acceptance": {}}

    hashes = iter(("a" * 64, "a" * 64))

    def fake_hash() -> str:
        events.append("hash")
        return next(hashes)

    monkeypatch.setattr(ablation, "require_evaluator_hash", fake_hash)
    monkeypatch.setattr(ablation, "run_all_inference", fake_inference)
    monkeypatch.setattr(ablation, "evaluate_predictions", fake_evaluation)
    monkeypatch.setattr(ablation, "_atomic_json", lambda *args: None)
    args = Namespace(
        data_root=tmp_path,
        manifest_dir=tmp_path,
        cache_dir=tmp_path,
        baseline_dir=tmp_path,
        prediction_root=tmp_path,
        run_root=tmp_path,
        cache_pattern=ablation.DEFAULT_CACHE_PATTERN,
        confidence_threshold=0.25,
        live_yolo_cpu_benchmark=False,
    )

    report = ablation.run_ablation(args)

    assert events == ["hash", "inference", "labels_and_metrics", "hash"]
    assert report["report_schema"] == ablation.REPORT_SCHEMA
    assert report["target_lifecycle_count_definitions"] == dict(
        ablation.TARGET_LIFECYCLE_COUNT_DEFINITIONS
    )
    assert (
        report["lane_source_coverage_definition"]
        == ablation.LANE_SOURCE_COVERAGE_DEFINITION
    )


def test_prediction_layout_requires_exact_six_files_and_3600_rows_per_variant(
    tmp_path: Path,
) -> None:
    for variant in ablation.ABLATION_VARIANTS:
        for trip_id in ablation.TRIP_IDS:
            _write_prediction(tmp_path / variant.value / f"{trip_id}.csv")

    reports = ablation._assert_prediction_layout(tmp_path)

    assert len(reports) == 30
    assert sum(int(report["rows"]) for report in reports) == 18000
    _write_prediction(tmp_path / P2Variant.FULL.value / "unexpected.csv")
    with pytest.raises(RuntimeError, match="expected exactly"):
        ablation._assert_prediction_layout(tmp_path)


def test_evaluator_hash_gate_and_prediction_extra_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator = tmp_path / "evaluation.py"
    evaluator.write_text("not the official evaluator\n", encoding="utf-8")
    monkeypatch.setattr(ablation, "OFFICIAL_EVALUATOR", evaluator)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        ablation.require_evaluator_hash()

    for variant in ablation.ABLATION_VARIANTS:
        (tmp_path / "predictions" / variant.value).mkdir(parents=True)
    (tmp_path / "predictions" / "sixth_variant").mkdir()
    with pytest.raises(RuntimeError, match="unexpected prediction variant"):
        ablation._assert_prediction_layout(tmp_path / "predictions")


def test_macro_and_confusion_counts_are_frame_exact() -> None:
    per_trip = [
        {
            "n_frames": 600,
            "mae_critical": float(index + 1),
            "f1": 0.1 * index,
            "composite_score": 50.0 + index,
        }
        for index in range(6)
    ]
    macro = ablation.macro_metrics(per_trip)
    pairs = {
        0: (1.0, 1.0),
        1: (1.0, 4.0),
        2: (4.0, 1.0),
        3: (4.0, 4.0),
    }

    assert macro["n_frames"] == 3600
    assert macro["mae_critical"] == 3.5
    assert macro["composite_score"] == 52.5
    assert ablation.confusion_counts(pairs) == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_invalid_ttc_requires_primary_and_is_separate_from_no_primary() -> None:
    common = dict(
        trip_id="T01-Sample",
        variant=P2Variant.FULL.value,
        timestamp=0.0,
        dangerous_track_ids=(),
        target_switched=False,
        held_by_hysteresis=False,
        warning=False,
        invalid_reason="test",
        ttc_source="test",
        lane_source="fixed",
        lane_confidence=0.0,
        candidate_count=0,
        detector_update=False,
        detection_count=0,
        ttc_jump=False,
        ttc_finite_toggle=False,
        warning_toggle=False,
        downstream_latency_ms=1.0,
        detector_latency_ms=None,
    )
    rows = (
        ablation.RuntimeRow(
            frame_id=0, predicted_ttc_s=float("inf"), primary_track_id=None, **common
        ),
        ablation.RuntimeRow(
            frame_id=1, predicted_ttc_s=float("inf"), primary_track_id=7, **common
        ),
        ablation.RuntimeRow(
            frame_id=2, predicted_ttc_s=3.0, primary_track_id=7, **common
        ),
    )

    summary = ablation.summarize_runtime(rows, primary_supported=True)

    assert summary["nonfinite_prediction_count"] == 2
    assert summary["no_primary_target_count"] == 1
    assert summary["invalid_ttc_count"] == 1
    assert summary["target_switch_count"] == 0
    assert summary["target_acquisition_count"] == 1
    assert summary["target_drop_count"] == 0
    assert summary["target_reacquisition_count"] == 0
    assert summary["target_identity_change_after_gap_count"] == 0
    assert summary["lane_source_counts"] == {"lane": 0, "fixed": 3}
    assert summary["lane_source_coverage"] == {"lane": 0.0, "fixed": 1.0}

    physics = ablation.summarize_runtime(rows, primary_supported=False)
    assert all(
        physics[name] is None
        for name in ablation.TARGET_LIFECYCLE_COUNT_DEFINITIONS
    )
    assert physics["lane_source_counts"] is None
    assert physics["lane_source_coverage"] is None


def test_target_lifecycle_counts_separate_direct_switches_from_gap_changes() -> None:
    template = ablation.RuntimeRow(
        trip_id="T01-Sample",
        variant=P2Variant.FULL.value,
        frame_id=0,
        timestamp=0.0,
        predicted_ttc_s=3.0,
        primary_track_id=None,
        dangerous_track_ids=(),
        target_switched=False,
        held_by_hysteresis=False,
        warning=False,
        invalid_reason="test",
        ttc_source="test",
        lane_source="lane",
        lane_confidence=0.9,
        candidate_count=1,
        detector_update=False,
        detection_count=0,
        ttc_jump=False,
        ttc_finite_toggle=False,
        warning_toggle=False,
        downstream_latency_ms=1.0,
        detector_latency_ms=None,
    )
    primary_ids = (None, 10, 10, 11, None, None, 11, None, 12, 12)
    rows = tuple(
        replace(
            template,
            frame_id=index,
            timestamp=index * 0.05,
            primary_track_id=current_id,
            target_switched=(
                index > 0
                and primary_ids[index - 1] is not None
                and current_id is not None
                and primary_ids[index - 1] != current_id
            ),
        )
        for index, current_id in enumerate(primary_ids)
    )

    summary = ablation.summarize_runtime(rows, primary_supported=True)

    assert summary["target_switch_count"] == 1
    assert summary["target_acquisition_count"] == 3
    assert summary["target_drop_count"] == 2
    assert summary["target_reacquisition_count"] == 2
    assert summary["target_identity_change_after_gap_count"] == 1

    mislabeled = list(rows)
    mislabeled[3] = replace(mislabeled[3], target_switched=False)
    with pytest.raises(RuntimeError, match="target_switched flags"):
        ablation.summarize_runtime(tuple(mislabeled), primary_supported=True)


def test_runtime_aggregate_sums_lifecycle_and_lane_counts_exactly() -> None:
    per_trip: dict[str, dict[str, object]] = {}
    for index, trip_id in enumerate(ablation.TRIP_IDS):
        summary = _runtime_summary()
        lane_frames = 50 * (index + 1)
        summary["lane_source_counts"] = {
            "lane": lane_frames,
            "fixed": 600 - lane_frames,
        }
        summary["lane_source_coverage"] = {
            "lane": lane_frames / 600,
            "fixed": (600 - lane_frames) / 600,
        }
        per_trip[trip_id] = summary

    pooled = ablation._aggregate_runtime_summaries(
        per_trip,
        downstream_latency_ms=(2.0,) * 3600,
    )

    assert pooled["target_switch_count"] == 6
    assert pooled["target_acquisition_count"] == 24
    assert pooled["target_drop_count"] == 18
    assert pooled["target_reacquisition_count"] == 18
    assert pooled["target_identity_change_after_gap_count"] == 12
    assert pooled["lane_source_counts"] == {"lane": 1050, "fixed": 2550}
    assert pooled["lane_source_coverage"] == {
        "lane": 1050 / 3600,
        "fixed": 2550 / 3600,
    }

    physics = {
        trip_id: _runtime_summary(primary_supported=False)
        for trip_id in ablation.TRIP_IDS
    }
    physics_pooled = ablation._aggregate_runtime_summaries(
        physics,
        downstream_latency_ms=(2.0,) * 3600,
    )
    assert all(
        physics_pooled[name] is None
        for name in ablation.TARGET_LIFECYCLE_COUNT_DEFINITIONS
    )
    assert physics_pooled["lane_source_counts"] is None
    assert physics_pooled["lane_source_coverage"] is None


def test_physics_runtime_parity_allows_only_documented_edge_drift() -> None:
    template = ablation.RuntimeRow(
        trip_id="T01-Sample",
        variant=P2Variant.PHYSICS_CURRENT.value,
        frame_id=0,
        timestamp=0.0,
        predicted_ttc_s=5.0,
        primary_track_id=None,
        dangerous_track_ids=(),
        target_switched=False,
        held_by_hysteresis=False,
        warning=False,
        invalid_reason="valid",
        ttc_source="physics",
        lane_source="legacy",
        lane_confidence=0.0,
        candidate_count=0,
        detector_update=True,
        detection_count=0,
        ttc_jump=False,
        ttc_finite_toggle=False,
        warning_toggle=False,
        downstream_latency_ms=1.0,
        detector_latency_ms=None,
    )
    rows = tuple(
        replace(template, frame_id=index, timestamp=index * 0.05)
        for index in range(600)
    )
    authoritative = tuple(
        (index, index * 0.05, float("inf") if index == 14 else 5.001)
        for index in range(600)
    )

    report = ablation.physics_runtime_parity(rows, authoritative)

    assert report["finite_mask_mismatch_frame_ids"] == [14]
    assert report["max_jointly_finite_abs_delta_s"] == pytest.approx(0.001)
    second_mismatch = list(authoritative)
    second_mismatch[15] = (15, 0.75, float("inf"))
    with pytest.raises(RuntimeError, match="drifted"):
        ablation.physics_runtime_parity(rows, tuple(second_mismatch))
    numeric_drift = list(rows)
    numeric_drift[20] = replace(numeric_drift[20], predicted_ttc_s=5.05)
    with pytest.raises(RuntimeError, match="drifted"):
        ablation.physics_runtime_parity(tuple(numeric_drift), authoritative)


def test_shared_detector_sample_combines_with_every_detector_update() -> None:
    estimate = ablation.estimate_end_to_end_cpu_timing(
        (1.0, 1.0, 1.0),
        (True, False, True),
        (9.0, 19.0),
    )

    assert estimate["detector_update_frames"] == 2
    # Combined values are 10, 1 and 20 ms.
    assert estimate["estimated_end_to_end_mean_latency_ms"] == pytest.approx(10.333)
    assert estimate["estimated_end_to_end_p95_latency_ms"] == pytest.approx(19.0)
    assert estimate["estimated_end_to_end_cpu_fps"] == pytest.approx(96.774)
    assert "latency_samples_ms" in estimate["method"]


def test_shared_detector_report_persists_raw_samples_for_exact_p95_replay() -> None:
    timing = ablation.SharedDetectorTiming(
        trip_id="T01-Sample",
        source_frames=12,
        stride=3,
        warmup_detector_updates=1,
        measured_detector_updates=2,
        latency_ms=(9.123456789, 19.987654321),
    )

    report = timing.report()

    assert report["schema"] == ablation.SHARED_DETECTOR_TIMING_SCHEMA
    assert report["latency_sample_unit"] == "ms"
    assert report["latency_samples_ms"] == [9.123456789, 19.987654321]
    assert report["detector_p95_latency_ms"] == round(
        float(ablation.np.percentile(report["latency_samples_ms"], 95)), 3
    )
    direct = ablation.estimate_end_to_end_cpu_timing(
        (1.0, 1.0, 1.0),
        (True, False, True),
        timing.latency_ms,
    )
    replayed = ablation.estimate_end_to_end_cpu_timing(
        (1.0, 1.0, 1.0),
        (True, False, True),
        report["latency_samples_ms"],
    )
    assert replayed == direct


def test_one_shared_detector_sample_is_attached_to_all_five_rungs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation: dict[str, object] = {
        "variants": {variant.value: {} for variant in ablation.ABLATION_VARIANTS}
    }
    token = _complete_token(tmp_path)
    timing = ablation.SharedDetectorTiming(
        trip_id="T01-Sample",
        source_frames=12,
        stride=3,
        warmup_detector_updates=1,
        measured_detector_updates=3,
        latency_ms=(9.0, 10.0, 11.0),
    )
    monkeypatch.setattr(
        ablation,
        "_read_diagnostic_timing",
        lambda root, variant: ((1.0,) * 3600, tuple(index % 3 == 0 for index in range(3600))),
    )

    ablation.add_shared_end_to_end_estimates(evaluation, token, timing)

    variants = evaluation["variants"]
    assert all(
        "estimated_end_to_end_cpu" in variants[variant.value]
        for variant in ablation.ABLATION_VARIANTS
    )


def test_diagnostic_csv_is_separate_and_contains_no_label_columns(tmp_path: Path) -> None:
    row = ablation.RuntimeRow(
        trip_id="T01-Sample",
        variant=P2Variant.FULL.value,
        frame_id=0,
        timestamp=0.0,
        predicted_ttc_s=float("inf"),
        primary_track_id=None,
        dangerous_track_ids=(),
        target_switched=False,
        held_by_hysteresis=False,
        warning=False,
        invalid_reason="no_tracks",
        ttc_source="invalid",
        lane_source="fallback",
        lane_confidence=0.0,
        candidate_count=0,
        detector_update=True,
        detection_count=0,
        ttc_jump=False,
        ttc_finite_toggle=False,
        warning_toggle=False,
        downstream_latency_ms=1.0,
        detector_latency_ms=None,
    )
    path = tmp_path / "diagnostics.csv"

    ablation.write_diagnostic_csv(path, (row,))

    with path.open("r", encoding="utf-8", newline="") as stream:
        header = tuple(csv.DictReader(stream).fieldnames or ())
    assert header == ablation.DIAGNOSTIC_COLUMNS
    assert not any(
        forbidden in column.lower()
        for column in header
        for forbidden in ("ground_truth", "target_label", "depth", "event")
    )


@dataclass
class _FakeMetric:
    trip_id: str
    n_frames: int
    mae_critical: float
    f1: float
    composite_score: float


def test_offline_evaluation_reports_3600_metrics_for_every_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = _complete_token(tmp_path)
    variant_marker = {
        variant.value: float(index + 3)
        for index, variant in enumerate(ablation.ABLATION_VARIANTS)
    }
    variant_marker[P2Variant.PHYSICS_CURRENT.value] = 10.0

    class Prediction:
        def __init__(self, value: float) -> None:
            self.predicted_ttc = value

    def load_predictions(path: Path) -> dict[int, Prediction]:
        marker = variant_marker[path.parent.name]
        return {frame_id: Prediction(marker) for frame_id in range(600)}

    def load_truth(path: Path) -> SimpleNamespace:
        del path
        return SimpleNamespace(ttc={frame_id: 10.0 for frame_id in range(600)})

    def compute_metrics(
        trip_id: str, pairs: dict[int, tuple[float, float]]
    ) -> _FakeMetric:
        marker = pairs[0][0]
        score = 55.2 if marker == 10.0 else 60.0 + marker / 100.0
        return _FakeMetric(trip_id, len(pairs), 2.0, 0.5, score)

    fake_evaluator = SimpleNamespace(
        load_predictions=load_predictions,
        load_ground_truth_from_trip=load_truth,
        compute_trip_metrics=compute_metrics,
    )
    monkeypatch.setattr(ablation, "_assert_prediction_layout", lambda path: ())
    monkeypatch.setattr(
        ablation,
        "_read_diagnostic_timing",
        lambda root, variant: ((2.0,) * 3600, tuple(index % 3 == 0 for index in range(3600))),
    )
    monkeypatch.setattr(
        ablation.importlib, "import_module", lambda name: fake_evaluator
    )

    report = ablation.evaluate_predictions(token, data_root=tmp_path)

    assert set(report["variants"]) == {
        variant.value for variant in ablation.ABLATION_VARIANTS
    }
    for variant_report in report["variants"].values():
        assert variant_report["n_predictions"] == 3600
        assert variant_report["macro"]["n_frames"] == 3600
        assert set(variant_report["fp_fn"]) == ablation.DIAGNOSTIC_TRIPS
    assert report["acceptance"]["baseline_macro_composite"] == 55.2
