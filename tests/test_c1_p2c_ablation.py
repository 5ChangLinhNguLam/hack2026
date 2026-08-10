from __future__ import annotations

import ast
import csv
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from safeloop.c1.temporal_features import CameraGeometry
from tools.c1_p2c_ablation import (
    DIAGNOSTIC_COLUMNS,
    EXPECTED_FRAMES,
    EXPECTED_PREDICTION_ROWS,
    TRIP_IDS,
    VARIANT_VALUES,
    DetectionStep,
    InferenceCompletion,
    RuntimeRow,
    _atomic_prediction_csv,
    _load_runtime_api,
    _runtime_row,
    aggregate_invalid_audits,
    build_acceptance,
    copy_frozen_prediction,
    evaluate_predictions,
    frozen_runtime_parity,
    invalid_primary_ttc_audit,
    macro_metrics,
    summarize_runtime,
    write_diagnostic_csv,
)


def _result(
    *,
    predicted_ttc_s: float = 3.0,
    primary_track_id: int | None = 7,
    primary_bbox: tuple[float, float, float, float] | None = (
        10.0,
        20.0,
        30.0,
        50.0,
    ),
    invalid_reason: str = "valid",
    warning: bool = False,
    estimator_uncertainty_s: float = 0.4,
    dangerous_track_ids: tuple[int, ...] | None = None,
) -> SimpleNamespace:
    if dangerous_track_ids is None:
        dangerous_track_ids = (
            (primary_track_id,)
            if primary_track_id is not None
            and math.isfinite(predicted_ttc_s)
            and predicted_ttc_s < 3.0
            else ()
        )
    return SimpleNamespace(
        predicted_ttc_s=predicted_ttc_s,
        primary_track_id=primary_track_id,
        primary_bbox=primary_bbox,
        dangerous_track_ids=dangerous_track_ids,
        target_switched=False,
        held_by_hysteresis=False,
        warning=warning,
        invalid_reason=invalid_reason,
        ttc_source="physics",
        lane_source="fixed",
        lane_confidence=0.3,
        candidate_count=1,
        downstream_latency_ms=1.25,
        estimator_reason="ok",
        estimator_uncertainty_s=estimator_uncertainty_s,
        estimator_sources=("ground_plane", "class_prior"),
    )


def _row(frame_id: int = 0, **result_updates: object) -> RuntimeRow:
    result = _result(**result_updates)
    return _runtime_row(
        trip_id="T01-Sample",
        variant="p2c_full_fusion",
        frame_id=frame_id,
        timestamp=frame_id * 0.05,
        result=result,
        detector_step=DetectionStep((), False),
        published_ttc_s=float(result.predicted_ttc_s),
        previous=None,
    )


def _rows() -> tuple[RuntimeRow, ...]:
    base = _row()
    return tuple(
        replace(base, frame_id=frame_id, timestamp=frame_id * 0.05)
        for frame_id in range(EXPECTED_FRAMES)
    )


def test_locked_protocol_requires_exact_21600_prediction_token(tmp_path: Path) -> None:
    assert VARIANT_VALUES == (
        "physics",
        "p2b_full",
        "p2b_ground_plane",
        "p2b_scale_expansion",
        "p2b_class_prior",
        "p2c_full_fusion",
    )
    complete = InferenceCompletion(
        variants=VARIANT_VALUES,
        trips=TRIP_IDS,
        prediction_rows=EXPECTED_PREDICTION_ROWS,
        prediction_root=tmp_path,
        diagnostics_root=tmp_path,
        summaries={},
    )
    complete.require_complete()

    with pytest.raises(RuntimeError, match="21,600"):
        replace(complete, prediction_rows=EXPECTED_PREDICTION_ROWS - 1).require_complete()


def test_harness_contract_accepts_one_frame_from_every_runtime_variant() -> None:
    runtime_type, variants = _load_runtime_api()
    geometry = CameraGeometry(64, 48, 50.0, 50.0, 32.0)
    image = np.zeros((48, 64, 3), dtype=np.uint8)

    for variant in VARIANT_VALUES:
        runtime = runtime_type(geometry, variants[variant])
        result = runtime.step(
            image,
            (),
            detector_update=True,
            timestamp=0.0,
            ego_speed_kmh=0.0,
        )
        row = _runtime_row(
            trip_id="Synthetic",
            variant=variant,
            frame_id=0,
            timestamp=0.0,
            result=result,
            detector_step=DetectionStep((), True),
            published_ttc_s=result.predicted_ttc_s,
            previous=None,
        )

        assert row.variant == variant
        assert row.primary_track_id is None
        assert row.primary_bbox is None


def test_runtime_row_requires_bbox_consistency_and_clear_invalid_reason() -> None:
    valid = _row()
    infinite_uncertainty = _row(estimator_uncertainty_s=float("inf"))

    assert valid.primary_bbox == (10.0, 20.0, 30.0, 50.0)
    assert valid.estimator_sources == "ground_plane;class_prior"
    assert math.isinf(infinite_uncertainty.estimator_uncertainty_s)
    assert _row(primary_bbox=None).primary_bbox is None

    with pytest.raises(ValueError, match="primary_bbox must be None"):
        _row(primary_track_id=None)
    with pytest.raises(RuntimeError, match="success reason"):
        _row(
            predicted_ttc_s=float("inf"),
            invalid_reason="ok",
            warning=False,
        )
    with pytest.raises(ValueError, match="non-negative or inf"):
        _row(estimator_uncertainty_s=-0.1)
    with pytest.raises(RuntimeError, match="safe/non-finite"):
        _row(predicted_ttc_s=4.0, dangerous_track_ids=(7,))
    with pytest.raises(RuntimeError, match="safe/non-finite"):
        _row(
            predicted_ttc_s=float("inf"),
            invalid_reason="high_uncertainty",
            dangerous_track_ids=(7,),
        )
    with pytest.raises(RuntimeError, match="warning and published TTC disagree"):
        _row(predicted_ttc_s=1.5, warning=False)

    legacy = _result(predicted_ttc_s=4.0, dangerous_track_ids=(7,))
    legacy_row = _runtime_row(
        trip_id="T01-Sample",
        variant="p2b_full",
        frame_id=0,
        timestamp=0.0,
        result=legacy,
        detector_step=DetectionStep((), False),
        published_ttc_s=4.0,
        previous=None,
    )
    assert legacy_row.primary_danger_ttc_consistent is False


def test_diagnostic_schema_contains_target_bbox_and_estimator_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "T01-Sample.csv"
    rows = _rows()

    write_diagnostic_csv(path, rows)

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        records = list(reader)
    assert tuple(reader.fieldnames or ()) == DIAGNOSTIC_COLUMNS
    assert len(records) == 600
    assert records[0]["primary_bbox_x1"] == "10.0"
    assert records[0]["estimator_reason"] == "ok"
    assert records[0]["estimator_uncertainty_s"] == "0.4"
    assert records[0]["estimator_sources"] == "ground_plane;class_prior"


def test_frozen_parity_uses_runtime_value_and_copy_is_byte_exact(
    tmp_path: Path,
) -> None:
    rows = tuple(
        replace(
            row,
            predicted_ttc_s=99.0,
            runtime_predicted_ttc_s=3.0 + row.frame_id * 0.001,
        )
        for row in _rows()
    )
    authoritative = tuple(
        (row.frame_id, row.timestamp, row.runtime_predicted_ttc_s) for row in rows
    )

    parity = frozen_runtime_parity(
        rows,
        authoritative,
        max_mask_mismatches=0,
        max_finite_delta_s=1e-9,
    )

    assert parity["finite_mask_mismatch_count"] == 0
    source = tmp_path / "source.csv"
    destination = tmp_path / "nested" / "copied.csv"
    _atomic_prediction_csv(source, rows)
    digest = copy_frozen_prediction(source, destination)
    assert source.read_bytes() == destination.read_bytes()
    assert len(digest) == 64


def test_frozen_parity_rejects_mask_or_numeric_drift() -> None:
    rows = _rows()
    authoritative = tuple(
        (row.frame_id, row.timestamp, row.runtime_predicted_ttc_s) for row in rows
    )
    mask_drift = (replace(rows[0], runtime_predicted_ttc_s=float("inf")), *rows[1:])
    value_drift = (replace(rows[0], runtime_predicted_ttc_s=4.0), *rows[1:])

    with pytest.raises(RuntimeError, match="parity failed"):
        frozen_runtime_parity(
            mask_drift,
            authoritative,
            max_mask_mismatches=0,
            max_finite_delta_s=0.02,
        )
    with pytest.raises(RuntimeError, match="parity failed"):
        frozen_runtime_parity(
            value_drift,
            authoritative,
            max_mask_mismatches=0,
            max_finite_delta_s=0.02,
        )


def test_macro_metrics_include_inverse_ttc_precision_and_recall() -> None:
    per_trip = [
        {
            "n_frames": 600,
            "mae_critical": float(index + 1),
            "inv_ttc_mae": 0.1 + index * 0.01,
            "precision": 0.2 + index * 0.1,
            "recall": 0.4 + index * 0.05,
            "f1": 0.3 + index * 0.05,
            "composite_score": 50.0 + index,
        }
        for index in range(6)
    ]

    macro = macro_metrics(per_trip)

    assert macro == {
        "n_trips": 6,
        "n_frames": 3600,
        "mae_critical": 3.5,
        "inv_ttc_mae": 0.125,
        "precision": 0.45,
        "recall": 0.525,
        "f1": 0.425,
        "composite_score": 52.5,
    }


def test_invalid_primary_ttc_is_disjoint_by_gt_zone_and_reason(
    tmp_path: Path,
) -> None:
    rows = list(_rows())
    for frame_id, reason in (
        (0, "high_uncertainty"),
        (1, "insufficient_history"),
        (2, "non_closing"),
    ):
        rows[frame_id] = replace(
            rows[frame_id],
            predicted_ttc_s=float("inf"),
            runtime_predicted_ttc_s=float("inf"),
            warning=False,
            invalid_reason=reason,
        )
    path = tmp_path / "T01-Sample.csv"
    write_diagnostic_csv(path, rows)
    truth = {frame_id: float("inf") for frame_id in range(EXPECTED_FRAMES)}
    truth.update({0: 1.5, 1: 2.5, 2: 3.0})

    report = invalid_primary_ttc_audit(path, truth)

    assert report == {
        "danger": 1,
        "critical": 1,
        "non_critical": 1,
        "total": 3,
        "reason_counts": {
            "high_uncertainty": 1,
            "insufficient_history": 1,
            "non_closing": 1,
        },
    }


def test_invalid_audit_aggregation_reports_pooled_and_macro_counts() -> None:
    per_trip = {
        trip_id: {
            "danger": index,
            "critical": 1,
            "non_critical": 2,
            "total": index + 3,
            "reason_counts": {"uncertain": index + 3},
        }
        for index, trip_id in enumerate(TRIP_IDS)
    }

    report = aggregate_invalid_audits(per_trip)

    assert report["pooled"] == {
        "danger": 15,
        "critical": 6,
        "non_critical": 12,
        "total": 33,
        "reason_counts": {"uncertain": 33},
    }
    assert report["macro_mean_per_trip"]["total"] == pytest.approx(5.5)


def _variant_report(
    *,
    macro_score: float,
    trip_score: float,
    t02_inverse: float,
    t06_fn: int,
    danger_invalid: int,
    p95_ms: float,
) -> dict[str, object]:
    per_trip = [
        {
            "trip_id": trip_id,
            "composite_score": trip_score,
            "inv_ttc_mae": t02_inverse if trip_id == "T02-Sample" else 0.2,
        }
        for trip_id in TRIP_IDS
    ]
    return {
        "macro": {"composite_score": macro_score},
        "per_trip": per_trip,
        "fp_fn": {
            trip_id: {"tp": 1, "fp": 0, "fn": t06_fn if trip_id == "T06-Sample" else 0, "tn": 599}
            for trip_id in TRIP_IDS
        },
        "invalid_primary_ttc": {"pooled": {"danger": danger_invalid}},
        "runtime": {
            "downstream_p95_latency_ms": p95_ms,
            "warning_ttc_consistency_violation_count": 0,
            "primary_danger_ttc_consistency_violation_count": 0,
        },
    }


def test_acceptance_gate_compares_full_fusion_to_accepted_p2b() -> None:
    physics = _variant_report(
        macro_score=55.2,
        trip_score=55.2,
        t02_inverse=0.4,
        t06_fn=8,
        danger_invalid=0,
        p95_ms=1.0,
    )
    p2b = _variant_report(
        macro_score=60.7,
        trip_score=60.7,
        t02_inverse=0.3279,
        t06_fn=1,
        danger_invalid=0,
        p95_ms=21.8,
    )
    full = _variant_report(
        macro_score=65.2,
        trip_score=65.2,
        t02_inverse=0.25,
        t06_fn=1,
        danger_invalid=0,
        p95_ms=24.9,
    )
    variants = {
        variant: p2b
        for variant in VARIANT_VALUES
    }
    variants.update(physics=physics, p2b_full=p2b, p2c_full_fusion=full)

    report = build_acceptance(variants)

    assert report["macro_strictly_above_60_7"] is True
    assert report["preferred_macro_at_least_65"] is True
    assert report["t02_inverse_ttc_improved"] is True
    assert report["t06_false_negatives_at_most_1"] is True
    assert report["danger_invalid_ttc_not_increased"] is True
    assert report["post_detector_p95_under_25_ms"] is True
    assert report["warning_ttc_consistent_all_rungs"] is True
    assert report["primary_danger_ttc_consistent_p2c_rungs"] is True
    assert report["danger_invalid_ttc"] == {
        "definition": (
            "primary target exists, published TTC is non-finite, and offline "
            "GT TTC is <2 seconds"
        ),
        "p2b_full": 0,
        "p2c_full_fusion": 0,
        "delta": 0,
        "gate_pass": True,
    }
    assert report["required_gate_pass"] is True


def test_consistency_gate_reports_legacy_p2b_but_blocks_p2c_rungs() -> None:
    physics = _variant_report(
        macro_score=55.2,
        trip_score=55.2,
        t02_inverse=0.4,
        t06_fn=8,
        danger_invalid=0,
        p95_ms=1.0,
    )
    p2b = _variant_report(
        macro_score=60.7,
        trip_score=60.7,
        t02_inverse=0.3279,
        t06_fn=1,
        danger_invalid=0,
        p95_ms=21.0,
    )
    full = _variant_report(
        macro_score=65.0,
        trip_score=65.0,
        t02_inverse=0.2,
        t06_fn=1,
        danger_invalid=0,
        p95_ms=20.0,
    )
    variants = {
        variant: _variant_report(
            macro_score=61.0,
            trip_score=61.0,
            t02_inverse=0.3,
            t06_fn=1,
            danger_invalid=0,
            p95_ms=20.0,
        )
        for variant in VARIANT_VALUES
    }
    variants.update(physics=physics, p2b_full=p2b, p2c_full_fusion=full)
    p2b["runtime"]["primary_danger_ttc_consistency_violation_count"] = 7  # type: ignore[index]

    legacy_report = build_acceptance(variants)

    assert legacy_report["primary_danger_ttc_consistent_p2c_rungs"] is True
    assert legacy_report["primary_danger_ttc_consistency_violations"]["p2b_full"] == 7

    variants["p2b_scale_expansion"]["runtime"][  # type: ignore[index]
        "primary_danger_ttc_consistency_violation_count"
    ] = 1
    blocked = build_acceptance(variants)
    assert blocked["primary_danger_ttc_consistent_p2c_rungs"] is False
    assert blocked["required_gate_pass"] is False


def test_runtime_summary_counts_lifecycle_jumps_and_infinite_uncertainty() -> None:
    rows = list(_rows())
    rows[1:] = [replace(row, primary_track_id=8) for row in rows[1:]]
    rows[1] = replace(
        rows[1],
        target_switched=True,
        estimator_uncertainty_s=float("inf"),
        ttc_jump=True,
    )

    report = summarize_runtime(rows)

    assert report["target_switch_count"] == 1
    assert report["ttc_jump_count"] == 1
    assert report["estimator_uncertainty_finite_count"] == 599
    assert report["estimator_uncertainty_infinite_count"] == 1


def test_incomplete_token_fails_before_late_evaluator_or_target_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    imported: list[str] = []

    def forbidden_import(name: str) -> object:
        imported.append(name)
        raise AssertionError(f"late module imported: {name}")

    monkeypatch.setattr("tools.c1_p2c_ablation.importlib.import_module", forbidden_import)
    incomplete = InferenceCompletion(
        variants=VARIANT_VALUES,
        trips=TRIP_IDS,
        prediction_rows=0,
        prediction_root=tmp_path,
        diagnostics_root=tmp_path,
        summaries={},
    )

    with pytest.raises(RuntimeError, match="21,600"):
        evaluate_predictions(incomplete, data_root=tmp_path)
    assert imported == []


def test_label_bearing_modules_are_not_top_level_imports() -> None:
    path = Path(__file__).resolve().parents[1] / "tools" / "c1_p2c_ablation.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert "team_kit.evaluation" not in imported
    assert "tools.c1_p2c_target_audit" not in imported
