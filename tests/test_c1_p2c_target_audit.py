from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from tools.c1_p2_oracle_diagnostics import KittiObject, OfflineFrame
from tools.c1_p2c_target_audit import (
    PrimaryDiagnostic,
    audit_trip,
    load_primary_diagnostics,
    map_primary_bbox,
    summarize_audits,
)


def _object(
    target_id: int, bbox: tuple[float, float, float, float]
) -> KittiObject:
    return KittiObject(
        target_id=target_id,
        kitti_label="Car",
        runtime_label="car",
        bbox=bbox,
        height_m=1.5,
        width_m=1.8,
        length_m=4.0,
        location_x_m=0.0,
        location_z_m=20.0,
        longitudinal_distance_m=20.0,
        lateral_distance_m=0.0,
        in_collision_cone=True,
    )


def _frame(
    frame_id: int,
    *,
    oracle_target_id: int | None,
    objects: tuple[KittiObject, ...],
) -> OfflineFrame:
    targets = ()
    if oracle_target_id is not None:
        targets = (
            {
                "target_id": oracle_target_id,
                "in_collision_cone": True,
                "ttc_2d": 1.0,
            },
        )
    return OfflineFrame(
        frame_id=frame_id,
        timestamp=frame_id * 0.05,
        ego_speed_kmh=30.0,
        ground_truth_ttc_s=1.0 if targets else float("inf"),
        targets=targets,
        objects=objects,
    )


def _diagnostic(
    frame_id: int,
    track_id: int | None,
    bbox: tuple[float, float, float, float] | None,
) -> PrimaryDiagnostic:
    return PrimaryDiagnostic("Trip-A", frame_id, track_id, bbox)


def test_primary_mapping_applies_iou_gate_and_ambiguity_margin() -> None:
    objects = (
        _object(1, (0.0, 0.0, 10.0, 10.0)),
        _object(2, (0.2, 0.0, 10.2, 10.0)),
    )

    ambiguous = map_primary_bbox((0.0, 0.0, 10.0, 10.0), objects)
    no_overlap = map_primary_bbox((50.0, 50.0, 60.0, 60.0), objects)
    clear = map_primary_bbox(
        (0.0, 0.0, 10.0, 10.0),
        (_object(1, (0.0, 0.0, 10.0, 10.0)), _object(2, (8.0, 0.0, 18.0, 10.0))),
    )

    assert ambiguous.status == "unknown_ambiguous"
    assert ambiguous.target_id is None
    assert no_overlap.status == "unknown_no_overlap"
    assert clear.status == "mapped"
    assert clear.target_id == 1
    assert clear.best_iou == pytest.approx(1.0)


def test_target_metrics_separate_mapping_coverage_from_selector_errors() -> None:
    first = _object(1, (0.0, 0.0, 10.0, 10.0))
    second = _object(2, (20.0, 0.0, 30.0, 10.0))
    frames = (
        _frame(0, oracle_target_id=1, objects=(first, second)),
        _frame(1, oracle_target_id=1, objects=(first, second)),
        _frame(2, oracle_target_id=1, objects=(first, second)),
        _frame(3, oracle_target_id=1, objects=(first, second)),
        _frame(4, oracle_target_id=None, objects=(first, second)),
        # The oracle target exists in privileged targets but is not projectable.
        _frame(5, oracle_target_id=1, objects=(second,)),
    )
    diagnostics = (
        _diagnostic(0, 10, first.bbox),       # TP
        _diagnostic(1, 11, second.bbox),      # wrong target: FP + FN
        _diagnostic(2, None, None),           # missing primary: FN
        _diagnostic(3, 12, (50.0, 0.0, 60.0, 10.0)),  # unknown: FN
        _diagnostic(4, 12, second.bbox),      # primary when oracle says none: FP
        _diagnostic(5, None, None),           # excluded: oracle projection absent
    )

    report = audit_trip("Trip-A", diagnostics, frames)

    assert report.oracle_target_frames == 5
    assert report.oracle_projected_frames == 4
    assert report.oracle_projection_coverage == pytest.approx(0.8)
    assert report.primary_present_frames == 4
    assert report.mapped_primary_frames == 3
    assert report.unknown_primary_frames == 1
    assert report.mapped_coverage == pytest.approx(0.75)
    assert report.unknown_coverage == pytest.approx(0.25)
    assert report.top1_evaluable_frames == 2
    assert report.top1_correct_frames == 1
    assert report.top1_target_agreement == pytest.approx(0.5)
    assert (report.target_tp, report.target_fp, report.target_fn) == (1, 2, 3)
    assert report.target_precision == pytest.approx(1.0 / 3.0)
    assert report.target_recall == pytest.approx(0.25)
    assert report.primary_track_switch_count == 1
    assert report.mapped_target_switch_count == 1


def test_macro_is_trip_balanced_and_switch_counts_are_summed() -> None:
    obj = _object(1, (0.0, 0.0, 10.0, 10.0))
    frames = (_frame(0, oracle_target_id=1, objects=(obj,)),)
    first = audit_trip("Trip-A", (_diagnostic(0, 1, obj.bbox),), frames)
    second = replace(
        first,
        trip_id="Trip-B",
        top1_target_agreement=0.0,
        target_precision=0.0,
        target_recall=0.0,
        mapped_coverage=0.5,
        unknown_coverage=0.5,
        primary_track_switch_count=3,
        mapped_target_switch_count=2,
    )

    report = summarize_audits((first, second))
    macro = report["macro"]

    assert macro["top1_target_agreement"] == pytest.approx(0.5)
    assert macro["target_precision"] == pytest.approx(0.5)
    assert macro["target_recall"] == pytest.approx(0.5)
    assert macro["mapped_coverage"] == pytest.approx(0.75)
    assert macro["primary_track_switch_count_total"] == 3
    assert macro["mapped_target_switch_count_total"] == 2


def test_diagnostic_loader_requires_primary_bbox_and_unique_frames(
    tmp_path: Path,
) -> None:
    path = tmp_path / "Trip-A.csv"
    fieldnames = (
        "trip_id",
        "frame_id",
        "primary_track_id",
        "primary_bbox_x1",
        "primary_bbox_y1",
        "primary_bbox_x2",
        "primary_bbox_y2",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "trip_id": "Trip-A",
                "frame_id": 0,
                "primary_track_id": 7,
                "primary_bbox_x1": 1,
                "primary_bbox_y1": 2,
                "primary_bbox_x2": 11,
                "primary_bbox_y2": 22,
            }
        )

    rows = load_primary_diagnostics(path)

    assert rows == (
        PrimaryDiagnostic("Trip-A", 0, 7, (1.0, 2.0, 11.0, 22.0)),
    )

    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writerow(
            {
                "trip_id": "Trip-A",
                "frame_id": 0,
                "primary_track_id": "",
                "primary_bbox_x1": "",
                "primary_bbox_y1": "",
                "primary_bbox_x2": "",
                "primary_bbox_y2": "",
            }
        )
    with pytest.raises(ValueError, match="duplicate frame_id"):
        load_primary_diagnostics(path)


def test_missing_bbox_for_retained_primary_is_unknown_coverage_not_false_positive(
    tmp_path: Path,
) -> None:
    obj = _object(1, (0.0, 0.0, 10.0, 10.0))
    frame = _frame(0, oracle_target_id=1, objects=(obj,))
    path = tmp_path / "Trip-A.csv"
    fieldnames = (
        "trip_id",
        "frame_id",
        "primary_track_id",
        "primary_bbox_x1",
        "primary_bbox_y1",
        "primary_bbox_x2",
        "primary_bbox_y2",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "trip_id": "Trip-A",
                "frame_id": 0,
                "primary_track_id": 7,
                "primary_bbox_x1": "",
                "primary_bbox_y1": "",
                "primary_bbox_x2": "",
                "primary_bbox_y2": "",
            }
        )

    diagnostics = load_primary_diagnostics(path)
    report = audit_trip("Trip-A", diagnostics, (frame,))

    assert diagnostics[0].primary_bbox is None
    assert report.primary_present_frames == 1
    assert report.mapped_primary_frames == 0
    assert report.unknown_primary_frames == 1
    assert report.unknown_missing_bbox_frames == 1
    assert report.mapped_coverage == 0.0
    assert report.unknown_coverage == 1.0
    assert (report.target_tp, report.target_fp, report.target_fn) == (0, 0, 1)


def test_target_audit_stays_outside_c1_runtime_imports() -> None:
    inference_root = Path(__file__).resolve().parents[1] / "safeloop" / "c1"
    offenders = [
        path.name
        for path in inference_root.glob("*.py")
        if "c1_p2c_target_audit" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
