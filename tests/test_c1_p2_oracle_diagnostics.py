from __future__ import annotations

import math
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from safeloop.c1.temporal_features import CameraGeometry
from safeloop.c1.types import Detection
from tools.c1_p2_oracle_diagnostics import (
    CausalOracleRange,
    KittiObject,
    OfflineFrame,
    OracleAssociationTracker,
    ReplayRow,
    _match_label_target,
    contiguous_segments,
    match_detections_to_objects,
    parse_kitti_label,
    primary_failure_cause,
    project_kitti_cuboid,
    write_prediction_csv,
)


def _object(target_id: int, bbox: tuple[float, float, float, float]) -> KittiObject:
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


def _row(frame_id: int, **updates: object) -> ReplayRow:
    integers = {
        "frame_id",
        "target_detection_count",
        "projected_object_count",
        "matched_object_count_iou_005",
        "matched_object_count_iou_010",
        "matched_object_count_iou_020",
        "projection_ambiguous_object_count",
    }
    identifiers = {
        "oracle_target_id",
        "selected_track_id",
        "selected_target_id",
        "target_track_id",
        "target_track_missed_updates",
    }
    text = {"matched_class_pairs"}
    booleans = {
        field.name
        for field in fields(ReplayRow)
        if field.name not in integers | identifiers | text
        and not field.name.endswith("_s")
        and field.name != "selected_range_m"
    }
    values: dict[str, object] = {}
    for field in fields(ReplayRow):
        if field.name in integers:
            values[field.name] = 0
        elif field.name in identifiers:
            values[field.name] = None
        elif field.name in text:
            values[field.name] = ""
        elif field.name in booleans:
            values[field.name] = False
        else:
            values[field.name] = float("inf")
    values.update(
        frame_id=frame_id,
        timestamp=frame_id * 0.05,
        ground_truth_ttc_s=float("inf"),
        baseline_ttc_s=float("inf"),
    )
    values.update(updates)
    return ReplayRow(**values)  # type: ignore[arg-type]


def test_project_kitti_cuboid_uses_3d_fields_when_2d_box_is_zero() -> None:
    line = "Car 0.00 0 0.00 0.00 0.00 0.00 0.00 1.50 1.80 4.00 0.00 1.50 20.00 0.00"
    projection = np.asarray(
        [[320.0, 0.0, 320.0, 0.0], [0.0, 320.0, 180.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )

    label, bbox = project_kitti_cuboid(
        line, projection, image_width=640, image_height=360
    ) or (None, None)

    assert label == "Car"
    assert bbox is not None
    assert bbox[0] < 320.0 < bbox[2]
    assert bbox[1] == pytest.approx(180.0)
    assert bbox[3] > bbox[1]
    assert parse_kitti_label(line)[1:4] == (1.5, 1.8, 4.0)


def test_label_target_join_requires_class_and_unambiguous_coordinates() -> None:
    targets = (
        {
            "target_id": 10,
            "target_class": "vehicle",
            "lateral_distance": 0.004,
            "longitudinal_distance": 20.003,
        },
        {
            "target_id": 20,
            "target_class": "walker",
            "lateral_distance": 0.0,
            "longitudinal_distance": 20.0,
        },
    )
    matched = _match_label_target(
        kitti_label="Car", location_x_m=0.0, location_z_m=20.0, targets=targets
    )
    assert matched["target_id"] == 10

    ambiguous = (
        targets[0],
        {
            "target_id": 11,
            "target_class": "vehicle",
            "lateral_distance": 0.01,
            "longitudinal_distance": 20.0,
        },
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        _match_label_target(
            kitti_label="Car", location_x_m=0.0, location_z_m=20.0, targets=ambiguous
        )


def test_detection_object_assignment_is_one_to_one_and_iou_gated() -> None:
    detections = (
        Detection(2, "car", 0.9, (0.0, 0.0, 10.0, 10.0)),
        Detection(2, "car", 0.8, (20.0, 0.0, 30.0, 10.0)),
    )
    objects = (_object(1, (0.0, 0.0, 10.0, 10.0)), _object(2, (20.0, 0.0, 30.0, 10.0)))

    assignments = match_detections_to_objects(detections, objects, min_iou=0.10)

    assert {(item.detection_index, item.object_index) for item in assignments} == {(0, 0), (1, 1)}
    assert all(item.iou == pytest.approx(1.0) for item in assignments)


def test_oracle_association_forces_same_live_track_and_keeps_false_positive() -> None:
    geometry = CameraGeometry(640, 360, 320.0, 320.0, 320.0)
    tracker = OracleAssociationTracker(geometry)
    common = {"image_shape": (360, 640), "ego_speed_kmh": 30.0}
    first = Detection(2, "car", 0.9, (100.0, 100.0, 150.0, 180.0))
    false_positive = Detection(0, "person", 0.8, (500.0, 100.0, 520.0, 180.0))
    tracker.update_with_identities(
        (first, false_positive), (7, None), timestamp=0.0, **common
    )
    target_track = next(track_id for track_id, target in tracker._target_by_track.items() if target == 7)

    # This jump is beyond the normal greedy gate, but the target identity is oracle-known.
    jumped = Detection(3, "motorcycle", 0.9, (350.0, 100.0, 420.0, 200.0))
    tracker.update_with_identities((jumped,), (7,), timestamp=0.15, **common)

    assert tracker._target_by_track[target_track] == 7
    assert next(track for track in tracker._tracks if track.track_id == target_track).bbox == jumped.bbox
    assert any(track.track_id != target_track for track in tracker._tracks)


def test_causal_range_uses_only_past_and_current_longitudinal_range() -> None:
    oracle = CausalOracleRange(history_size=4, min_history=3)
    predictions = []
    for frame_id, distance in enumerate((20.0, 19.5, 19.0)):
        target = {
            "target_id": 7,
            "target_class": "vehicle",
            "longitudinal_distance": distance,
            # Poisoned forbidden fields must not affect the result.
            "closing_speed": -9999.0,
            "rel_velocity": {"x": 9999.0},
            "ttc_simple": 0.01,
            "ttc_2d": 0.01,
        }
        frame = OfflineFrame(
            frame_id=frame_id,
            timestamp=frame_id * 0.1,
            ego_speed_kmh=0.0,
            ground_truth_ttc_s=0.01,
            targets=(target,),
            objects=(_object(7, (0.0, 0.0, 10.0, 10.0)),),
        )
        predictions.append(oracle.update(frame, 7))

    assert math.isinf(predictions[0])
    assert math.isinf(predictions[1])
    # Closing from ranges is 5 m/s; contact gap is 19 - (4.5+4.0)/2.
    assert predictions[2] == pytest.approx((19.0 - 4.25) / 5.0)


def test_segments_do_not_bridge_false_frame_and_primary_cause_is_outcome_gated() -> None:
    quiet_event = _row(0, detector_missed_object=True)
    error_event = replace(
        quiet_event,
        frame_id=1,
        timestamp=0.05,
        ground_truth_ttc_s=1.0,
        gt_critical_prediction_not_critical=True,
        gt_mae_critical_prediction_not_critical=True,
    )
    gap = _row(2)
    second_error = replace(error_event, frame_id=3, timestamp=0.15)

    segments = contiguous_segments(
        "T01-Sample", (quiet_event, error_event, gap, second_error), "detector_missed_object"
    )

    assert [(item["start_frame"], item["end_frame"]) for item in segments] == [(0, 1), (3, 3)]
    assert primary_failure_cause(quiet_event) == ""
    assert primary_failure_cause(error_event) == "detector_missed_object"


def test_prediction_writer_enforces_complete_600_frame_output(tmp_path: Path) -> None:
    rows = tuple(_row(frame_id) for frame_id in range(600))
    path = tmp_path / "T01-Sample.csv"

    report = write_prediction_csv(path, rows, [float("inf")] * 600)

    assert report["rows"] == 600
    assert report["infinite_ttc"] == 600


def test_offline_tool_is_not_imported_by_c1_inference_modules() -> None:
    inference_root = Path(__file__).resolve().parents[1] / "safeloop" / "c1"
    offenders = [
        path.name
        for path in inference_root.glob("*.py")
        if "c1_p2_oracle_diagnostics" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
