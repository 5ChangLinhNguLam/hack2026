from __future__ import annotations

import math

import numpy as np
import pytest

from safeloop.c1.p2b_tracker import (
    P2AssociationConfig,
    P2CausalTracker,
    P2RangePolicyConfig,
    _class_penalty,
    _hungarian,
)
from safeloop.c1.range_ttc import RangeTTCReason, RangeTTCResult
from safeloop.c1.temporal_features import CameraGeometry, baseline_tracker
from safeloop.c1.types import Detection


GEOMETRY = CameraGeometry(640, 360, 320.0, 320.0, 320.0)
SHAPE = (360, 640)


def _detection(label: str, box: tuple[float, float, float, float]) -> Detection:
    class_id = {
        "person": 0,
        "bicycle": 1,
        "car": 2,
        "motorcycle": 3,
        "bus": 5,
        "truck": 7,
    }[label]
    return Detection(class_id, label, 0.85, box)


def _tracker(
    *,
    association: bool = False,
    kalman: bool = True,
    robust: bool = False,
    fallback: bool = False,
):
    frozen = baseline_tracker(GEOMETRY)
    return P2CausalTracker(
        frozen.config,
        association=P2AssociationConfig(
            enabled=association,
            enable_kalman=kalman,
        ),
        range_policy=P2RangePolicyConfig(
            enabled=robust,
            enable_last_good_fallback=fallback,
        ),
        focal_y_px=GEOMETRY.focal_y_px,
        focal_x_px=GEOMETRY.focal_x_px,
        principal_x_px=GEOMETRY.principal_x_px,
    )


def _risk_signature(risks):
    return [
        (
            item.track_id,
            item.label,
            item.bbox,
            item.predicted_ttc_s,
            item.scale_ttc_s,
            item.range_ttc_s,
            item.lateral_ttc_s,
            item.estimated_distance_m,
        )
        for item in risks
    ]


def _seed_closing_range(tracker: P2CausalTracker) -> float:
    timestamps = (0.0, 0.11, 0.27, 0.46, 0.68, 0.93)
    for index, timestamp in enumerate(timestamps):
        height = 42.0 + 5.0 * index
        tracker.update(
            (_detection("car", (280.0, 250.0 - height, 360.0, 250.0)),),
            timestamp=timestamp,
            image_shape=SHAPE,
        )
    return timestamps[-1]


def _synthetic_range_result(
    *,
    timestamp: float,
    reason: str,
    ttc_s: float,
) -> RangeTTCResult:
    return RangeTTCResult(
        evaluation_timestamp_s=timestamp,
        range_m=20.0,
        closing_speed_mps=(2.0 if math.isfinite(ttc_s) else -1.0),
        ttc_s=ttc_s,
        range_uncertainty_m=0.1,
        closing_speed_uncertainty_mps=0.1,
        ttc_uncertainty_s=(0.2 if math.isfinite(ttc_s) else float("inf")),
        reason_code=reason,
        sample_count=6,
        inlier_count=6,
        outlier_count=0,
        last_observation_age_s=0.0,
    )


def test_disabled_extension_is_exact_physics_tracker_on_updates_and_stride_frames():
    frozen = baseline_tracker(GEOMETRY)
    extended = _tracker()
    for index in range(12):
        timestamp = index * 0.05
        if index % 3 == 0:
            box = (285.0 - index, 140.0, 355.0 + index, 215.0 + 2 * index)
            detections = (_detection("car", box),)
            expected = frozen.update(
                detections,
                timestamp=timestamp,
                image_shape=SHAPE,
                ego_speed_kmh=32.0,
            )
            actual = extended.update(
                detections,
                timestamp=timestamp,
                image_shape=SHAPE,
                ego_speed_kmh=32.0,
            )
        else:
            expected = frozen.predict(
                timestamp=timestamp, image_shape=SHAPE, ego_speed_kmh=32.0
            )
            actual = extended.predict(
                timestamp=timestamp, image_shape=SHAPE, ego_speed_kmh=32.0
            )
        assert _risk_signature(actual) == _risk_signature(expected)


def test_single_vehicle_class_flip_does_not_replace_stable_label_or_track():
    tracker = _tracker(association=True)
    first = tracker.update(
        (_detection("car", (260.0, 130.0, 340.0, 230.0)),),
        timestamp=0.0,
        image_shape=SHAPE,
    )
    second = tracker.update(
        (_detection("truck", (262.0, 131.0, 342.0, 231.0)),),
        timestamp=0.15,
        image_shape=SHAPE,
    )
    snapshot = tracker.candidate_states()[0]
    assert first[0].track_id == second[0].track_id == snapshot.track_id
    assert snapshot.label == "car"
    assert snapshot.class_posterior["truck"] > 0.0


def test_vru_transition_is_allowed_but_penalised_more_than_two_wheeler_flip():
    assert _class_penalty("bicycle", "motorcycle") < _class_penalty(
        "person", "motorcycle"
    )
    assert _class_penalty("person", "motorcycle") < _class_penalty(
        "person", "car"
    )
    tracker = _tracker(association=True)
    tracker.update(
        (_detection("person", (100.0, 120.0, 145.0, 245.0)),),
        timestamp=0.0,
        image_shape=SHAPE,
    )
    tracker.update(
        (_detection("motorcycle", (104.0, 121.0, 151.0, 246.0)),),
        timestamp=0.15,
        image_shape=SHAPE,
    )
    assert len(tracker.candidate_states()) == 1
    assert tracker.candidate_states()[0].track_id == 1


def test_incompatible_raw_class_flip_cannot_create_false_scale_ttc() -> None:
    tracker = _tracker(association=True)
    tracker.update(
        (_detection("car", (280.0, 180.0, 340.0, 240.0)),),
        timestamp=0.0,
        image_shape=SHAPE,
    )
    tracker.update(
        (_detection("person", (275.0, 170.0, 345.0, 250.0)),),
        timestamp=0.15,
        image_shape=SHAPE,
    )

    snapshot = tracker.candidate_states()[0]

    assert snapshot.label == "car"
    assert tracker._tracks[0].history[-1].label == "person"
    assert not math.isfinite(snapshot.scale_ttc_s)


def test_low_confidence_incompatible_flip_preserves_causal_scale_history() -> None:
    tracker = _tracker(association=True)
    observations = (
        (0.00, "person", 0.85, (285.0, 180.0, 345.0, 240.0)),
        (0.15, "person", 0.85, (280.0, 175.0, 350.0, 245.0)),
        (0.30, "car", 0.29, (275.0, 170.0, 355.0, 250.0)),
        (0.45, "person", 0.85, (270.0, 165.0, 360.0, 255.0)),
    )
    for timestamp, label, confidence, box in observations:
        class_id = 0 if label == "person" else 2
        tracker.update(
            (Detection(class_id, label, confidence, box),),
            timestamp=timestamp,
            image_shape=SHAPE,
        )

    snapshot = tracker.candidate_states()[0]
    meta = tracker._meta[snapshot.track_id]

    assert snapshot.label == "person"
    assert [item.label for item in tracker._tracks[0].history] == ["person"] * 4
    assert list(meta.raw_labels) == ["person", "person", "car", "person"]
    assert snapshot.class_posterior["car"] > 0.0
    assert math.isfinite(snapshot.scale_ttc_s)


def test_hungarian_assignment_is_global_not_row_greedy():
    # Row 0's cheapest column is also row 1's only good choice.  A row-greedy
    # pass costs 1.01; the global solution costs 0.04.
    pairs = _hungarian(
        np.asarray([[0.01, 0.02], [0.02, 1.00]], dtype=float)
    )
    assert sorted(pairs) == [(0, 1), (1, 0)]


def test_kalman_tracks_irregular_motion_and_log_size_with_pure_projection() -> None:
    tracker = _tracker(association=True)

    def moving_box(timestamp: float) -> tuple[float, float, float, float]:
        center_x = 120.0 + 45.0 * timestamp
        center_y = 150.0 + 12.0 * timestamp
        width = 45.0 * math.exp(0.35 * timestamp)
        height = 65.0 * math.exp(0.25 * timestamp)
        return (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        )

    for timestamp in (0.0, 0.11, 0.29, 0.52):
        tracker.update(
            (_detection("car", moving_box(timestamp)),),
            timestamp=timestamp,
            image_shape=SHAPE,
        )

    track = tracker._tracks[0]
    motion_filter = tracker._meta[track.track_id].motion_filter
    assert motion_filter is not None
    state_before = motion_filter.state.copy()
    covariance_before = motion_filter.covariance.copy()
    first_projection = tracker._predicted_bbox(track, 0.72)
    second_projection = tracker._predicted_bbox(track, 0.72)
    projected_center_x = 0.5 * (first_projection[0] + first_projection[2])
    projected_width = first_projection[2] - first_projection[0]
    measured_width = track.bbox[2] - track.bbox[0]

    assert first_projection == pytest.approx(second_projection)
    assert projected_center_x == pytest.approx(120.0 + 45.0 * 0.72, abs=2.0)
    assert projected_width > measured_width
    assert np.array_equal(motion_filter.state, state_before)
    assert np.array_equal(motion_filter.covariance, covariance_before)

    snapshot = tracker.candidate_states()[0]
    assert snapshot.track_id == 1
    assert snapshot.foot_velocity_x_px_s == pytest.approx(45.0, abs=3.0)
    # Bottom-foot velocity includes centre-y translation and height growth.
    assert snapshot.foot_velocity_y_px_s > 12.0


def test_kalman_coast_increases_uncertainty_without_moving_measurement_bbox() -> None:
    tracker = _tracker(association=True)
    for timestamp, center_x in ((0.0, 160.0), (0.12, 168.0), (0.31, 181.0)):
        tracker.update(
            (
                _detection(
                    "car",
                    (center_x - 30.0, 150.0, center_x + 30.0, 230.0),
                ),
            ),
            timestamp=timestamp,
            image_shape=SHAPE,
        )

    track = tracker._tracks[0]
    motion_filter = tracker._meta[track.track_id].motion_filter
    assert motion_filter is not None
    measurement_bbox = track.bbox
    trace_before = float(np.trace(motion_filter.covariance))
    before = tracker.candidate_states()[0]
    confidence_before = before.association_confidence
    motion_confidence_before = before.motion_state_confidence
    velocity_uncertainty_before = before.foot_velocity_uncertainty_px_s
    assert motion_confidence_before is not None
    assert velocity_uncertainty_before is not None

    tracker.predict(timestamp=0.41, image_shape=SHAPE)
    tracker.predict(timestamp=0.51, image_shape=SHAPE)

    snapshot = tracker.candidate_states()[0]
    trace_after = float(np.trace(motion_filter.covariance))
    assert track.bbox == measurement_bbox
    assert len(track.history) == 3
    assert track.missed == 0
    assert trace_after > trace_before
    # Scheduled detector-stride prediction grows state uncertainty, but it is
    # not an association failure and must not relabel association evidence.
    assert snapshot.association_confidence == pytest.approx(confidence_before)
    assert snapshot.motion_state_confidence is not None
    assert snapshot.motion_state_confidence < motion_confidence_before
    assert snapshot.foot_velocity_uncertainty_px_s is not None
    assert (
        snapshot.foot_velocity_uncertainty_px_s
        > velocity_uncertainty_before
    )
    assert snapshot.observed_this_call is False

    tracker.update((), timestamp=0.61, image_shape=SHAPE)
    missed = tracker.candidate_states()[0]
    assert missed.missed_updates == 1
    assert missed.association_confidence < confidence_before


def test_kalman_prediction_recovers_same_track_after_irregular_stride_gap() -> None:
    tracker = _tracker(association=True)
    tracker.update(
        (_detection("car", (80.0, 150.0, 120.0, 220.0)),),
        timestamp=0.0,
        image_shape=SHAPE,
    )
    tracker.update(
        (_detection("car", (100.0, 150.0, 140.0, 220.0)),),
        timestamp=0.20,
        image_shape=SHAPE,
    )
    tracker.predict(timestamp=0.31, image_shape=SHAPE)
    risks = tracker.update(
        (_detection("car", (125.0, 150.0, 165.0, 220.0)),),
        timestamp=0.45,
        image_shape=SHAPE,
    )

    assert len(tracker.candidate_states()) == 1
    assert risks[0].track_id == tracker.candidate_states()[0].track_id == 1
    assert tracker._tracks[0].bbox == (125.0, 150.0, 165.0, 220.0)
    assert tracker._meta[1].motion_filter is not None
    assert tracker._meta[1].motion_filter.timestamp == pytest.approx(0.45)


def test_kalman_covariance_is_independent_of_predict_callback_density() -> None:
    dense = _tracker(association=True)
    sparse = _tracker(association=True)
    detection = (_detection("car", (100.0, 140.0, 150.0, 220.0)),)
    dense.update(detection, timestamp=0.0, image_shape=SHAPE)
    sparse.update(detection, timestamp=0.0, image_shape=SHAPE)

    for timestamp in (0.07, 0.18, 0.30):
        dense.predict(timestamp=timestamp, image_shape=SHAPE)
    sparse.predict(timestamp=0.30, image_shape=SHAPE)

    dense_filter = dense._meta[1].motion_filter
    sparse_filter = sparse._meta[1].motion_filter
    assert dense_filter is not None and sparse_filter is not None
    assert dense_filter.state == pytest.approx(sparse_filter.state, abs=1e-12)
    assert dense_filter.covariance == pytest.approx(
        sparse_filter.covariance, abs=1e-10
    )
    assert dense.candidate_states()[0].foot_velocity_uncertainty_px_s == pytest.approx(
        sparse.candidate_states()[0].foot_velocity_uncertainty_px_s,
        abs=1e-10,
    )
    assert dense.candidate_states()[0].motion_state_confidence == pytest.approx(
        sparse.candidate_states()[0].motion_state_confidence,
        abs=1e-10,
    )


def test_kalman_state_uncertainty_does_not_relabel_association_evidence() -> None:
    kalman = _tracker(association=True, kalman=True)
    class_only = _tracker(association=True, kalman=False)
    detection = (_detection("car", (100.0, 140.0, 150.0, 220.0)),)
    for tracker in (kalman, class_only):
        tracker.update(detection, timestamp=0.0, image_shape=SHAPE)
        tracker.update((), timestamp=0.15, image_shape=SHAPE)

    filtered = kalman.candidate_states()[0]
    unfiltered = class_only.candidate_states()[0]
    assert filtered.missed_updates == unfiltered.missed_updates == 1
    assert filtered.association_confidence == pytest.approx(
        unfiltered.association_confidence
    )
    assert filtered.association_confidence == pytest.approx(0.55 * math.exp(-0.30))
    assert filtered.motion_state_confidence is not None
    assert unfiltered.motion_state_confidence is None


def test_kalman_is_toggleable_single_hit_is_stationary_and_reset_cleans_state() -> None:
    class_only = _tracker(association=True, kalman=False)
    class_only.update(
        (_detection("car", (100.0, 140.0, 150.0, 220.0)),),
        timestamp=0.0,
        image_shape=SHAPE,
    )
    assert class_only._meta[1].motion_filter is None

    tracker = _tracker(association=True)
    tracker.update(
        (_detection("car", (100.0, 140.0, 150.0, 220.0)),),
        timestamp=0.0,
        image_shape=SHAPE,
    )
    snapshot = tracker.candidate_states()[0]
    assert snapshot.foot_velocity_x_px_s == pytest.approx(0.0)
    assert snapshot.foot_velocity_y_px_s == pytest.approx(0.0)
    assert tracker._meta[1].motion_filter is not None

    tracker.reset()

    assert tracker._meta == {}
    assert tracker._tracks == []
    tracker.update(
        (_detection("person", (200.0, 120.0, 240.0, 230.0)),),
        timestamp=1.0,
        image_shape=SHAPE,
    )
    assert tracker.candidate_states()[0].track_id == 1
    assert tracker._meta[1].motion_filter is not None


def test_candidate_states_include_raw_physics_before_legacy_corridor_gate():
    tracker = _tracker()
    for index in range(5):
        tracker.update(
            (_detection("car", (20.0, 120.0, 70.0 + 8 * index, 210.0 + 8 * index)),),
            timestamp=index * 0.15,
            image_shape=SHAPE,
            ego_speed_kmh=25.0,
        )
    snapshot = tracker.candidate_states()[0]
    # The box is far off the old centre corridor, so final TrackRisk can be
    # gated to infinity while its causal scale/range evidence remains exposed.
    risk = tracker.predict(timestamp=0.65, image_shape=SHAPE, ego_speed_kmh=25.0)[0]
    assert not math.isfinite(risk.predicted_ttc_s)
    assert math.isfinite(snapshot.raw_physics_ttc_s)


def test_robust_range_uses_detector_observations_and_reports_uncertainty():
    tracker = _tracker(robust=True)
    _seed_closing_range(tracker)
    snapshot = tracker.candidate_states()[0]
    assert math.isfinite(snapshot.robust_range_ttc_s)
    assert snapshot.closing_speed_mps > 0.0
    assert math.isfinite(snapshot.range_uncertainty_m)
    assert snapshot.reason_code in {"ok", "jump_limited"}


def test_predict_does_not_add_range_observation():
    tracker = _tracker(robust=True)
    tracker.update(
        (_detection("car", (280.0, 180.0, 360.0, 250.0)),),
        timestamp=0.0,
        image_shape=SHAPE,
    )
    history_before = len(tracker._meta[1].range_estimator.history)  # type: ignore[union-attr]
    tracker.predict(timestamp=0.05, image_shape=SHAPE)
    tracker.predict(timestamp=0.10, image_shape=SHAPE)
    history_after = len(tracker._meta[1].range_estimator.history)  # type: ignore[union-attr]
    assert history_after == history_before == 1


def test_detector_miss_advances_range_fit_without_appending_observation():
    tracker = _tracker(robust=True, fallback=True)
    last_timestamp = _seed_closing_range(tracker)
    meta = tracker._meta[1]
    assert meta.range_estimator is not None
    history_before = len(meta.range_estimator.history)
    ttc_before = meta.robust_ttc_s
    anchor_before = meta.last_good_timestamp

    tracker.update((), timestamp=last_timestamp + 0.10, image_shape=SHAPE)

    assert len(meta.range_estimator.history) == history_before
    assert meta.robust_result is not None
    assert meta.robust_result.evaluation_timestamp_s == pytest.approx(
        last_timestamp + 0.10
    )
    assert meta.robust_ttc_s == pytest.approx(ttc_before - 0.10, abs=1e-6)
    assert meta.last_good_timestamp == anchor_before


def test_stride_density_does_not_refresh_jump_or_fallback_anchor():
    dense = _tracker(robust=True, fallback=True)
    sparse = _tracker(robust=True, fallback=True)
    last_timestamp = _seed_closing_range(dense)
    _seed_closing_range(sparse)
    dense_anchor = dense._meta[1].last_good_timestamp
    sparse_anchor = sparse._meta[1].last_good_timestamp

    for timestamp in (0.98, 1.03, 1.08, 1.13, 1.18, 1.20):
        dense.predict(timestamp=timestamp, image_shape=SHAPE)
    sparse.predict(timestamp=1.20, image_shape=SHAPE)

    assert dense._meta[1].last_good_timestamp == dense_anchor == last_timestamp
    assert sparse._meta[1].last_good_timestamp == sparse_anchor == last_timestamp
    assert dense._meta[1].robust_ttc_s == pytest.approx(
        sparse._meta[1].robust_ttc_s, abs=1e-9
    )


def test_tracker_owns_counted_down_hold_and_expires_from_last_real_anchor():
    tracker = _tracker(robust=True, fallback=True)
    last_timestamp = _seed_closing_range(tracker)
    anchor_ttc = tracker._meta[1].last_good_ttc_s

    tracker.predict(timestamp=last_timestamp + 0.42, image_shape=SHAPE)
    held = tracker.candidate_states()[0]

    assert held.reason_code == "held_previous"
    assert held.robust_range_ttc_s == pytest.approx(anchor_ttc - 0.42, abs=1e-6)
    assert held.last_good_ttc_s == pytest.approx(held.robust_range_ttc_s)

    tracker.predict(timestamp=last_timestamp + 0.46, image_shape=SHAPE)
    expired = tracker.candidate_states()[0]

    assert expired.reason_code == RangeTTCReason.STALE_HISTORY
    assert math.isinf(expired.robust_range_ttc_s)
    assert math.isinf(expired.last_good_ttc_s)


def test_fallback_never_holds_nonclosing_or_coherent_beyond_monitor_result():
    tracker = _tracker(robust=True, fallback=True)
    last_timestamp = _seed_closing_range(tracker)
    meta = tracker._meta[1]
    anchor_timestamp = meta.last_good_timestamp

    nonclosing = _synthetic_range_result(
        timestamp=last_timestamp + 0.10,
        reason=RangeTTCReason.NON_CLOSING,
        ttc_s=float("inf"),
    )
    nonclosing_ttc, nonclosing_reason = tracker._bounded_robust_ttc(
        meta,
        nonclosing,
        timestamp=last_timestamp + 0.10,
        refresh_last_good=True,
    )

    assert math.isinf(nonclosing_ttc)
    assert nonclosing_reason == RangeTTCReason.NON_CLOSING
    assert meta.last_good_timestamp == anchor_timestamp

    beyond_monitor = _synthetic_range_result(
        timestamp=last_timestamp + 0.15,
        reason=RangeTTCReason.OK,
        ttc_s=tracker.range_policy.max_ttc_s + 1.0,
    )
    safe_ttc, safe_reason = tracker._bounded_robust_ttc(
        meta,
        beyond_monitor,
        timestamp=last_timestamp + 0.15,
        refresh_last_good=True,
    )

    assert math.isinf(safe_ttc)
    assert safe_reason == RangeTTCReason.OK
    assert meta.last_good_timestamp == anchor_timestamp


def test_robust_ttc_never_delays_danger_but_limits_upward_jump() -> None:
    tracker = _tracker(robust=True, fallback=True)
    tracker.update(
        (_detection("car", (280.0, 180.0, 360.0, 250.0)),),
        timestamp=0.0,
        image_shape=SHAPE,
    )
    meta = tracker._meta[1]
    meta.last_good_ttc_s = 5.0
    meta.last_good_timestamp = 0.0

    cut_in = _synthetic_range_result(
        timestamp=0.1,
        reason=RangeTTCReason.OK,
        ttc_s=0.6,
    )
    cut_in_ttc, cut_in_reason = tracker._bounded_robust_ttc(
        meta,
        cut_in,
        timestamp=0.1,
        refresh_last_good=True,
    )

    assert cut_in_ttc == pytest.approx(0.6)
    assert cut_in_reason == RangeTTCReason.OK

    meta.last_good_ttc_s = 1.0
    meta.last_good_timestamp = 0.1
    upward = _synthetic_range_result(
        timestamp=0.2,
        reason=RangeTTCReason.OK,
        ttc_s=5.0,
    )
    upward_ttc, upward_reason = tracker._bounded_robust_ttc(
        meta,
        upward,
        timestamp=0.2,
        refresh_last_good=True,
    )

    assert upward_ttc == pytest.approx(
        0.9 + tracker.range_policy.jump_residual_limit_s
    )
    assert upward_reason == "jump_limited"

def test_tracker_rejects_nonfinite_or_nonincreasing_runtime_timestamps():
    tracker = _tracker(robust=True)
    tracker.update(
        (_detection("car", (280.0, 180.0, 360.0, 250.0)),),
        timestamp=1.0,
        image_shape=SHAPE,
    )

    with pytest.raises(ValueError, match="increase strictly"):
        tracker.predict(timestamp=1.0, image_shape=SHAPE)
    with pytest.raises(ValueError, match="finite"):
        tracker.predict(timestamp=float("nan"), image_shape=SHAPE)

    tracker.reset()
    tracker.update(
        (_detection("car", (280.0, 180.0, 360.0, 250.0)),),
        timestamp=0.0,
        image_shape=SHAPE,
    )


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        P2AssociationConfig(posterior_decay=1.0)
    with pytest.raises(ValueError):
        P2AssociationConfig(min_incompatible_physics_confidence=1.1)
    with pytest.raises(ValueError):
        P2RangePolicyConfig(jump_residual_limit_s=0.0)
