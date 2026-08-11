from __future__ import annotations

import math

import pytest

from safeloop.c1.lane import LaneEstimate
from safeloop.c1.p2c_range import (
    CausalScaleExpansionEstimator,
    CausalTTCFusion,
    GroundPlaneConfig,
    P2CReason,
    ScaleExpansionConfig,
    ScaleObservation,
    TTCFusionConfig,
    TTCFusionInput,
    causal_lane_horizon_y,
    estimate_causal_scale_expansion_ttc,
    estimate_class_prior_range,
    estimate_ground_plane_range,
    fuse_ttc_estimates,
    project_scale_expansion_result,
)


IMAGE_SHAPE = (360, 640)
FOCAL_Y = 320.0
PRINCIPAL_Y = 180.0


def _bbox(*, height: float, width: float = 40.0, bottom: float = 300.0):
    return (300.0, bottom - height, 300.0 + width, bottom)


def _lane(*, confidence: float = 0.90, inferred_side: str | None = None) -> LaneEstimate:
    # Two boundaries intersect at exactly (320, 160).
    rows = (160, 200, 240, 280, 320, 359)
    left = tuple((round(320.0 - 1.1 * (y - 160.0)), y) for y in rows)
    right = tuple((round(320.0 + 1.1 * (y - 160.0)), y) for y in rows)
    return LaneEstimate(
        valid=True,
        confidence=confidence,
        left_points=left,
        right_points=right,
        lane_center_offset=0.0,
        heading_error_deg=0.0,
        lane_width_px=438.0,
        departure_warning=False,
        inferred_side=inferred_side,
    )


def _scale_observations(
    timestamps: list[float],
    *,
    inverse_ttc_s: float = 0.25,
) -> list[ScaleObservation]:
    observations: list[ScaleObservation] = []
    for timestamp in timestamps:
        growth = math.exp(inverse_ttc_s * timestamp)
        observations.append(
            ScaleObservation(
                timestamp_s=timestamp,
                bbox=_bbox(height=24.0 * growth, width=42.0 * growth),
                confidence=0.85,
            )
        )
    return observations


def test_ground_plane_fixed_horizon_uses_bbox_footpoint() -> None:
    bbox = _bbox(height=70.0, bottom=300.0)
    result = estimate_ground_plane_range(
        bbox,
        timestamp_s=1.0,
        image_shape=IMAGE_SHAPE,
        focal_y_px=FOCAL_Y,
        principal_y_px=PRINCIPAL_Y,
    )

    assert result.reason_code == P2CReason.FIXED_HORIZON
    assert result.horizon_source == "fixed"
    assert result.footpoint_y_px == 300.0
    assert result.horizon_y_px == PRINCIPAL_Y
    assert result.range_m == pytest.approx(FOCAL_Y * 1.55 / 120.0)
    assert 0.0 < result.range_uncertainty_m < result.range_m

    # Horizontal position must not affect flat-ground longitudinal range.
    shifted = estimate_ground_plane_range(
        (20.0, 230.0, 60.0, 300.0),
        timestamp_s=1.0,
        image_shape=IMAGE_SHAPE,
        focal_y_px=FOCAL_Y,
        principal_y_px=PRINCIPAL_Y,
    )
    assert shifted.range_m == result.range_m


def test_ground_plane_uses_reliable_causal_lane_horizon() -> None:
    horizon = causal_lane_horizon_y(_lane(), image_height=IMAGE_SHAPE[0])
    assert horizon is not None
    assert horizon[0] == pytest.approx(160.0, abs=0.3)

    result = estimate_ground_plane_range(
        _bbox(height=70.0, bottom=300.0),
        timestamp_s=2.0,
        image_shape=IMAGE_SHAPE,
        focal_y_px=FOCAL_Y,
        principal_y_px=PRINCIPAL_Y,
        lane=_lane(),
    )

    assert result.reason_code == P2CReason.OK
    assert result.horizon_source == "lane"
    assert result.range_m == pytest.approx(FOCAL_Y * 1.55 / 140.0, rel=0.01)


def test_low_confidence_lane_falls_back_to_global_calibration() -> None:
    result = estimate_ground_plane_range(
        _bbox(height=50.0),
        timestamp_s=0.0,
        image_shape=IMAGE_SHAPE,
        focal_y_px=FOCAL_Y,
        principal_y_px=PRINCIPAL_Y,
        lane=_lane(confidence=0.40),
    )

    assert result.horizon_source == "fixed"
    assert result.reason_code == P2CReason.FIXED_HORIZON


def test_ground_plane_rejects_footpoint_above_horizon() -> None:
    result = estimate_ground_plane_range(
        _bbox(height=20.0, bottom=181.0),
        timestamp_s=0.0,
        image_shape=IMAGE_SHAPE,
        focal_y_px=FOCAL_Y,
        principal_y_px=PRINCIPAL_Y,
    )

    assert result.reason_code == P2CReason.FOOTPOINT_ABOVE_HORIZON
    assert not result.reliable
    assert math.isnan(result.range_m)


def test_ground_plane_marks_bottom_truncated_measurement_unreliable() -> None:
    result = estimate_ground_plane_range(
        _bbox(height=100.0, bottom=359.0),
        timestamp_s=0.0,
        image_shape=IMAGE_SHAPE,
        focal_y_px=FOCAL_Y,
        principal_y_px=PRINCIPAL_Y,
    )

    assert result.reason_code == P2CReason.FOOTPOINT_TRUNCATED
    assert math.isfinite(result.range_m)
    assert not result.reliable


def test_inferred_lane_has_more_horizon_uncertainty() -> None:
    full = causal_lane_horizon_y(_lane(), image_height=IMAGE_SHAPE[0])
    inferred = causal_lane_horizon_y(
        _lane(inferred_side="left"), image_height=IMAGE_SHAPE[0]
    )

    assert full is not None and inferred is not None
    assert inferred[1] > full[1]


def test_class_prior_range_uses_full_posterior_mixture() -> None:
    pure_car = estimate_class_prior_range(
        _bbox(height=60.0),
        {"car": 1.0},
        timestamp_s=0.0,
        focal_y_px=FOCAL_Y,
    )
    mixed = estimate_class_prior_range(
        _bbox(height=60.0),
        {"car": 0.75, "truck": 0.25},
        timestamp_s=0.0,
        focal_y_px=FOCAL_Y,
    )

    assert pure_car.reason_code == P2CReason.OK
    assert pure_car.range_m == pytest.approx(8.0)
    assert mixed.effective_height_m == pytest.approx(1.875)
    assert mixed.range_m == pytest.approx(10.0)
    assert mixed.range_uncertainty_m > pure_car.range_uncertainty_m
    assert mixed.posterior_entropy > pure_car.posterior_entropy


def test_class_prior_changes_continuously_when_posterior_changes() -> None:
    first = estimate_class_prior_range(
        _bbox(height=50.0),
        {"person": 0.55, "bicycle": 0.45},
        timestamp_s=0.0,
        focal_y_px=FOCAL_Y,
    )
    second = estimate_class_prior_range(
        _bbox(height=50.0),
        {"person": 0.45, "bicycle": 0.55},
        timestamp_s=0.1,
        focal_y_px=FOCAL_Y,
    )

    # There is no winner-takes-all class switch; only the posterior weights move.
    assert abs(first.range_m - second.range_m) < 0.04


def test_class_prior_has_explicit_empty_and_small_box_reasons() -> None:
    empty = estimate_class_prior_range(
        _bbox(height=40.0),
        {},
        timestamp_s=0.0,
        focal_y_px=FOCAL_Y,
    )
    small = estimate_class_prior_range(
        _bbox(height=5.0),
        {"person": 1.0},
        timestamp_s=0.0,
        focal_y_px=FOCAL_Y,
    )

    assert empty.reason_code == P2CReason.EMPTY_CLASS_POSTERIOR
    assert small.reason_code == P2CReason.BOX_TOO_SMALL
    assert not empty.reliable and not small.reliable


def test_unknown_class_mass_is_preserved_as_uncertainty() -> None:
    result = estimate_class_prior_range(
        _bbox(height=50.0),
        {"car": 0.4, "unknown-road-user": 0.6},
        timestamp_s=0.0,
        focal_y_px=FOCAL_Y,
    )

    assert result.supported_probability == pytest.approx(0.4)
    assert result.effective_height_uncertainty_m > 0.5


def test_scale_expansion_recovers_irregular_timestamp_rate() -> None:
    observations = _scale_observations([0.0, 0.06, 0.17, 0.31, 0.52, 0.80])
    result = estimate_causal_scale_expansion_ttc(
        observations,
        evaluation_timestamp_s=0.80,
    )

    assert result.reason_code == P2CReason.OK
    assert result.source == "height_area_fusion"
    assert result.inverse_ttc_s == pytest.approx(0.25, abs=1e-8)
    assert result.ttc_s == pytest.approx(4.0, abs=1e-8)
    assert math.isfinite(result.ttc_uncertainty_s)
    assert result.height_fit.inlier_count == 6
    assert result.area_fit.inlier_count == 6


def test_scale_expansion_mad_huber_rejects_bbox_outlier() -> None:
    observations = _scale_observations(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    )
    bad = observations[3]
    observations[3] = ScaleObservation(
        bad.timestamp_s,
        _bbox(height=150.0, width=250.0),
        bad.confidence,
    )

    result = estimate_causal_scale_expansion_ttc(
        observations,
        evaluation_timestamp_s=0.7,
    )

    assert result.reason_code == P2CReason.OK
    assert result.inverse_ttc_s == pytest.approx(0.25, abs=1e-6)
    assert result.height_fit.outlier_count == 1
    assert result.area_fit.outlier_count == 1


def test_future_scale_observation_cannot_change_past_estimate() -> None:
    causal = _scale_observations([0.0, 0.1, 0.2, 0.3, 0.4])
    baseline = estimate_causal_scale_expansion_ttc(
        causal,
        evaluation_timestamp_s=0.4,
    )
    poisoned = estimate_causal_scale_expansion_ttc(
        [
            *causal,
            ScaleObservation(0.41, _bbox(height=300.0, width=500.0)),
            ScaleObservation(1.00, _bbox(height=9.0, width=9.0)),
        ],
        evaluation_timestamp_s=0.4,
    )

    assert poisoned == baseline


def test_scale_coast_counts_down_without_synthetic_observation() -> None:
    estimator = CausalScaleExpansionEstimator()
    last = None
    for observation in _scale_observations([0.0, 0.1, 0.2, 0.3, 0.4]):
        last = estimator.update(
            timestamp_s=observation.timestamp_s,
            bbox=observation.bbox,
            confidence=observation.confidence,
        )
    assert last is not None and last.reliable
    history = estimator.history

    coast = estimator.predict(timestamp_s=0.55)

    assert estimator.history == history
    assert coast.reason_code == P2CReason.OK
    assert coast.ttc_s == pytest.approx(last.ttc_s - 0.15, abs=1e-8)


def test_scale_observe_retains_real_boxes_without_eager_fit() -> None:
    estimator = CausalScaleExpansionEstimator()

    estimator.observe(timestamp_s=0.1, bbox=_bbox(height=30.0), confidence=0.8)
    estimator.observe(timestamp_s=0.2, bbox=_bbox(height=32.0), confidence=0.9)

    assert [item.timestamp_s for item in estimator.history] == [0.1, 0.2]
    assert estimator.predict(timestamp_s=0.2).height_fit.sample_count == 2

    with pytest.raises(ValueError, match="increase strictly"):
        estimator.observe(timestamp_s=0.2, bbox=_bbox(height=34.0))


def test_cached_scale_projection_matches_direct_predict() -> None:
    estimator = CausalScaleExpansionEstimator(
        ScaleExpansionConfig(maximum_extrapolation_s=0.2)
    )
    anchor = None
    for observation in _scale_observations([0.0, 0.1, 0.2, 0.3, 0.4]):
        anchor = estimator.update(
            timestamp_s=observation.timestamp_s,
            bbox=observation.bbox,
            confidence=observation.confidence,
        )
    assert anchor is not None

    direct = estimator.predict(timestamp_s=0.55)
    projected = project_scale_expansion_result(
        anchor,
        evaluation_timestamp_s=0.55,
        config=estimator.config,
    )

    assert projected.reason_code == direct.reason_code
    assert projected.ttc_s == pytest.approx(direct.ttc_s, abs=1e-9)
    assert projected.height_fit == anchor.height_fit
    stale = project_scale_expansion_result(
        anchor,
        evaluation_timestamp_s=0.61,
        config=estimator.config,
    )
    assert stale.reason_code == P2CReason.STALE_HISTORY
    assert math.isinf(stale.ttc_s)


def test_scale_reacquisition_flash_is_treated_as_outlier() -> None:
    estimator = CausalScaleExpansionEstimator()
    observations = _scale_observations([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    for observation in observations:
        estimator.update(
            timestamp_s=observation.timestamp_s,
            bbox=observation.bbox,
            confidence=observation.confidence,
        )
    result = estimator.update(
        timestamp_s=0.8,
        bbox=_bbox(height=250.0, width=400.0),
        confidence=0.30,
    )

    assert result.reason_code == P2CReason.OK
    assert result.height_fit.outlier_count == 1
    assert result.inverse_ttc_s == pytest.approx(0.25, abs=1e-6)


def test_nonexpanding_and_stale_scale_have_reason_codes() -> None:
    receding = _scale_observations(
        [0.0, 0.1, 0.2, 0.3, 0.4], inverse_ttc_s=-0.20
    )
    nonexpanding = estimate_causal_scale_expansion_ttc(
        receding,
        evaluation_timestamp_s=0.4,
    )
    estimator = CausalScaleExpansionEstimator(
        ScaleExpansionConfig(maximum_extrapolation_s=0.2)
    )
    for observation in _scale_observations([0.0, 0.1, 0.2, 0.3]):
        estimator.update(
            timestamp_s=observation.timestamp_s,
            bbox=observation.bbox,
        )
    stale = estimator.predict(timestamp_s=0.51)

    assert nonexpanding.reason_code == P2CReason.NON_EXPANDING
    assert stale.reason_code == P2CReason.STALE_HISTORY
    assert math.isinf(nonexpanding.ttc_s) and math.isinf(stale.ttc_s)


def test_scale_state_rejects_noncausal_update_order() -> None:
    estimator = CausalScaleExpansionEstimator()
    estimator.update(timestamp_s=0.1, bbox=_bbox(height=30.0))

    with pytest.raises(ValueError, match="increase strictly"):
        estimator.update(timestamp_s=0.1, bbox=_bbox(height=31.0))


def test_fusion_uses_inverse_ttc_uncertainty_weighting() -> None:
    result = fuse_ttc_estimates(
        [
            TTCFusionInput("ground", 2.0, 0.2),
            TTCFusionInput("scale", 4.0, 1.0),
        ],
        timestamp_s=1.0,
    )

    assert result.reason_code == P2CReason.FUSED
    assert result.contributing_sources == ("ground", "scale")
    assert 2.0 < result.ttc_s < 4.0
    assert result.ttc_s < 2.5  # lower-uncertainty ground estimate dominates
    assert math.isfinite(result.uncertainty_s)


def test_fusion_disagreement_inflates_uncertainty() -> None:
    close = fuse_ttc_estimates(
        [
            TTCFusionInput("a", 3.0, 0.3),
            TTCFusionInput("b", 3.2, 0.3),
        ],
        timestamp_s=0.0,
    )
    disagree = fuse_ttc_estimates(
        [
            TTCFusionInput("a", 1.5, 0.3),
            TTCFusionInput("b", 6.0, 0.3),
        ],
        timestamp_s=0.0,
    )

    assert close.uncertainty_s < disagree.uncertainty_s or not disagree.reliable


def test_fusion_has_explicit_physics_fallback_reason() -> None:
    physics = TTCFusionInput(
        "physics", 2.4, float("inf"), reason_code="legacy_physics"
    )
    result = fuse_ttc_estimates(
        [TTCFusionInput("scale", float("inf"), float("inf"), "unstable")],
        timestamp_s=0.0,
        fallback=physics,
    )

    assert result.reason_code == P2CReason.FALLBACK
    assert result.ttc_s == 2.4
    assert result.fallback_used
    assert "legacy_physics" in result.source


def test_contact_ttc_is_clamped_to_finite_danger_not_discarded() -> None:
    result = fuse_ttc_estimates(
        [
            TTCFusionInput(
                "ground",
                0.0,
                0.15,
                reason_code="within_safety_buffer",
            )
        ],
        timestamp_s=0.0,
    )

    assert result.reason_code == P2CReason.OK
    assert result.ttc_s == pytest.approx(0.1)
    assert result.reliable


def test_causal_fusion_never_delays_downward_danger_jump() -> None:
    fusion = CausalTTCFusion(TTCFusionConfig(maximum_upward_jump_s=0.5))
    first = fusion.update(
        [TTCFusionInput("ground", 4.0, 0.2)], timestamp_s=0.0
    )
    danger = fusion.update(
        [TTCFusionInput("ground", 1.0, 0.2)], timestamp_s=0.1
    )

    assert first.ttc_s == 4.0
    assert danger.ttc_s == 1.0
    assert danger.reason_code == P2CReason.OK


def test_causal_fusion_limits_upward_jump_and_holds_brief_gap() -> None:
    fusion = CausalTTCFusion(
        TTCFusionConfig(maximum_upward_jump_s=0.5, fallback_max_age_s=0.3)
    )
    fusion.update([TTCFusionInput("scale", 2.0, 0.2)], timestamp_s=0.0)
    limited = fusion.update(
        [TTCFusionInput("scale", 8.0, 0.2)], timestamp_s=0.1
    )
    held = fusion.update([], timestamp_s=0.2)

    assert limited.reason_code == P2CReason.JUMP_LIMITED
    assert limited.ttc_s == pytest.approx(2.4)
    assert held.reason_code == P2CReason.HELD_PREVIOUS
    assert held.ttc_s == pytest.approx(2.3)


def test_dense_hold_callbacks_do_not_refresh_the_evidence_anchor() -> None:
    fusion = CausalTTCFusion(TTCFusionConfig(fallback_max_age_s=0.30))
    fusion.update([TTCFusionInput("scale", 2.0, 0.2)], timestamp_s=0.0)

    for timestamp in (0.05, 0.10, 0.20, 0.30):
        held = fusion.update([], timestamp_s=timestamp)
        assert held.reason_code == P2CReason.HELD_PREVIOUS
        assert held.ttc_s == pytest.approx(2.0 - timestamp)

    expired = fusion.update([], timestamp_s=0.301)
    assert expired.reason_code == P2CReason.NO_RELIABLE_INPUT
    assert math.isinf(expired.ttc_s)
