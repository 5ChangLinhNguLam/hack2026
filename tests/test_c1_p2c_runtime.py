from __future__ import annotations

import inspect
import math
from dataclasses import replace

import numpy as np
import pytest

from safeloop.c1.p2b_runtime import P2DeployableRuntime, P2FrameResult, P2Variant
from safeloop.c1.p2b_tracker import P2TrackSnapshot
from safeloop.c1.p2c_range import (
    P2CReason,
    ScaleExpansionResult,
    ScaleFitResult,
    TTCFusionInput,
    TTCFusionResult,
)
from safeloop.c1.p2c_runtime import (
    LOCKED_P2C_VARIANTS,
    P2CDeployableRuntime,
    P2CVariant,
)
from safeloop.c1.temporal_features import CameraGeometry
from safeloop.c1.target_selector import TargetAssessment
from safeloop.c1.types import Detection


GEOMETRY = CameraGeometry(640, 360, 320.0, 320.0, 320.0)
IMAGE = np.zeros((360, 640, 3), dtype=np.uint8)
BOX = (280.0, 170.0, 360.0, 300.0)


def _detection(
    bbox: tuple[float, float, float, float] = BOX,
) -> Detection:
    return Detection(2, "car", 0.90, bbox)


def _snapshot(
    *,
    raw_ttc_s: float = float("inf"),
    observed: bool = False,
) -> P2TrackSnapshot:
    return P2TrackSnapshot(
        track_id=1,
        label="car",
        class_posterior={"car": 1.0},
        bbox=BOX,
        confidence=0.9,
        foot_velocity_x_px_s=0.0,
        foot_velocity_y_px_s=0.0,
        foot_velocity_uncertainty_px_s=5.0,
        motion_state_confidence=0.9,
        raw_physics_ttc_s=raw_ttc_s,
        ego_motion_ttc_s=float("inf"),
        scale_ttc_s=raw_ttc_s,
        legacy_range_ttc_s=float("inf"),
        lateral_ttc_s=float("inf"),
        robust_range_ttc_s=float("inf"),
        closing_speed_mps=0.0,
        estimated_range_m=20.0,
        range_stability=0.8,
        range_uncertainty_m=1.0,
        ttc_uncertainty_s=1.0,
        hits=8,
        missed_updates=0,
        association_confidence=0.9,
        observed_this_call=observed,
        last_seen_age_s=0.0,
        last_good_ttc_s=float("inf"),
        reason_code="non_approaching_or_unstable",
    )


def _base_result(*, source: str = "last_good", ttc_s: float = 1.5) -> P2FrameResult:
    return P2FrameResult(
        timestamp=1.0,
        predicted_ttc_s=ttc_s,
        primary_track_id=1,
        dangerous_track_ids=(1,),
        target_switched=False,
        held_by_hysteresis=False,
        warning=ttc_s < 2.0,
        invalid_reason="valid",
        ttc_source=source,
        lane_source="fixed",
        lane_confidence=0.3,
        candidate_count=1,
        assessments=(),
        risks=(),
        downstream_latency_ms=1.0,
    )


def _assessment(
    *,
    ttc_s: float = 1.5,
    observed: bool = True,
    confirmed: bool = True,
    uncertainty: float = 0.20,
) -> TargetAssessment:
    return TargetAssessment(
        track_id=1,
        evidence_score=0.9,
        priority_score=0.9,
        path_score=0.9,
        closing_score=0.9,
        range_score=0.9,
        maturity_score=0.9,
        association_score=0.9,
        uncertainty=uncertainty,
        selected_ttc_s=ttc_s,
        ttc_source="physics",
        confirmed=confirmed,
        cold_start_eligible=not confirmed,
        eligible=True,
        dangerous=ttc_s < 3.0,
        observed_this_update=observed,
        hits=8,
        missed_updates=0,
    )


def _scale_result(
    *,
    reason: str,
    ttc_s: float = float("inf"),
    age_s: float = 0.1,
) -> ScaleExpansionResult:
    fit_reason = P2CReason.NON_EXPANDING if reason == P2CReason.NON_EXPANDING else P2CReason.OK
    fit = ScaleFitResult(
        source="log_height",
        inverse_ttc_s=0.0 if fit_reason == P2CReason.NON_EXPANDING else 1.0 / ttc_s,
        inverse_ttc_uncertainty_s=0.02,
        reason_code=fit_reason,
        sample_count=6,
        inlier_count=6,
        outlier_count=0,
    )
    area_fit = ScaleFitResult(
        source="log_sqrt_area",
        inverse_ttc_s=fit.inverse_ttc_s,
        inverse_ttc_uncertainty_s=fit.inverse_ttc_uncertainty_s,
        reason_code=fit.reason_code,
        sample_count=fit.sample_count,
        inlier_count=fit.inlier_count,
        outlier_count=fit.outlier_count,
    )
    return ScaleExpansionResult(
        evaluation_timestamp_s=1.0,
        ttc_s=ttc_s,
        ttc_uncertainty_s=0.25 if math.isfinite(ttc_s) else float("inf"),
        inverse_ttc_s=fit.inverse_ttc_s,
        inverse_ttc_uncertainty_s=fit.inverse_ttc_uncertainty_s,
        reason_code=reason,
        source="height_area_fusion" if math.isfinite(ttc_s) else "none",
        height_fit=fit,
        area_fit=area_fit,
        last_observation_age_s=age_s,
    )


def test_six_variants_are_locked_and_isolated() -> None:
    assert tuple(variant.value for variant in P2CVariant) == (
        "physics",
        "p2b_full",
        "p2b_ground_plane",
        "p2b_scale_expansion",
        "p2b_class_prior",
        "p2c_full_fusion",
    )
    assert len(LOCKED_P2C_VARIANTS) == 6
    assert not LOCKED_P2C_VARIANTS[P2CVariant.PHYSICS].uses_p2c_estimators
    assert not LOCKED_P2C_VARIANTS[P2CVariant.P2B_FULL].uses_p2c_estimators

    ground = LOCKED_P2C_VARIANTS[P2CVariant.GROUND_PLANE]
    scale = LOCKED_P2C_VARIANTS[P2CVariant.SCALE_EXPANSION]
    class_prior = LOCKED_P2C_VARIANTS[P2CVariant.CLASS_PRIOR]
    full = LOCKED_P2C_VARIANTS[P2CVariant.FULL_FUSION]
    assert (ground.use_ground_plane, ground.use_scale_expansion, ground.use_class_prior) == (
        True,
        False,
        False,
    )
    assert (scale.use_ground_plane, scale.use_scale_expansion, scale.use_class_prior) == (
        False,
        True,
        False,
    )
    assert (
        class_prior.use_ground_plane,
        class_prior.use_scale_expansion,
        class_prior.use_class_prior,
    ) == (False, False, True)
    assert full.use_ground_plane and full.use_scale_expansion and full.use_class_prior
    assert full.use_safety_veto
    assert not any((ground.use_safety_veto, scale.use_safety_veto, class_prior.use_safety_veto))


@pytest.mark.parametrize(
    ("variant", "delegated"),
    (
        (P2CVariant.PHYSICS, P2Variant.PHYSICS_CURRENT),
        (P2CVariant.P2B_FULL, P2Variant.FULL),
    ),
)
def test_delegated_variants_preserve_p2_output_exactly(
    variant: P2CVariant,
    delegated: P2Variant,
) -> None:
    runtime = P2CDeployableRuntime(GEOMETRY, variant)
    reference = P2DeployableRuntime(GEOMETRY, delegated)
    sequence = (
        (0.00, True, (_detection(),)),
        (0.05, False, ()),
        (0.15, True, (_detection((279.0, 166.0, 361.0, 301.0)),)),
    )
    comparable_fields = tuple(
        field
        for field in P2FrameResult.__dataclass_fields__
        if field != "downstream_latency_ms"
    )
    for timestamp, detector_update, detections in sequence:
        actual = runtime.step(
            IMAGE,
            detections,
            detector_update=detector_update,
            timestamp=timestamp,
            ego_speed_kmh=30.0,
        )
        expected = reference.step(
            IMAGE,
            detections,
            detector_update=detector_update,
            timestamp=timestamp,
            ego_speed_kmh=30.0,
        )
        assert {
            field: getattr(actual, field) for field in comparable_fields
        } == {
            field: getattr(expected, field) for field in comparable_fields
        }


def test_runtime_interface_excludes_non_runtime_modalities() -> None:
    parameters = tuple(inspect.signature(P2CDeployableRuntime.step).parameters)
    assert parameters == (
        "self",
        "image_bgr",
        "detections",
        "detector_update",
        "timestamp",
        "ego_speed_kmh",
    )

    runtime = P2CDeployableRuntime(GEOMETRY, P2CVariant.FULL_FUSION)
    runtime.step(
        IMAGE,
        (_detection(),),
        detector_update=True,
        timestamp=1.0,
        ego_speed_kmh=20.0,
    )
    with pytest.raises(ValueError, match="increase strictly"):
        runtime.step(
            IMAGE,
            (_detection(),),
            detector_update=True,
            timestamp=0.9,
            ego_speed_kmh=20.0,
        )


def test_estimators_append_only_real_detection_observations() -> None:
    runtime = P2CDeployableRuntime(GEOMETRY, P2CVariant.FULL_FUSION)
    runtime.step(
        IMAGE,
        (_detection(),),
        detector_update=True,
        timestamp=0.0,
        ego_speed_kmh=20.0,
    )
    state = runtime._track_states[1]
    assert state.ground_range is not None
    assert state.class_range is not None
    assert state.scale is not None
    initial_lengths = (
        len(state.ground_range.history),
        len(state.class_range.history),
        len(state.scale.history),
    )

    runtime.step(
        IMAGE,
        (),
        detector_update=False,
        timestamp=0.05,
        ego_speed_kmh=20.0,
    )
    assert (
        len(state.ground_range.history),
        len(state.class_range.history),
        len(state.scale.history),
    ) == initial_lengths

    runtime.step(
        IMAGE,
        (_detection((279.0, 168.0, 361.0, 302.0)),),
        detector_update=True,
        timestamp=0.15,
        ego_speed_kmh=20.0,
    )
    assert len(state.ground_range.history) == initial_lengths[0] + 1
    assert len(state.class_range.history) == initial_lengths[1] + 1
    assert len(state.scale.history) == initial_lengths[2] + 1


def test_one_observation_cold_start_cannot_publish_warning() -> None:
    runtime = P2CDeployableRuntime(GEOMETRY, P2CVariant.FULL_FUSION)
    result = runtime.step(
        IMAGE,
        (_detection(),),
        detector_update=True,
        timestamp=0.0,
        ego_speed_kmh=80.0,
    )

    assert result.primary_track_id == 1
    assert result.base_predicted_ttc_s < 2.0
    assert result.assessments[0].cold_start_eligible
    assert not result.assessments[0].confirmed
    assert not result.warning
    assert result.predicted_ttc_s == 2.0


def test_second_real_observation_can_confirm_existing_physics_warning() -> None:
    runtime = P2CDeployableRuntime(GEOMETRY, P2CVariant.SCALE_EXPANSION)
    first = runtime.step(
        IMAGE,
        (_detection(),),
        detector_update=True,
        timestamp=0.0,
        ego_speed_kmh=80.0,
    )
    second = runtime.step(
        IMAGE,
        (_detection(),),
        detector_update=True,
        timestamp=0.15,
        ego_speed_kmh=80.0,
    )

    assert not first.warning
    assert second.assessments[0].confirmed
    assert second.base_predicted_ttc_s < 2.0
    assert second.warning
    assert second.predicted_ttc_s < 2.0


def test_unconfirmed_cold_start_does_not_anchor_confirmed_safe_output() -> None:
    runtime = P2CDeployableRuntime(GEOMETRY, P2CVariant.SCALE_EXPANSION)
    first = runtime.step(
        IMAGE,
        (_detection(),),
        detector_update=True,
        timestamp=0.0,
        ego_speed_kmh=80.0,
    )
    safe = runtime.step(
        IMAGE,
        (_detection((279.0, 169.0, 361.0, 301.0)),),
        detector_update=True,
        timestamp=0.15,
        ego_speed_kmh=80.0,
    )

    assert first.predicted_ttc_s == 2.0 and not first.warning
    assert safe.assessments[0].confirmed
    assert safe.base_predicted_ttc_s > 3.0
    assert safe.predicted_ttc_s == safe.base_predicted_ttc_s
    assert not safe.warning


def test_reset_and_tracker_pruning_remove_estimator_state() -> None:
    runtime = P2CDeployableRuntime(GEOMETRY, P2CVariant.FULL_FUSION)
    runtime.step(
        IMAGE,
        (_detection(),),
        detector_update=True,
        timestamp=0.0,
        ego_speed_kmh=20.0,
    )
    assert runtime.active_estimator_track_ids == (1,)

    for index in range(1, 7):
        runtime.step(
            IMAGE,
            (),
            detector_update=True,
            timestamp=0.1 * index,
            ego_speed_kmh=20.0,
        )
    assert runtime.active_estimator_track_ids == ()

    runtime.reset()
    assert runtime.active_estimator_track_ids == ()
    restarted = runtime.step(
        IMAGE,
        (_detection(),),
        detector_update=True,
        timestamp=0.0,
        ego_speed_kmh=20.0,
    )
    assert restarted.primary_track_id == 1


def test_result_exposes_primary_bbox_components_and_reason_codes() -> None:
    runtime = P2CDeployableRuntime(GEOMETRY, P2CVariant.FULL_FUSION)
    result = runtime.step(
        IMAGE,
        (_detection(),),
        detector_update=True,
        timestamp=0.0,
        ego_speed_kmh=30.0,
    )

    assert result.primary_bbox == BOX
    assert tuple(item.source for item in result.component_diagnostics) == (
        "ground_plane",
        "scale_expansion",
        "class_prior",
    )
    assert result.estimator_reason == P2CReason.FALLBACK
    assert result.estimator_sources == ("p2b_full",)
    assert math.isinf(result.estimator_uncertainty_s)
    assert all(item.reason == P2CReason.INSUFFICIENT_HISTORY for item in result.component_diagnostics)


def test_lane_detector_is_evaluated_once_per_p2c_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = P2CDeployableRuntime(GEOMETRY, P2CVariant.GROUND_PLANE)
    assert runtime._lane_capture is not None
    calls = 0
    original = runtime._lane_capture._detector.detect

    def counted(image: np.ndarray):
        nonlocal calls
        calls += 1
        return original(image)

    monkeypatch.setattr(runtime._lane_capture._detector, "detect", counted)
    runtime.step(
        IMAGE,
        (_detection(),),
        detector_update=True,
        timestamp=0.0,
        ego_speed_kmh=20.0,
    )
    assert calls == 1


def test_strict_nonexpanding_scale_clears_only_stale_physics_fallback() -> None:
    runtime = P2CDeployableRuntime(GEOMETRY, P2CVariant.FULL_FUSION)
    state = runtime._new_track_state()
    state.scale_result = _scale_result(reason=P2CReason.NON_EXPANDING)

    cleared = runtime._fuse(
        state,
        _snapshot(raw_ttc_s=float("inf"), observed=False),
        _base_result(source="last_good"),
        (),
        timestamp=1.0,
        assessment=_assessment(observed=False),
    )
    assert cleared.reason_code == P2CReason.SCALE_SAFETY_VETO
    assert math.isinf(cleared.ttc_s)
    assert not cleared.fallback_used

    protected = runtime._fuse(
        state,
        _snapshot(raw_ttc_s=1.2, observed=True),
        _base_result(source="last_good"),
        (),
        timestamp=1.1,
        assessment=_assessment(),
    )
    assert protected.reason_code == P2CReason.OK
    assert protected.source == "fresh_raw_physics"
    assert protected.ttc_s == 1.2


def test_strict_scale_bound_can_reject_correlated_monocular_danger() -> None:
    runtime = P2CDeployableRuntime(GEOMETRY, P2CVariant.FULL_FUSION)
    state = runtime._new_track_state()
    state.scale_result = _scale_result(reason=P2CReason.NON_EXPANDING)
    ground_danger = TTCFusionInput("ground_plane", 1.4, 0.2)

    result = runtime._fuse(
        state,
        _snapshot(raw_ttc_s=float("inf"), observed=False),
        _base_result(source="last_good"),
        (ground_danger,),
        timestamp=1.0,
        assessment=_assessment(observed=False),
    )

    assert math.isinf(result.ttc_s)
    assert result.reason_code == P2CReason.SCALE_SAFETY_VETO


def test_reliable_safe_scale_bypasses_stale_danger_jump_hold() -> None:
    runtime = P2CDeployableRuntime(GEOMETRY, P2CVariant.FULL_FUSION)
    state = runtime._new_track_state()
    # Establish a previous finite-danger fusion anchor.
    state.fusion.update(
        (TTCFusionInput("ground_plane", 1.5, 0.2),), timestamp_s=0.8
    )
    state.scale_result = _scale_result(reason=P2CReason.OK, ttc_s=3.2)
    safe_scale = TTCFusionInput("scale_expansion", 3.2, 0.25)

    result = runtime._fuse(
        state,
        _snapshot(raw_ttc_s=float("inf"), observed=False),
        _base_result(source="robust_range"),
        (safe_scale,),
        timestamp=1.0,
        assessment=_assessment(observed=False, ttc_s=3.2),
    )

    assert result.reason_code == P2CReason.SCALE_SAFETY_VETO
    assert result.ttc_s == pytest.approx(3.2)


def test_uncertain_nonexpanding_scale_cannot_clear_fallback_danger() -> None:
    runtime = P2CDeployableRuntime(GEOMETRY, P2CVariant.FULL_FUSION)
    state = runtime._new_track_state()
    result = _scale_result(reason=P2CReason.NON_EXPANDING)
    uncertain_height = replace(
        result.height_fit, inverse_ttc_uncertainty_s=0.30
    )
    uncertain_area = replace(
        result.area_fit, inverse_ttc_uncertainty_s=0.30
    )
    state.scale_result = replace(
        result, height_fit=uncertain_height, area_fit=uncertain_area
    )

    protected = runtime._fuse(
        state,
        _snapshot(raw_ttc_s=float("inf"), observed=False),
        _base_result(source="last_good"),
        (),
        timestamp=1.0,
        assessment=_assessment(observed=False),
    )

    assert protected.reason_code == P2CReason.FALLBACK
    assert protected.ttc_s == pytest.approx(1.5)


def test_fresh_confirmed_raw_physics_is_a_causal_fusion_input() -> None:
    runtime = P2CDeployableRuntime(GEOMETRY, P2CVariant.SCALE_EXPANSION)
    state = runtime._new_track_state()
    base = _base_result(source="physics", ttc_s=4.0)

    current = runtime._fuse(
        state,
        _snapshot(raw_ttc_s=1.4, observed=True),
        base,
        (),
        timestamp=1.0,
        assessment=_assessment(ttc_s=1.4),
    )

    assert current.reason_code == P2CReason.OK
    assert current.source == "fresh_raw_physics"
    assert current.ttc_s == pytest.approx(1.4)
    assert current.contributing_sources == ("fresh_raw_physics",)

    coast_runtime = P2CDeployableRuntime(GEOMETRY, P2CVariant.SCALE_EXPANSION)
    coast = coast_runtime._fuse(
        coast_runtime._new_track_state(),
        _snapshot(raw_ttc_s=1.4, observed=False),
        base,
        (),
        timestamp=1.0,
        assessment=_assessment(ttc_s=1.4, observed=False),
    )
    assert coast.reason_code == P2CReason.FALLBACK
    assert coast.ttc_s == 4.0
    assert "fresh_raw_physics" not in coast.contributing_sources


def test_single_safe_estimator_cannot_erase_p2b_danger() -> None:
    runtime = P2CDeployableRuntime(GEOMETRY, P2CVariant.GROUND_PLANE)
    state = runtime._new_track_state()

    result = runtime._fuse(
        state,
        _snapshot(observed=False),
        _base_result(source="last_good", ttc_s=1.5),
        (TTCFusionInput("ground_plane", 5.0, 0.2),),
        timestamp=1.0,
        assessment=_assessment(observed=False),
    )

    assert result.reason_code == P2CReason.FALLBACK
    assert result.source == "safety_envelope:last_good"
    assert result.ttc_s == 1.5
    assert result.fallback_used


def test_fresh_physics_danger_blocks_strict_scale_clear() -> None:
    runtime = P2CDeployableRuntime(GEOMETRY, P2CVariant.FULL_FUSION)
    state = runtime._new_track_state()
    state.scale_result = _scale_result(reason=P2CReason.OK, ttc_s=3.2)

    result = runtime._fuse(
        state,
        _snapshot(raw_ttc_s=1.2, observed=True),
        _base_result(source="last_good", ttc_s=1.5),
        (TTCFusionInput("scale_expansion", 3.2, 0.25),),
        timestamp=1.0,
        assessment=_assessment(ttc_s=1.2),
    )

    assert result.ttc_s < 2.0
    assert "fresh_raw_physics" in result.contributing_sources


def test_scale_clear_requires_both_individual_fits_to_be_safe() -> None:
    runtime = P2CDeployableRuntime(GEOMETRY, P2CVariant.FULL_FUSION)
    state = runtime._new_track_state()
    nominal = _scale_result(reason=P2CReason.OK, ttc_s=3.2)
    state.scale_result = replace(
        nominal,
        area_fit=replace(nominal.area_fit, inverse_ttc_s=0.5),
    )

    result = runtime._fuse(
        state,
        _snapshot(observed=False),
        _base_result(source="last_good", ttc_s=1.5),
        (TTCFusionInput("scale_expansion", 3.2, 0.25),),
        timestamp=1.0,
        assessment=_assessment(ttc_s=3.2, observed=False),
    )

    assert result.source == "safety_envelope:last_good"
    assert result.ttc_s == 1.5


def test_correlated_signal_inflation_prevents_false_precision_gain() -> None:
    inputs = (
        TTCFusionInput("ground_plane", 2.0, 0.2),
        TTCFusionInput("scale_expansion", 2.2, 0.3),
        TTCFusionInput("class_prior", 2.1, 0.4),
    )

    inflated = P2CDeployableRuntime._inflate_correlated_signals(inputs)

    assert tuple(item.source for item in inflated) == tuple(
        item.source for item in inputs
    )
    assert tuple(item.ttc_s for item in inflated) == tuple(
        item.ttc_s for item in inputs
    )
    assert tuple(item.uncertainty_s for item in inflated) == pytest.approx(
        tuple(item.uncertainty_s * math.sqrt(3.0) for item in inputs)
    )


def test_warning_latch_clears_only_with_strong_safe_evidence() -> None:
    runtime = P2CDeployableRuntime(GEOMETRY, P2CVariant.FULL_FUSION)
    runtime._warning_latched_tracks.add(1)
    safe = TTCFusionResult(
        timestamp_s=1.0,
        ttc_s=4.0,
        uncertainty_s=0.2,
        reason_code=P2CReason.OK,
        source="scale_expansion",
        contributing_sources=("scale_expansion",),
        fallback_used=False,
    )

    _, warning = runtime._warning(
        _base_result(),
        _assessment(ttc_s=4.0),
        safe,
        strong_safe_evidence=False,
    )
    assert not warning
    assert 1 in runtime._warning_latched_tracks

    runtime._warning(
        _base_result(),
        _assessment(ttc_s=4.0),
        safe,
        strong_safe_evidence=True,
    )
    assert 1 not in runtime._warning_latched_tracks


def test_primary_danger_id_is_removed_for_safe_or_invalid_ttc() -> None:
    runtime = P2CDeployableRuntime(GEOMETRY, P2CVariant.P2B_FULL)
    safe = replace(
        _base_result(ttc_s=4.0), dangerous_track_ids=(1, 7)
    )
    invalid = replace(
        _base_result(ttc_s=float("inf")),
        dangerous_track_ids=(1, 7),
        invalid_reason="unstable_trend",
    )

    assert runtime._from_base(safe).dangerous_track_ids == (7,)
    assert runtime._from_base(invalid).dangerous_track_ids == (7,)

    full = P2CDeployableRuntime(GEOMETRY, P2CVariant.FULL_FUSION)
    assessment = _assessment()
    assert full._published_dangerous_ids(safe, assessment, 4.0) == (7,)
    assert full._published_dangerous_ids(safe, assessment, float("inf")) == (7,)
    assert full._published_dangerous_ids(safe, assessment, 1.8) == (7, 1)


def test_runtime_observes_all_tracks_but_fits_only_primary_and_caches_coast() -> None:
    runtime = P2CDeployableRuntime(GEOMETRY, P2CVariant.FULL_FUSION)
    first = _snapshot(observed=True)
    second = replace(first, track_id=2, bbox=(100.0, 160.0, 160.0, 280.0))

    runtime._update_track_states(
        (first, second),
        lane=None,
        timestamp=0.0,
        primary_track_id=1,
    )

    primary = runtime._track_states[1]
    secondary = runtime._track_states[2]
    assert len(primary.scale.history) == len(secondary.scale.history) == 1
    assert len(primary.ground_range.history) == len(secondary.ground_range.history) == 1
    assert primary.scale_fit_anchor is not None
    assert primary.ground_fit_anchor is not None
    assert secondary.scale_fit_anchor is None
    assert secondary.ground_fit_anchor is None

    scale_anchor = primary.scale_fit_anchor
    ground_anchor = primary.ground_fit_anchor
    histories = (primary.scale.history, primary.ground_range.history)
    runtime._update_track_states(
        (replace(first, observed_this_call=False), replace(second, observed_this_call=False)),
        lane=None,
        timestamp=0.05,
        primary_track_id=1,
    )

    assert primary.scale_fit_anchor is scale_anchor
    assert primary.ground_fit_anchor is ground_anchor
    assert (primary.scale.history, primary.ground_range.history) == histories
    assert primary.scale_result.evaluation_timestamp_s == pytest.approx(0.05)
    assert primary.ground_trend.evaluation_timestamp_s == pytest.approx(0.05)
