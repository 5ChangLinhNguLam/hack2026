from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from safeloop.c1.corridor import CorridorEvidence
from safeloop.c1.p2b_runtime import (
    LOCKED_P2_VARIANTS,
    P2DeployableRuntime,
    P2Variant,
)
from safeloop.c1.p2b_tracker import P2TrackSnapshot
from safeloop.c1.target_selector import (
    RuntimeTargetCandidate,
    TargetAssessment,
    TargetSelection,
)
from safeloop.c1.temporal_features import CameraGeometry, baseline_tracker
from safeloop.c1.types import Detection


GEOMETRY = CameraGeometry(640, 360, 320.0, 320.0, 320.0)
IMAGE = np.zeros((360, 640, 3), dtype=np.uint8)


def _detection(box: tuple[float, float, float, float]) -> Detection:
    return Detection(2, "car", 0.90, box)


def _corridor() -> CorridorEvidence:
    return CorridorEvidence(
        footpoint=(320.0, 300.0),
        predicted_footpoint=(320.0, 300.0),
        current_bounds=(240.0, 400.0),
        predicted_bounds=(240.0, 400.0),
        corridor_overlap=0.9,
        predicted_corridor_overlap=0.9,
        in_path_probability=0.9,
        cut_in_probability=0.0,
        lateral_uncertainty_px=8.0,
        track_confidence=0.9,
        corridor_source="fixed",
        corridor_confidence=0.3,
    )


def _candidate(*, uncertainty: float = 0.1) -> RuntimeTargetCandidate:
    return RuntimeTargetCandidate(
        track_id=7,
        bbox=(280.0, 180.0, 360.0, 300.0),
        corridor=_corridor(),
        label="car",
        confidence=0.9,
        raw_physics_ttc_s=1.5,
        closing_speed_mps=5.0,
        range_m=18.0,
        range_stability=0.8,
        range_uncertainty_m=0.5,
        hits=8,
        association_confidence=0.9,
        uncertainty=uncertainty,
    )


def _selection(ttc_s: float, *, uncertainty: float = 0.1) -> TargetSelection:
    assessment = TargetAssessment(
        track_id=7,
        evidence_score=0.8,
        priority_score=0.8,
        path_score=0.9,
        closing_score=0.9,
        range_score=0.7,
        maturity_score=1.0,
        association_score=0.9,
        uncertainty=uncertainty,
        selected_ttc_s=ttc_s,
        ttc_source="physics" if math.isfinite(ttc_s) else "invalid",
        confirmed=True,
        cold_start_eligible=False,
        eligible=math.isfinite(ttc_s),
        dangerous=math.isfinite(ttc_s) and ttc_s < 3.0,
        observed_this_update=True,
        hits=8,
        missed_updates=0,
    )
    return TargetSelection(
        primary_track_id=7,
        primary_ttc_s=ttc_s,
        warning=math.isfinite(ttc_s) and ttc_s < 2.0,
        dangerous_track_ids=(7,) if math.isfinite(ttc_s) and ttc_s < 3.0 else (),
        assessments=(assessment,),
        switched=False,
        held_by_hysteresis=False,
    )


def test_locked_variants_isolate_every_ablation_toggle() -> None:
    physics = LOCKED_P2_VARIANTS[P2Variant.PHYSICS_CURRENT]
    corridor = LOCKED_P2_VARIANTS[P2Variant.CORRIDOR_SELECTOR]
    class_history = LOCKED_P2_VARIANTS[P2Variant.SELECTOR_CLASS_HISTORY]
    robust = LOCKED_P2_VARIANTS[P2Variant.SELECTOR_ROBUST_RANGE]
    full = LOCKED_P2_VARIANTS[P2Variant.FULL]

    assert not physics.use_corridor_selector
    assert corridor.use_corridor_selector and not corridor.enable_cold_start_ttc
    assert not corridor.use_class_history_association
    assert not corridor.use_robust_range
    assert class_history.use_class_history_association and not class_history.use_robust_range
    assert robust.use_robust_range and not robust.use_class_history_association
    assert not class_history.enable_cold_start_ttc
    assert not robust.enable_cold_start_ttc
    assert full.use_class_history_association and full.use_robust_range
    assert full.enable_last_good_fallback and full.enable_hysteresis
    assert full.enable_cold_start_ttc


def test_physics_runtime_wires_update_and_stride_predict_exactly() -> None:
    runtime = P2DeployableRuntime(GEOMETRY, P2Variant.PHYSICS_CURRENT)
    reference = baseline_tracker(GEOMETRY)
    detections = (_detection((285.0, 150.0, 355.0, 230.0)),)

    first = runtime.step(
        IMAGE,
        detections,
        detector_update=True,
        timestamp=0.0,
        ego_speed_kmh=30.0,
    )
    first_reference = reference.update(
        detections,
        timestamp=0.0,
        image_shape=(360, 640),
        ego_speed_kmh=30.0,
    )
    stride = runtime.step(
        IMAGE,
        (),
        detector_update=False,
        timestamp=0.05,
        ego_speed_kmh=30.0,
    )
    stride_reference = reference.predict(
        timestamp=0.05,
        image_shape=(360, 640),
        ego_speed_kmh=30.0,
    )

    assert first.risks == tuple(first_reference)
    assert stride.risks == tuple(stride_reference)
    assert runtime.lane_detector is None


def test_class_posterior_entropy_and_motion_covariance_are_variant_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = P2TrackSnapshot(
        track_id=1,
        label="car",
        class_posterior={"car": 0.5, "person": 0.5},
        bbox=(280.0, 180.0, 360.0, 300.0),
        confidence=0.9,
        foot_velocity_x_px_s=0.0,
        foot_velocity_y_px_s=0.0,
        foot_velocity_uncertainty_px_s=None,
        motion_state_confidence=None,
        raw_physics_ttc_s=3.0,
        ego_motion_ttc_s=float("inf"),
        scale_ttc_s=3.0,
        legacy_range_ttc_s=float("inf"),
        lateral_ttc_s=float("inf"),
        robust_range_ttc_s=float("inf"),
        closing_speed_mps=2.0,
        estimated_range_m=20.0,
        range_stability=0.8,
        range_uncertainty_m=0.5,
        ttc_uncertainty_s=0.5,
        hits=8,
        missed_updates=0,
        association_confidence=0.9,
        observed_this_call=True,
        last_seen_age_s=0.0,
        last_good_ttc_s=float("inf"),
        reason_code="valid_legacy_physics",
    )
    corridor_runtime = P2DeployableRuntime(GEOMETRY, P2Variant.CORRIDOR_SELECTOR)
    class_runtime = P2DeployableRuntime(GEOMETRY, P2Variant.SELECTOR_CLASS_HISTORY)
    corridor = corridor_runtime.corridor_estimator.estimate(None, (360, 640))

    plain = corridor_runtime._candidate(snapshot, corridor)
    history = class_runtime._candidate(snapshot, corridor)

    assert history.uncertainty > plain.uncertainty

    captured_uncertainties: list[float] = []
    assert class_runtime.corridor_estimator is not None
    original_assess = class_runtime.corridor_estimator.assess

    def recording_assess(*args: object, **kwargs: object) -> CorridorEvidence:
        captured_uncertainties.append(float(kwargs["velocity_uncertainty_px_s"]))
        return original_assess(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(class_runtime.corridor_estimator, "assess", recording_assess)
    class_runtime._candidate(snapshot, corridor)
    class_runtime._candidate(
        replace(snapshot, foot_velocity_uncertainty_px_s=17.5), corridor
    )

    assert captured_uncertainties == [36.0 / math.sqrt(snapshot.hits), 17.5]

    covariance_aware = class_runtime._candidate(
        replace(snapshot, motion_state_confidence=0.50), corridor
    )
    assert covariance_aware.association_confidence == pytest.approx(
        snapshot.association_confidence
    )
    assert covariance_aware.corridor.track_confidence == pytest.approx(
        snapshot.confidence * snapshot.association_confidence
    )
    assert covariance_aware.uncertainty > history.uncertainty


def test_full_output_limits_upward_jump_but_not_genuine_cut_in() -> None:
    runtime = P2DeployableRuntime(GEOMETRY, P2Variant.FULL)
    candidate = _candidate()
    runtime._remember_full_output(
        7,
        timestamp=0.0,
        predicted_ttc_s=1.5,
        warning=True,
        source="physics",
    )

    upward, source, _ = runtime._stabilize_full_output(
        _selection(5.0), (candidate,), timestamp=0.05, source="physics"
    )
    downward, _, _ = runtime._stabilize_full_output(
        _selection(0.4), (candidate,), timestamp=0.05, source="physics"
    )

    assert upward == pytest.approx(2.45)
    assert source == "jump_limited_physics"
    assert downward == pytest.approx(0.4)


def test_full_output_holds_only_recent_warning_physics_with_strong_evidence() -> None:
    runtime = P2DeployableRuntime(GEOMETRY, P2Variant.FULL)
    runtime._remember_full_output(
        7,
        timestamp=0.0,
        predicted_ttc_s=1.5,
        warning=True,
        source="physics",
    )

    held, source, warning = runtime._stabilize_full_output(
        _selection(float("inf")),
        (_candidate(),),
        timestamp=0.10,
        source="invalid",
    )
    rejected, _, rejected_warning = runtime._stabilize_full_output(
        _selection(float("inf"), uncertainty=0.95),
        (_candidate(uncertainty=0.95),),
        timestamp=0.10,
        source="invalid",
    )

    assert held == pytest.approx(1.4)
    assert source == "physics_hold"
    assert warning
    assert not math.isfinite(rejected)
    assert not rejected_warning


def test_high_uncertainty_physics_requires_previous_stable_fallback() -> None:
    runtime = P2DeployableRuntime(GEOMETRY, P2Variant.FULL)
    uncertain_selection = _selection(0.4, uncertainty=0.80)
    uncertain_candidate = _candidate(uncertainty=0.80)

    without_anchor, source, warning = runtime._stabilize_full_output(
        uncertain_selection,
        (uncertain_candidate,),
        timestamp=0.05,
        source="physics",
    )

    assert not math.isfinite(without_anchor)
    assert source == "uncertain_physics"
    assert not warning

    runtime._remember_full_output(
        7,
        timestamp=0.0,
        predicted_ttc_s=1.5,
        warning=True,
        source="physics",
    )
    held, source, warning = runtime._stabilize_full_output(
        uncertain_selection,
        (uncertain_candidate,),
        timestamp=0.05,
        source="physics",
    )

    assert held == pytest.approx(1.45)
    assert source == "physics_hold"
    assert warning


def test_nonphysics_output_cannot_be_laundered_into_physics_fallback() -> None:
    runtime = P2DeployableRuntime(GEOMETRY, P2Variant.FULL)
    runtime._remember_full_output(
        7,
        timestamp=0.0,
        predicted_ttc_s=1.5,
        warning=True,
        source="robust_range",
    )

    current, source, _ = runtime._stabilize_full_output(
        _selection(5.0),
        (_candidate(),),
        timestamp=0.05,
        source="ego_motion_cold_start",
    )
    runtime._remember_full_output(
        7,
        timestamp=0.05,
        predicted_ttc_s=current,
        warning=False,
        source=source,
    )
    held, held_source, _ = runtime._stabilize_full_output(
        _selection(float("inf")),
        (_candidate(),),
        timestamp=0.10,
        source="invalid",
    )

    assert current == pytest.approx(5.0)
    assert source == "ego_motion_cold_start"
    assert not math.isfinite(held)
    assert held_source == "invalid"


def test_finite_robust_danger_is_not_overwritten_by_previous_physics() -> None:
    runtime = P2DeployableRuntime(GEOMETRY, P2Variant.FULL)
    runtime._remember_full_output(
        7,
        timestamp=0.0,
        predicted_ttc_s=5.0,
        warning=False,
        source="physics",
    )

    current, source, warning = runtime._stabilize_full_output(
        _selection(1.0),
        (_candidate(),),
        timestamp=0.05,
        source="robust_range",
    )

    assert current == pytest.approx(1.0)
    assert source == "robust_range"
    assert warning


def test_reset_clears_runtime_continuity_state() -> None:
    runtime = P2DeployableRuntime(GEOMETRY, P2Variant.FULL)
    runtime._remember_full_output(
        7,
        timestamp=0.0,
        predicted_ttc_s=1.5,
        warning=True,
        source="physics",
    )

    runtime.reset()

    assert runtime._published_ttc_by_track == {}
