from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from safeloop.c1.corridor import CorridorEvidence
from safeloop.c1.target_selector import (
    DeployableTargetSelector,
    RuntimeTargetCandidate,
    TargetSelectorConfig,
)


def _corridor(
    *,
    in_path: float = 0.85,
    cut_in: float = 0.0,
    overlap: float = 0.75,
    predicted_overlap: float | None = None,
    lateral_uncertainty_px: float = 8.0,
) -> CorridorEvidence:
    predicted = overlap if predicted_overlap is None else predicted_overlap
    return CorridorEvidence(
        footpoint=(320.0, 300.0),
        predicted_footpoint=(320.0, 300.0),
        current_bounds=(250.0, 390.0),
        predicted_bounds=(250.0, 390.0),
        corridor_overlap=overlap,
        predicted_corridor_overlap=predicted,
        in_path_probability=in_path,
        cut_in_probability=cut_in,
        lateral_uncertainty_px=lateral_uncertainty_px,
        track_confidence=0.90,
        corridor_source="lane",
        corridor_confidence=0.80,
    )


def _candidate(
    track_id: int,
    *,
    ttc_s: float = 2.4,
    corridor: CorridorEvidence | None = None,
    range_m: float = 22.0,
    closing_speed_mps: float = 4.0,
    observed: bool = True,
    missed: int = 0,
) -> RuntimeTargetCandidate:
    return RuntimeTargetCandidate(
        track_id=track_id,
        label="car",
        bbox=(280.0, 190.0, 360.0, 310.0),
        corridor=corridor or _corridor(),
        confidence=0.90,
        foot_velocity_x_px_s=0.0,
        foot_velocity_y_px_s=2.0,
        raw_physics_ttc_s=ttc_s,
        scale_ttc_s=ttc_s,
        robust_ttc_s=ttc_s + 0.1,
        closing_speed_mps=closing_speed_mps,
        range_m=range_m,
        range_stability=0.85,
        range_uncertainty_m=0.4,
        hits=8,
        missed_updates=missed,
        association_confidence=0.88,
        observed_this_update=observed,
        robust_reason_code="ok",
        uncertainty=0.08,
    )


def test_selector_uses_corridor_evidence_not_box_size_or_shortest_ttc() -> None:
    selector = DeployableTargetSelector()
    in_path = _candidate(10, ttc_s=2.4)
    off_path_large = replace(
        _candidate(
            20,
            ttc_s=0.7,
            corridor=_corridor(
                in_path=0.01,
                overlap=0.0,
                predicted_overlap=0.0,
            ),
        ),
        bbox=(20.0, 30.0, 620.0, 350.0),
        range_m=3.0,
    )

    result = selector.select([off_path_large, in_path])

    assert result.primary_track_id == 10
    assert result.primary_ttc_s == pytest.approx(2.4)
    off_path_assessment = next(
        item for item in result.assessments if item.track_id == 20
    )
    assert not off_path_assessment.eligible


def test_future_footpoint_cut_in_can_become_primary() -> None:
    cut_in = _candidate(
        7,
        ttc_s=1.7,
        corridor=_corridor(
            in_path=0.50,
            cut_in=0.78,
            overlap=0.0,
            predicted_overlap=0.72,
        ),
    )

    result = DeployableTargetSelector().select([cut_in])

    assert result.primary_track_id == 7
    assert result.warning
    assert result.dangerous_track_ids == (7,)


def test_hysteresis_counts_only_real_detector_observations() -> None:
    selector = DeployableTargetSelector(
        TargetSelectorConfig(
            enable_hysteresis=True,
            switch_confirmation_observations=2,
            switch_score_margin=0.04,
        )
    )
    current = _candidate(
        1,
        ttc_s=4.5,
        corridor=_corridor(in_path=0.55, overlap=0.35),
        range_m=38.0,
        closing_speed_mps=1.0,
    )
    # Keep this challenger outside the danger threshold so the dedicated
    # confirmed-danger handover does not bypass ordinary hysteresis.
    challenger = replace(_candidate(2, ttc_s=3.2), hits=2)

    assert selector.select([current]).primary_track_id == 1
    first = selector.select([current, challenger])
    skipped_detector = selector.select(
        [current, replace(challenger, observed_this_update=False)]
    )
    second_observation = selector.select([current, replace(challenger, hits=3)])

    assert first.primary_track_id == 1
    assert first.held_by_hysteresis
    assert skipped_detector.primary_track_id == 1
    assert skipped_detector.held_by_hysteresis
    assert second_observation.primary_track_id == 2
    assert second_observation.switched


def test_hysteresis_disabled_switches_immediately() -> None:
    selector = DeployableTargetSelector(
        TargetSelectorConfig(enable_hysteresis=False)
    )
    current = _candidate(
        1,
        ttc_s=4.5,
        corridor=_corridor(in_path=0.55, overlap=0.35),
        range_m=38.0,
        closing_speed_mps=1.0,
    )
    challenger = _candidate(2, ttc_s=1.4)

    assert selector.select([current]).primary_track_id == 1
    result = selector.select([current, challenger])

    assert result.primary_track_id == 2
    assert result.switched
    assert not result.held_by_hysteresis


def test_confirmed_in_path_danger_replaces_safe_incumbent_without_extra_delay() -> None:
    selector = DeployableTargetSelector(
        TargetSelectorConfig(enable_hysteresis=True)
    )
    safe = _candidate(1, ttc_s=5.0)
    danger = replace(_candidate(2, ttc_s=1.2), hits=2)

    assert selector.select([safe]).primary_track_id == 1
    result = selector.select([safe, danger])

    assert result.primary_track_id == 2
    assert result.switched
    assert result.warning


def test_unconfirmed_cold_start_challenger_does_not_bypass_hysteresis_on_stride() -> None:
    selector = DeployableTargetSelector(
        TargetSelectorConfig(enable_hysteresis=True)
    )
    incumbent = _candidate(1, ttc_s=5.0)
    first_evidence = _corridor(
        in_path=0.30,
        overlap=0.0,
        predicted_overlap=0.0,
    )
    observed_danger = replace(
        _candidate(
            2,
            ttc_s=float("inf"),
            corridor=first_evidence,
            range_m=10.0,
            closing_speed_mps=0.0,
        ),
        ego_motion_ttc_s=0.75,
        hits=1,
    )

    assert selector.select([incumbent]).primary_track_id == 1
    observed = selector.select([incumbent, observed_danger])
    first_stride = selector.select(
        [
            incumbent,
            replace(
                observed_danger,
                corridor=_corridor(),
                ego_motion_ttc_s=0.70,
                observed_this_update=False,
            ),
        ]
    )
    second_stride = selector.select(
        [
            incumbent,
            replace(
                observed_danger,
                corridor=_corridor(),
                ego_motion_ttc_s=0.65,
                observed_this_update=False,
            ),
        ]
    )

    assert all(
        result.primary_track_id == 1
        for result in (observed, first_stride, second_stride)
    )
    assert not any(
        result.warning for result in (observed, first_stride, second_stride)
    )
    challenger = next(
        item for item in second_stride.assessments if item.track_id == 2
    )
    assert not challenger.confirmed
    assert challenger.ttc_source == "ego_motion_cold_start"


@pytest.mark.parametrize("source", ("physics", "robust_range", "last_good"))
def test_normal_ttc_sources_require_confirmed_track(source: str) -> None:
    selector = DeployableTargetSelector(
        TargetSelectorConfig(
            use_robust_ttc=source == "robust_range",
            enable_last_good_fallback=source == "last_good",
            enable_cold_start_ttc=False,
        )
    )
    candidate = replace(
        _candidate(30, ttc_s=1.5 if source == "physics" else float("inf")),
        robust_ttc_s=1.5,
        robust_reason_code="held_previous" if source == "last_good" else "ok",
        observed_this_update=source != "last_good",
        missed_updates=1 if source == "last_good" else 0,
        hits=1,
    )

    unconfirmed = selector.select([candidate])
    confirmed = selector.select([replace(candidate, hits=2)])

    first = unconfirmed.assessments[0]
    second = confirmed.assessments[0]
    assert not first.confirmed
    assert first.ttc_source == "invalid"
    assert not first.eligible
    assert unconfirmed.primary_track_id is None
    assert second.confirmed
    assert second.ttc_source == source
    assert second.eligible
    assert confirmed.primary_track_id == 30


def test_missing_detection_cannot_create_a_new_warning() -> None:
    selector = DeployableTargetSelector()
    observed_safe = _candidate(5, ttc_s=2.2)
    coasted_danger = _candidate(
        5,
        ttc_s=1.7,
        observed=False,
        missed=1,
    )

    assert not selector.select([observed_safe]).warning
    coast_result = selector.select([coasted_danger])

    assert coast_result.primary_ttc_s == pytest.approx(1.7)
    assert not coast_result.warning


def test_detector_observed_danger_can_cross_warning_on_stride_countdown() -> None:
    selector = DeployableTargetSelector()
    observed = _candidate(8, ttc_s=2.05)

    assert not selector.select([observed]).warning
    stride = selector.select(
        [replace(observed, raw_physics_ttc_s=1.95, observed_this_update=False)]
    )

    assert stride.warning
    assert stride.primary_ttc_s == pytest.approx(1.95)


def test_short_strong_coast_retains_but_does_not_extend_warning_forever() -> None:
    selector = DeployableTargetSelector()
    observed_danger = _candidate(9, ttc_s=1.6)

    assert selector.select([observed_danger]).warning
    strong_coast = selector.select(
        [replace(observed_danger, observed_this_update=False, missed_updates=1)]
    )
    stale_coast = selector.select(
        [replace(observed_danger, observed_this_update=False, missed_updates=4)]
    )

    assert strong_coast.warning
    assert not stale_coast.warning


def test_selector_reports_all_dangers_but_one_consistent_primary() -> None:
    selector = DeployableTargetSelector()
    later_collision = _candidate(4, ttc_s=1.8)
    sooner_collision = _candidate(3, ttc_s=1.1)

    result = selector.select([later_collision, sooner_collision])

    assert result.primary_track_id == 3
    assert result.primary_ttc_s == pytest.approx(1.1)
    assert result.dangerous_track_ids == (3, 4)


def test_robust_ttc_requires_stable_low_uncertainty_range() -> None:
    selector = DeployableTargetSelector(
        TargetSelectorConfig(use_robust_ttc=True)
    )
    stable = replace(
        _candidate(1, ttc_s=float("inf")),
        robust_ttc_s=1.8,
        range_stability=0.9,
        range_uncertainty_m=0.3,
    )
    unstable = replace(
        _candidate(2, ttc_s=2.6),
        robust_ttc_s=0.8,
        range_stability=0.1,
        range_uncertainty_m=20.0,
    )

    result = selector.select([stable, unstable])
    assessments = {item.track_id: item for item in result.assessments}

    assert assessments[1].selected_ttc_s == pytest.approx(1.8)
    assert assessments[1].ttc_source == "robust_range"
    assert assessments[2].selected_ttc_s == pytest.approx(2.6)
    assert assessments[2].ttc_source == "physics"


def test_robust_ttc_cannot_suppress_a_more_urgent_physics_estimate() -> None:
    selector = DeployableTargetSelector(
        TargetSelectorConfig(use_robust_ttc=True)
    )
    candidate = replace(
        _candidate(6, ttc_s=1.6),
        robust_ttc_s=2.4,
        range_stability=0.9,
        range_uncertainty_m=0.3,
    )

    result = selector.select([candidate])

    assert result.primary_ttc_s == pytest.approx(1.6)
    assert result.assessments[0].ttc_source == "physics"
    assert result.warning


def test_last_good_ttc_fallback_is_explicit_and_still_cannot_create_warning() -> None:
    selector = DeployableTargetSelector(
        TargetSelectorConfig(enable_last_good_fallback=True)
    )
    coast = replace(
        _candidate(12, ttc_s=float("inf"), observed=False, missed=1),
        robust_ttc_s=1.5,
        robust_reason_code="held_previous",
        range_uncertainty_m=0.2,
    )

    result = selector.select([coast])
    assessment = result.assessments[0]

    assert assessment.ttc_source == "last_good"
    assert assessment.selected_ttc_s == pytest.approx(1.5)
    assert not result.warning


def test_unconfirmed_last_good_challenger_cannot_enter_hysteresis() -> None:
    selector = DeployableTargetSelector(
        TargetSelectorConfig(
            enable_hysteresis=True,
            enable_last_good_fallback=True,
        )
    )
    incumbent = _candidate(1, ttc_s=5.0)
    observed_last_good = replace(
        _candidate(2, ttc_s=float("inf")),
        robust_ttc_s=1.2,
        robust_reason_code="held_previous",
        hits=1,
    )

    assert selector.select([incumbent]).primary_track_id == 1
    observed = selector.select([incumbent, observed_last_good])
    challenger = next(item for item in observed.assessments if item.track_id == 2)

    assert observed.primary_track_id == 1
    assert not observed.warning
    assert not challenger.confirmed
    assert challenger.ttc_source == "invalid"
    assert not challenger.eligible


def test_snapshot_adapter_maps_runtime_fields_without_annotation_inputs() -> None:
    snapshot = SimpleNamespace(
        track_id=21,
        label="bicycle",
        class_posterior={"bicycle": 0.8, "person": 0.2},
        bbox=(250.0, 180.0, 310.0, 300.0),
        confidence=0.75,
        foot_velocity_x_px_s=16.0,
        foot_velocity_y_px_s=1.0,
        raw_physics_ttc_s=2.4,
        ego_motion_ttc_s=2.1,
        scale_ttc_s=2.4,
        legacy_range_ttc_s=float("inf"),
        robust_range_ttc_s=2.2,
        closing_speed_mps=3.0,
        estimated_range_m=18.0,
        range_stability=0.8,
        range_uncertainty_m=0.5,
        hits=7,
        missed_updates=0,
        association_confidence=0.7,
        observed_this_call=True,
        last_good_ttc_s=2.3,
        reason_code="ok",
    )

    candidate = RuntimeTargetCandidate.from_snapshot(snapshot, _corridor())

    assert candidate.track_id == 21
    assert candidate.label == "bicycle"
    assert candidate.robust_ttc_s == pytest.approx(2.2)
    assert candidate.ego_motion_ttc_s == pytest.approx(2.1)
    assert candidate.observed_this_update


def test_ineligible_primary_expires_after_bounded_hysteresis_hold() -> None:
    selector = DeployableTargetSelector(
        TargetSelectorConfig(enable_hysteresis=True, primary_hold_updates=1)
    )
    active = _candidate(6, ttc_s=2.5)
    no_ttc = replace(active, raw_physics_ttc_s=float("inf"))

    assert selector.select([active]).primary_track_id == 6
    held = selector.select([no_ttc])
    expired = selector.select([no_ttc])

    assert held.primary_track_id == 6
    assert held.held_by_hysteresis
    assert expired.primary_track_id is None


def test_hysteresis_hold_preserves_finite_primary_ttc_after_path_gate_drop() -> None:
    selector = DeployableTargetSelector(
        TargetSelectorConfig(enable_hysteresis=True, primary_hold_updates=2)
    )
    active = _candidate(31, ttc_s=2.4)
    path_drop = replace(
        active,
        raw_physics_ttc_s=1.7,
        corridor=_corridor(
            in_path=0.0,
            cut_in=0.0,
            overlap=0.0,
            predicted_overlap=0.0,
        ),
    )

    assert selector.select([active]).primary_track_id == 31
    first_hold = selector.select([path_drop])
    second_hold = selector.select([path_drop])
    expired = selector.select([path_drop])

    assert first_hold.primary_track_id == 31
    assert first_hold.primary_ttc_s == pytest.approx(1.7)
    assert first_hold.held_by_hysteresis
    assert not first_hold.warning
    assert second_hold.primary_ttc_s == pytest.approx(1.7)
    assert expired.primary_track_id is None
    assert not math.isfinite(expired.primary_ttc_s)


def test_observed_high_confidence_near_corridor_enables_ego_motion_cold_start() -> None:
    cold_start = replace(
        _candidate(41, ttc_s=float("inf")),
        ego_motion_ttc_s=1.6,
        hits=1,
    )

    result = DeployableTargetSelector().select([cold_start])
    assessment = result.assessments[0]

    assert not assessment.confirmed
    assert assessment.cold_start_eligible
    assert assessment.ttc_source == "ego_motion_cold_start"
    assert result.primary_ttc_s == pytest.approx(1.6)
    assert result.warning


@pytest.mark.parametrize(
    "change",
    (
        {"observed_this_update": False},
        {"missed_updates": 1},
        {"hits": 3},
        {"confidence": 0.39},
        {"association_confidence": 0.39},
        {"uncertainty": 0.90},
    ),
)
def test_ego_motion_cold_start_fails_closed_outside_runtime_gates(change) -> None:
    updates = {"ego_motion_ttc_s": 1.5, "hits": 1, **change}
    candidate = replace(
        _candidate(42, ttc_s=float("inf")),
        **updates,
    )

    result = DeployableTargetSelector().select([candidate])

    assert result.primary_track_id is None
    assert result.assessments[0].ttc_source == "invalid"
    assert not result.assessments[0].cold_start_eligible
    assert not result.warning


def test_ego_motion_cold_start_requires_foot_region_near_corridor() -> None:
    far_evidence = replace(
        _corridor(in_path=0.8, overlap=0.0, predicted_overlap=0.0),
        footpoint=(80.0, 300.0),
        predicted_footpoint=(90.0, 300.0),
    )
    candidate = replace(
        _candidate(
            43,
            ttc_s=float("inf"),
            corridor=far_evidence,
        ),
        ego_motion_ttc_s=1.4,
        hits=1,
    )

    result = DeployableTargetSelector().select([candidate])

    assert result.primary_track_id is None
    assert not result.assessments[0].cold_start_eligible


def test_ego_motion_signal_on_coast_cannot_create_warning() -> None:
    unobserved = replace(
        _candidate(44, ttc_s=float("inf"), observed=False, missed=1),
        ego_motion_ttc_s=1.2,
        hits=1,
    )

    result = DeployableTargetSelector().select([unobserved])

    assert result.primary_track_id is None
    assert not result.warning


def test_armed_cold_start_continues_across_stride_and_one_detector_miss() -> None:
    evidence = replace(
        _corridor(in_path=0.14, overlap=0.0, predicted_overlap=0.0),
        footpoint=(502.0, 264.0),
        predicted_footpoint=(502.0, 264.0),
        current_bounds=(213.0, 427.0),
        predicted_bounds=(213.0, 427.0),
        lateral_uncertainty_px=48.0,
    )
    observed = replace(
        _candidate(31, ttc_s=float("inf"), corridor=evidence),
        bbox=(452.0, 189.0, 552.0, 264.0),
        ego_motion_ttc_s=1.5,
        hits=1,
        confidence=0.85,
        association_confidence=0.55,
        range_m=12.0,
        range_stability=0.0,
        range_uncertainty_m=12.0,
        uncertainty=0.20,
    )
    selector = DeployableTargetSelector()

    fresh = selector.select([observed])
    stride = selector.select(
        [replace(observed, observed_this_update=False, ego_motion_ttc_s=1.45)]
    )
    one_miss = selector.select(
        [
            replace(
                observed,
                observed_this_update=False,
                missed_updates=1,
                ego_motion_ttc_s=1.40,
            )
        ]
    )
    expired = selector.select(
        [
            replace(
                observed,
                observed_this_update=False,
                missed_updates=2,
                ego_motion_ttc_s=1.35,
            )
        ]
    )

    assert fresh.primary_ttc_s == pytest.approx(1.5)
    assert fresh.warning
    assert stride.primary_ttc_s == pytest.approx(1.45)
    assert stride.warning
    assert one_miss.primary_ttc_s == pytest.approx(1.40)
    assert one_miss.warning
    assert not math.isfinite(expired.primary_ttc_s)
    assert not expired.warning


def test_invalid_or_duplicate_candidates_fail_closed() -> None:
    with pytest.raises(ValueError, match="hit thresholds"):
        TargetSelectorConfig(min_confirmed_hits=0)
    with pytest.raises(ValueError, match="positive width"):
        replace(_candidate(1), bbox=(2.0, 2.0, 1.0, 3.0))

    selector = DeployableTargetSelector()
    duplicated = _candidate(1)
    with pytest.raises(ValueError, match="unique"):
        selector.select([duplicated, duplicated])
