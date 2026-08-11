"""Production C3 score and contextual-risk contract tests.

These tests deliberately use synthetic telemetry only.  In particular, the
runtime must not inspect practice-only ground truth while reconstructing the
evaluator-compatible trip score.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import pytest

from safeloop.c3 import (
    Challenge3Accumulator,
    HARSH_ACCEL_THRESHOLD_MPS2,
    HARSH_BRAKE_THRESHOLD_MPS2,
    HARSH_CORNER_THRESHOLD_MPS2,
    NEAR_MISS_TTC_SECONDS,
)
from safeloop.contextual_risk import ContextualRiskPolicy
from team_kit.evaluation import TripGroundTruth, compute_challenge3_metrics


class SyntheticBundle:
    """Minimal source frame that fails loudly on any label/GT access."""

    _FORBIDDEN = {"gt", "depth", "targets", "labels", "events_active"}

    def __init__(
        self,
        frame_id: int,
        *,
        speed_kmh: float = 0.0,
        longitudinal_accel: float = 0.0,
        lateral_accel: float = 0.0,
        timestamp: float | None = None,
    ) -> None:
        self.frame_id = frame_id
        self.timestamp = frame_id * 0.05 if timestamp is None else timestamp
        self.ego = {
            "speed_kmh": speed_kmh,
            "longitudinal_accel": longitudinal_accel,
            "lateral_accel": lateral_accel,
        }

    def __getattr__(self, name: str) -> object:
        if name in self._FORBIDDEN:
            raise AssertionError(f"C3 must not read {name}")
        raise AttributeError(name)


def test_exact_evaluator_thresholds_are_strict() -> None:
    accumulator = Challenge3Accumulator(speed_limit_kmh=40.0, expected_frames=10)

    boundary_cases = (
        SyntheticBundle(0, longitudinal_accel=HARSH_BRAKE_THRESHOLD_MPS2),
        SyntheticBundle(1, longitudinal_accel=HARSH_ACCEL_THRESHOLD_MPS2),
        SyntheticBundle(2, lateral_accel=HARSH_CORNER_THRESHOLD_MPS2),
        SyntheticBundle(3, lateral_accel=-HARSH_CORNER_THRESHOLD_MPS2),
        SyntheticBundle(4, speed_kmh=45.0),
    )
    for bundle in boundary_cases:
        result = accumulator.update(
            bundle, predicted_ttc_s=NEAR_MISS_TTC_SECONDS
        )
        assert not result.is_harsh_brake
        assert not result.is_harsh_accel
        assert not result.is_harsh_corner
        assert not result.is_speeding
        assert not result.is_near_miss

    beyond_cases = (
        SyntheticBundle(
            5,
            longitudinal_accel=math.nextafter(
                HARSH_BRAKE_THRESHOLD_MPS2, -math.inf
            ),
        ),
        SyntheticBundle(
            6,
            longitudinal_accel=math.nextafter(
                HARSH_ACCEL_THRESHOLD_MPS2, math.inf
            ),
        ),
        SyntheticBundle(
            7,
            lateral_accel=math.nextafter(
                HARSH_CORNER_THRESHOLD_MPS2, math.inf
            ),
        ),
        SyntheticBundle(8, speed_kmh=math.nextafter(45.0, math.inf)),
    )
    results = [
        accumulator.update(bundle, predicted_ttc_s=math.inf)
        for bundle in beyond_cases
    ]
    near_miss = accumulator.update(
        SyntheticBundle(9),
        predicted_ttc_s=math.nextafter(NEAR_MISS_TTC_SECONDS, -math.inf),
    )

    assert results[0].is_harsh_brake
    assert results[1].is_harsh_accel
    assert results[2].is_harsh_corner
    assert results[3].is_speeding
    assert near_miss.is_near_miss
    assert near_miss.trip_complete


def test_formula_uses_all_source_frames_for_speeding_denominator() -> None:
    accumulator = Challenge3Accumulator(speed_limit_kmh=40.0, expected_frames=5)
    frames = (
        (SyntheticBundle(0, longitudinal_accel=-4.0), 1.0),
        (SyntheticBundle(1, longitudinal_accel=4.0), math.inf),
        (SyntheticBundle(2, lateral_accel=3.0), math.inf),
        (SyntheticBundle(3, speed_kmh=46.0), math.inf),
        (SyntheticBundle(4), math.inf),
    )

    for bundle, ttc in frames:
        result = accumulator.update(bundle, predicted_ttc_s=ttc)

    # 3 brake + 2 accel + 2 corner + 5 near-miss + 20% * 0.15 = 15.
    assert result.speeding_frames == 1
    assert result.speeding_pct_time == pytest.approx(20.0)
    assert result.total_penalty == pytest.approx(15.0)
    assert result.safe_score_estimate == pytest.approx(85.0)
    assert result.risk_score_pct == pytest.approx(15.0)
    assert result.trip_complete
    summary = accumulator.summary()
    assert summary["counts"] == {
        "near_miss_frames": 1,
        "harsh_brake_frames": 1,
        "harsh_accel_frames": 1,
        "harsh_corner_frames": 1,
        "speeding_frames": 1,
    }
    assert summary["penalties"] == {
        "near_miss": 5.0,
        "harsh_brake": 3.0,
        "harsh_accel": 2.0,
        "harsh_corner": 2.0,
        "speeding": 3.0,
    }

    evaluator = compute_challenge3_metrics(
        "synthetic",
        True,
        [ttc for _bundle, ttc in frames],
        TripGroundTruth(
            ttc={},
            driver_state={},
            safe_driving_score=85.0,
            harsh_brake_count=1,
            harsh_accel_count=1,
            harsh_corner_count=1,
            speeding_pct_time=20.0,
        ),
        85.0,
    )
    assert evaluator is not None
    assert evaluator.predicted_safe_score == result.safe_score_estimate
    assert evaluator.composite_score == 100.0


@pytest.mark.parametrize("ttc", [math.inf, -math.inf, math.nan, "not-a-number"])
def test_nonfinite_or_unparseable_ttc_is_not_a_near_miss(ttc: object) -> None:
    result = Challenge3Accumulator(
        speed_limit_kmh=40.0, expected_frames=1
    ).update(SyntheticBundle(0), predicted_ttc_s=ttc)

    assert not result.is_near_miss
    assert result.near_miss_frames == 0


def test_raw_finite_ttc_below_threshold_matches_evaluator() -> None:
    accumulator = Challenge3Accumulator(speed_limit_kmh=40.0, expected_frames=2)
    first = accumulator.update(SyntheticBundle(0), predicted_ttc_s=1.499)
    second = accumulator.update(SyntheticBundle(1), predicted_ttc_s=-0.1)

    # The evaluator's exact contract is ``finite TTC < 1.5``.  It does not
    # sanitize negative model output before counting it.
    assert first.is_near_miss
    assert second.is_near_miss
    assert second.near_miss_frames == 2


def test_score_clamps_and_reset_starts_a_new_trip() -> None:
    accumulator = Challenge3Accumulator(speed_limit_kmh=40.0, expected_frames=21)
    for frame_id in range(21):
        result = accumulator.update(
            SyntheticBundle(frame_id), predicted_ttc_s=1.0
        )

    assert result.safe_score_estimate == 0.0
    assert result.risk_score_pct == 100.0
    assert result.total_penalty == 105.0
    assert result.trip_complete

    accumulator.reset()
    summary = accumulator.summary()
    assert summary["processed_frames"] == 0
    assert summary["final_safe_score_estimate"] == 100.0
    assert summary["final_risk_score_pct"] == 0.0
    assert not summary["trip_complete"]
    assert all(count == 0 for count in summary["counts"].values())

    restarted = accumulator.update(SyntheticBundle(0), predicted_ttc_s=math.inf)
    assert restarted.frame_id == 0
    assert restarted.safe_score_estimate == 100.0


def test_frame_stream_must_start_at_zero_and_remain_contiguous() -> None:
    with pytest.raises(ValueError, match="start at frame 0"):
        Challenge3Accumulator(speed_limit_kmh=40.0).update(
            SyntheticBundle(1), predicted_ttc_s=math.inf
        )

    accumulator = Challenge3Accumulator(speed_limit_kmh=40.0)
    accumulator.update(SyntheticBundle(0), predicted_ttc_s=math.inf)
    with pytest.raises(ValueError, match="contiguous and ordered"):
        accumulator.update(SyntheticBundle(2), predicted_ttc_s=math.inf)


def test_complete_trip_rejects_extra_frames() -> None:
    accumulator = Challenge3Accumulator(speed_limit_kmh=40.0, expected_frames=1)
    accumulator.update(SyntheticBundle(0), predicted_ttc_s=math.inf)

    with pytest.raises(ValueError, match="more frames than expected"):
        accumulator.update(SyntheticBundle(1), predicted_ttc_s=math.inf)


def test_runtime_uses_only_ego_telemetry_and_reports_tailgating_omission() -> None:
    accumulator = Challenge3Accumulator(speed_limit_kmh=40.0, expected_frames=1)
    result = accumulator.update(SyntheticBundle(0), predicted_ttc_s=math.inf)
    summary = accumulator.summary()

    assert result.safe_score_estimate == 100.0
    assert result.tailgating_penalty_omitted is True
    assert summary["tailgating_status"] == "unavailable"
    assert summary["tailgating_penalty_omitted"] is True


@dataclass(frozen=True)
class FakeC1:
    predicted_ttc_s: float


@dataclass(frozen=True)
class FakeVssSignals:
    attentive_probability: float
    distraction_level: float
    fatigue_level: float


@dataclass(frozen=True)
class FakeC2:
    state: str
    attentive_probability: float
    distraction_level: float
    fatigue_level: float

    def vss_signals(self) -> FakeVssSignals:
        return FakeVssSignals(
            attentive_probability=self.attentive_probability,
            distraction_level=self.distraction_level,
            fatigue_level=self.fatigue_level,
        )


def test_c2_changes_contextual_risk_but_not_challenge3_score() -> None:
    policy = ContextualRiskPolicy()
    c1 = FakeC1(predicted_ttc_s=2.2)
    alert = FakeC2("alert", 95.0, 5.0, 5.0)
    microsleep = FakeC2("microsleep", 5.0, 8.0, 98.0)

    alert_risk = policy.evaluate(c1, alert)
    microsleep_risk = policy.evaluate(c1, microsleep)
    assert microsleep_risk.score_pct > alert_risk.score_pct
    assert microsleep_risk.level in {"HIGH", "CRITICAL"}
    assert "MICROSLEEP" in microsleep_risk.reasons

    # Official C3 consumes TTC and ego only, so changing the C2 result cannot
    # change its score.  Contextual product risk remains a separate output.
    scores = []
    for _driver_state in (alert, microsleep):
        scores.append(
            Challenge3Accumulator(
                speed_limit_kmh=40.0, expected_frames=1
            ).update(SyntheticBundle(0), predicted_ttc_s=c1.predicted_ttc_s)
        )
    assert scores[0].safe_score_estimate == scores[1].safe_score_estimate
    assert scores[0].near_miss_frames == scores[1].near_miss_frames


def test_critical_contextual_risk_is_warning_only() -> None:
    decision = ContextualRiskPolicy().evaluate(
        FakeC1(predicted_ttc_s=1.0),
        FakeC2("microsleep", 5.0, 8.0, 98.0),
    )

    assert decision.level == "CRITICAL"
    assert decision.action == "VISUAL_AUDIO_HAPTIC_WARNING"
    assert decision.brake_request_pct == 0.0
    assert decision.diagnostic_row()["contextual_risk_brake_request_pct"] == 0.0
