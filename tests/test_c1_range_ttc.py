from __future__ import annotations

import math

import pytest

from safeloop.c1.range_ttc import (
    CausalRangeTTCEstimator,
    RangeObservation,
    RangeTTCConfig,
    RangeTTCReason,
    estimate_causal_range_ttc,
)


def _observations(
    timestamps: list[float],
    *,
    initial_range_m: float = 25.0,
    closing_speed_mps: float = 5.0,
) -> list[RangeObservation]:
    return [
        RangeObservation(timestamp, initial_range_m - closing_speed_mps * timestamp)
        for timestamp in timestamps
    ]


def test_irregular_real_timestamps_recover_range_rate_and_ttc() -> None:
    observations = _observations([0.0, 0.04, 0.13, 0.31, 0.48, 0.73])

    result = estimate_causal_range_ttc(
        observations,
        evaluation_timestamp_s=0.73,
        safety_buffer_m=4.5,
    )

    assert result.reason_code == RangeTTCReason.OK
    assert result.closing_speed_mps == pytest.approx(5.0, abs=1e-8)
    assert result.range_m == pytest.approx(21.35, abs=1e-8)
    assert result.ttc_s == pytest.approx((21.35 - 4.5) / 5.0, abs=1e-8)
    assert math.isfinite(result.range_uncertainty_m)
    assert math.isfinite(result.closing_speed_uncertainty_mps)
    assert math.isfinite(result.ttc_uncertainty_s)


def test_mad_gate_removes_a_large_range_outlier() -> None:
    observations = _observations([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    observations[3] = RangeObservation(0.3, 80.0)

    result = estimate_causal_range_ttc(
        observations,
        evaluation_timestamp_s=0.6,
        safety_buffer_m=4.5,
    )

    assert result.reason_code == RangeTTCReason.OK
    assert result.outlier_count == 1
    assert result.inlier_count == 6
    assert result.range_m == pytest.approx(22.0, abs=1e-6)
    assert result.closing_speed_mps == pytest.approx(5.0, abs=1e-6)


def test_insufficient_history_has_explicit_nonfinite_reason() -> None:
    result = estimate_causal_range_ttc(
        _observations([0.0, 0.1, 0.2]),
        evaluation_timestamp_s=0.2,
    )

    assert result.reason_code == RangeTTCReason.INSUFFICIENT_HISTORY
    assert math.isinf(result.ttc_s)
    assert result.sample_count == 3


def test_nonclosing_history_has_explicit_nonfinite_reason() -> None:
    observations = [
        RangeObservation(timestamp, 20.0 + 2.0 * timestamp)
        for timestamp in [0.0, 0.1, 0.25, 0.4, 0.7]
    ]

    result = estimate_causal_range_ttc(
        observations,
        evaluation_timestamp_s=0.7,
    )

    assert result.reason_code == RangeTTCReason.NON_CLOSING
    assert result.closing_speed_mps == pytest.approx(-2.0, abs=1e-8)
    assert math.isinf(result.ttc_s)


def test_inconsistent_slopes_are_reported_as_unstable() -> None:
    observations = [
        RangeObservation(timestamp, value)
        for timestamp, value in zip(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            [20.0, 18.0, 20.5, 17.8, 20.8, 17.6],
            strict=True,
        )
    ]

    result = estimate_causal_range_ttc(
        observations,
        evaluation_timestamp_s=0.5,
    )

    assert result.reason_code == RangeTTCReason.UNSTABLE_TREND
    assert math.isinf(result.ttc_s)


def test_future_poison_cannot_change_a_past_estimate() -> None:
    causal = _observations([0.0, 0.1, 0.2, 0.3, 0.4])
    baseline = estimate_causal_range_ttc(
        causal,
        evaluation_timestamp_s=0.4,
        safety_buffer_m=4.5,
    )
    poisoned = estimate_causal_range_ttc(
        [
            *causal,
            RangeObservation(0.41, 1_000_000.0),
            RangeObservation(1.00, 0.01),
        ],
        evaluation_timestamp_s=0.4,
        safety_buffer_m=4.5,
    )

    assert poisoned == baseline


def test_stateful_predict_does_not_append_synthetic_range() -> None:
    estimator = CausalRangeTTCEstimator(
        RangeTTCConfig(max_extrapolation_s=0.5)
    )
    result = None
    for observation in _observations([0.0, 0.1, 0.2, 0.3]):
        result = estimator.update(
            timestamp_s=observation.timestamp_s,
            range_m=observation.range_m,
            safety_buffer_m=4.5,
        )

    assert result is not None and result.reason_code == RangeTTCReason.OK
    history_before = estimator.history
    coast = estimator.predict(timestamp_s=0.4, safety_buffer_m=4.5)
    assert estimator.history == history_before
    assert coast.reason_code == RangeTTCReason.OK
    assert coast.ttc_s < result.ttc_s


def test_stale_coast_returns_reason_instead_of_unbounded_extrapolation() -> None:
    estimator = CausalRangeTTCEstimator(
        RangeTTCConfig(max_extrapolation_s=0.2)
    )
    for observation in _observations([0.0, 0.1, 0.2, 0.3]):
        estimator.update(
            timestamp_s=observation.timestamp_s,
            range_m=observation.range_m,
        )

    stale = estimator.predict(timestamp_s=0.51)

    assert stale.reason_code == RangeTTCReason.STALE_HISTORY
    assert math.isinf(stale.ttc_s)


def test_stateful_estimator_rejects_noncausal_update_order() -> None:
    estimator = CausalRangeTTCEstimator()
    estimator.update(timestamp_s=0.1, range_m=20.0)

    with pytest.raises(ValueError, match="increase strictly"):
        estimator.update(timestamp_s=0.1, range_m=19.0)
