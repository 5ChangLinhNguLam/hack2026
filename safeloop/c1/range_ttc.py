"""Causal robust range-rate and time-to-collision estimation.

The estimator intentionally accepts only timestamped monocular range estimates.
It has no dependency on depth, labels, events, or challenge targets.  A robust
line is fitted in a numerically centred time coordinate using a Theil--Sen
initial estimate, a MAD residual gate, and Huber iteratively reweighted least
squares (IRLS).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Sequence

import numpy as np


class RangeTTCReason:
    """Stable, serialization-friendly reason codes returned by the estimator."""

    OK = "ok"
    INSUFFICIENT_HISTORY = "insufficient_history"
    INSUFFICIENT_TIME_SPAN = "insufficient_time_span"
    NON_CLOSING = "non_closing"
    UNSTABLE_TREND = "unstable_trend"
    STALE_HISTORY = "stale_history"
    WITHIN_SAFETY_BUFFER = "within_safety_buffer"


@dataclass(frozen=True)
class RangeObservation:
    """One causal range observation in SI units."""

    timestamp_s: float
    range_m: float


@dataclass(frozen=True)
class RangeTTCConfig:
    """Configuration for :func:`estimate_causal_range_ttc`."""

    history_size: int = 12
    min_samples: int = 4
    min_time_span_s: float = 0.10
    min_pair_dt_s: float = 1e-4
    mad_gate_sigma: float = 3.5
    min_residual_gate_m: float = 0.20
    min_inlier_fraction: float = 0.60
    huber_delta_sigma: float = 1.5
    huber_iterations: int = 12
    min_closing_speed_mps: float = 0.30
    min_closing_fraction: float = 0.60
    max_relative_closing_uncertainty: float = 1.0
    max_range_uncertainty_m: float = 5.0
    max_extrapolation_s: float = 0.50

    def __post_init__(self) -> None:
        if self.history_size < 2:
            raise ValueError("history_size must be at least 2")
        if not 2 <= self.min_samples <= self.history_size:
            raise ValueError("min_samples must be in [2, history_size]")
        if self.min_time_span_s <= 0.0 or self.min_pair_dt_s <= 0.0:
            raise ValueError("time-span thresholds must be positive")
        if self.mad_gate_sigma <= 0.0 or self.min_residual_gate_m <= 0.0:
            raise ValueError("MAD gate parameters must be positive")
        if not 0.0 < self.min_inlier_fraction <= 1.0:
            raise ValueError("min_inlier_fraction must be in (0, 1]")
        if self.huber_delta_sigma <= 0.0 or self.huber_iterations < 1:
            raise ValueError("Huber parameters must be positive")
        if self.min_closing_speed_mps < 0.0:
            raise ValueError("min_closing_speed_mps must be non-negative")
        if not 0.0 <= self.min_closing_fraction <= 1.0:
            raise ValueError("min_closing_fraction must be in [0, 1]")
        if self.max_relative_closing_uncertainty < 0.0:
            raise ValueError("max_relative_closing_uncertainty must be non-negative")
        if self.max_range_uncertainty_m <= 0.0:
            raise ValueError("max_range_uncertainty_m must be positive")
        if self.max_extrapolation_s < 0.0:
            raise ValueError("max_extrapolation_s must be non-negative")


@dataclass(frozen=True)
class RangeTTCResult:
    """Robust range-rate estimate at one causal evaluation timestamp."""

    evaluation_timestamp_s: float
    range_m: float
    closing_speed_mps: float
    ttc_s: float
    range_uncertainty_m: float
    closing_speed_uncertainty_mps: float
    ttc_uncertainty_s: float
    reason_code: str
    sample_count: int
    inlier_count: int
    outlier_count: int
    last_observation_age_s: float

    @property
    def reliable(self) -> bool:
        return self.reason_code in {
            RangeTTCReason.OK,
            RangeTTCReason.WITHIN_SAFETY_BUFFER,
        }

    @property
    def stable(self) -> bool:
        """Alias used by selector/fallback code when gating a fitted trend."""

        return self.reliable

    @property
    def accepted_count(self) -> int:
        """Number of samples accepted by the MAD outlier gate."""

        return self.inlier_count


def _invalid_result(
    timestamp_s: float,
    reason_code: str,
    *,
    sample_count: int,
    range_m: float = float("nan"),
    closing_speed_mps: float = 0.0,
    range_uncertainty_m: float = float("inf"),
    closing_speed_uncertainty_mps: float = float("inf"),
    inlier_count: int = 0,
    last_observation_age_s: float = float("inf"),
) -> RangeTTCResult:
    return RangeTTCResult(
        evaluation_timestamp_s=timestamp_s,
        range_m=range_m,
        closing_speed_mps=closing_speed_mps,
        ttc_s=float("inf"),
        range_uncertainty_m=range_uncertainty_m,
        closing_speed_uncertainty_mps=closing_speed_uncertainty_mps,
        ttc_uncertainty_s=float("inf"),
        reason_code=reason_code,
        sample_count=sample_count,
        inlier_count=inlier_count,
        outlier_count=max(0, sample_count - inlier_count),
        last_observation_age_s=last_observation_age_s,
    )


def _median_absolute_deviation(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _pairwise_slopes(
    times_s: np.ndarray,
    ranges_m: np.ndarray,
    *,
    min_pair_dt_s: float,
) -> np.ndarray:
    slopes: list[float] = []
    for left in range(times_s.size - 1):
        delta_t = times_s[left + 1 :] - times_s[left]
        valid = delta_t > min_pair_dt_s
        if np.any(valid):
            delta_range = ranges_m[left + 1 :] - ranges_m[left]
            slopes.extend((delta_range[valid] / delta_t[valid]).tolist())
    return np.asarray(slopes, dtype=np.float64)


def _weighted_line(
    centred_times_s: np.ndarray,
    ranges_m: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float] | None:
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return None
    weighted_time = float(np.sum(weights * centred_times_s) / weight_sum)
    weighted_range = float(np.sum(weights * ranges_m) / weight_sum)
    time_delta = centred_times_s - weighted_time
    denominator = float(np.sum(weights * time_delta * time_delta))
    if denominator <= np.finfo(np.float64).eps:
        return None
    slope = float(
        np.sum(weights * time_delta * (ranges_m - weighted_range)) / denominator
    )
    intercept = weighted_range - slope * weighted_time
    return intercept, slope


def _deduplicate_observations(
    observations: Sequence[RangeObservation],
    *,
    evaluation_timestamp_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Invalid values are measurements with no physical meaning, not zero-range
    # collision evidence.  Ignore them before counting usable history.
    causal = sorted(
        (
            (float(item.timestamp_s), float(item.range_m))
            for item in observations
            if math.isfinite(item.timestamp_s)
            and math.isfinite(item.range_m)
            and item.range_m > 0.0
            and item.timestamp_s <= evaluation_timestamp_s
        ),
        key=lambda item: item[0],
    )
    if not causal:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    if all(
        causal[index - 1][0] != causal[index][0]
        for index in range(1, len(causal))
    ):
        # Stateful runtime histories are strictly increasing.  Avoid hundreds
        # of tiny NumPy median calls while preserving duplicate handling for
        # the public stateless API.
        return (
            np.fromiter((row[0] for row in causal), dtype=np.float64),
            np.fromiter((row[1] for row in causal), dtype=np.float64),
        )

    unique_times: list[float] = []
    unique_ranges: list[float] = []
    index = 0
    while index < len(causal):
        timestamp = causal[index][0]
        same_time: list[float] = []
        while index < len(causal) and causal[index][0] == timestamp:
            same_time.append(causal[index][1])
            index += 1
        unique_times.append(timestamp)
        unique_ranges.append(float(np.median(same_time)))
    return (
        np.asarray(unique_times, dtype=np.float64),
        np.asarray(unique_ranges, dtype=np.float64),
    )


def estimate_causal_range_ttc(
    observations: Sequence[RangeObservation],
    *,
    evaluation_timestamp_s: float,
    safety_buffer_m: float = 0.0,
    config: RangeTTCConfig | None = None,
) -> RangeTTCResult:
    """Estimate range rate and TTC using observations available by ``evaluation``.

    Observations later than ``evaluation_timestamp_s`` are unconditionally
    excluded.  This explicit boundary makes offline replay unable to leak a
    future sample into a past prediction.
    """

    cfg = config or RangeTTCConfig()
    evaluation_timestamp_s = float(evaluation_timestamp_s)
    safety_buffer_m = float(safety_buffer_m)
    if not math.isfinite(evaluation_timestamp_s):
        raise ValueError("evaluation_timestamp_s must be finite")
    if not math.isfinite(safety_buffer_m) or safety_buffer_m < 0.0:
        raise ValueError("safety_buffer_m must be finite and non-negative")

    times_s, ranges_m = _deduplicate_observations(
        observations,
        evaluation_timestamp_s=evaluation_timestamp_s,
    )
    if times_s.size > cfg.history_size:
        times_s = times_s[-cfg.history_size :]
        ranges_m = ranges_m[-cfg.history_size :]
    sample_count = int(times_s.size)
    if sample_count < cfg.min_samples:
        return _invalid_result(
            evaluation_timestamp_s,
            RangeTTCReason.INSUFFICIENT_HISTORY,
            sample_count=sample_count,
        )

    last_age_s = max(0.0, evaluation_timestamp_s - float(times_s[-1]))
    time_span_s = float(times_s[-1] - times_s[0])
    if time_span_s < cfg.min_time_span_s:
        return _invalid_result(
            evaluation_timestamp_s,
            RangeTTCReason.INSUFFICIENT_TIME_SPAN,
            sample_count=sample_count,
            range_m=float(ranges_m[-1]),
            last_observation_age_s=last_age_s,
        )

    pairwise_slopes = _pairwise_slopes(
        times_s,
        ranges_m,
        min_pair_dt_s=cfg.min_pair_dt_s,
    )
    if pairwise_slopes.size == 0:
        return _invalid_result(
            evaluation_timestamp_s,
            RangeTTCReason.INSUFFICIENT_TIME_SPAN,
            sample_count=sample_count,
            range_m=float(ranges_m[-1]),
            last_observation_age_s=last_age_s,
        )

    # Centre at the evaluation time so the fitted intercept is directly the
    # range prediction used for TTC and remains well-conditioned for epoch time.
    centred_times_s = times_s - evaluation_timestamp_s
    initial_slope = float(np.median(pairwise_slopes))
    initial_intercept = float(
        np.median(ranges_m - initial_slope * centred_times_s)
    )
    initial_residuals = ranges_m - (
        initial_intercept + initial_slope * centred_times_s
    )
    residual_center = float(np.median(initial_residuals))
    residual_mad = _median_absolute_deviation(initial_residuals)
    residual_sigma = 1.4826 * residual_mad
    gate_m = max(cfg.min_residual_gate_m, cfg.mad_gate_sigma * residual_sigma)
    inlier_mask = np.abs(initial_residuals - residual_center) <= gate_m
    inlier_count = int(np.count_nonzero(inlier_mask))
    inlier_fraction = inlier_count / sample_count
    if inlier_count < cfg.min_samples or inlier_fraction < cfg.min_inlier_fraction:
        return _invalid_result(
            evaluation_timestamp_s,
            RangeTTCReason.UNSTABLE_TREND,
            sample_count=sample_count,
            range_m=initial_intercept,
            closing_speed_mps=-initial_slope,
            inlier_count=inlier_count,
            last_observation_age_s=last_age_s,
        )

    fit_times = centred_times_s[inlier_mask]
    fit_ranges = ranges_m[inlier_mask]
    weights = np.ones(inlier_count, dtype=np.float64)
    fitted = _weighted_line(fit_times, fit_ranges, weights)
    if fitted is None:
        return _invalid_result(
            evaluation_timestamp_s,
            RangeTTCReason.INSUFFICIENT_TIME_SPAN,
            sample_count=sample_count,
            range_m=initial_intercept,
            inlier_count=inlier_count,
            last_observation_age_s=last_age_s,
        )

    intercept, slope = fitted
    for _ in range(cfg.huber_iterations):
        residuals = fit_ranges - (intercept + slope * fit_times)
        centered_residuals = residuals - float(np.median(residuals))
        robust_sigma = max(
            1.4826 * _median_absolute_deviation(centered_residuals),
            np.finfo(np.float64).eps,
        )
        huber_delta = cfg.huber_delta_sigma * robust_sigma
        absolute_residuals = np.abs(centered_residuals)
        weights = np.ones_like(absolute_residuals)
        outside = absolute_residuals > huber_delta
        weights[outside] = huber_delta / absolute_residuals[outside]
        updated = _weighted_line(fit_times, fit_ranges, weights)
        if updated is None:
            break
        new_intercept, new_slope = updated
        converged = (
            abs(new_intercept - intercept) <= 1e-8
            and abs(new_slope - slope) <= 1e-8
        )
        intercept, slope = new_intercept, new_slope
        if converged:
            break

    residuals = fit_ranges - (intercept + slope * fit_times)
    residual_sigma = 1.4826 * _median_absolute_deviation(residuals)
    weighted_count = max(float(np.sum(weights)), 1.0)
    weighted_time = float(np.sum(weights * fit_times) / weighted_count)
    time_spread = float(np.sum(weights * (fit_times - weighted_time) ** 2))
    leverage = 1.0 / weighted_count
    if time_spread > np.finfo(np.float64).eps:
        leverage += weighted_time * weighted_time / time_spread
    range_uncertainty_m = float(
        max(residual_sigma, np.finfo(np.float64).eps) * math.sqrt(1.0 + leverage)
    )

    inlier_slopes = _pairwise_slopes(
        times_s[inlier_mask],
        ranges_m[inlier_mask],
        min_pair_dt_s=cfg.min_pair_dt_s,
    )
    slope_mad = _median_absolute_deviation(inlier_slopes)
    closing_speed_uncertainty_mps = float(
        1.4826 * slope_mad / math.sqrt(max(1, inlier_count))
    )
    closing_speed_mps = -float(slope)
    predicted_range_m = float(intercept)

    closing_samples = -inlier_slopes
    closing_fraction = float(
        np.mean(closing_samples >= cfg.min_closing_speed_mps)
    )
    relative_closing_uncertainty = closing_speed_uncertainty_mps / max(
        closing_speed_mps,
        cfg.min_closing_speed_mps,
        np.finfo(np.float64).eps,
    )
    # A coherent receding/static fit is not an unstable closing fit.  Preserve
    # that distinction in the reason code before checking directional support.
    if (
        closing_speed_mps < cfg.min_closing_speed_mps
        and relative_closing_uncertainty
        <= cfg.max_relative_closing_uncertainty
        and range_uncertainty_m <= cfg.max_range_uncertainty_m
    ):
        return _invalid_result(
            evaluation_timestamp_s,
            RangeTTCReason.NON_CLOSING,
            sample_count=sample_count,
            range_m=predicted_range_m,
            closing_speed_mps=closing_speed_mps,
            range_uncertainty_m=range_uncertainty_m,
            closing_speed_uncertainty_mps=closing_speed_uncertainty_mps,
            inlier_count=inlier_count,
            last_observation_age_s=last_age_s,
        )
    unstable = (
        closing_fraction < cfg.min_closing_fraction
        or relative_closing_uncertainty > cfg.max_relative_closing_uncertainty
        or range_uncertainty_m > cfg.max_range_uncertainty_m
    )
    if unstable:
        return _invalid_result(
            evaluation_timestamp_s,
            RangeTTCReason.UNSTABLE_TREND,
            sample_count=sample_count,
            range_m=predicted_range_m,
            closing_speed_mps=closing_speed_mps,
            range_uncertainty_m=range_uncertainty_m,
            closing_speed_uncertainty_mps=closing_speed_uncertainty_mps,
            inlier_count=inlier_count,
            last_observation_age_s=last_age_s,
        )
    if last_age_s > cfg.max_extrapolation_s:
        return _invalid_result(
            evaluation_timestamp_s,
            RangeTTCReason.STALE_HISTORY,
            sample_count=sample_count,
            range_m=predicted_range_m,
            closing_speed_mps=closing_speed_mps,
            range_uncertainty_m=range_uncertainty_m,
            closing_speed_uncertainty_mps=closing_speed_uncertainty_mps,
            inlier_count=inlier_count,
            last_observation_age_s=last_age_s,
        )

    clearance_m = predicted_range_m - safety_buffer_m
    if clearance_m <= 0.0:
        # TTC is clamped at contact, but range noise still makes the contact
        # time uncertain.  Do not report a misleading zero uncertainty.
        contact_uncertainty_s = range_uncertainty_m / closing_speed_mps
        return RangeTTCResult(
            evaluation_timestamp_s=evaluation_timestamp_s,
            range_m=predicted_range_m,
            closing_speed_mps=closing_speed_mps,
            ttc_s=0.0,
            range_uncertainty_m=range_uncertainty_m,
            closing_speed_uncertainty_mps=closing_speed_uncertainty_mps,
            ttc_uncertainty_s=contact_uncertainty_s,
            reason_code=RangeTTCReason.WITHIN_SAFETY_BUFFER,
            sample_count=sample_count,
            inlier_count=inlier_count,
            outlier_count=sample_count - inlier_count,
            last_observation_age_s=last_age_s,
        )

    ttc_s = clearance_m / closing_speed_mps
    ttc_uncertainty_s = math.hypot(
        range_uncertainty_m / closing_speed_mps,
        clearance_m
        * closing_speed_uncertainty_mps
        / (closing_speed_mps * closing_speed_mps),
    )
    return RangeTTCResult(
        evaluation_timestamp_s=evaluation_timestamp_s,
        range_m=predicted_range_m,
        closing_speed_mps=closing_speed_mps,
        ttc_s=ttc_s,
        range_uncertainty_m=range_uncertainty_m,
        closing_speed_uncertainty_mps=closing_speed_uncertainty_mps,
        ttc_uncertainty_s=ttc_uncertainty_s,
        reason_code=RangeTTCReason.OK,
        sample_count=sample_count,
        inlier_count=inlier_count,
        outlier_count=sample_count - inlier_count,
        last_observation_age_s=last_age_s,
    )


def project_range_ttc_result(
    anchor: RangeTTCResult,
    *,
    evaluation_timestamp_s: float,
    safety_buffer_m: float = 0.0,
    config: RangeTTCConfig | None = None,
) -> RangeTTCResult:
    """Project an unchanged robust fit to a later causal timestamp in O(1).

    The robust slope and residual statistics only change when a real range
    observation is appended.  Between detector updates this helper advances
    the fitted range, propagates slope uncertainty, and applies the same
    safety-buffer/staleness semantics without refitting identical history.
    """

    cfg = config or RangeTTCConfig()
    timestamp = float(evaluation_timestamp_s)
    safety_buffer = float(safety_buffer_m)
    if not math.isfinite(timestamp):
        raise ValueError("evaluation_timestamp_s must be finite")
    if not math.isfinite(safety_buffer) or safety_buffer < 0.0:
        raise ValueError("safety_buffer_m must be finite and non-negative")
    elapsed = timestamp - float(anchor.evaluation_timestamp_s)
    if elapsed < 0.0:
        raise ValueError("projection timestamp cannot precede the fit anchor")
    if elapsed == 0.0:
        return anchor

    last_age = float(anchor.last_observation_age_s) + elapsed
    closing_speed = float(anchor.closing_speed_mps)
    range_m = float(anchor.range_m)
    if math.isfinite(range_m) and math.isfinite(closing_speed):
        range_m -= closing_speed * elapsed
    range_uncertainty = float(anchor.range_uncertainty_m)
    closing_uncertainty = float(anchor.closing_speed_uncertainty_mps)
    if math.isfinite(range_uncertainty) and math.isfinite(closing_uncertainty):
        range_uncertainty = math.hypot(
            range_uncertainty, elapsed * closing_uncertainty
        )

    reason = anchor.reason_code
    ttc_s = float("inf")
    ttc_uncertainty = float("inf")
    if reason in {RangeTTCReason.OK, RangeTTCReason.WITHIN_SAFETY_BUFFER}:
        if last_age > cfg.max_extrapolation_s:
            reason = RangeTTCReason.STALE_HISTORY
        elif not math.isfinite(closing_speed) or closing_speed <= 0.0:
            reason = RangeTTCReason.NON_CLOSING
        else:
            clearance = range_m - safety_buffer
            if clearance <= 0.0:
                reason = RangeTTCReason.WITHIN_SAFETY_BUFFER
                ttc_s = 0.0
                ttc_uncertainty = range_uncertainty / closing_speed
            else:
                reason = RangeTTCReason.OK
                ttc_s = clearance / closing_speed
                ttc_uncertainty = math.hypot(
                    range_uncertainty / closing_speed,
                    clearance
                    * closing_uncertainty
                    / (closing_speed * closing_speed),
                )

    return RangeTTCResult(
        evaluation_timestamp_s=timestamp,
        range_m=range_m,
        closing_speed_mps=closing_speed,
        ttc_s=ttc_s,
        range_uncertainty_m=range_uncertainty,
        closing_speed_uncertainty_mps=closing_uncertainty,
        ttc_uncertainty_s=ttc_uncertainty,
        reason_code=reason,
        sample_count=anchor.sample_count,
        inlier_count=anchor.inlier_count,
        outlier_count=anchor.outlier_count,
        last_observation_age_s=last_age,
    )


class CausalRangeTTCEstimator:
    """Small stateful wrapper that retains only a bounded causal history."""

    def __init__(self, config: RangeTTCConfig | None = None):
        self.config = config or RangeTTCConfig()
        self._history: Deque[RangeObservation] = deque(
            maxlen=self.config.history_size
        )

    @property
    def history(self) -> tuple[RangeObservation, ...]:
        return tuple(self._history)

    def reset(self) -> None:
        self._history.clear()

    def observe(self, *, timestamp_s: float, range_m: float) -> None:
        """Append one real measurement without fitting the history.

        Runtime policies that maintain many candidate tracks can retain every
        causal observation cheaply, then evaluate only the selected target.
        ``predict`` remains side-effect free and never invents a coast sample.
        """

        timestamp_s = float(timestamp_s)
        range_m = float(range_m)
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if not math.isfinite(range_m) or range_m <= 0.0:
            raise ValueError("range_m must be finite and positive")
        if self._history and timestamp_s <= self._history[-1].timestamp_s:
            raise ValueError("timestamps must increase strictly")
        self._history.append(RangeObservation(timestamp_s, range_m))

    def update(
        self,
        *,
        timestamp_s: float,
        range_m: float,
        safety_buffer_m: float = 0.0,
    ) -> RangeTTCResult:
        self.observe(timestamp_s=timestamp_s, range_m=range_m)
        return estimate_causal_range_ttc(
            self.history,
            evaluation_timestamp_s=timestamp_s,
            safety_buffer_m=safety_buffer_m,
            config=self.config,
        )

    def predict(
        self,
        *,
        timestamp_s: float,
        safety_buffer_m: float = 0.0,
    ) -> RangeTTCResult:
        """Extrapolate without appending a synthetic observation."""

        return estimate_causal_range_ttc(
            self.history,
            evaluation_timestamp_s=timestamp_s,
            safety_buffer_m=safety_buffer_m,
            config=self.config,
        )
