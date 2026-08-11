"""Causal, deployable monocular range/TTC estimators for C1 P2-C.

This module is deliberately independent from the official evaluator and from
all label-bearing dataset records.  Runtime inputs are limited to camera
intrinsics, the current/previous detector boxes, causal lane geometry, a
track's class posterior, and real timestamps.

The estimators are small NumPy-only components:

* ground-plane range from a bbox footpoint and a fixed or causal lane horizon;
* class-size-prior range from the complete class posterior (never a raw label);
* scale-expansion TTC from robust log-height/log-sqrt-area trends; and
* non-learned uncertainty fusion in inverse-TTC space.

Ground-plane and class-prior range observations can be passed to the existing
``CausalRangeTTCEstimator`` to obtain independent range-rate TTC signals.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Mapping, Sequence

import numpy as np

from .lane import LaneEstimate
from .types import BBox


class P2CReason:
    """Stable reason codes shared by the P2-C estimator boundary."""

    OK = "ok"
    FIXED_HORIZON = "fixed_horizon"
    INSUFFICIENT_HISTORY = "insufficient_history"
    INSUFFICIENT_TIME_SPAN = "insufficient_time_span"
    INVALID_BBOX = "invalid_bbox"
    FOOTPOINT_ABOVE_HORIZON = "footpoint_above_horizon"
    FOOTPOINT_TRUNCATED = "footpoint_truncated"
    RANGE_OUT_OF_BOUNDS = "range_out_of_bounds"
    HIGH_RANGE_UNCERTAINTY = "high_range_uncertainty"
    EMPTY_CLASS_POSTERIOR = "empty_class_posterior"
    BOX_TOO_SMALL = "box_too_small"
    NON_EXPANDING = "non_expanding"
    UNSTABLE_SCALE = "unstable_scale"
    STALE_HISTORY = "stale_history"
    TTC_OUT_OF_BOUNDS = "ttc_out_of_bounds"
    NO_RELIABLE_INPUT = "no_reliable_input"
    HIGH_FUSION_UNCERTAINTY = "high_fusion_uncertainty"
    FUSED = "fused"
    FALLBACK = "fallback"
    HELD_PREVIOUS = "held_previous"
    JUMP_LIMITED = "jump_limited"
    SCALE_SAFETY_VETO = "scale_safety_veto"


def _finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def _finite_nonnegative(value: float) -> bool:
    return math.isfinite(value) and value >= 0.0


def _valid_bbox(bbox: BBox) -> bool:
    return (
        len(bbox) == 4
        and all(math.isfinite(float(value)) for value in bbox)
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    )


@dataclass(frozen=True, slots=True)
class GroundPlaneConfig:
    """Global camera/uncertainty assumptions for footpoint ranging.

    Camera height and pitch are not present in ``calibration_info.txt``.  They
    therefore remain explicit, global configuration values rather than hidden
    per-trip constants.  ``principal_y_px`` is supplied by the caller from
    ``K_left[1][2]``; a zero pitch uses it as the fixed horizon.
    """

    camera_height_m: float = 1.55
    camera_height_uncertainty_m: float = 0.15
    fixed_pitch_down_deg: float = 0.0
    fixed_horizon_uncertainty_fraction: float = 0.045
    minimum_lane_confidence: float = 0.60
    lane_horizon_uncertainty_fraction: float = 0.020
    inferred_lane_uncertainty_scale: float = 1.60
    minimum_footpoint_uncertainty_px: float = 2.0
    bbox_height_uncertainty_fraction: float = 0.045
    focal_uncertainty_fraction: float = 0.01
    minimum_horizon_gap_px: float = 4.0
    minimum_horizon_fraction: float = 0.12
    maximum_horizon_fraction: float = 0.72
    maximum_range_m: float = 120.0
    maximum_relative_uncertainty: float = 0.65
    bottom_truncation_margin_px: float = 1.0
    bottom_truncation_uncertainty_scale: float = 2.50

    def __post_init__(self) -> None:
        positive = {
            "camera_height_m": self.camera_height_m,
            "camera_height_uncertainty_m": self.camera_height_uncertainty_m,
            "fixed_horizon_uncertainty_fraction": (
                self.fixed_horizon_uncertainty_fraction
            ),
            "lane_horizon_uncertainty_fraction": (
                self.lane_horizon_uncertainty_fraction
            ),
            "inferred_lane_uncertainty_scale": (
                self.inferred_lane_uncertainty_scale
            ),
            "minimum_footpoint_uncertainty_px": (
                self.minimum_footpoint_uncertainty_px
            ),
            "bbox_height_uncertainty_fraction": (
                self.bbox_height_uncertainty_fraction
            ),
            "focal_uncertainty_fraction": self.focal_uncertainty_fraction,
            "minimum_horizon_gap_px": self.minimum_horizon_gap_px,
            "maximum_range_m": self.maximum_range_m,
            "bottom_truncation_uncertainty_scale": (
                self.bottom_truncation_uncertainty_scale
            ),
        }
        if any(not _finite_positive(value) for value in positive.values()):
            raise ValueError("ground-plane scales must be finite and positive")
        probabilities = (
            self.minimum_lane_confidence,
            self.minimum_horizon_fraction,
            self.maximum_horizon_fraction,
            self.maximum_relative_uncertainty,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("ground-plane fractions must be in [0, 1]")
        if self.maximum_horizon_fraction <= self.minimum_horizon_fraction:
            raise ValueError("maximum horizon fraction must exceed minimum")
        if not math.isfinite(self.fixed_pitch_down_deg):
            raise ValueError("fixed pitch must be finite")
        if self.bottom_truncation_margin_px < 0.0:
            raise ValueError("bottom truncation margin must be non-negative")


@dataclass(frozen=True, slots=True)
class GroundPlaneRangeResult:
    timestamp_s: float
    range_m: float
    range_uncertainty_m: float
    footpoint_y_px: float
    horizon_y_px: float
    horizon_source: str
    reason_code: str

    @property
    def reliable(self) -> bool:
        return self.reason_code in {P2CReason.OK, P2CReason.FIXED_HORIZON}


@dataclass(frozen=True, slots=True)
class GroundPlaneHorizon:
    y_px: float
    uncertainty_px: float
    source: str


def _invalid_ground_result(
    timestamp_s: float,
    reason_code: str,
    *,
    footpoint_y_px: float = float("nan"),
    horizon_y_px: float = float("nan"),
    horizon_source: str = "invalid",
    range_m: float = float("nan"),
    uncertainty_m: float = float("inf"),
) -> GroundPlaneRangeResult:
    return GroundPlaneRangeResult(
        timestamp_s=float(timestamp_s),
        range_m=float(range_m),
        range_uncertainty_m=float(uncertainty_m),
        footpoint_y_px=float(footpoint_y_px),
        horizon_y_px=float(horizon_y_px),
        horizon_source=horizon_source,
        reason_code=reason_code,
    )


def _fit_x_from_y(points: Sequence[tuple[int, int]]) -> tuple[float, float] | None:
    if len(points) < 2:
        return None
    y = np.asarray([float(point[1]) for point in points], dtype=np.float64)
    x = np.asarray([float(point[0]) for point in points], dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    matrix = np.column_stack((y, np.ones_like(y)))
    if np.linalg.matrix_rank(matrix) < 2:
        return None
    slope, intercept = np.linalg.lstsq(matrix, x, rcond=None)[0]
    return float(slope), float(intercept)


def causal_lane_horizon_y(
    lane: LaneEstimate | None,
    *,
    image_height: int,
    config: GroundPlaneConfig | None = None,
) -> tuple[float, float] | None:
    """Return a causal vanishing-row estimate and its pixel uncertainty.

    ``LaneEstimate`` is already current/past-only and temporally smoothed by
    ``LaneDetector``.  No image from a future frame enters this calculation.
    """

    cfg = config or GroundPlaneConfig()
    if (
        lane is None
        or not lane.valid
        or lane.confidence < cfg.minimum_lane_confidence
    ):
        return None
    left = _fit_x_from_y(lane.left_points)
    right = _fit_x_from_y(lane.right_points)
    if left is None or right is None:
        return None
    denominator = left[0] - right[0]
    if abs(denominator) <= 1e-6:
        return None
    horizon_y = (right[1] - left[1]) / denominator
    minimum = cfg.minimum_horizon_fraction * image_height
    maximum = cfg.maximum_horizon_fraction * image_height
    if not math.isfinite(horizon_y) or not minimum <= horizon_y <= maximum:
        return None
    uncertainty = max(
        1.0,
        cfg.lane_horizon_uncertainty_fraction
        * image_height
        / max(lane.confidence, 0.05),
    )
    if lane.inferred_side is not None:
        uncertainty *= cfg.inferred_lane_uncertainty_scale
    return float(horizon_y), float(uncertainty)


def resolve_ground_plane_horizon(
    lane: LaneEstimate | None,
    *,
    image_height: int,
    focal_y_px: float,
    principal_y_px: float,
    config: GroundPlaneConfig | None = None,
) -> GroundPlaneHorizon:
    """Resolve the shared causal lane/fixed horizon once for a video frame."""

    cfg = config or GroundPlaneConfig()
    lane_horizon = causal_lane_horizon_y(
        lane, image_height=image_height, config=cfg
    )
    if lane_horizon is not None:
        return GroundPlaneHorizon(lane_horizon[0], lane_horizon[1], "lane")
    pitch_radians = math.radians(cfg.fixed_pitch_down_deg)
    horizon_y = float(principal_y_px - focal_y_px * math.tan(pitch_radians))
    return GroundPlaneHorizon(
        horizon_y,
        cfg.fixed_horizon_uncertainty_fraction * image_height,
        "fixed",
    )


def estimate_ground_plane_range(
    bbox: BBox,
    *,
    timestamp_s: float,
    image_shape: tuple[int, int],
    focal_y_px: float,
    principal_y_px: float,
    lane: LaneEstimate | None = None,
    resolved_horizon: GroundPlaneHorizon | None = None,
    bbox_footpoint_uncertainty_px: float | None = None,
    config: GroundPlaneConfig | None = None,
) -> GroundPlaneRangeResult:
    """Estimate longitudinal range from the bbox bottom-centre contact row."""

    cfg = config or GroundPlaneConfig()
    timestamp_s = float(timestamp_s)
    if not math.isfinite(timestamp_s):
        raise ValueError("timestamp_s must be finite")
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("image shape must be positive")
    if not _finite_positive(float(focal_y_px)):
        raise ValueError("focal_y_px must be finite and positive")
    if not math.isfinite(float(principal_y_px)):
        raise ValueError("principal_y_px must be finite")
    if not _valid_bbox(bbox):
        return _invalid_ground_result(timestamp_s, P2CReason.INVALID_BBOX)

    foot_y = float(bbox[3])
    box_height = float(bbox[3] - bbox[1])
    horizon = resolved_horizon or resolve_ground_plane_horizon(
        lane,
        image_height=height,
        focal_y_px=float(focal_y_px),
        principal_y_px=float(principal_y_px),
        config=cfg,
    )
    horizon_y = float(horizon.y_px)
    horizon_sigma = float(horizon.uncertainty_px)
    source = horizon.source
    if (
        not math.isfinite(horizon_y)
        or not math.isfinite(horizon_sigma)
        or horizon_sigma < 0.0
        or source not in {"lane", "fixed"}
    ):
        raise ValueError("resolved_horizon must be finite lane/fixed geometry")

    gap_px = foot_y - horizon_y
    if gap_px < cfg.minimum_horizon_gap_px:
        return _invalid_ground_result(
            timestamp_s,
            P2CReason.FOOTPOINT_ABOVE_HORIZON,
            footpoint_y_px=foot_y,
            horizon_y_px=horizon_y,
            horizon_source=source,
        )

    if bbox_footpoint_uncertainty_px is None:
        foot_sigma = max(
            cfg.minimum_footpoint_uncertainty_px,
            cfg.bbox_height_uncertainty_fraction * box_height,
        )
    else:
        foot_sigma = float(bbox_footpoint_uncertainty_px)
        if not math.isfinite(foot_sigma) or foot_sigma < 0.0:
            raise ValueError("bbox footpoint uncertainty must be non-negative")

    truncated = foot_y >= height - cfg.bottom_truncation_margin_px
    if truncated:
        foot_sigma *= cfg.bottom_truncation_uncertainty_scale
    range_m = float(focal_y_px * cfg.camera_height_m / gap_px)
    relative_variance = (
        (cfg.camera_height_uncertainty_m / cfg.camera_height_m) ** 2
        + cfg.focal_uncertainty_fraction**2
        + (math.hypot(foot_sigma, horizon_sigma) / gap_px) ** 2
    )
    uncertainty_m = float(range_m * math.sqrt(relative_variance))
    if not _finite_positive(range_m) or range_m > cfg.maximum_range_m:
        reason = P2CReason.RANGE_OUT_OF_BOUNDS
    elif truncated:
        reason = P2CReason.FOOTPOINT_TRUNCATED
    elif uncertainty_m / range_m > cfg.maximum_relative_uncertainty:
        reason = P2CReason.HIGH_RANGE_UNCERTAINTY
    else:
        reason = P2CReason.OK if source == "lane" else P2CReason.FIXED_HORIZON
    return GroundPlaneRangeResult(
        timestamp_s=timestamp_s,
        range_m=range_m,
        range_uncertainty_m=uncertainty_m,
        footpoint_y_px=foot_y,
        horizon_y_px=horizon_y,
        horizon_source=source,
        reason_code=reason,
    )


@dataclass(frozen=True, slots=True)
class ObjectSizePrior:
    mean_height_m: float
    std_height_m: float

    def __post_init__(self) -> None:
        if not _finite_positive(self.mean_height_m) or not _finite_positive(
            self.std_height_m
        ):
            raise ValueError("object-size prior parameters must be positive")


def _default_size_priors() -> Mapping[str, ObjectSizePrior]:
    # Generic physical dimensions, deliberately not fitted to a trip/frame.
    return {
        "person": ObjectSizePrior(1.70, 0.18),
        "bicycle": ObjectSizePrior(1.65, 0.30),
        "motorcycle": ObjectSizePrior(1.55, 0.32),
        "car": ObjectSizePrior(1.50, 0.25),
        "truck": ObjectSizePrior(3.00, 0.80),
        "bus": ObjectSizePrior(3.20, 0.70),
    }


@dataclass(frozen=True)
class ClassSizePriorConfig:
    priors: Mapping[str, ObjectSizePrior] = field(default_factory=_default_size_priors)
    unknown_prior: ObjectSizePrior = field(
        default_factory=lambda: ObjectSizePrior(1.80, 0.85)
    )
    minimum_box_height_px: float = 8.0
    minimum_bbox_height_uncertainty_px: float = 2.0
    bbox_height_uncertainty_fraction: float = 0.055
    focal_uncertainty_fraction: float = 0.01
    maximum_range_m: float = 120.0
    maximum_relative_uncertainty: float = 0.80

    def __post_init__(self) -> None:
        if not self.priors:
            raise ValueError("at least one class-size prior is required")
        if any(not isinstance(prior, ObjectSizePrior) for prior in self.priors.values()):
            raise TypeError("priors must map labels to ObjectSizePrior")
        positive = (
            self.minimum_box_height_px,
            self.minimum_bbox_height_uncertainty_px,
            self.bbox_height_uncertainty_fraction,
            self.focal_uncertainty_fraction,
            self.maximum_range_m,
            self.maximum_relative_uncertainty,
        )
        if any(not _finite_positive(value) for value in positive):
            raise ValueError("class-prior scales must be finite and positive")


@dataclass(frozen=True, slots=True)
class ClassPriorRangeResult:
    timestamp_s: float
    range_m: float
    range_uncertainty_m: float
    effective_height_m: float
    effective_height_uncertainty_m: float
    posterior_entropy: float
    supported_probability: float
    reason_code: str

    @property
    def reliable(self) -> bool:
        return self.reason_code == P2CReason.OK


def _normalised_probability_items(
    posterior: Mapping[str, float],
) -> tuple[tuple[str, float], ...]:
    values: list[tuple[str, float]] = []
    for label, raw_probability in posterior.items():
        probability = float(raw_probability)
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError("class posterior probabilities must be finite and non-negative")
        if probability > 0.0:
            values.append((str(label), probability))
    total = sum(probability for _, probability in values)
    if total <= 0.0:
        return ()
    return tuple((label, probability / total) for label, probability in values)


def estimate_class_prior_range(
    bbox: BBox,
    class_posterior: Mapping[str, float],
    *,
    timestamp_s: float,
    focal_y_px: float,
    bbox_height_uncertainty_px: float | None = None,
    config: ClassSizePriorConfig | None = None,
) -> ClassPriorRangeResult:
    """Estimate range from a posterior mixture of physical-height priors.

    Mixing all posterior classes avoids abrupt estimator switches when the
    detector alternates compatible labels on successive frames.
    """

    cfg = config or ClassSizePriorConfig()
    timestamp_s = float(timestamp_s)
    if not math.isfinite(timestamp_s):
        raise ValueError("timestamp_s must be finite")
    if not _finite_positive(float(focal_y_px)):
        raise ValueError("focal_y_px must be finite and positive")
    if not _valid_bbox(bbox):
        return ClassPriorRangeResult(
            timestamp_s,
            float("nan"),
            float("inf"),
            float("nan"),
            float("inf"),
            1.0,
            0.0,
            P2CReason.INVALID_BBOX,
        )
    items = _normalised_probability_items(class_posterior)
    if not items:
        return ClassPriorRangeResult(
            timestamp_s,
            float("nan"),
            float("inf"),
            float("nan"),
            float("inf"),
            1.0,
            0.0,
            P2CReason.EMPTY_CLASS_POSTERIOR,
        )
    box_height = float(bbox[3] - bbox[1])
    if box_height < cfg.minimum_box_height_px:
        return ClassPriorRangeResult(
            timestamp_s,
            float("nan"),
            float("inf"),
            float("nan"),
            float("inf"),
            1.0,
            sum(probability for label, probability in items if label in cfg.priors),
            P2CReason.BOX_TOO_SMALL,
        )

    supported_probability = sum(
        probability for label, probability in items if label in cfg.priors
    )
    components = [
        (probability, cfg.priors.get(label, cfg.unknown_prior))
        for label, probability in items
    ]
    height_mean = sum(weight * prior.mean_height_m for weight, prior in components)
    height_second_moment = sum(
        weight * (prior.std_height_m**2 + prior.mean_height_m**2)
        for weight, prior in components
    )
    height_variance = max(0.0, height_second_moment - height_mean**2)
    height_sigma = math.sqrt(height_variance)
    probabilities = np.asarray([probability for _, probability in items], dtype=float)
    entropy = (
        -float(np.sum(probabilities * np.log(probabilities)))
        / math.log(len(probabilities))
        if len(probabilities) > 1
        else 0.0
    )
    if bbox_height_uncertainty_px is None:
        bbox_sigma = max(
            cfg.minimum_bbox_height_uncertainty_px,
            cfg.bbox_height_uncertainty_fraction * box_height,
        )
    else:
        bbox_sigma = float(bbox_height_uncertainty_px)
        if not math.isfinite(bbox_sigma) or bbox_sigma < 0.0:
            raise ValueError("bbox height uncertainty must be non-negative")

    range_m = float(focal_y_px * height_mean / box_height)
    relative_variance = (
        (height_sigma / height_mean) ** 2
        + (bbox_sigma / box_height) ** 2
        + cfg.focal_uncertainty_fraction**2
    )
    uncertainty_m = float(range_m * math.sqrt(relative_variance))
    if not _finite_positive(range_m) or range_m > cfg.maximum_range_m:
        reason = P2CReason.RANGE_OUT_OF_BOUNDS
    elif uncertainty_m / range_m > cfg.maximum_relative_uncertainty:
        reason = P2CReason.HIGH_RANGE_UNCERTAINTY
    else:
        reason = P2CReason.OK
    return ClassPriorRangeResult(
        timestamp_s=timestamp_s,
        range_m=range_m,
        range_uncertainty_m=uncertainty_m,
        effective_height_m=float(height_mean),
        effective_height_uncertainty_m=float(height_sigma),
        posterior_entropy=float(np.clip(entropy, 0.0, 1.0)),
        supported_probability=float(np.clip(supported_probability, 0.0, 1.0)),
        reason_code=reason,
    )


@dataclass(frozen=True, slots=True)
class ScaleObservation:
    timestamp_s: float
    bbox: BBox
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class ScaleExpansionConfig:
    history_size: int = 12
    min_samples: int = 4
    min_time_span_s: float = 0.15
    min_pair_dt_s: float = 1e-4
    minimum_box_height_px: float = 8.0
    mad_gate_sigma: float = 3.5
    minimum_log_residual_gate: float = 0.025
    minimum_inlier_fraction: float = 0.60
    huber_delta_sigma: float = 1.5
    huber_iterations: int = 12
    minimum_inverse_ttc_s: float = 0.08
    maximum_inverse_ttc_uncertainty_s: float = 0.30
    maximum_relative_inverse_uncertainty: float = 0.80
    maximum_ttc_s: float = 10.0
    maximum_extrapolation_s: float = 0.50
    inverse_uncertainty_floor_s: float = 0.01

    def __post_init__(self) -> None:
        if self.history_size < 2:
            raise ValueError("history_size must be at least two")
        if not 2 <= self.min_samples <= self.history_size:
            raise ValueError("min_samples must be in [2, history_size]")
        positive = (
            self.min_time_span_s,
            self.min_pair_dt_s,
            self.minimum_box_height_px,
            self.mad_gate_sigma,
            self.minimum_log_residual_gate,
            self.huber_delta_sigma,
            self.minimum_inverse_ttc_s,
            self.maximum_inverse_ttc_uncertainty_s,
            self.maximum_relative_inverse_uncertainty,
            self.maximum_ttc_s,
            self.inverse_uncertainty_floor_s,
        )
        if any(not _finite_positive(value) for value in positive):
            raise ValueError("scale-expansion thresholds must be positive")
        if not 0.0 < self.minimum_inlier_fraction <= 1.0:
            raise ValueError("minimum_inlier_fraction must be in (0, 1]")
        if self.huber_iterations < 1:
            raise ValueError("huber_iterations must be positive")
        if self.maximum_extrapolation_s < 0.0:
            raise ValueError("maximum_extrapolation_s must be non-negative")


@dataclass(frozen=True, slots=True)
class ScaleFitResult:
    source: str
    inverse_ttc_s: float
    inverse_ttc_uncertainty_s: float
    reason_code: str
    sample_count: int
    inlier_count: int
    outlier_count: int

    @property
    def reliable(self) -> bool:
        return self.reason_code == P2CReason.OK


@dataclass(frozen=True, slots=True)
class ScaleExpansionResult:
    evaluation_timestamp_s: float
    ttc_s: float
    ttc_uncertainty_s: float
    inverse_ttc_s: float
    inverse_ttc_uncertainty_s: float
    reason_code: str
    source: str
    height_fit: ScaleFitResult
    area_fit: ScaleFitResult
    last_observation_age_s: float

    @property
    def reliable(self) -> bool:
        return self.reason_code == P2CReason.OK


def _invalid_scale_fit(source: str, reason: str, sample_count: int = 0) -> ScaleFitResult:
    return ScaleFitResult(
        source=source,
        inverse_ttc_s=0.0,
        inverse_ttc_uncertainty_s=float("inf"),
        reason_code=reason,
        sample_count=sample_count,
        inlier_count=0,
        outlier_count=sample_count,
    )


def _median_absolute_deviation(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _pairwise_slopes(
    times_s: np.ndarray, values: np.ndarray, *, minimum_dt_s: float
) -> np.ndarray:
    slopes: list[float] = []
    for left in range(times_s.size - 1):
        delta_t = times_s[left + 1 :] - times_s[left]
        valid = delta_t > minimum_dt_s
        if np.any(valid):
            slopes.extend(((values[left + 1 :] - values[left])[valid] / delta_t[valid]).tolist())
    return np.asarray(slopes, dtype=np.float64)


def _weighted_line(
    times_s: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float] | None:
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return None
    mean_time = float(np.sum(weights * times_s) / weight_sum)
    mean_value = float(np.sum(weights * values) / weight_sum)
    delta = times_s - mean_time
    denominator = float(np.sum(weights * delta * delta))
    if denominator <= np.finfo(np.float64).eps:
        return None
    slope = float(np.sum(weights * delta * (values - mean_value)) / denominator)
    return mean_value - slope * mean_time, slope


def _robust_log_scale_fit(
    times_s: np.ndarray,
    log_values: np.ndarray,
    confidences: np.ndarray,
    *,
    source: str,
    config: ScaleExpansionConfig,
) -> ScaleFitResult:
    sample_count = int(times_s.size)
    if sample_count < config.min_samples:
        return _invalid_scale_fit(source, P2CReason.INSUFFICIENT_HISTORY, sample_count)
    if float(times_s[-1] - times_s[0]) < config.min_time_span_s:
        return _invalid_scale_fit(source, P2CReason.INSUFFICIENT_TIME_SPAN, sample_count)
    slopes = _pairwise_slopes(
        times_s, log_values, minimum_dt_s=config.min_pair_dt_s
    )
    if slopes.size == 0:
        return _invalid_scale_fit(source, P2CReason.INSUFFICIENT_TIME_SPAN, sample_count)
    initial_slope = float(np.median(slopes))
    initial_intercept = float(np.median(log_values - initial_slope * times_s))
    residuals = log_values - (initial_intercept + initial_slope * times_s)
    residual_center = float(np.median(residuals))
    residual_sigma = 1.4826 * _median_absolute_deviation(residuals)
    gate = max(
        config.minimum_log_residual_gate,
        config.mad_gate_sigma * residual_sigma,
    )
    inliers = np.abs(residuals - residual_center) <= gate
    inlier_count = int(np.count_nonzero(inliers))
    if (
        inlier_count < config.min_samples
        or inlier_count / sample_count < config.minimum_inlier_fraction
    ):
        return ScaleFitResult(
            source,
            initial_slope,
            float("inf"),
            P2CReason.UNSTABLE_SCALE,
            sample_count,
            inlier_count,
            sample_count - inlier_count,
        )

    fit_times = times_s[inliers]
    fit_values = log_values[inliers]
    base_weights = np.clip(confidences[inliers], 0.05, 1.0)
    weights = base_weights.copy()
    fitted = _weighted_line(fit_times, fit_values, weights)
    if fitted is None:
        return _invalid_scale_fit(source, P2CReason.INSUFFICIENT_TIME_SPAN, sample_count)
    intercept, slope = fitted
    for _ in range(config.huber_iterations):
        residuals = fit_values - (intercept + slope * fit_times)
        centered = residuals - float(np.median(residuals))
        sigma = max(
            1.4826 * _median_absolute_deviation(centered),
            np.finfo(np.float64).eps,
        )
        delta = config.huber_delta_sigma * sigma
        robust_weights = np.ones_like(centered)
        outside = np.abs(centered) > delta
        robust_weights[outside] = delta / np.abs(centered[outside])
        weights = base_weights * robust_weights
        updated = _weighted_line(fit_times, fit_values, weights)
        if updated is None:
            break
        new_intercept, new_slope = updated
        converged = abs(new_intercept - intercept) <= 1e-9 and abs(new_slope - slope) <= 1e-9
        intercept, slope = new_intercept, new_slope
        if converged:
            break

    inlier_slopes = _pairwise_slopes(
        fit_times, fit_values, minimum_dt_s=config.min_pair_dt_s
    )
    slope_sigma = float(
        1.4826
        * _median_absolute_deviation(inlier_slopes)
        / math.sqrt(max(1, inlier_count))
    )
    slope_sigma = max(config.inverse_uncertainty_floor_s, slope_sigma)
    relative_uncertainty = slope_sigma / max(
        slope, config.minimum_inverse_ttc_s
    )
    if slope < config.minimum_inverse_ttc_s:
        reason = P2CReason.NON_EXPANDING
    elif (
        slope_sigma > config.maximum_inverse_ttc_uncertainty_s
        or relative_uncertainty > config.maximum_relative_inverse_uncertainty
    ):
        reason = P2CReason.UNSTABLE_SCALE
    else:
        reason = P2CReason.OK
    return ScaleFitResult(
        source=source,
        inverse_ttc_s=float(slope),
        inverse_ttc_uncertainty_s=slope_sigma,
        reason_code=reason,
        sample_count=sample_count,
        inlier_count=inlier_count,
        outlier_count=sample_count - inlier_count,
    )


def _deduplicate_scale_observations(
    observations: Sequence[ScaleObservation],
    *,
    evaluation_timestamp_s: float,
    config: ScaleExpansionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    usable: list[tuple[float, float, float, float]] = []
    for observation in observations:
        timestamp = float(observation.timestamp_s)
        if (
            not math.isfinite(timestamp)
            or timestamp > evaluation_timestamp_s
            or not _valid_bbox(observation.bbox)
        ):
            continue
        x1, y1, x2, y2 = observation.bbox
        width, height = float(x2 - x1), float(y2 - y1)
        if height < config.minimum_box_height_px:
            continue
        confidence = float(observation.confidence)
        if not math.isfinite(confidence) or confidence < 0.0:
            continue
        usable.append(
            (
                timestamp,
                math.log(height),
                math.log(math.sqrt(width * height)),
                float(np.clip(confidence, 0.05, 1.0)),
            )
        )
    usable.sort(key=lambda item: item[0])
    if usable and all(
        usable[index - 1][0] != usable[index][0]
        for index in range(1, len(usable))
    ):
        rows = usable[-config.history_size :]
        return tuple(
            np.fromiter((row[column] for row in rows), dtype=np.float64)
            for column in range(4)
        )  # type: ignore[return-value]
    times: list[float] = []
    log_heights: list[float] = []
    log_areas: list[float] = []
    confidences: list[float] = []
    index = 0
    while index < len(usable):
        timestamp = usable[index][0]
        rows: list[tuple[float, float, float]] = []
        while index < len(usable) and usable[index][0] == timestamp:
            rows.append(usable[index][1:])
            index += 1
        times.append(timestamp)
        log_heights.append(float(np.median([row[0] for row in rows])))
        log_areas.append(float(np.median([row[1] for row in rows])))
        confidences.append(float(np.median([row[2] for row in rows])))
    if len(times) > config.history_size:
        times = times[-config.history_size :]
        log_heights = log_heights[-config.history_size :]
        log_areas = log_areas[-config.history_size :]
        confidences = confidences[-config.history_size :]
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(log_heights, dtype=np.float64),
        np.asarray(log_areas, dtype=np.float64),
        np.asarray(confidences, dtype=np.float64),
    )


def estimate_causal_scale_expansion_ttc(
    observations: Sequence[ScaleObservation],
    *,
    evaluation_timestamp_s: float,
    config: ScaleExpansionConfig | None = None,
) -> ScaleExpansionResult:
    """Estimate TTC from robust causal image-scale expansion."""

    cfg = config or ScaleExpansionConfig()
    evaluation_timestamp_s = float(evaluation_timestamp_s)
    if not math.isfinite(evaluation_timestamp_s):
        raise ValueError("evaluation_timestamp_s must be finite")
    times, log_heights, log_areas, confidences = _deduplicate_scale_observations(
        observations,
        evaluation_timestamp_s=evaluation_timestamp_s,
        config=cfg,
    )
    sample_count = int(times.size)
    last_age = (
        max(0.0, evaluation_timestamp_s - float(times[-1]))
        if sample_count
        else float("inf")
    )
    height_fit = _robust_log_scale_fit(
        times,
        log_heights,
        confidences,
        source="log_height",
        config=cfg,
    )
    area_fit = _robust_log_scale_fit(
        times,
        log_areas,
        confidences,
        source="log_sqrt_area",
        config=cfg,
    )
    reliable = [fit for fit in (height_fit, area_fit) if fit.reliable]
    if last_age > cfg.maximum_extrapolation_s:
        reason = P2CReason.STALE_HISTORY
        source = "none"
        inverse_ttc = 0.0
        inverse_uncertainty = float("inf")
    elif not reliable:
        # Prefer the most informative stable reason when both components fail.
        reasons = {height_fit.reason_code, area_fit.reason_code}
        if P2CReason.UNSTABLE_SCALE in reasons:
            reason = P2CReason.UNSTABLE_SCALE
        elif P2CReason.NON_EXPANDING in reasons:
            reason = P2CReason.NON_EXPANDING
        elif P2CReason.INSUFFICIENT_TIME_SPAN in reasons:
            reason = P2CReason.INSUFFICIENT_TIME_SPAN
        else:
            reason = P2CReason.INSUFFICIENT_HISTORY
        source = "none"
        inverse_ttc = 0.0
        inverse_uncertainty = float("inf")
    elif len(reliable) == 1:
        fit = reliable[0]
        reason = P2CReason.OK
        source = fit.source
        inverse_ttc = fit.inverse_ttc_s
        inverse_uncertainty = fit.inverse_ttc_uncertainty_s
    else:
        values = np.asarray([fit.inverse_ttc_s for fit in reliable], dtype=float)
        variances = np.asarray(
            [fit.inverse_ttc_uncertainty_s**2 for fit in reliable], dtype=float
        )
        weights = 1.0 / np.maximum(variances, cfg.inverse_uncertainty_floor_s**2)
        weights /= float(np.sum(weights))
        inverse_ttc = float(np.sum(weights * values))
        within_variance = 1.0 / float(
            np.sum(1.0 / np.maximum(variances, cfg.inverse_uncertainty_floor_s**2))
        )
        disagreement_variance = float(np.sum(weights * (values - inverse_ttc) ** 2))
        inverse_uncertainty = math.sqrt(within_variance + disagreement_variance)
        relative = inverse_uncertainty / max(
            inverse_ttc, cfg.minimum_inverse_ttc_s
        )
        reason = (
            P2CReason.OK
            if inverse_uncertainty <= cfg.maximum_inverse_ttc_uncertainty_s
            and relative <= cfg.maximum_relative_inverse_uncertainty
            else P2CReason.UNSTABLE_SCALE
        )
        source = "height_area_fusion"

    if reason == P2CReason.OK:
        raw_ttc = 1.0 / inverse_ttc
        ttc_s = raw_ttc - last_age
        ttc_uncertainty = inverse_uncertainty / (inverse_ttc * inverse_ttc)
        if not 0.1 <= ttc_s <= cfg.maximum_ttc_s:
            reason = P2CReason.TTC_OUT_OF_BOUNDS
            ttc_s = float("inf")
            ttc_uncertainty = float("inf")
    else:
        ttc_s = float("inf")
        ttc_uncertainty = float("inf")
    return ScaleExpansionResult(
        evaluation_timestamp_s=evaluation_timestamp_s,
        ttc_s=float(ttc_s),
        ttc_uncertainty_s=float(ttc_uncertainty),
        inverse_ttc_s=float(inverse_ttc),
        inverse_ttc_uncertainty_s=float(inverse_uncertainty),
        reason_code=reason,
        source=source,
        height_fit=height_fit,
        area_fit=area_fit,
        last_observation_age_s=float(last_age),
    )


def project_scale_expansion_result(
    anchor: ScaleExpansionResult,
    *,
    evaluation_timestamp_s: float,
    config: ScaleExpansionConfig | None = None,
) -> ScaleExpansionResult:
    """Advance an unchanged scale fit without appending or refitting boxes."""

    cfg = config or ScaleExpansionConfig()
    timestamp = float(evaluation_timestamp_s)
    if not math.isfinite(timestamp):
        raise ValueError("evaluation_timestamp_s must be finite")
    elapsed = timestamp - float(anchor.evaluation_timestamp_s)
    if elapsed < 0.0:
        raise ValueError("projection timestamp cannot precede the fit anchor")
    if elapsed == 0.0:
        return anchor

    last_age = float(anchor.last_observation_age_s) + elapsed
    reason = anchor.reason_code
    ttc_s = float("inf")
    ttc_uncertainty = float("inf")
    if last_age > cfg.maximum_extrapolation_s:
        reason = P2CReason.STALE_HISTORY
    elif reason in {P2CReason.OK, P2CReason.TTC_OUT_OF_BOUNDS}:
        inverse_ttc = float(anchor.inverse_ttc_s)
        if inverse_ttc > 0.0 and math.isfinite(inverse_ttc):
            ttc_s = 1.0 / inverse_ttc - last_age
            ttc_uncertainty = (
                float(anchor.inverse_ttc_uncertainty_s)
                / (inverse_ttc * inverse_ttc)
            )
            if 0.1 <= ttc_s <= cfg.maximum_ttc_s:
                reason = P2CReason.OK
            else:
                reason = P2CReason.TTC_OUT_OF_BOUNDS
                ttc_s = float("inf")
                ttc_uncertainty = float("inf")

    return ScaleExpansionResult(
        evaluation_timestamp_s=timestamp,
        ttc_s=ttc_s,
        ttc_uncertainty_s=ttc_uncertainty,
        inverse_ttc_s=anchor.inverse_ttc_s,
        inverse_ttc_uncertainty_s=anchor.inverse_ttc_uncertainty_s,
        reason_code=reason,
        source=anchor.source,
        height_fit=anchor.height_fit,
        area_fit=anchor.area_fit,
        last_observation_age_s=last_age,
    )


class CausalScaleExpansionEstimator:
    """Bounded per-track history; predict/coast never invents a bbox sample."""

    def __init__(self, config: ScaleExpansionConfig | None = None) -> None:
        self.config = config or ScaleExpansionConfig()
        self._history: Deque[ScaleObservation] = deque(maxlen=self.config.history_size)

    @property
    def history(self) -> tuple[ScaleObservation, ...]:
        return tuple(self._history)

    def reset(self) -> None:
        self._history.clear()

    def observe(
        self,
        *,
        timestamp_s: float,
        bbox: BBox,
        confidence: float = 1.0,
    ) -> None:
        """Append one detector observation without running the robust fit."""

        timestamp_s = float(timestamp_s)
        confidence = float(confidence)
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if not _valid_bbox(bbox):
            raise ValueError("bbox must be finite with positive width and height")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self._history and timestamp_s <= self._history[-1].timestamp_s:
            raise ValueError("timestamps must increase strictly")
        self._history.append(ScaleObservation(timestamp_s, bbox, confidence))

    def update(
        self,
        *,
        timestamp_s: float,
        bbox: BBox,
        confidence: float = 1.0,
    ) -> ScaleExpansionResult:
        self.observe(
            timestamp_s=timestamp_s,
            bbox=bbox,
            confidence=confidence,
        )
        return estimate_causal_scale_expansion_ttc(
            self.history,
            evaluation_timestamp_s=timestamp_s,
            config=self.config,
        )

    def predict(self, *, timestamp_s: float) -> ScaleExpansionResult:
        timestamp_s = float(timestamp_s)
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if self._history and timestamp_s < self._history[-1].timestamp_s:
            raise ValueError("prediction timestamp cannot precede the last observation")
        return estimate_causal_scale_expansion_ttc(
            self.history,
            evaluation_timestamp_s=timestamp_s,
            config=self.config,
        )


@dataclass(frozen=True, slots=True)
class TTCFusionInput:
    source: str
    ttc_s: float
    uncertainty_s: float
    reason_code: str = P2CReason.OK

    @property
    def reliable(self) -> bool:
        return (
            self.reason_code
            in {
                P2CReason.OK,
                "within_safety_buffer",
                P2CReason.JUMP_LIMITED,
                P2CReason.HELD_PREVIOUS,
                "valid_legacy_physics",
            }
            and _finite_nonnegative(self.ttc_s)
            and _finite_positive(self.uncertainty_s)
        )


@dataclass(frozen=True, slots=True)
class TTCFusionConfig:
    minimum_ttc_s: float = 0.10
    maximum_ttc_s: float = 10.0
    inverse_ttc_uncertainty_floor_s: float = 0.01
    relative_uncertainty_ttc_floor_s: float = 0.50
    maximum_relative_ttc_uncertainty: float = 0.80
    maximum_ttc_uncertainty_s: float = 3.0
    maximum_upward_jump_s: float = 1.0
    fallback_max_age_s: float = 0.45

    def __post_init__(self) -> None:
        values = (
            self.minimum_ttc_s,
            self.maximum_ttc_s,
            self.inverse_ttc_uncertainty_floor_s,
            self.relative_uncertainty_ttc_floor_s,
            self.maximum_relative_ttc_uncertainty,
            self.maximum_ttc_uncertainty_s,
            self.maximum_upward_jump_s,
            self.fallback_max_age_s,
        )
        if any(not _finite_positive(value) for value in values):
            raise ValueError("fusion limits must be finite and positive")
        if self.maximum_ttc_s <= self.minimum_ttc_s:
            raise ValueError("maximum TTC must exceed minimum TTC")


@dataclass(frozen=True, slots=True)
class TTCFusionResult:
    timestamp_s: float
    ttc_s: float
    uncertainty_s: float
    reason_code: str
    source: str
    contributing_sources: tuple[str, ...]
    fallback_used: bool

    @property
    def reliable(self) -> bool:
        return self.reason_code in {
            P2CReason.OK,
            P2CReason.FUSED,
            P2CReason.FALLBACK,
            P2CReason.HELD_PREVIOUS,
            P2CReason.JUMP_LIMITED,
        }


def fuse_ttc_estimates(
    inputs: Sequence[TTCFusionInput],
    *,
    timestamp_s: float,
    fallback: TTCFusionInput | None = None,
    config: TTCFusionConfig | None = None,
) -> TTCFusionResult:
    """Fuse independent TTC signals by inverse-variance in inverse-TTC space."""

    cfg = config or TTCFusionConfig()
    timestamp_s = float(timestamp_s)
    if not math.isfinite(timestamp_s):
        raise ValueError("timestamp_s must be finite")
    accepted = [
        item
        for item in inputs
        if item.reliable and item.ttc_s <= cfg.maximum_ttc_s
    ]
    if not accepted:
        if fallback is not None and _finite_nonnegative(fallback.ttc_s):
            return TTCFusionResult(
                timestamp_s,
                max(cfg.minimum_ttc_s, float(fallback.ttc_s)),
                float(fallback.uncertainty_s),
                P2CReason.FALLBACK,
                f"fallback:{fallback.source}:{fallback.reason_code}",
                (),
                True,
            )
        return TTCFusionResult(
            timestamp_s,
            float("inf"),
            float("inf"),
            P2CReason.NO_RELIABLE_INPUT,
            "invalid",
            (),
            False,
        )

    bounded_ttc = np.asarray(
        [max(cfg.minimum_ttc_s, item.ttc_s) for item in accepted], dtype=float
    )
    inverse_values = 1.0 / bounded_ttc
    inverse_sigmas = np.asarray(
        [
            item.uncertainty_s / (ttc_s * ttc_s)
            for item, ttc_s in zip(accepted, bounded_ttc, strict=True)
        ],
        dtype=float,
    )
    variances = np.maximum(
        inverse_sigmas**2, cfg.inverse_ttc_uncertainty_floor_s**2
    )
    raw_weights = 1.0 / variances
    weights = raw_weights / float(np.sum(raw_weights))
    inverse_ttc = float(np.sum(weights * inverse_values))
    within_variance = 1.0 / float(np.sum(raw_weights))
    disagreement_variance = float(
        np.sum(weights * (inverse_values - inverse_ttc) ** 2)
    )
    inverse_uncertainty = math.sqrt(within_variance + disagreement_variance)
    ttc_s = 1.0 / inverse_ttc
    uncertainty_s = inverse_uncertainty / (inverse_ttc * inverse_ttc)
    reason = P2CReason.OK if len(accepted) == 1 else P2CReason.FUSED
    # Relative error becomes singular at contact even when the absolute error
    # is small.  A fixed sub-second denominator preserves a valid contact
    # warning while the independent absolute-uncertainty gate still applies.
    relative = uncertainty_s / max(ttc_s, cfg.relative_uncertainty_ttc_floor_s)
    if (
        uncertainty_s > cfg.maximum_ttc_uncertainty_s
        or relative > cfg.maximum_relative_ttc_uncertainty
    ):
        if fallback is not None and _finite_nonnegative(fallback.ttc_s):
            return TTCFusionResult(
                timestamp_s,
                max(cfg.minimum_ttc_s, float(fallback.ttc_s)),
                float(fallback.uncertainty_s),
                P2CReason.FALLBACK,
                f"fallback:high_fusion_uncertainty:{fallback.source}",
                tuple(item.source for item in accepted),
                True,
            )
        reason = P2CReason.HIGH_FUSION_UNCERTAINTY
        ttc_s = float("inf")
        uncertainty_s = float("inf")
    return TTCFusionResult(
        timestamp_s=timestamp_s,
        ttc_s=float(ttc_s),
        uncertainty_s=float(uncertainty_s),
        reason_code=reason,
        source=(accepted[0].source if len(accepted) == 1 else "uncertainty_fusion"),
        contributing_sources=tuple(item.source for item in accepted),
        fallback_used=False,
    )


class CausalTTCFusion:
    """Per-track fusion state with bounded upward jumps and causal holding."""

    def __init__(self, config: TTCFusionConfig | None = None) -> None:
        self.config = config or TTCFusionConfig()
        self._previous: TTCFusionResult | None = None

    def reset(self) -> None:
        self._previous = None

    def update(
        self,
        inputs: Sequence[TTCFusionInput],
        *,
        timestamp_s: float,
        fallback: TTCFusionInput | None = None,
    ) -> TTCFusionResult:
        result = fuse_ttc_estimates(
            inputs,
            timestamp_s=timestamp_s,
            fallback=fallback,
            config=self.config,
        )
        previous = self._previous
        if previous is not None:
            elapsed = float(timestamp_s) - previous.timestamp_s
            if elapsed < 0.0:
                raise ValueError("fusion timestamps cannot move backwards")
            expected = max(self.config.minimum_ttc_s, previous.ttc_s - elapsed)
            if math.isfinite(result.ttc_s) and math.isfinite(previous.ttc_s):
                maximum = expected + self.config.maximum_upward_jump_s
                if result.ttc_s > maximum:
                    result = TTCFusionResult(
                        timestamp_s=float(timestamp_s),
                        ttc_s=maximum,
                        uncertainty_s=max(result.uncertainty_s, previous.uncertainty_s),
                        reason_code=P2CReason.JUMP_LIMITED,
                        source=f"jump_limited:{result.source}",
                        contributing_sources=result.contributing_sources,
                        fallback_used=result.fallback_used,
                    )
            elif (
                not math.isfinite(result.ttc_s)
                and math.isfinite(previous.ttc_s)
                and elapsed <= self.config.fallback_max_age_s
            ):
                result = TTCFusionResult(
                    timestamp_s=float(timestamp_s),
                    ttc_s=expected,
                    uncertainty_s=previous.uncertainty_s + elapsed,
                    reason_code=P2CReason.HELD_PREVIOUS,
                    source=f"held:{previous.source}",
                    contributing_sources=previous.contributing_sources,
                    fallback_used=True,
                )
        # A held value is an output derived from the previous evidence anchor,
        # not new evidence.  Keeping the original anchor timestamp makes the
        # hold expire by wall-clock time even when runtime callbacks are dense.
        if (
            math.isfinite(result.ttc_s)
            and result.reason_code != P2CReason.HELD_PREVIOUS
        ):
            self._previous = result
        return result
