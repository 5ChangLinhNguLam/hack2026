"""Causal tracker state for the deployable C1 P2-B policy.

The accepted 55.2 physics tracker remains untouched.  This module subclasses
it so the ablation runner can expose the physics components *before* the old
collision-corridor gate and, when requested, replace only association/class
stabilisation.  Every input is available at runtime: detector boxes, their
timestamps, camera calibration and ego speed.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .tracker import (
    MonocularTTCTracker,
    TrackerConfig,
    _SAFETY_BUFFER_M,
    _Observation,
    _Track,
    _bbox_geometry,
    _class_group,
    _iou,
    _median_pairwise_slope,
)
from .range_ttc import (
    CausalRangeTTCEstimator,
    RangeTTCConfig,
    RangeTTCReason,
    RangeTTCResult,
)
from .types import BBox, Detection, TrackRisk


_VEHICLES = frozenset({"car", "truck", "bus"})
_TWO_WHEELERS = frozenset({"bicycle", "motorcycle"})
_ROAD_USERS = frozenset({"person", "bicycle", "motorcycle", "car", "truck", "bus"})
_FALLBACK_REASON_CODES = frozenset(
    {
        RangeTTCReason.INSUFFICIENT_HISTORY,
        RangeTTCReason.INSUFFICIENT_TIME_SPAN,
        RangeTTCReason.UNSTABLE_TREND,
        RangeTTCReason.STALE_HISTORY,
    }
)


@dataclass(frozen=True)
class P2AssociationConfig:
    """Geometry-first association and causal class posterior settings."""

    enabled: bool = False
    enable_kalman: bool = True
    unmatched_cost: float = 0.82
    max_center_distance: float = 2.20
    min_predicted_iou: float = 0.01
    max_log_size_delta: float = 1.20
    posterior_decay: float = 0.88
    posterior_switch_margin: float = 0.16
    posterior_min_evidence: float = 1.80
    max_prediction_horizon_s: float = 0.45
    min_incompatible_physics_confidence: float = 0.35
    kalman_position_measurement_std_px: float = 4.0
    kalman_log_size_measurement_std: float = 0.08
    kalman_acceleration_std_px_s2: float = 80.0
    kalman_log_size_acceleration_std_s2: float = 1.0
    kalman_initial_velocity_std_px_s: float = 120.0
    kalman_initial_log_size_rate_std_s: float = 1.0
    kalman_position_uncertainty_scale_px: float = 80.0
    kalman_log_size_uncertainty_scale: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 < self.unmatched_cost < 3.0:
            raise ValueError("unmatched_cost must be positive and bounded")
        if self.max_center_distance <= 0.0:
            raise ValueError("max_center_distance must be positive")
        if self.max_prediction_horizon_s <= 0.0:
            raise ValueError("max_prediction_horizon_s must be positive")
        if not 0.0 <= self.posterior_decay < 1.0:
            raise ValueError("posterior_decay must be in [0, 1)")
        if self.posterior_switch_margin < 0.0:
            raise ValueError("posterior_switch_margin must be non-negative")
        if not 0.0 <= self.min_incompatible_physics_confidence <= 1.0:
            raise ValueError(
                "min_incompatible_physics_confidence must be in [0, 1]"
            )
        kalman_scales = (
            self.kalman_position_measurement_std_px,
            self.kalman_log_size_measurement_std,
            self.kalman_acceleration_std_px_s2,
            self.kalman_log_size_acceleration_std_s2,
            self.kalman_initial_velocity_std_px_s,
            self.kalman_initial_log_size_rate_std_s,
            self.kalman_position_uncertainty_scale_px,
            self.kalman_log_size_uncertainty_scale,
        )
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in kalman_scales
        ):
            raise ValueError("Kalman noise and uncertainty scales must be positive")


@dataclass(frozen=True)
class P2RangePolicyConfig:
    """Robust monocular range policy and bounded causal fallback."""

    enabled: bool = False
    enable_last_good_fallback: bool = False
    estimator: RangeTTCConfig = field(
        default_factory=lambda: RangeTTCConfig(
            history_size=12,
            min_samples=4,
            min_time_span_s=0.20,
            min_closing_speed_mps=0.50,
            min_closing_fraction=0.65,
            max_relative_closing_uncertainty=0.80,
            max_range_uncertainty_m=3.0,
            max_extrapolation_s=0.40,
        )
    )
    safety_buffer_scale: float = 1.0
    max_ttc_s: float = 10.0
    maximum_ttc_uncertainty_s: float = 2.0
    maximum_range_uncertainty_m: float = 3.0
    jump_residual_limit_s: float = 1.0
    fallback_max_age_s: float = 0.45

    def __post_init__(self) -> None:
        positive = (
            self.max_ttc_s,
            self.maximum_ttc_uncertainty_s,
            self.maximum_range_uncertainty_m,
            self.jump_residual_limit_s,
            self.fallback_max_age_s,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("range policy limits must be finite and positive")
        if not math.isfinite(self.safety_buffer_scale) or self.safety_buffer_scale < 0:
            raise ValueError("safety_buffer_scale must be finite and non-negative")


@dataclass
class _CausalBoxKalman:
    """Tiny constant-velocity filter over centre and logarithmic box size.

    State order is ``cx, cy, vx, vy, log(w), log(h), dlog(w), dlog(h)``.
    Projection is side-effect free because one track is compared with several
    detections during Hungarian cost construction.  Only an actual runtime
    step commits a prediction or measurement update.
    """

    state: np.ndarray
    covariance: np.ndarray
    timestamp: float

    _MEASUREMENT_INDICES = (0, 1, 4, 5)

    @staticmethod
    def _measurement(bbox: BBox) -> np.ndarray:
        cx, cy, width, height, _ = _bbox_geometry(bbox)
        return np.asarray(
            [cx, cy, math.log(width), math.log(height)], dtype=np.float64
        )

    @staticmethod
    def _measurement_covariance(
        confidence: float,
        config: P2AssociationConfig,
    ) -> np.ndarray:
        confidence_scale = 1.0 / math.sqrt(max(0.05, min(1.0, confidence)))
        position_std = config.kalman_position_measurement_std_px * confidence_scale
        log_size_std = config.kalman_log_size_measurement_std * confidence_scale
        return np.diag(
            [
                position_std * position_std,
                position_std * position_std,
                log_size_std * log_size_std,
                log_size_std * log_size_std,
            ]
        ).astype(np.float64)

    @classmethod
    def initialise(
        cls,
        bbox: BBox,
        *,
        timestamp: float,
        confidence: float,
        config: P2AssociationConfig,
    ) -> _CausalBoxKalman:
        measurement = cls._measurement(bbox)
        state = np.zeros(8, dtype=np.float64)
        state[[0, 1, 4, 5]] = measurement
        measurement_covariance = cls._measurement_covariance(confidence, config)
        covariance = np.zeros((8, 8), dtype=np.float64)
        covariance[np.ix_((0, 1, 4, 5), (0, 1, 4, 5))] = measurement_covariance
        covariance[2, 2] = config.kalman_initial_velocity_std_px_s**2
        covariance[3, 3] = config.kalman_initial_velocity_std_px_s**2
        covariance[6, 6] = config.kalman_initial_log_size_rate_std_s**2
        covariance[7, 7] = config.kalman_initial_log_size_rate_std_s**2
        return cls(state=state, covariance=covariance, timestamp=float(timestamp))

    @staticmethod
    def _transition(
        delta_s: float,
        config: P2AssociationConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        transition = np.eye(8, dtype=np.float64)
        transition[0, 2] = delta_s
        transition[1, 3] = delta_s
        transition[4, 6] = delta_s
        transition[5, 7] = delta_s
        process = np.zeros((8, 8), dtype=np.float64)

        def add_white_acceleration(
            position_index: int,
            velocity_index: int,
            acceleration_std: float,
        ) -> None:
            # Exact integral of a continuous white-acceleration process.  This
            # semigroup form makes covariance depend on elapsed time, not on
            # how many detector-stride predict callbacks happen in between.
            variance = acceleration_std * acceleration_std
            dt2 = delta_s * delta_s
            dt3 = dt2 * delta_s
            process[position_index, position_index] = variance * dt3 / 3.0
            process[position_index, velocity_index] = variance * dt2 / 2.0
            process[velocity_index, position_index] = variance * dt2 / 2.0
            process[velocity_index, velocity_index] = variance * delta_s

        add_white_acceleration(0, 2, config.kalman_acceleration_std_px_s2)
        add_white_acceleration(1, 3, config.kalman_acceleration_std_px_s2)
        add_white_acceleration(4, 6, config.kalman_log_size_acceleration_std_s2)
        add_white_acceleration(5, 7, config.kalman_log_size_acceleration_std_s2)
        return transition, process

    def project(
        self,
        timestamp: float,
        config: P2AssociationConfig,
        *,
        maximum_horizon_s: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a pure projection; repeated calls are exactly idempotent."""

        delta_s = max(0.0, float(timestamp) - self.timestamp)
        if maximum_horizon_s is not None:
            delta_s = min(delta_s, maximum_horizon_s)
        transition, process = self._transition(delta_s, config)
        state = transition @ self.state
        covariance = transition @ self.covariance @ transition.T + process
        return state, 0.5 * (covariance + covariance.T)

    def predict(self, timestamp: float, config: P2AssociationConfig) -> None:
        if timestamp < self.timestamp:
            raise ValueError("Kalman timestamp cannot move backwards")
        self.state, self.covariance = self.project(timestamp, config)
        self.timestamp = float(timestamp)

    def update(
        self,
        bbox: BBox,
        *,
        timestamp: float,
        confidence: float,
        config: P2AssociationConfig,
    ) -> None:
        self.predict(timestamp, config)
        measurement = self._measurement(bbox)
        measurement_matrix = np.zeros((4, 8), dtype=np.float64)
        for row, column in enumerate(self._MEASUREMENT_INDICES):
            measurement_matrix[row, column] = 1.0
        measurement_covariance = self._measurement_covariance(confidence, config)
        innovation = measurement - measurement_matrix @ self.state
        innovation_covariance = (
            measurement_matrix @ self.covariance @ measurement_matrix.T
            + measurement_covariance
        )
        gain = np.linalg.solve(
            innovation_covariance,
            measurement_matrix @ self.covariance,
        ).T
        self.state = self.state + gain @ innovation
        identity_minus_gain = np.eye(8) - gain @ measurement_matrix
        # Joseph form remains symmetric positive semidefinite under finite
        # precision, which matters when many coast/predict calls accumulate.
        self.covariance = (
            identity_minus_gain @ self.covariance @ identity_minus_gain.T
            + gain @ measurement_covariance @ gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        diagonal = np.maximum(np.diag(self.covariance), 1e-12)
        self.covariance[np.diag_indices(8)] = diagonal

    @staticmethod
    def bbox_from_state(state: np.ndarray) -> BBox:
        width = math.exp(float(np.clip(state[4], math.log(1.0), math.log(8192.0))))
        height = math.exp(float(np.clip(state[5], math.log(1.0), math.log(8192.0))))
        cx, cy = float(state[0]), float(state[1])
        return (
            cx - 0.5 * width,
            cy - 0.5 * height,
            cx + 0.5 * width,
            cy + 0.5 * height,
        )

    @classmethod
    def normalised_innovation(
        cls,
        projection: tuple[np.ndarray, np.ndarray],
        detection: Detection,
        config: P2AssociationConfig,
    ) -> float:
        state, covariance = projection
        measurement = cls._measurement(detection.bbox)
        predicted = state[[0, 1, 4, 5]]
        measurement_covariance = cls._measurement_covariance(
            detection.confidence, config
        )
        projected_covariance = covariance[np.ix_((0, 1, 4, 5), (0, 1, 4, 5))]
        innovation_covariance = projected_covariance + measurement_covariance
        residual = measurement - predicted
        squared = float(
            residual @ np.linalg.solve(innovation_covariance, residual)
        )
        if not math.isfinite(squared):
            return float("inf")
        return math.sqrt(max(0.0, squared) / 4.0)

    def confidence(self, config: P2AssociationConfig) -> float:
        position_std = math.sqrt(
            max(0.0, 0.5 * (self.covariance[0, 0] + self.covariance[1, 1]))
        )
        log_size_std = math.sqrt(
            max(0.0, 0.5 * (self.covariance[4, 4] + self.covariance[5, 5]))
        )
        position_term = (
            position_std / config.kalman_position_uncertainty_scale_px
        ) ** 2
        size_term = (
            log_size_std / config.kalman_log_size_uncertainty_scale
        ) ** 2
        return float(np.clip(1.0 / (1.0 + position_term + size_term), 0.0, 1.0))

    def lateral_velocity_uncertainty_px_s(
        self, config: P2AssociationConfig
    ) -> float:
        """Return the causal standard deviation of the footpoint x velocity."""

        variance = float(self.covariance[2, 2])
        if not math.isfinite(variance) or variance < 0.0:
            # A conservative finite fallback keeps the edge runtime usable if
            # a platform-specific linear algebra failure corrupts covariance.
            return config.kalman_initial_velocity_std_px_s
        return math.sqrt(variance)


@dataclass
class _TrackMeta:
    posterior: dict[str, float]
    stable_label: str
    hits: int = 1
    association_confidence: float = 0.55
    observed_on_last_call: bool = True
    last_seen_timestamp: float = 0.0
    raw_labels: deque[str] = field(default_factory=lambda: deque(maxlen=12))
    range_estimator: CausalRangeTTCEstimator | None = None
    robust_result: RangeTTCResult | None = None
    robust_ttc_s: float = float("inf")
    robust_reason: str = RangeTTCReason.INSUFFICIENT_HISTORY
    last_good_ttc_s: float = float("inf")
    last_good_timestamp: float = float("-inf")
    motion_filter: _CausalBoxKalman | None = None


@dataclass(frozen=True)
class P2TrackSnapshot:
    """Read-only, label-free candidate supplied to the target selector."""

    track_id: int
    label: str
    class_posterior: Mapping[str, float]
    bbox: BBox
    confidence: float
    foot_velocity_x_px_s: float
    foot_velocity_y_px_s: float
    foot_velocity_uncertainty_px_s: float | None
    motion_state_confidence: float | None
    raw_physics_ttc_s: float
    ego_motion_ttc_s: float
    scale_ttc_s: float
    legacy_range_ttc_s: float
    lateral_ttc_s: float
    robust_range_ttc_s: float
    closing_speed_mps: float
    estimated_range_m: float
    range_stability: float
    range_uncertainty_m: float
    ttc_uncertainty_s: float
    hits: int
    missed_updates: int
    association_confidence: float
    observed_this_call: bool
    last_seen_age_s: float
    last_good_ttc_s: float
    reason_code: str

    @property
    def confirmed(self) -> bool:
        return self.hits >= 3


def _normalised_posterior(values: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(value)) for value in values.values())
    if total <= 1e-9:
        return {}
    return {
        label: max(0.0, float(value)) / total
        for label, value in sorted(values.items())
        if value > 0.0
    }


def _class_penalty(stable_label: str, detected_label: str) -> float:
    """Soft compatibility; geometry remains the dominant association term."""

    if stable_label == detected_label:
        return 0.0
    if stable_label in _VEHICLES and detected_label in _VEHICLES:
        return 0.08
    if stable_label in _TWO_WHEELERS and detected_label in _TWO_WHEELERS:
        return 0.12
    if (
        stable_label == "person" and detected_label in _TWO_WHEELERS
    ) or (
        detected_label == "person" and stable_label in _TWO_WHEELERS
    ):
        # A rider is often split or relabelled, but this transition must not be
        # as cheap as bicycle <-> motorcycle.
        return 0.30
    return 0.90


def _physics_history_label(
    stable_label: str,
    detected_label: str,
    detection_confidence: float,
    *,
    min_incompatible_confidence: float,
) -> str:
    """Gate only weak cross-group classifier noise from physics history.

    Association and the class posterior still consume ``detected_label``.
    A confident incompatible observation also remains raw so a real class or
    identity transition cannot create a false like-for-like scale slope.
    """

    if (
        _class_group(stable_label) != _class_group(detected_label)
        and detection_confidence < min_incompatible_confidence
    ):
        return stable_label
    return detected_label


def _hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Dependency-free minimum-cost assignment for a rectangular matrix."""

    if cost.ndim != 2:
        raise ValueError("Hungarian cost must be a matrix")
    n_rows, n_columns = cost.shape
    if n_rows == 0 or n_columns == 0:
        return []
    transposed = n_rows > n_columns
    matrix = cost.T if transposed else cost
    rows, columns = matrix.shape
    # Classic shortest augmenting-path Hungarian algorithm, 1-indexed.
    u = np.zeros(rows + 1, dtype=np.float64)
    v = np.zeros(columns + 1, dtype=np.float64)
    p = np.zeros(columns + 1, dtype=np.int64)
    way = np.zeros(columns + 1, dtype=np.int64)
    for row in range(1, rows + 1):
        p[0] = row
        minimum = np.full(columns + 1, np.inf, dtype=np.float64)
        used = np.zeros(columns + 1, dtype=bool)
        column0 = 0
        while True:
            used[column0] = True
            row0 = int(p[column0])
            delta = float("inf")
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = float(matrix[row0 - 1, column - 1] - u[row0] - v[column])
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = float(minimum[column])
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = int(way[column0])
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    pairs: list[tuple[int, int]] = []
    for column in range(1, columns + 1):
        if p[column] == 0:
            continue
        left, right = int(p[column] - 1), column - 1
        pairs.append((right, left) if transposed else (left, right))
    return pairs


class P2CausalTracker(MonocularTTCTracker):
    """Physics tracker with optional geometry-first/class-history association."""

    def __init__(
        self,
        config: TrackerConfig,
        *,
        association: P2AssociationConfig | None = None,
        range_policy: P2RangePolicyConfig | None = None,
        focal_y_px: float = 320.0,
        focal_x_px: float | None = None,
        principal_x_px: float | None = None,
    ) -> None:
        super().__init__(
            config,
            focal_y_px=focal_y_px,
            focal_x_px=focal_x_px,
            principal_x_px=principal_x_px,
        )
        self.association = association or P2AssociationConfig()
        self.range_policy = range_policy or P2RangePolicyConfig()
        self._meta: dict[int, _TrackMeta] = {}
        self._match_quality: dict[tuple[int, int], float] = {}
        self._last_timestamp = float("-inf")
        self._last_image_shape = (360, 640)
        self._last_ego_speed_kmh = 0.0

    def reset(self) -> None:
        super().reset()
        self._meta.clear()
        self._match_quality.clear()
        self._last_timestamp = float("-inf")

    def _advance_timestamp(self, timestamp: float) -> float:
        """Validate the single forward-only runtime clock."""

        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if timestamp <= self._last_timestamp:
            raise ValueError("P2 tracker timestamps must increase strictly")
        self._last_timestamp = timestamp
        return timestamp

    def _predicted_bbox(self, track: _Track, timestamp: float) -> BBox:
        meta = self._meta.get(track.track_id)
        if (
            self.association.enabled
            and self.association.enable_kalman
            and meta is not None
            and meta.motion_filter is not None
        ):
            projection = meta.motion_filter.project(
                timestamp,
                self.association,
                maximum_horizon_s=self.association.max_prediction_horizon_s,
            )
            return _CausalBoxKalman.bbox_from_state(projection[0])

        observations = list(track.history)
        if len(observations) < 2:
            return track.bbox
        previous, current = observations[-2], observations[-1]
        delta_t = current.timestamp - previous.timestamp
        horizon = min(
            max(0.0, timestamp - current.timestamp),
            self.association.max_prediction_horizon_s,
        )
        if delta_t <= 1e-4 or horizon <= 0.0:
            return track.bbox
        velocity = [
            (current.bbox[index] - previous.bbox[index]) / delta_t
            for index in range(4)
        ]
        predicted = tuple(
            current.bbox[index] + velocity[index] * horizon for index in range(4)
        )
        if predicted[2] <= predicted[0] or predicted[3] <= predicted[1]:
            return track.bbox
        return predicted  # type: ignore[return-value]

    def _association_cost(
        self,
        track: _Track,
        detection: Detection,
        *,
        timestamp: float,
        predicted_bbox: BBox | None = None,
        motion_projection: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> float:
        predicted = (
            predicted_bbox
            if predicted_bbox is not None
            else self._predicted_bbox(track, timestamp)
        )
        overlap = _iou(predicted, detection.bbox)
        tx, ty, tw, th, _ = _bbox_geometry(predicted)
        dx, dy, dw, dh, _ = _bbox_geometry(detection.bbox)
        normaliser = max(10.0, math.sqrt(max(tw * th, dw * dh)))
        centre_distance = math.hypot(dx - tx, dy - ty) / normaliser
        log_size_delta = abs(math.log(dw / tw)) + abs(math.log(dh / th))
        meta = self._meta.get(track.track_id)
        stable_label = meta.stable_label if meta is not None else track.label
        class_cost = _class_penalty(stable_label, detection.label)
        if motion_projection is not None:
            normalised_innovation = _CausalBoxKalman.normalised_innovation(
                motion_projection, detection, self.association
            )
            motion_cost = min(1.0, normalised_innovation / 3.0)
        else:
            motion_cost = 0.0
        if (
            overlap < self.association.min_predicted_iou
            and centre_distance > self.association.max_center_distance
        ) or log_size_delta > self.association.max_log_size_delta:
            return 2.5
        # Geometry/motion contributes 92%; class is deliberately only a soft
        # term so a one-frame label flip cannot destroy motion history.  The
        # innovation term is covariance-normalised, so an uncertain coast is
        # not treated as a precise miss.
        if motion_projection is not None:
            return (
                0.50 * (1.0 - overlap)
                + 0.24
                * min(1.0, centre_distance / self.association.max_center_distance)
                + 0.10
                * min(1.0, log_size_delta / self.association.max_log_size_delta)
                + 0.08 * motion_cost
                + 0.08 * class_cost
            )
        return (
            0.54 * (1.0 - overlap)
            + 0.28 * min(1.0, centre_distance / self.association.max_center_distance)
            + 0.10 * min(1.0, log_size_delta / self.association.max_log_size_delta)
            + 0.08 * class_cost
        )

    def _match(
        self,
        detections: Sequence[Detection],
        image_shape: tuple[int, int],
    ) -> list[tuple[int, int]]:
        self._match_quality = {}
        if not self.association.enabled:
            matches = super()._match(detections, image_shape)
            for track_index, detection_index in matches:
                overlap = _iou(
                    self._tracks[track_index].bbox,
                    detections[detection_index].bbox,
                )
                self._match_quality[(track_index, detection_index)] = max(0.0, overlap)
            return matches
        if not self._tracks or not detections:
            return []

        n_tracks, n_detections = len(self._tracks), len(detections)
        motion_projections: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        predicted_boxes: dict[int, BBox] = {}
        if self.association.enable_kalman:
            for track_index, track in enumerate(self._tracks):
                motion_filter = self._meta[track.track_id].motion_filter
                if motion_filter is None:
                    continue
                projection = motion_filter.project(
                    self._last_timestamp,
                    self.association,
                    maximum_horizon_s=self.association.max_prediction_horizon_s,
                )
                motion_projections[track_index] = projection
                predicted_boxes[track_index] = _CausalBoxKalman.bbox_from_state(
                    projection[0]
                )
        size = n_tracks + n_detections
        matrix = np.full((size, size), 3.0, dtype=np.float64)
        for track_index, track in enumerate(self._tracks):
            for detection_index, detection in enumerate(detections):
                matrix[track_index, detection_index] = self._association_cost(
                    track,
                    detection,
                    timestamp=self._last_timestamp,
                    predicted_bbox=predicted_boxes.get(track_index),
                    motion_projection=motion_projections.get(track_index),
                )
            matrix[track_index, n_detections + track_index] = self.association.unmatched_cost
        for detection_index in range(n_detections):
            matrix[n_tracks + detection_index, detection_index] = self.association.unmatched_cost
        matrix[n_tracks:, n_detections:] = 0.0

        matches: list[tuple[int, int]] = []
        for track_index, detection_index in _hungarian(matrix):
            if track_index >= n_tracks or detection_index >= n_detections:
                continue
            cost = float(matrix[track_index, detection_index])
            if cost >= self.association.unmatched_cost:
                continue
            matches.append((track_index, detection_index))
            self._match_quality[(track_index, detection_index)] = max(
                0.0, 1.0 - cost / self.association.unmatched_cost
            )
        return matches

    def _update_class(self, meta: _TrackMeta, raw_label: str) -> str:
        if raw_label not in _ROAD_USERS:
            return meta.stable_label
        decay = self.association.posterior_decay
        for label in tuple(meta.posterior):
            meta.posterior[label] *= decay
        meta.posterior[raw_label] = meta.posterior.get(raw_label, 0.0) + 1.0
        meta.raw_labels.append(raw_label)
        if not self.association.enabled:
            meta.stable_label = raw_label
            return raw_label

        current = meta.posterior.get(meta.stable_label, 0.0)
        best_label, best_evidence = max(
            meta.posterior.items(), key=lambda item: (item[1], item[0])
        )
        total = sum(meta.posterior.values())
        required_margin = self.association.posterior_switch_margin * max(total, 1.0)
        if (
            best_label != meta.stable_label
            and best_evidence >= self.association.posterior_min_evidence
            and best_evidence - current >= required_margin
        ):
            meta.stable_label = best_label
        return meta.stable_label

    def _range_estimator(self) -> CausalRangeTTCEstimator:
        return CausalRangeTTCEstimator(self.range_policy.estimator)

    def _safety_buffer(self, label: str) -> float:
        return (
            _SAFETY_BUFFER_M.get(label, 4.0)
            * self.range_policy.safety_buffer_scale
        )

    def _bounded_robust_ttc(
        self,
        meta: _TrackMeta,
        result: RangeTTCResult,
        *,
        timestamp: float,
        refresh_last_good: bool,
    ) -> tuple[float, str]:
        """Limit jumps and hold only observations missing reliable evidence.

        Detector-stride predictions may use the anchor but cannot refresh it.
        This keeps both the jump filter and fallback expiry independent of the
        number of intermediate runtime frames.
        """

        within_monitor = (
            result.reliable
            and math.isfinite(result.ttc_s)
            and 0.0 <= result.ttc_s <= self.range_policy.max_ttc_s
        )
        uncertainty_accepted = (
            result.range_uncertainty_m
            <= self.range_policy.maximum_range_uncertainty_m
            and result.ttc_uncertainty_s
            <= self.range_policy.maximum_ttc_uncertainty_s
        )
        reliable = within_monitor and uncertainty_accepted
        if reliable:
            candidate = max(0.1, float(result.ttc_s))
            reason = result.reason_code
            if math.isfinite(meta.last_good_ttc_s):
                elapsed = max(0.0, timestamp - meta.last_good_timestamp)
                expected = max(0.1, meta.last_good_ttc_s - elapsed)
                residual = candidate - expected
                limit = self.range_policy.jump_residual_limit_s
                # Never postpone a causal increase in danger.  Only bound an
                # upward residual, where danger appears to disappear faster
                # than elapsed time and the configured physical jump limit.
                if residual > limit:
                    candidate = expected + limit
                    candidate = max(0.1, min(self.range_policy.max_ttc_s, candidate))
                    reason = "jump_limited"
            if refresh_last_good:
                meta.last_good_ttc_s = candidate
                meta.last_good_timestamp = timestamp
            return candidate, reason

        uncertainty_rejected = within_monitor and not uncertainty_accepted
        missing_or_uncertain = (
            result.reason_code in _FALLBACK_REASON_CODES
            or uncertainty_rejected
        )
        if (
            missing_or_uncertain
            and self.range_policy.enable_last_good_fallback
            and math.isfinite(meta.last_good_ttc_s)
            and timestamp - meta.last_good_timestamp
            <= self.range_policy.fallback_max_age_s
        ):
            held = max(
                0.1,
                meta.last_good_ttc_s - max(0.0, timestamp - meta.last_good_timestamp),
            )
            return held, "held_previous"
        return float("inf"), result.reason_code

    def _update_robust_range(
        self,
        track: _Track,
        meta: _TrackMeta,
        *,
        timestamp: float,
        append_observation: bool,
    ) -> None:
        if not self.range_policy.enabled:
            return
        if meta.range_estimator is None:
            meta.range_estimator = self._range_estimator()
        if append_observation:
            result = meta.range_estimator.update(
                timestamp_s=timestamp,
                range_m=self._estimated_distance(track.label, track.bbox),
                safety_buffer_m=self._safety_buffer(track.label),
            )
        else:
            result = meta.range_estimator.predict(
                timestamp_s=timestamp,
                safety_buffer_m=self._safety_buffer(track.label),
            )
        meta.robust_result = result
        meta.robust_ttc_s, meta.robust_reason = self._bounded_robust_ttc(
            meta,
            result,
            timestamp=timestamp,
            refresh_last_good=append_observation,
        )

    def update(
        self,
        detections: Sequence[Detection],
        *,
        timestamp: float,
        image_shape: tuple[int, int],
        ego_speed_kmh: float = 0.0,
    ) -> list[TrackRisk]:
        timestamp = self._advance_timestamp(timestamp)
        self._last_image_shape = image_shape
        self._last_ego_speed_kmh = float(ego_speed_kmh)
        original_track_count = len(self._tracks)
        matches = self._match(detections, image_shape)
        matched_tracks = {track_index for track_index, _ in matches}
        matched_detections = {detection_index for _, detection_index in matches}

        for index, track in enumerate(self._tracks):
            meta = self._meta[track.track_id]
            meta.observed_on_last_call = False
            if index not in matched_tracks:
                track.missed += 1
                if meta.motion_filter is not None:
                    meta.motion_filter.predict(timestamp, self.association)

        for track_index, detection_index in matches:
            track = self._tracks[track_index]
            detection = detections[detection_index]
            meta = self._meta[track.track_id]
            stable_label = self._update_class(meta, detection.label)
            track.class_id = detection.class_id
            track.label = stable_label
            track.confidence = detection.confidence
            track.bbox = detection.bbox
            track.missed = 0
            if meta.motion_filter is not None:
                meta.motion_filter.update(
                    detection.bbox,
                    timestamp=timestamp,
                    confidence=detection.confidence,
                    config=self.association,
                )
            # Keep credible raw labels in physics history so the accepted
            # class-group guard rejects false scale slopes across real
            # incompatible transitions.  A weak one-frame cross-group label
            # is gated by the causal class posterior instead of truncating an
            # otherwise coherent geometry series.  Association/posterior state
            # above still receives the raw detector label.
            physics_label = _physics_history_label(
                stable_label,
                detection.label,
                detection.confidence,
                min_incompatible_confidence=(
                    self.association.min_incompatible_physics_confidence
                ),
            )
            track.history.append(
                _Observation(timestamp, detection.bbox, physics_label)
            )
            meta.hits += 1
            meta.association_confidence = (
                0.72 * meta.association_confidence
                + 0.28 * self._match_quality.get((track_index, detection_index), 0.5)
            )
            meta.observed_on_last_call = True
            meta.last_seen_timestamp = timestamp
            self._update_robust_range(
                track,
                meta,
                timestamp=timestamp,
                append_observation=True,
            )

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            history = deque(maxlen=self.config.history_size)
            history.append(_Observation(timestamp, detection.bbox, detection.label))
            track = _Track(
                track_id=self._next_track_id,
                class_id=detection.class_id,
                label=detection.label,
                confidence=detection.confidence,
                bbox=detection.bbox,
                history=history,
            )
            self._tracks.append(track)
            self._meta[track.track_id] = _TrackMeta(
                posterior={detection.label: 1.0},
                stable_label=detection.label,
                hits=1,
                association_confidence=0.55,
                observed_on_last_call=True,
                last_seen_timestamp=timestamp,
                raw_labels=deque((detection.label,), maxlen=12),
                range_estimator=(
                    self._range_estimator() if self.range_policy.enabled else None
                ),
                motion_filter=(
                    _CausalBoxKalman.initialise(
                        detection.bbox,
                        timestamp=timestamp,
                        confidence=detection.confidence,
                        config=self.association,
                    )
                    if self.association.enabled and self.association.enable_kalman
                    else None
                ),
            )
            self._update_robust_range(
                track,
                self._meta[track.track_id],
                timestamp=timestamp,
                append_observation=True,
            )
            self._next_track_id += 1

        # A real detector call with no match is still a causal runtime step.
        # Age the fit and bounded fallback without inventing a range sample.
        for track_index, track in enumerate(self._tracks[:original_track_count]):
            if track_index in matched_tracks:
                continue
            if track.missed <= self.config.max_missed_frames:
                self._update_robust_range(
                    track,
                    self._meta[track.track_id],
                    timestamp=timestamp,
                    append_observation=False,
                )

        self._tracks = [
            track
            for track in self._tracks
            if track.missed <= self.config.max_missed_frames
        ]
        active_ids = {track.track_id for track in self._tracks}
        self._meta = {
            track_id: meta
            for track_id, meta in self._meta.items()
            if track_id in active_ids
        }
        return self._emit_risks(image_shape, ego_speed_kmh, timestamp)

    def predict(
        self,
        *,
        timestamp: float,
        image_shape: tuple[int, int],
        ego_speed_kmh: float = 0.0,
    ) -> list[TrackRisk]:
        timestamp = self._advance_timestamp(timestamp)
        self._last_image_shape = image_shape
        self._last_ego_speed_kmh = float(ego_speed_kmh)
        for meta in self._meta.values():
            meta.observed_on_last_call = False
        for track in self._tracks:
            meta = self._meta[track.track_id]
            if meta.motion_filter is not None:
                meta.motion_filter.predict(timestamp, self.association)
            self._update_robust_range(
                track,
                meta,
                timestamp=timestamp,
                append_observation=False,
            )
        return self._emit_risks(image_shape, ego_speed_kmh, timestamp)

    def candidate_states(self) -> tuple[P2TrackSnapshot, ...]:
        """Return all live tracks, including candidates hidden by old gating."""

        output: list[P2TrackSnapshot] = []
        for track in self._tracks:
            meta = self._meta[track.track_id]
            risk = self._risk(
                track,
                self._last_image_shape,
                self._last_ego_speed_kmh,
                self._last_timestamp,
            )
            observations = list(track.history)
            times = [item.timestamp for item in observations]
            foot_x = [(item.bbox[0] + item.bbox[2]) * 0.5 for item in observations]
            foot_y = [item.bbox[3] for item in observations]
            if meta.motion_filter is not None:
                motion_state = meta.motion_filter.state
                velocity_x = float(motion_state[2])
                velocity_uncertainty = (
                    meta.motion_filter.lateral_velocity_uncertainty_px_s(
                        self.association
                    )
                )
                box_height = math.exp(
                    float(
                        np.clip(
                            motion_state[5], math.log(1.0), math.log(8192.0)
                        )
                    )
                )
                # A bbox footpoint moves with both centre translation and
                # half of the logarithmic height-growth rate.
                velocity_y = float(
                    motion_state[3] + 0.5 * box_height * motion_state[7]
                )
            else:
                velocity_x = _median_pairwise_slope(times[-5:], foot_x[-5:])
                velocity_y = _median_pairwise_slope(times[-5:], foot_y[-5:])
                velocity_uncertainty = None
            raw_components = (
                risk.scale_ttc_s,
                risk.range_ttc_s,
                risk.lateral_ttc_s,
            )
            raw_ttc = min(
                (value for value in raw_components if math.isfinite(value)),
                default=float("inf"),
            )
            if math.isfinite(raw_ttc):
                reason = "valid_legacy_physics"
            elif meta.hits < self.config.min_history:
                reason = "insufficient_history"
            elif track.missed > self.config.max_coast_updates:
                reason = "coast_expired"
            else:
                reason = "non_approaching_or_unstable"
            robust = meta.robust_result
            range_uncertainty_m = (
                robust.range_uncertainty_m
                if robust is not None
                else float("inf")
            )
            ttc_uncertainty_s = (
                robust.ttc_uncertainty_s
                if robust is not None
                else float("inf")
            )
            closing_speed = (
                robust.closing_speed_mps
                if robust is not None and math.isfinite(robust.closing_speed_mps)
                else risk.range_closing_speed_mps
            )
            estimated_range = (
                robust.range_m
                if robust is not None and math.isfinite(robust.range_m)
                else risk.estimated_distance_m
            )
            ego_motion_ttc = float("inf")
            ego_speed_mps = max(0.0, self._last_ego_speed_kmh) / 3.6
            box_height = max(0.0, track.bbox[3] - track.bbox[1])
            if (
                ego_speed_mps > 0.5
                and box_height >= self.config.fast_attack_min_box_height_px
                and track.confidence >= self.config.fast_attack_min_confidence
            ):
                observation_age_s = (
                    max(0.0, self._last_timestamp - observations[-1].timestamp)
                    if observations
                    else 0.0
                )
                candidate = risk.estimated_distance_m / ego_speed_mps - observation_age_s
                if 0.1 <= candidate <= self.config.max_ttc_s:
                    ego_motion_ttc = candidate
            robust_reason = meta.robust_reason if self.range_policy.enabled else reason
            motion_state_confidence = (
                meta.motion_filter.confidence(self.association)
                if meta.motion_filter is not None
                else None
            )
            output.append(
                P2TrackSnapshot(
                    track_id=track.track_id,
                    label=meta.stable_label,
                    class_posterior=_normalised_posterior(meta.posterior),
                    bbox=track.bbox,
                    confidence=track.confidence,
                    foot_velocity_x_px_s=float(velocity_x),
                    foot_velocity_y_px_s=float(velocity_y),
                    foot_velocity_uncertainty_px_s=velocity_uncertainty,
                    motion_state_confidence=motion_state_confidence,
                    raw_physics_ttc_s=float(raw_ttc),
                    ego_motion_ttc_s=float(ego_motion_ttc),
                    scale_ttc_s=risk.scale_ttc_s,
                    legacy_range_ttc_s=risk.range_ttc_s,
                    lateral_ttc_s=risk.lateral_ttc_s,
                    robust_range_ttc_s=meta.robust_ttc_s,
                    closing_speed_mps=closing_speed,
                    estimated_range_m=estimated_range,
                    range_stability=(
                        float(np.clip(1.0 / (1.0 + range_uncertainty_m), 0.0, 1.0))
                        if math.isfinite(range_uncertainty_m)
                        else risk.range_trend_confidence
                    ),
                    range_uncertainty_m=range_uncertainty_m,
                    ttc_uncertainty_s=ttc_uncertainty_s,
                    hits=meta.hits,
                    missed_updates=track.missed,
                    association_confidence=float(
                        np.clip(
                            meta.association_confidence
                            * math.exp(-0.30 * track.missed),
                            0.0,
                            1.0,
                        )
                    ),
                    observed_this_call=meta.observed_on_last_call,
                    last_seen_age_s=max(
                        0.0, self._last_timestamp - meta.last_seen_timestamp
                    ),
                    # Expose only the canonical, already-counted-down hold.
                    # Historical anchors are never selectable directly.
                    last_good_ttc_s=(
                        meta.robust_ttc_s
                        if meta.robust_reason == "held_previous"
                        else float("inf")
                    ),
                    reason_code=robust_reason,
                )
            )
        return tuple(output)
