"""Causal ego-corridor geometry for deployable C1 target selection.

The module consumes only the current lane estimate, image geometry and a
tracker bbox/motion estimate.  It deliberately operates in image space so it
does not require depth, object labels or trip-specific state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .lane import LaneEstimate
from .types import BBox


CorridorSource = Literal["lane", "fixed"]


@dataclass(frozen=True, slots=True)
class CorridorConfig:
    """Normalized geometry and uncertainty controls shared by all trips."""

    minimum_lane_confidence: float = 0.60
    minimum_lane_width_fraction: float = 0.08
    maximum_lane_width_fraction: float = 0.92
    fallback_top_y_fraction: float = 0.35
    fallback_top_half_width_fraction: float = 0.075
    fallback_bottom_half_width_fraction: float = 0.23
    fallback_confidence: float = 0.30
    footprint_half_width_fraction: float = 0.18
    prediction_horizon_s: float = 0.75
    future_in_path_weight: float = 0.90
    minimum_lateral_uncertainty_px: float = 5.0
    bbox_uncertainty_fraction: float = 0.08
    lane_uncertainty_fraction: float = 0.12
    inferred_lane_uncertainty_scale: float = 1.35
    fixed_corridor_uncertainty_scale: float = 1.55
    coast_uncertainty_px_s: float = 18.0
    minimum_cut_in_speed_px_s: float = 12.0

    def __post_init__(self) -> None:
        fractions = {
            "minimum_lane_confidence": self.minimum_lane_confidence,
            "minimum_lane_width_fraction": self.minimum_lane_width_fraction,
            "maximum_lane_width_fraction": self.maximum_lane_width_fraction,
            "fallback_top_y_fraction": self.fallback_top_y_fraction,
            "fallback_top_half_width_fraction": self.fallback_top_half_width_fraction,
            "fallback_bottom_half_width_fraction": self.fallback_bottom_half_width_fraction,
            "fallback_confidence": self.fallback_confidence,
            "footprint_half_width_fraction": self.footprint_half_width_fraction,
            "future_in_path_weight": self.future_in_path_weight,
            "bbox_uncertainty_fraction": self.bbox_uncertainty_fraction,
            "lane_uncertainty_fraction": self.lane_uncertainty_fraction,
        }
        for name, value in fractions.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 < self.fallback_top_y_fraction < 1.0:
            raise ValueError("fallback_top_y_fraction must be strictly inside (0, 1)")
        if self.maximum_lane_width_fraction <= self.minimum_lane_width_fraction:
            raise ValueError("maximum_lane_width_fraction must exceed the minimum")
        if self.fallback_bottom_half_width_fraction <= self.fallback_top_half_width_fraction:
            raise ValueError("fixed corridor must widen towards the image bottom")
        if (
            not math.isfinite(self.prediction_horizon_s)
            or self.prediction_horizon_s <= 0.0
        ):
            raise ValueError("prediction_horizon_s must be positive")
        positive_values = {
            "minimum_lateral_uncertainty_px": self.minimum_lateral_uncertainty_px,
            "inferred_lane_uncertainty_scale": self.inferred_lane_uncertainty_scale,
            "fixed_corridor_uncertainty_scale": self.fixed_corridor_uncertainty_scale,
            "minimum_cut_in_speed_px_s": self.minimum_cut_in_speed_px_s,
        }
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in positive_values.values()
        ):
            names = ", ".join(positive_values)
            raise ValueError(f"{names} must be finite and positive")
        if (
            not math.isfinite(self.coast_uncertainty_px_s)
            or self.coast_uncertainty_px_s < 0.0
        ):
            raise ValueError("coast_uncertainty_px_s must be non-negative")


@dataclass(frozen=True, slots=True)
class EgoCorridor:
    """Two image-space boundaries represented as ``x = a*y + b``."""

    image_shape: tuple[int, int]
    source: CorridorSource
    confidence: float
    left_line: tuple[float, float]
    right_line: tuple[float, float]
    inferred_side: str | None = None

    def bounds_at(self, y_px: float) -> tuple[float, float]:
        """Return clipped left/right bounds at one image row."""

        height, width = self.image_shape
        y = float(np.clip(y_px, 0.0, max(0.0, height - 1.0)))
        left = self.left_line[0] * y + self.left_line[1]
        right = self.right_line[0] * y + self.right_line[1]
        left = float(np.clip(left, 0.0, float(width)))
        right = float(np.clip(right, 0.0, float(width)))
        if right < left:
            left, right = right, left
        return left, right


@dataclass(frozen=True, slots=True)
class CorridorEvidence:
    """Per-track causal path evidence consumed by the target selector."""

    footpoint: tuple[float, float]
    predicted_footpoint: tuple[float, float]
    current_bounds: tuple[float, float]
    predicted_bounds: tuple[float, float]
    corridor_overlap: float
    predicted_corridor_overlap: float
    in_path_probability: float
    cut_in_probability: float
    lateral_uncertainty_px: float
    track_confidence: float
    corridor_source: CorridorSource
    corridor_confidence: float


class CorridorEstimator:
    """Build a lane-backed corridor and score bbox footpoints against it."""

    def __init__(
        self,
        config: CorridorConfig | None = None,
        *,
        principal_x_px: float | None = None,
    ) -> None:
        self.config = config or CorridorConfig()
        if principal_x_px is not None and not math.isfinite(principal_x_px):
            raise ValueError("principal_x_px must be finite")
        self.principal_x_px = principal_x_px

    def estimate(
        self,
        lane: LaneEstimate | None,
        image_shape: tuple[int, int],
    ) -> EgoCorridor:
        """Use reliable lane boundaries, otherwise a calibrated fixed shape."""

        height, width = self._validated_shape(image_shape)
        if lane is not None:
            lane_corridor = self._from_lane(lane, (height, width))
            if lane_corridor is not None:
                return lane_corridor
        return self._fixed((height, width))

    def assess(
        self,
        corridor: EgoCorridor,
        bbox: BBox,
        *,
        velocity_px_s: tuple[float, float] = (0.0, 0.0),
        velocity_uncertainty_px_s: float = 0.0,
        coast_age_s: float = 0.0,
        track_confidence: float = 1.0,
        horizon_s: float | None = None,
    ) -> CorridorEvidence:
        """Score current and short-horizon bbox footpoints.

        A bbox contributes its bottom-centre contact point.  Its width is used
        only to form a small contact interval and an uncertainty term; the box
        centre is never used as a road position.
        """

        x1, y1, x2, y2 = (float(value) for value in bbox)
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError("bbox must contain finite coordinates")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must have positive width and height")
        if (
            not math.isfinite(velocity_uncertainty_px_s)
            or not math.isfinite(coast_age_s)
            or velocity_uncertainty_px_s < 0.0
            or coast_age_s < 0.0
        ):
            raise ValueError("velocity uncertainty and coast age must be non-negative")
        vx, vy = (float(value) for value in velocity_px_s)
        if not math.isfinite(vx) or not math.isfinite(vy):
            raise ValueError("velocity must be finite")
        if not math.isfinite(track_confidence):
            raise ValueError("track_confidence must be finite")

        horizon = self.config.prediction_horizon_s if horizon_s is None else horizon_s
        if not math.isfinite(horizon) or horizon < 0.0:
            raise ValueError("horizon_s must be finite and non-negative")

        height, _ = corridor.image_shape
        observed_foot_x = 0.5 * (x1 + x2)
        observed_foot_y = float(np.clip(y2, 0.0, max(0.0, height - 1.0)))
        # The tracker deliberately does not append synthetic observations on
        # detector-stride/coast frames.  Advance the scoring point from the
        # last real observation to the current timestamp using causal motion.
        foot_x = observed_foot_x + vx * coast_age_s
        foot_y = float(
            np.clip(
                observed_foot_y + vy * coast_age_s,
                0.0,
                max(0.0, height - 1.0),
            )
        )
        future_x = foot_x + vx * horizon
        future_y = float(np.clip(foot_y + vy * horizon, 0.0, max(0.0, height - 1.0)))
        current_bounds = corridor.bounds_at(foot_y)
        predicted_bounds = corridor.bounds_at(future_y)

        box_width = x2 - x1
        contact_half_width = max(
            1.0, self.config.footprint_half_width_fraction * box_width
        )
        current_overlap = self._interval_overlap(
            foot_x - contact_half_width,
            foot_x + contact_half_width,
            *current_bounds,
        )
        predicted_overlap = self._interval_overlap(
            future_x - contact_half_width,
            future_x + contact_half_width,
            *predicted_bounds,
        )

        corridor_width = max(1.0, current_bounds[1] - current_bounds[0])
        lane_uncertainty = (
            (1.0 - float(np.clip(corridor.confidence, 0.0, 1.0)))
            * self.config.lane_uncertainty_fraction
            * corridor_width
        )
        source_scale = (
            self.config.fixed_corridor_uncertainty_scale
            if corridor.source == "fixed"
            else (
                self.config.inferred_lane_uncertainty_scale
                if corridor.inferred_side is not None
                else 1.0
            )
        )
        lateral_uncertainty = source_scale * (
            self.config.minimum_lateral_uncertainty_px
            + self.config.bbox_uncertainty_fraction * box_width
            + lane_uncertainty
            + self.config.coast_uncertainty_px_s * coast_age_s
        )
        future_uncertainty = lateral_uncertainty + velocity_uncertainty_px_s * horizon
        current_probability = self._interval_probability(
            foot_x, lateral_uncertainty, *current_bounds
        )
        future_probability = self._interval_probability(
            future_x, future_uncertainty, *predicted_bounds
        )

        toward_speed = self._toward_corridor_speed(foot_x, current_bounds, vx)
        motion_support = float(
            np.clip(
                toward_speed / self.config.minimum_cut_in_speed_px_s,
                0.0,
                1.0,
            )
        )
        cut_in_probability = float(
            np.clip(
                (future_probability - current_probability)
                * (1.0 - current_probability)
                * motion_support,
                0.0,
                1.0,
            )
        )
        in_path_probability = float(
            np.clip(
                max(
                    current_probability,
                    self.config.future_in_path_weight * future_probability,
                ),
                0.0,
                1.0,
            )
        )

        return CorridorEvidence(
            footpoint=(foot_x, foot_y),
            predicted_footpoint=(future_x, future_y),
            current_bounds=current_bounds,
            predicted_bounds=predicted_bounds,
            corridor_overlap=current_overlap,
            predicted_corridor_overlap=predicted_overlap,
            in_path_probability=in_path_probability,
            cut_in_probability=cut_in_probability,
            lateral_uncertainty_px=float(lateral_uncertainty),
            track_confidence=float(np.clip(track_confidence, 0.0, 1.0)),
            corridor_source=corridor.source,
            corridor_confidence=float(np.clip(corridor.confidence, 0.0, 1.0)),
        )

    def _from_lane(
        self,
        lane: LaneEstimate,
        image_shape: tuple[int, int],
    ) -> EgoCorridor | None:
        height, width = image_shape
        if (
            not lane.valid
            or lane.confidence < self.config.minimum_lane_confidence
            or len(lane.left_points) < 2
            or len(lane.right_points) < 2
        ):
            return None
        left = self._fit_boundary(lane.left_points)
        right = self._fit_boundary(lane.right_points)
        if left is None or right is None:
            return None

        observed_top = max(
            min(point[1] for point in lane.left_points),
            min(point[1] for point in lane.right_points),
        )
        rows = (
            float(observed_top),
            0.70 * height,
            height - 1.0,
        )
        for y in rows:
            left_x = left[0] * y + left[1]
            right_x = right[0] * y + right[1]
            width_fraction = (right_x - left_x) / width
            if not (
                self.config.minimum_lane_width_fraction
                <= width_fraction
                <= self.config.maximum_lane_width_fraction
            ):
                return None
        return EgoCorridor(
            image_shape=image_shape,
            source="lane",
            confidence=float(np.clip(lane.confidence, 0.0, 1.0)),
            left_line=left,
            right_line=right,
            inferred_side=lane.inferred_side,
        )

    def _fixed(self, image_shape: tuple[int, int]) -> EgoCorridor:
        height, width = image_shape
        center = (
            float(self.principal_x_px)
            if self.principal_x_px is not None
            else 0.5 * width
        )
        center = float(np.clip(center, 0.0, float(width)))
        top_y = self.config.fallback_top_y_fraction * height
        bottom_y = height - 1.0
        top_half = self.config.fallback_top_half_width_fraction * width
        bottom_half = self.config.fallback_bottom_half_width_fraction * width
        left = self._line_through(
            (top_y, center - top_half), (bottom_y, center - bottom_half)
        )
        right = self._line_through(
            (top_y, center + top_half), (bottom_y, center + bottom_half)
        )
        return EgoCorridor(
            image_shape=image_shape,
            source="fixed",
            confidence=self.config.fallback_confidence,
            left_line=left,
            right_line=right,
        )

    @staticmethod
    def _fit_boundary(
        points: tuple[tuple[int, int], ...],
    ) -> tuple[float, float] | None:
        array = np.asarray(points, dtype=np.float64)
        xs = array[:, 0]
        ys = array[:, 1]
        if not np.isfinite(array).all() or float(np.ptp(ys)) < 1.0:
            return None
        design = np.column_stack((ys, np.ones_like(ys)))
        slope, intercept = np.linalg.lstsq(design, xs, rcond=None)[0]
        if not math.isfinite(float(slope)) or not math.isfinite(float(intercept)):
            return None
        return float(slope), float(intercept)

    @staticmethod
    def _line_through(
        first: tuple[float, float], second: tuple[float, float]
    ) -> tuple[float, float]:
        y1, x1 = first
        y2, x2 = second
        slope = (x2 - x1) / max(y2 - y1, 1e-6)
        return slope, x1 - slope * y1

    @staticmethod
    def _interval_overlap(
        first_left: float,
        first_right: float,
        second_left: float,
        second_right: float,
    ) -> float:
        width = max(first_right - first_left, 1e-6)
        intersection = max(
            0.0, min(first_right, second_right) - max(first_left, second_left)
        )
        return float(np.clip(intersection / width, 0.0, 1.0))

    @staticmethod
    def _interval_probability(
        mean: float, sigma: float, left: float, right: float
    ) -> float:
        sigma = max(float(sigma), 1e-6)
        scale = sigma * math.sqrt(2.0)
        probability = 0.5 * (
            math.erf((right - mean) / scale)
            - math.erf((left - mean) / scale)
        )
        return float(np.clip(probability, 0.0, 1.0))

    @staticmethod
    def _toward_corridor_speed(
        foot_x: float, bounds: tuple[float, float], velocity_x: float
    ) -> float:
        left, right = bounds
        if foot_x < left and velocity_x > 0.0:
            return velocity_x
        if foot_x > right and velocity_x < 0.0:
            return -velocity_x
        return 0.0

    @staticmethod
    def _validated_shape(image_shape: tuple[int, int]) -> tuple[int, int]:
        if len(image_shape) != 2:
            raise ValueError("image_shape must be (height, width)")
        height, width = image_shape
        if height <= 1 or width <= 1:
            raise ValueError("image_shape must be positive")
        return int(height), int(width)
