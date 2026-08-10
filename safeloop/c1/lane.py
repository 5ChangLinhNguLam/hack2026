"""Causal, single-camera lane geometry for the C1 edge pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class LaneConfig:
    roi_top_fraction: float = 0.53
    canny_low: int = 45
    canny_high: int = 145
    hough_threshold: int = 18
    min_line_length_px: int = 20
    max_line_gap_px: int = 32
    min_abs_slope_y_over_x: float = 0.35
    smoothing_alpha: float = 0.28
    max_coast_frames: int = 5
    default_lane_width_fraction: float = 0.46
    min_lane_width_fraction: float = 0.24
    max_lane_width_fraction: float = 0.90
    departure_offset_threshold: float = 0.55


@dataclass(frozen=True)
class LaneEstimate:
    valid: bool
    confidence: float
    left_points: tuple[tuple[int, int], ...]
    right_points: tuple[tuple[int, int], ...]
    lane_center_offset: float
    heading_error_deg: float
    lane_width_px: float
    departure_warning: bool
    inferred_side: str | None


@dataclass(frozen=True)
class LaneProxyReport:
    frame_count: int
    valid_fraction: float
    mean_confidence: float
    plausible_width_fraction: float
    center_jitter_px: float
    proxy_score: float
    metric_name: str = "temporal_geometry_proxy_not_lane_accuracy"


class LaneDetector:
    """White/yellow mask + Hough geometry with causal temporal smoothing."""

    def __init__(self, config: LaneConfig | None = None) -> None:
        self.config = config or LaneConfig()
        self._left: tuple[float, float] | None = None  # x = a*y + b
        self._right: tuple[float, float] | None = None
        self._left_missed = 0
        self._right_missed = 0

    def reset(self) -> None:
        self._left = self._right = None
        self._left_missed = self._right_missed = 0

    def detect(self, image_bgr: np.ndarray) -> LaneEstimate:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Lane detector cần ảnh BGR HxWx3")
        h, w = image_bgr.shape[:2]
        binary = self._binary_lane_pixels(image_bgr)
        roi = np.zeros_like(binary)
        top_y = int(h * self.config.roi_top_fraction)
        polygon = np.array([[
            (int(0.03 * w), h - 1),
            (int(0.38 * w), top_y),
            (int(0.62 * w), top_y),
            (int(0.97 * w), h - 1),
        ]], dtype=np.int32)
        cv2.fillPoly(roi, polygon, 255)
        masked = cv2.bitwise_and(binary, roi)
        lines = cv2.HoughLinesP(
            masked,
            1,
            np.pi / 180,
            self.config.hough_threshold,
            minLineLength=self.config.min_line_length_px,
            maxLineGap=self.config.max_line_gap_px,
        )
        left_lines: list[tuple[float, float, float]] = []
        right_lines: list[tuple[float, float, float]] = []
        if lines is not None:
            for raw in lines.reshape(-1, 4):
                x1, y1, x2, y2 = (float(value) for value in raw)
                dx, dy = x2 - x1, y2 - y1
                if abs(dx) < 1e-3 or abs(dy / dx) < self.config.min_abs_slope_y_over_x:
                    continue
                a = dx / dy
                b = x1 - a * y1
                length = math.hypot(dx, dy)
                center_x = (x1 + x2) / 2.0
                if a < 0 and center_x < 0.68 * w:
                    left_lines.append((a, b, length))
                elif a > 0 and center_x > 0.32 * w:
                    right_lines.append((a, b, length))

        new_left = self._robust_line(left_lines)
        new_right = self._robust_line(right_lines)
        self._left, self._left_missed = self._update_side(
            self._left, new_left, self._left_missed
        )
        self._right, self._right_missed = self._update_side(
            self._right, new_right, self._right_missed
        )

        inferred: str | None = None
        left, right = self._left, self._right
        default_width = self.config.default_lane_width_fraction * w
        if left is None and right is not None:
            left = (right[0], right[1] - default_width)
            inferred = "left"
        elif right is None and left is not None:
            right = (left[0], left[1] + default_width)
            inferred = "right"
        if left is None or right is None:
            return LaneEstimate(False, 0.0, (), (), 0.0, 0.0, 0.0, False, None)

        eval_y = 0.90 * h
        left_x = left[0] * eval_y + left[1]
        right_x = right[0] * eval_y + right[1]
        lane_width = right_x - left_x
        plausible = (
            self.config.min_lane_width_fraction * w
            <= lane_width
            <= self.config.max_lane_width_fraction * w
        )
        measured_sides = int(new_left is not None) + int(new_right is not None)
        line_support = min(1.0, (len(left_lines) + len(right_lines)) / 8.0)
        confidence = 0.30 * measured_sides + 0.35 * line_support + 0.25 * float(plausible)
        if inferred is not None:
            confidence *= 0.62
        confidence = min(1.0, confidence)

        lane_center = (left_x + right_x) / 2.0
        offset = (w / 2.0 - lane_center) / max(lane_width / 2.0, 1.0)
        center_slope = (left[0] + right[0]) / 2.0
        heading = math.degrees(math.atan(center_slope))
        y_values = np.linspace(top_y, h - 1, 12)
        left_points = tuple((int(left[0] * y + left[1]), int(y)) for y in y_values)
        right_points = tuple((int(right[0] * y + right[1]), int(y)) for y in y_values)
        valid = confidence >= 0.20 and plausible
        departure = (
            valid
            and confidence >= 0.40
            and abs(offset) >= self.config.departure_offset_threshold
        )
        return LaneEstimate(
            valid,
            confidence,
            left_points,
            right_points,
            float(offset),
            float(heading),
            float(lane_width),
            departure,
            inferred,
        )

    def _binary_lane_pixels(self, image: np.ndarray) -> np.ndarray:
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        hue, light, sat = cv2.split(hls)
        white = ((light >= 155) & (sat <= 125)).astype(np.uint8) * 255
        yellow = (
            (hue >= 10) & (hue <= 42) & (sat >= 55) & (light >= 65)
        ).astype(np.uint8) * 255
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, self.config.canny_low, self.config.canny_high)
        color = cv2.bitwise_or(white, yellow)
        color_edges = cv2.Canny(color, 40, 120)
        return cv2.bitwise_or(edges, color_edges)

    @staticmethod
    def _robust_line(lines: Sequence[tuple[float, float, float]]) -> tuple[float, float] | None:
        if not lines:
            return None
        slopes = np.asarray([item[0] for item in lines])
        intercepts = np.asarray([item[1] for item in lines])
        weights = np.asarray([item[2] for item in lines])
        median_slope = float(np.median(slopes))
        median_intercept = float(np.median(intercepts))
        slope_mad = float(np.median(np.abs(slopes - median_slope))) + 1e-3
        intercept_mad = float(np.median(np.abs(intercepts - median_intercept))) + 1.0
        keep = (
            (np.abs(slopes - median_slope) <= 2.5 * slope_mad)
            & (np.abs(intercepts - median_intercept) <= 2.5 * intercept_mad)
        )
        if not np.any(keep):
            keep[:] = True
        return (
            float(np.average(slopes[keep], weights=weights[keep])),
            float(np.average(intercepts[keep], weights=weights[keep])),
        )

    def _update_side(
        self,
        previous: tuple[float, float] | None,
        current: tuple[float, float] | None,
        missed: int,
    ) -> tuple[tuple[float, float] | None, int]:
        if current is None:
            missed += 1
            return (previous if missed <= self.config.max_coast_frames else None), missed
        if previous is None:
            return current, 0
        alpha = self.config.smoothing_alpha
        return (
            alpha * current[0] + (1.0 - alpha) * previous[0],
            alpha * current[1] + (1.0 - alpha) * previous[1],
        ), 0


class LaneSelfEvaluator:
    """Report temporal/geometry consistency when no true lane labels exist."""

    def __init__(self, image_width: int) -> None:
        self.image_width = image_width
        self.estimates: list[LaneEstimate] = []

    def add(self, estimate: LaneEstimate) -> None:
        self.estimates.append(estimate)

    def report(self) -> LaneProxyReport:
        count = len(self.estimates)
        if count == 0:
            return LaneProxyReport(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        valid = [item for item in self.estimates if item.valid]
        valid_fraction = len(valid) / count
        mean_confidence = float(np.mean([item.confidence for item in self.estimates]))
        plausible_fraction = sum(
            0.24 * self.image_width <= item.lane_width_px <= 0.90 * self.image_width
            for item in valid
        ) / max(1, len(valid))
        centers = np.asarray([
            self.image_width / 2.0 - item.lane_center_offset * item.lane_width_px / 2.0
            for item in valid
        ])
        jitter = float(np.median(np.abs(np.diff(centers)))) if len(centers) >= 2 else 0.0
        stability = math.exp(-jitter / 20.0)
        score = 100.0 * (
            0.40 * valid_fraction
            + 0.30 * mean_confidence
            + 0.20 * plausible_fraction
            + 0.10 * stability
        )
        return LaneProxyReport(
            count,
            round(valid_fraction, 4),
            round(mean_confidence, 4),
            round(plausible_fraction, 4),
            round(jitter, 3),
            round(score, 1),
        )


def render_lane(image: np.ndarray, estimate: LaneEstimate) -> np.ndarray:
    panel = image.copy()
    if estimate.left_points:
        cv2.polylines(panel, [np.asarray(estimate.left_points)], False, (0, 255, 255), 3)
    if estimate.right_points:
        cv2.polylines(panel, [np.asarray(estimate.right_points)], False, (0, 255, 255), 3)
    color = (0, 0, 255) if estimate.departure_warning else (0, 255, 0)
    cv2.putText(
        panel,
        f"LANE conf={estimate.confidence:.2f} offset={estimate.lane_center_offset:+.2f}",
        (10, panel.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        2,
        cv2.LINE_AA,
    )
    return panel
